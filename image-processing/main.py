import os
import json
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from pipeline import (
    load_nifti_file,
    get_voxel_spacing,
    apply_hu_window,
    resample_volume,
    get_lung_mask,
    get_analysis_mask,
    detect_ggo,
    detect_lung_opacity,
    min_volume_to_voxels,
    load_ggo_gt_mask,
    compute_dice,
    compute_iou,
    compute_sensitivity_precision,
    compute_case_hit,
    compute_slice_hit_rate,
    compute_zone_hits,
    compute_centroid_distance_mm,
    compute_volume_ratio,
    save_nifti,
)

from viewer import CTViewer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_JSON = PROJECT_ROOT / "data" / "rexgrounding-ct" / "dataset_img_processing.json"
OUTPUT_DIR = PROJECT_ROOT / "image-processing" / "outputs"

SPLIT = "train"
START_INDEX = 0
MAX_CASES = 5

VIEW_RESULT = True
SAVE_MASK = True
SAVE_CSV = True
OVERWRITE_CSV = True

RESAMPLE_MM = None

LUNG_THRESHOLD = -400
ANALYSIS_PERMISSIVE_THRESHOLD = -250
ANALYSIS_UPPER_HU = 100
ANALYSIS_DILATION_ITERATIONS = 2
ANALYSIS_CLOSING_ITERATIONS = 2
DETECTION_MODE = "ggo"
LOWER_HU = -750
UPPER_HU = -250
SMOOTHING_SIGMA_MM = 1.0
ADAPTIVE_PERCENTILE = 88
MIN_LESION_VOLUME_MM3 = 200.0


def print_case_info(sample):
    print("\n=== Selected Case ===")
    print("Name:", sample["name"])
    print("CT path:", sample["ct_path"])
    print("SEG path:", sample["seg_path"])
    print("GGO channels:", sample["ggo_channels"])

    print("\nFindings:")
    for ch in sample["ggo_channels"]:
        print(f"  Channel {ch}: {sample['findings'][str(ch)]}")


def save_metrics_csv(output_path, row):
    file_exists = os.path.exists(output_path)

    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def run_detection(ct_data, lung_mask, spacing):
    min_component_voxels = min_volume_to_voxels(
        MIN_LESION_VOLUME_MM3,
        spacing=spacing
    )

    if DETECTION_MODE == "ggo":
        pred_mask = detect_ggo(
            ct_data,
            lung_mask,
            spacing=spacing,
            lower_hu=LOWER_HU,
            upper_hu=UPPER_HU,
            min_size=min_component_voxels,
            smoothing_sigma_mm=SMOOTHING_SIGMA_MM,
            adaptive_percentile=ADAPTIVE_PERCENTILE
        )
    elif DETECTION_MODE == "opacity":
        pred_mask = detect_lung_opacity(
            ct_data,
            lung_mask,
            spacing=spacing,
            lower_hu=LOWER_HU,
            upper_hu=UPPER_HU,
            min_size=min_component_voxels,
            smoothing_sigma_mm=SMOOTHING_SIGMA_MM,
            adaptive_percentile=ADAPTIVE_PERCENTILE
        )
    else:
        raise ValueError(f"Unsupported DETECTION_MODE: {DETECTION_MODE}")

    return pred_mask, min_component_voxels


def evaluate_case(sample, case_index):
    print_case_info(sample)

    name = sample["name"]
    ct_path = sample["ct_path"]
    seg_path = sample["seg_path"]
    ggo_channels = sample["ggo_channels"]

    print("\nLoading CT volume...")
    ct_data, ct_affine, ct_header = load_nifti_file(ct_path)

    print("Loading GGO ground truth mask...")
    gt_mask, _, _ = load_ggo_gt_mask(seg_path, ggo_channels)

    print("\nCT shape:", ct_data.shape)
    print("GT shape:", gt_mask.shape)

    original_spacing = get_voxel_spacing(ct_header)
    print("Original spacing:", original_spacing)
    working_spacing = original_spacing

    if RESAMPLE_MM is not None:
        working_spacing = (RESAMPLE_MM, RESAMPLE_MM, RESAMPLE_MM)

        print("\nResampling CT...")
        ct_data = resample_volume(
            ct_data,
            current_spacing=original_spacing,
            target_spacing=working_spacing,
            order=1
        )

        print("\nResampling GT mask...")
        gt_mask = resample_volume(
            gt_mask.astype("float32"),
            current_spacing=original_spacing,
            target_spacing=working_spacing,
            order=0
        ).astype(bool)

    if ct_data.shape != gt_mask.shape:
        raise ValueError(f"Shape mismatch: CT {ct_data.shape}, GT {gt_mask.shape}")

    print("\nGenerating lung mask...")
    lung_mask = get_lung_mask(
        ct_data,
        lung_threshold=LUNG_THRESHOLD
    )

    print("Generating analysis mask...")
    analysis_mask = get_analysis_mask(
        ct_data,
        lung_threshold=LUNG_THRESHOLD,
        permissive_threshold=ANALYSIS_PERMISSIVE_THRESHOLD,
        upper_hu=ANALYSIS_UPPER_HU,
        dilation_iterations=ANALYSIS_DILATION_ITERATIONS,
        closing_iterations=ANALYSIS_CLOSING_ITERATIONS
    )

    gt_inside_lung_ratio = float((gt_mask & lung_mask).sum() / (gt_mask.sum() + 1e-8))
    gt_inside_analysis_ratio = float(
        (gt_mask & analysis_mask).sum() / (gt_mask.sum() + 1e-8)
    )

    print("GT inside lung mask ratio    :", f"{gt_inside_lung_ratio:.4f}")
    print("GT inside analysis mask ratio:", f"{gt_inside_analysis_ratio:.4f}")

    print(f"Detecting {DETECTION_MODE} candidates...")
    pred_mask, min_component_voxels = run_detection(
        ct_data,
        analysis_mask,
        spacing=working_spacing
    )

    print("\nComputing evaluation metrics...")
    dice = compute_dice(pred_mask, gt_mask)
    iou = compute_iou(pred_mask, gt_mask)
    sensitivity, precision = compute_sensitivity_precision(pred_mask, gt_mask)
    case_hit = compute_case_hit(pred_mask, gt_mask)
    slice_hit_rate = compute_slice_hit_rate(pred_mask, gt_mask)
    zone_hits = compute_zone_hits(pred_mask, gt_mask, analysis_mask)
    centroid_distance_mm = compute_centroid_distance_mm(
        pred_mask,
        gt_mask,
        spacing=working_spacing
    )
    volume_ratio = compute_volume_ratio(pred_mask, gt_mask)

    print("\n=== Evaluation Results ===")
    print(f"Dice        : {dice:.4f}")
    print(f"IoU         : {iou:.4f}")
    print(f"Sensitivity : {sensitivity:.4f}")
    print(f"Precision   : {precision:.4f}")
    print(f"Case hit    : {case_hit}")
    print(f"Slice hit   : {slice_hit_rate:.4f}")
    print(
        "Zone hits   :",
        {
            k: v for k, v in zone_hits.items()
        }
    )
    print(
        "Centroid mm :",
        "NA" if centroid_distance_mm is None else f"{centroid_distance_mm:.2f}"
    )
    print(
        "Vol ratio   :",
        "NA" if volume_ratio is None else f"{volume_ratio:.4f}"
    )

    print("\nVoxel counts:")
    print("GT voxels       :", int(gt_mask.sum()))
    print("Predicted voxels:", int(pred_mask.sum()))
    print("Lung voxels     :", int(lung_mask.sum()))
    print("Analysis voxels :", int(analysis_mask.sum()))

    if SAVE_MASK:
        pred_output_path = os.path.join(
            OUTPUT_DIR,
            name.replace(".nii.gz", "_pred_ggo.nii.gz")
        )

        save_nifti(pred_mask, ct_affine, pred_output_path)
        print("\nSaved predicted mask:", pred_output_path)

    if SAVE_CSV:
        csv_path = os.path.join(OUTPUT_DIR, "results.csv")

        row = {
            "name": name,
            "split": SPLIT,
            "index": case_index,
            "ggo_channels": str(ggo_channels),
            "detection_mode": DETECTION_MODE,
            "lung_threshold": LUNG_THRESHOLD,
            "analysis_permissive_threshold": ANALYSIS_PERMISSIVE_THRESHOLD,
            "analysis_upper_hu": ANALYSIS_UPPER_HU,
            "analysis_dilation_iterations": ANALYSIS_DILATION_ITERATIONS,
            "analysis_closing_iterations": ANALYSIS_CLOSING_ITERATIONS,
            "lower_hu": LOWER_HU,
            "upper_hu": UPPER_HU,
            "smoothing_sigma_mm": SMOOTHING_SIGMA_MM,
            "adaptive_percentile": ADAPTIVE_PERCENTILE,
            "min_lesion_volume_mm3": MIN_LESION_VOLUME_MM3,
            "min_component_voxels": min_component_voxels,
            "dice": dice,
            "iou": iou,
            "sensitivity": sensitivity,
            "precision": precision,
            "case_hit": case_hit,
            "slice_hit_rate": slice_hit_rate,
            "left_hit": zone_hits["left_hit"],
            "right_hit": zone_hits["right_hit"],
            "upper_hit": zone_hits["upper_hit"],
            "mid_hit": zone_hits["mid_hit"],
            "lower_hit": zone_hits["lower_hit"],
            "centroid_distance_mm": centroid_distance_mm,
            "pred_to_gt_volume_ratio": volume_ratio,
            "gt_voxels": int(gt_mask.sum()),
            "pred_voxels": int(pred_mask.sum()),
            "lung_voxels": int(lung_mask.sum()),
            "analysis_voxels": int(analysis_mask.sum()),
            "gt_inside_lung_ratio": gt_inside_lung_ratio,
            "gt_inside_analysis_ratio": gt_inside_analysis_ratio,
        }

        save_metrics_csv(csv_path, row)
        print("Saved metrics:", csv_path)

    if VIEW_RESULT:
        print("\nOpening viewer...")
        ct_windowed = apply_hu_window(ct_data)

        viewer = CTViewer(
            ct_windowed,
            pred_mask=pred_mask,
            gt_mask=gt_mask
        )

        plt.show()

    return {
        "dice": dice,
        "iou": iou,
        "sensitivity": sensitivity,
        "precision": precision,
        "case_hit": case_hit,
        "slice_hit_rate": slice_hit_rate,
        "left_hit": zone_hits["left_hit"],
        "right_hit": zone_hits["right_hit"],
        "upper_hit": zone_hits["upper_hit"],
        "mid_hit": zone_hits["mid_hit"],
        "lower_hit": zone_hits["lower_hit"],
        "centroid_distance_mm": centroid_distance_mm,
        "pred_to_gt_volume_ratio": volume_ratio,
        "gt_voxels": int(gt_mask.sum()),
        "pred_voxels": int(pred_mask.sum()),
        "lung_voxels": int(lung_mask.sum()),
        "analysis_voxels": int(analysis_mask.sum()),
        "gt_inside_lung_ratio": gt_inside_lung_ratio,
        "gt_inside_analysis_ratio": gt_inside_analysis_ratio,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(DATASET_JSON, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    samples = dataset[SPLIT]

    if len(samples) == 0:
        raise ValueError(f"No samples found in split: {SPLIT}")

    end_index = min(len(samples), START_INDEX + MAX_CASES)
    selected_cases = list(enumerate(samples[START_INDEX:end_index], start=START_INDEX))

    if SAVE_CSV and OVERWRITE_CSV:
        csv_path = os.path.join(OUTPUT_DIR, "results.csv")
        if os.path.exists(csv_path):
            os.remove(csv_path)

    metrics = []

    print("\n=== Run Configuration ===")
    print("Split:", SPLIT)
    print("Cases:", len(selected_cases))
    print("Detection mode:", DETECTION_MODE)
    print("Analysis permissive threshold:", ANALYSIS_PERMISSIVE_THRESHOLD)
    print("Analysis upper HU:", ANALYSIS_UPPER_HU)
    print("Analysis dilation iterations:", ANALYSIS_DILATION_ITERATIONS)
    print("HU range:", (LOWER_HU, UPPER_HU))
    print("Adaptive percentile:", ADAPTIVE_PERCENTILE)
    print("Smoothing sigma (mm):", SMOOTHING_SIGMA_MM)
    print("Min lesion volume (mm^3):", MIN_LESION_VOLUME_MM3)

    for case_index, sample in selected_cases:
        print("\n" + "=" * 80)
        print(f"Evaluating case {case_index + 1}/{len(samples)}")
        metrics.append(evaluate_case(sample, case_index))

    if metrics:
        mean_dice = sum(m["dice"] for m in metrics) / len(metrics)
        mean_iou = sum(m["iou"] for m in metrics) / len(metrics)
        mean_sensitivity = sum(m["sensitivity"] for m in metrics) / len(metrics)
        mean_precision = sum(m["precision"] for m in metrics) / len(metrics)
        mean_case_hit = sum(m["case_hit"] for m in metrics) / len(metrics)
        mean_slice_hit = sum(m["slice_hit_rate"] for m in metrics) / len(metrics)

        print("\n=== Aggregate Results ===")
        print(f"Mean Dice        : {mean_dice:.4f}")
        print(f"Mean IoU         : {mean_iou:.4f}")
        print(f"Mean Sensitivity : {mean_sensitivity:.4f}")
        print(f"Mean Precision   : {mean_precision:.4f}")
        print(f"Mean Case Hit    : {mean_case_hit:.4f}")
        print(f"Mean Slice Hit   : {mean_slice_hit:.4f}")


if __name__ == "__main__":
    main()
