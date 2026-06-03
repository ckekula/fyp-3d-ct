import os
import json
import csv

import matplotlib.pyplot as plt

from pipeline import (
    load_nifti_file,
    get_voxel_spacing,
    apply_hu_window,
    resample_volume,
    get_lung_mask,
    detect_ggo,
    detect_lung_opacity,
    min_volume_to_voxels,
    load_ggo_gt_mask,
    compute_dice,
    compute_iou,
    compute_sensitivity_precision,
    save_nifti,
)

from viewer import CTViewer


DATASET_JSON = r"D:\My\Projects\fyp-3d-ct\data\rexgrounding-ct\dataset_img_processing.json"
OUTPUT_DIR = r"D:\My\Projects\fyp-3d-ct\image-processing\outputs"

SPLIT = "train"
START_INDEX = 0
MAX_CASES = 5

VIEW_RESULT = True
SAVE_MASK = True
SAVE_CSV = True
OVERWRITE_CSV = True

RESAMPLE_MM = None

LUNG_THRESHOLD = -400
DETECTION_MODE = "ggo"
LOWER_HU = -700
UPPER_HU = -250
SMOOTHING_SIGMA_MM = 1.5
ADAPTIVE_PERCENTILE = 92
MIN_LESION_VOLUME_MM3 = 500.0


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

    print(f"Detecting {DETECTION_MODE} candidates...")
    pred_mask, min_component_voxels = run_detection(
        ct_data,
        lung_mask,
        spacing=working_spacing
    )

    print("\nComputing evaluation metrics...")
    dice = compute_dice(pred_mask, gt_mask)
    iou = compute_iou(pred_mask, gt_mask)
    sensitivity, precision = compute_sensitivity_precision(pred_mask, gt_mask)

    print("\n=== Evaluation Results ===")
    print(f"Dice        : {dice:.4f}")
    print(f"IoU         : {iou:.4f}")
    print(f"Sensitivity : {sensitivity:.4f}")
    print(f"Precision   : {precision:.4f}")

    print("\nVoxel counts:")
    print("GT voxels       :", int(gt_mask.sum()))
    print("Predicted voxels:", int(pred_mask.sum()))
    print("Lung voxels     :", int(lung_mask.sum()))

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
            "gt_voxels": int(gt_mask.sum()),
            "pred_voxels": int(pred_mask.sum()),
            "lung_voxels": int(lung_mask.sum()),
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
        "gt_voxels": int(gt_mask.sum()),
        "pred_voxels": int(pred_mask.sum()),
        "lung_voxels": int(lung_mask.sum()),
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

        print("\n=== Aggregate Results ===")
        print(f"Mean Dice        : {mean_dice:.4f}")
        print(f"Mean IoU         : {mean_iou:.4f}")
        print(f"Mean Sensitivity : {mean_sensitivity:.4f}")
        print(f"Mean Precision   : {mean_precision:.4f}")


if __name__ == "__main__":
    main()
