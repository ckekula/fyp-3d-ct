import json
from pathlib import Path

from pipeline_abnormal import (
    load_nifti_file,
    load_target_gt_mask,
    get_voxel_spacing,
    get_analysis_mask,
    detect_abnormal_lung_regions,
    compute_dice,
    compute_iou,
    compute_sensitivity_precision,
    compute_gt_coverage,
)


DATASET_JSON = Path(
    r"D:\My\Projects\fyp-3d-ct\data\rexgrounding-ct\dataset_img_processing.json"
)

SPLIT = "train"
START_INDEX = 0
MAX_CASES = 5

LUNG_THRESHOLD = -400
ANALYSIS_UPPER_HU = 200
ANALYSIS_DILATION_ITERATIONS = 4
ANALYSIS_CLOSING_ITERATIONS = 2

LOWER_HU = -800
UPPER_HU = 100
ADAPTIVE_PERCENTILE = 88
SMOOTHING_SIGMA_MM = 1.0
MIN_LESION_VOLUME_MM3 = 100.0
USE_VESSEL_SUPPRESSION = True
VESSEL_HU = -200


def load_samples():
    with open(DATASET_JSON, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    samples = dataset[SPLIT]
    if not samples:
        raise ValueError(f"No samples found in split: {SPLIT}")

    if START_INDEX < 0 or START_INDEX >= len(samples):
        raise IndexError(
            f"START_INDEX {START_INDEX} is out of range for split {SPLIT} "
            f"with {len(samples)} samples."
        )

    end_index = min(len(samples), START_INDEX + MAX_CASES)
    return list(enumerate(samples[START_INDEX:end_index], start=START_INDEX))


def evaluate_case(case_index, sample):
    name = sample["name"]
    ct_path = sample["ct_path"]
    seg_path = sample["seg_path"]
    target_channels = sample.get("ggo_channels", [])

    if not target_channels:
        raise ValueError(
            f"Sample {name} does not define any target channels in `ggo_channels`."
        )

    print("=== Selected Case ===")
    print("Name:", name)
    print("Split:", SPLIT)
    print("Case index:", case_index)
    print("CT path:", ct_path)
    print("SEG path:", seg_path)
    print("Target channels:", target_channels)

    print("\nLoading CT...")
    ct_data, ct_affine, ct_header = load_nifti_file(ct_path)

    print("Loading GT...")
    gt_mask, _, _ = load_target_gt_mask(seg_path, target_channels)

    spacing = get_voxel_spacing(ct_header)
    print("CT shape:", ct_data.shape)
    print("GT shape:", gt_mask.shape)
    print("Spacing:", spacing)

    print("\nBuilding analysis mask...")
    analysis_mask = get_analysis_mask(
        ct_data,
        lung_threshold=LUNG_THRESHOLD,
        upper_hu=ANALYSIS_UPPER_HU,
        dilation_iterations=ANALYSIS_DILATION_ITERATIONS,
        closing_iterations=ANALYSIS_CLOSING_ITERATIONS,
    )

    gt_inside_analysis = compute_gt_coverage(gt_mask, analysis_mask)
    print("GT inside analysis mask ratio:", f"{gt_inside_analysis:.4f}")

    print("\nDetecting abnormal lung regions...")
    pred_mask, debug = detect_abnormal_lung_regions(
        ct_data,
        analysis_mask,
        spacing=spacing,
        lower_hu=LOWER_HU,
        upper_hu=UPPER_HU,
        adaptive_percentile=ADAPTIVE_PERCENTILE,
        smoothing_sigma_mm=SMOOTHING_SIGMA_MM,
        min_lesion_volume_mm3=MIN_LESION_VOLUME_MM3,
        use_vessel_suppression=USE_VESSEL_SUPPRESSION,
        vessel_hu=VESSEL_HU,
    )

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
    print("Analysis voxels :", int(analysis_mask.sum()))

    print("\nDebug info:")
    for key, value in debug.items():
        print(f"{key}: {value}")

    return {
        "dice": dice,
        "iou": iou,
        "sensitivity": sensitivity,
        "precision": precision,
        "gt_inside_analysis": gt_inside_analysis,
    }


def main():
    selected_cases = load_samples()

    print("=== Run Configuration ===")
    print("Split:", SPLIT)
    print("Start index:", START_INDEX)
    print("Cases selected:", len(selected_cases))

    metrics = []

    for display_index, (case_index, sample) in enumerate(selected_cases, start=1):
        print("\n" + "=" * 80)
        print(f"Evaluating case {display_index}/{len(selected_cases)}")
        metrics.append(evaluate_case(case_index, sample))

    if metrics:
        mean_dice = sum(m["dice"] for m in metrics) / len(metrics)
        mean_iou = sum(m["iou"] for m in metrics) / len(metrics)
        mean_sensitivity = sum(m["sensitivity"] for m in metrics) / len(metrics)
        mean_precision = sum(m["precision"] for m in metrics) / len(metrics)
        mean_gt_inside_analysis = (
            sum(m["gt_inside_analysis"] for m in metrics) / len(metrics)
        )

        print("\n" + "=" * 80)
        print("=== Aggregate Results ===")
        print(f"Mean Dice               : {mean_dice:.4f}")
        print(f"Mean IoU                : {mean_iou:.4f}")
        print(f"Mean Sensitivity        : {mean_sensitivity:.4f}")
        print(f"Mean Precision          : {mean_precision:.4f}")
        print(f"Mean GT Inside Analysis : {mean_gt_inside_analysis:.4f}")


if __name__ == "__main__":
    main()
