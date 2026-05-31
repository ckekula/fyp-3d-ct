import argparse
import os

import matplotlib.pyplot as plt

from pipeline import (
    load_nifti_file,
    load_nifti_with_meta,
    normalize_binary_mask,
    apply_hu_window,
    get_lung_mask,
    detect_ggo,
    suppress_vessels,
    compute_dice,
    compute_jaccard,
    get_voxel_spacing,
    resample_scan,
)
from viewer import CTViewer


def parse_args():
    parser = argparse.ArgumentParser(description="Run a CT lung segmentation and GGO detection demo.")
    parser.add_argument("--ct", type=str, required=False,
                        default=r"D:\My\Projects\fyp-3d-ct\data\data_volumes\train_6\train_6_a\train_6_a_2.nii.gz",
                        help="Path to the chest CT NIfTI file.")
    parser.add_argument("--gt", type=str, required=False,
                        default=r"D:\My\Projects\fyp-3d-ct\data\segmentations\train_6_a_2.nii.gz",
                        help="Path to the ground truth mask NIfTI file (optional).")
    parser.add_argument("--resample", type=float, default=None,
                        help="Target isotropic voxel spacing in millimeters. Example: 1.0")
    parser.add_argument("--lung-threshold", type=int, default=-400,
                        help="HU threshold for lung segmentation.")
    parser.add_argument("--ggo-range", type=int, nargs=2, default=[-800, -300],
                        help="HU range for GGO detection.")
    parser.add_argument("--suppress-vessels", action="store_true",
                        help="Apply morphological vessel suppression to reduce false positives.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.ct):
        raise FileNotFoundError(f"CT scan not found: {args.ct}")

    print("Loading CT scan...")
    ct_data, affine, header = load_nifti_with_meta(args.ct)

    if args.resample is not None:
        print(f"Resampling scan to {args.resample}mm isotropic spacing...")
        current_spacing = get_voxel_spacing(header)
        ct_data = resample_scan(ct_data, current_spacing, target_spacing=(args.resample,) * 3)

    print("Applying lung HU windowing for visualization...")
    ct_windowed = apply_hu_window(ct_data, window_level=-600, window_width=1500)

    print("Generating lung mask...")
    lung_mask = get_lung_mask(ct_data, threshold=args.lung_threshold)

    print("Detecting ground glass opacities (GGOs)...")
    ggo_mask = detect_ggo(ct_data, lung_mask, lower_hu=args.ggo_range[0], upper_hu=args.ggo_range[1])

    if args.suppress_vessels:
        print("Suppressing vessel-like false positives...")
        ggo_mask = suppress_vessels(ggo_mask, iterations=2)

    gt_mask = None
    if args.gt and os.path.exists(args.gt):
        print("Loading ground truth mask...")
        gt_mask = load_nifti_file(args.gt)
        if args.resample is not None:
            gt_mask = resample_scan(gt_mask.astype(np.float32), current_spacing, target_spacing=(args.resample,) * 3, order=0)
        gt_mask = normalize_binary_mask(gt_mask, reference_shape=ct_data.shape)

    if gt_mask is not None:
        print("Evaluating detection quality...")
        dice = compute_dice(ggo_mask, gt_mask)
        jaccard = compute_jaccard(ggo_mask, gt_mask)
        print(f"Dice score: {dice:.4f}")
        print(f"Jaccard index: {jaccard:.4f}")

    print("Launching interactive CT viewer...")
    viewer = CTViewer(ct_windowed, pred_mask=ggo_mask, gt_mask=gt_mask)
    plt.show()


if __name__ == "__main__":
    main()
