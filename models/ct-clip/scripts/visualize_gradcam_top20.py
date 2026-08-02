import sys
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rexground_pilot_pipeline import preprocess_ct, TRAIN_META, VALID_META, SEG_DIR, ROOT, TARGET_SPACING, TARGET_SHAPE
from data import resize_array
import torch

OUT_DIR = ROOT / "rexground_predictions/full"
VIZ_DIR = OUT_DIR / "visualizations"
VIZ_DIR.mkdir(parents=True, exist_ok=True)


def aligned_gt_mask(name, crop_info, meta_row):
    """Reproduce the same resample -> crop -> pad pipeline used for the
    image, applied to the segmentation mask, so it lines up with the CAM."""
    seg_nii = nib.load(str(SEG_DIR / name))
    seg = seg_nii.get_fdata()
    if seg.ndim == 4:
        seg = seg[0]
    seg_mask = seg > 0

    seg_zxy = np.transpose(seg_mask.astype(np.float32), (2, 0, 1))
    seg_t = torch.tensor(seg_zxy).unsqueeze(0).unsqueeze(0)
    xy_spacing = float(meta_row["XYSpacing"][1:][:-2].split(",")[0])
    z_spacing = float(meta_row["ZSpacing"])
    seg_resized = resize_array(seg_t, (z_spacing, xy_spacing, xy_spacing), TARGET_SPACING)[0][0]
    seg_resized = np.transpose(seg_resized, (1, 2, 0))  # X,Y,Z

    dh, dw, dd = TARGET_SHAPE
    hs, ws, ds = crop_info["h_start"], crop_info["w_start"], crop_info["d_start"]
    seg_cropped = seg_resized[hs:hs + dh, ws:ws + dw, ds:ds + dd]
    seg_padded = np.pad(
        seg_cropped,
        ((crop_info["pad_h_before"], dh - seg_cropped.shape[0] - crop_info["pad_h_before"]),
         (crop_info["pad_w_before"], dw - seg_cropped.shape[1] - crop_info["pad_w_before"]),
         (crop_info["pad_d_before"], dd - seg_cropped.shape[2] - crop_info["pad_d_before"])),
        mode="constant", constant_values=0,
    )
    return seg_padded > 0.5  # (X, Y, Z)


def best_dice_slice(cam_xyz, gt_mask, best_pct):
    """Pick the z-slice with the highest per-slice Dice at the CAM's own
    best threshold, restricted to slices that actually contain GT voxels."""
    thr = np.percentile(cam_xyz, best_pct)
    cam_mask = cam_xyz >= thr

    z_candidates = np.where(gt_mask.sum(axis=(0, 1)) > 0)[0]
    if len(z_candidates) == 0:
        return int(np.argmax(gt_mask.sum(axis=(0, 1)))), 0.0

    best_z, best_slice_dice = z_candidates[0], -1.0
    for z in z_candidates:
        inter = np.logical_and(cam_mask[:, :, z], gt_mask[:, :, z]).sum()
        d = 2 * inter / (cam_mask[:, :, z].sum() + gt_mask[:, :, z].sum() + 1e-8)
        if d > best_slice_dice:
            best_slice_dice, best_z = d, z
    return int(best_z), float(best_slice_dice)


def main():
    summary = pd.read_csv(OUT_DIR / "gradcam_summary_top20.csv")
    train_meta = pd.read_csv(TRAIN_META).set_index("VolumeName")
    valid_meta = pd.read_csv(VALID_META).set_index("VolumeName")

    from rexground_pilot_pipeline import VOLUME_DIR

    for _, row in summary.iterrows():
        name = row["name"]
        target = row["gradcam_target"]
        best_pct = row["gradcam_best_pct"]
        overall_dice = row["gradcam_best_dice"]

        meta_row = train_meta.loc[name] if name in train_meta.index else valid_meta.loc[name]
        image_tensor, crop_info = preprocess_ct(VOLUME_DIR / name, meta_row)

        img_zxy = image_tensor.squeeze().numpy()          # (Z, X, Y)
        img_xyz = np.transpose(img_zxy, (1, 2, 0))         # (X, Y, Z)

        cam_zxy = np.load(row["gradcam_npy"])
        cam_xyz = np.transpose(cam_zxy, (1, 2, 0))         # (X, Y, Z)

        gt_mask = aligned_gt_mask(name, crop_info, meta_row)

        z, slice_dice = best_dice_slice(cam_xyz, gt_mask, best_pct)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(img_xyz[:, :, z], cmap="gray")
        axes[0].set_title(f"{name}\nCT slice {z}")
        axes[0].axis("off")

        axes[1].imshow(img_xyz[:, :, z], cmap="gray")
        axes[1].imshow(
            np.ma.masked_where(gt_mask[:, :, z] == 0, gt_mask[:, :, z]),
            cmap="Reds", alpha=0.6,
        )
        axes[1].set_title("Ground truth (segmentation)")
        axes[1].axis("off")

        axes[2].imshow(img_xyz[:, :, z], cmap="gray")
        axes[2].imshow(cam_xyz[:, :, z], cmap="jet", alpha=0.5, vmin=0, vmax=1)
        axes[2].set_title(
            f"CT-CLIP Grad-CAM ({target})\nslice Dice={slice_dice:.4f}  |  case-best Dice={overall_dice:.4f}"
        )
        axes[2].axis("off")

        plt.tight_layout()
        out_path = VIZ_DIR / f"{target}_{name.replace('.nii.gz', '')}_slice{z}.png"
        plt.savefig(out_path, dpi=110)
        plt.close(fig)
        print(f"saved {out_path.name}  (slice_dice={slice_dice:.4f}, pointing_hit={row['gradcam_pointing_hit']})")


if __name__ == "__main__":
    main()
