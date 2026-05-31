import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import json

base_dir = Path("/home/chest_ct/code")
ct_path = base_dir / "data/segmentations/segmentations/train_13492_b_2.nii.gz"
npz_path = base_dir / "models/merlin/results_full/train_13492_b_2/localization_masks.npz"
gt_path = base_dir / "data/segmentations/segmentations/train_13492_b_2.nii.gz"

# Load CT
ct_img = nib.load(str(ct_path))
ct_data = ct_img.get_fdata(dtype=np.float32)

if ct_data.ndim == 4:
    ct_data = np.mean(ct_data, axis=0)

ct_data = np.squeeze(ct_data)

# Normalize CT for display
ct_min, ct_max = -1000, 400
ct_disp = np.clip(ct_data, ct_min, ct_max)
ct_disp = (ct_disp - ct_min) / (ct_max - ct_min)

# Load Pred
pred_npz = np.load(str(npz_path))
# Get the first class
cls_name = list(pred_npz.files)[0]
pred_mask = pred_npz[cls_name]

# If pred_mask is ZYX and ct_data is XYZ
if pred_mask.ndim == 3 and pred_mask.shape != ct_data.shape:
    if np.transpose(pred_mask, (2, 0, 1)).shape == ct_data.shape:
        pred_mask = np.transpose(pred_mask, (2, 0, 1))

# Load GT
gt_img = nib.load(str(gt_path))
gt_data = gt_img.get_fdata(dtype=np.float32)
gt_data = np.squeeze(gt_data)

# For unified masks, gt_data might contain multiple classes, let's just make it binary
gt_data = (gt_data > 0).astype(np.float32)

# Find a good slice
z_slices = np.where(gt_data.sum(axis=(0, 1)) > 0)[0]
if len(z_slices) > 0:
    target_z = int(np.median(z_slices))
else:
    target_z = ct_data.shape[2] // 2

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(ct_disp[:, :, target_z].T, cmap='gray', origin='lower')
axes[0].set_title('CT Scan')

axes[1].imshow(ct_disp[:, :, target_z].T, cmap='gray', origin='lower')
axes[1].imshow(gt_data[:, :, target_z].T, cmap='Reds', alpha=0.5, origin='lower')
axes[1].set_title('Ground Truth Abnormalities')

axes[2].imshow(ct_disp[:, :, target_z].T, cmap='gray', origin='lower')
axes[2].imshow(pred_mask[:, :, target_z].T, cmap='hot', alpha=0.5, origin='lower')
axes[2].set_title(f'Merlin Prediction: {cls_name}')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.savefig(base_dir / "localization_demo.png")
print("Saved localization_demo.png")
