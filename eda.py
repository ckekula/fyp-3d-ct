import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

demo_ct_path = "/home/chest_ct/code/data/data_volumes/dataset/train_fixed/train_12991/train_12991_a/train_12991_a_1.nii.gz"
demo_segmentation_path = "/home/chest_ct/code/data/segmentations/segmentations/train_12991_a_1.nii.gz"

# Load data
ct_data = nib.load(demo_ct_path).get_fdata()
seg_data = nib.load(demo_segmentation_path).get_fdata()

print("CT shape:", ct_data.shape)
print("SEG shape:", seg_data.shape)

# Select a finding/channel
finding_idx = 0
seg_mask = seg_data[finding_idx]

print("Selected mask shape:", seg_mask.shape)

# Find slices containing the segmentation
seg_slices = np.where(seg_mask.sum(axis=(0, 1)) > 0)[0]

if len(seg_slices) == 0:
    raise ValueError("No segmentation found for this finding")

# Middle slice containing lesion
middle_slice = seg_slices[len(seg_slices) // 2]

# Plot
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(ct_data[:, :, middle_slice], cmap="gray")
plt.title("CT Scan")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(ct_data[:, :, middle_slice], cmap="gray")
plt.imshow(seg_mask[:, :, middle_slice], cmap="Reds", alpha=0.5)
plt.title(f"Overlay - Finding {finding_idx}")
plt.axis("off")

plt.tight_layout()
plt.show()