import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.segmentation import clear_border
from tqdm import tqdm

from lc_ksvd.config import HU_MIN, HU_MAX
from lc_ksvd.data_loader import LabelRegistry, MetadataRegistry, load_volume, resample_volume

print(">>> SCRIPT STARTED")
# -------------------------------------------------------
# Lung mask (your method)
# -------------------------------------------------------
def build_lung_mask(vol):
    mask = (vol >= HU_MIN) & (vol <= HU_MAX)

    mask_clear = np.zeros_like(mask, dtype=bool)
    for z in range(mask.shape[2]):
        mask_clear[:, :, z] = clear_border(mask[:, :, z])

    labels, num = ndimage.label(mask_clear)

    if num > 0:
        sizes = ndimage.sum(mask_clear, labels, range(1, num + 1))

        if len(sizes) >= 2:
            largest_two = np.argsort(sizes)[-2:] + 1
            lung_mask = np.isin(labels, largest_two)
        else:
            lung_mask = labels > 0
    else:
        lung_mask = mask_clear

    structure = ndimage.generate_binary_structure(3, 2)
    lung_mask = ndimage.binary_closing(lung_mask, structure=structure, iterations=2)
    lung_mask = ndimage.binary_fill_holes(lung_mask)

    return lung_mask


# -------------------------------------------------------
# Init
# -------------------------------------------------------
metadata = MetadataRegistry(split=None)
labels = LabelRegistry(metadata, split=None)

abnormalities = ["2b", "2c", "2d"]

output_dir = Path("/home/chest_ct/code/models/lc-ksvd/eda_outputs/npz_voxels")
output_dir.mkdir(parents=True, exist_ok=True)

bins = np.arange(-1000, 251, 10)


# -------------------------------------------------------
# MAIN LOOP
# -------------------------------------------------------
for category in abnormalities:

    print(f"\nProcessing category: {category}")

    volume_names = labels.get_positive_volume_names(category)

    all_lung_voxels = []

    # ---------------------------
    # Collect voxels
    # ---------------------------
    for volume_name in tqdm(volume_names, desc=f"Processing {category}", unit="vol"):

        try:
            vol_hu, spacing = load_volume(volume_name)
            vol_rs = resample_volume(vol_hu, spacing)

            lung_mask = build_lung_mask(vol_rs)
            lung_voxels = vol_rs[lung_mask].astype(np.float32)

            if lung_voxels.size > 0:
                all_lung_voxels.append(lung_voxels)

        except Exception as e:
            print(f"Skipped {volume_name}: {e}")

    if not all_lung_voxels:
        print(f"No data for {category}")
        continue

    # ---------------------------
    # Merge voxels
    # ---------------------------
    all_lung_voxels = np.concatenate(all_lung_voxels).astype(np.float32)

    print(f"{category}: total voxels = {all_lung_voxels.size:,}")

    # ---------------------------
    # Save npz
    # ---------------------------
    save_path = output_dir / f"{category}_lung_voxels.npz"

    np.savez_compressed(save_path, voxels=all_lung_voxels)

    print(f"Saved: {save_path}")

    # ---------------------------
    # Plot distribution
    # ---------------------------
    plt.figure(figsize=(10, 5))

    hist, _ = np.histogram(all_lung_voxels, bins=bins, density=True)

    plt.plot(bins[:-1], hist, linewidth=2, label=f"{category}")

    plt.axvline(all_lung_voxels.mean(), color="red", linestyle="--",
                label=f"mean = {all_lung_voxels.mean():.1f}")

    plt.axvline(np.median(all_lung_voxels), color="blue", linestyle="--",
                label=f"median = {np.median(all_lung_voxels):.1f}")

    plt.title(f"HU Distribution (Lung Voxels) - {category}")
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()