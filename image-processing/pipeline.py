import os

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage


def load_nifti_file(filepath):
    """Loads a NIfTI file and returns its image data as a numpy array."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    scan = nib.load(filepath)
    return scan.get_fdata()


def load_nifti_with_meta(filepath):
    """Loads a NIfTI file and returns the image data, affine, and header."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    scan = nib.load(filepath)
    return scan.get_fdata(), scan.affine, scan.header


def save_nifti_file(data, affine, output_path):
    """Saves a numpy array back to a NIfTI file."""
    image = nib.Nifti1Image(data.astype(np.float32), affine)
    nib.save(image, output_path)


def get_voxel_spacing(header):
    """Extracts the voxel spacing from a NIfTI header."""
    spacing = header.get_zooms()[:3]
    return tuple(float(s) for s in spacing)


def normalize_binary_mask(mask, reference_shape=None):
    """Converts a segmentation mask to 3D binary format and validates its shape."""
    mask = np.asarray(mask)
    if mask.ndim == 4 and mask.shape[0] <= 8:
        mask = np.any(mask != 0, axis=0)
    elif mask.ndim == 4 and mask.shape[-1] == 1:
        mask = np.squeeze(mask, axis=-1)

    if mask.ndim != 3:
        raise ValueError(
            f"Mask must be 3D after normalization, got shape {mask.shape}. "
            "Provide a 3D binary mask or a 4D channel-first mask."
        )

    if reference_shape is not None and tuple(mask.shape) != tuple(reference_shape):
        raise ValueError(
            f"Mask shape {mask.shape} does not match reference shape {reference_shape}."
        )

    return mask.astype(bool)


def resample_scan(scan_data, current_spacing, target_spacing=(1.0, 1.0, 1.0), order=1):
    """Resamples a 3D scan to isotropic voxel spacing."""
    current_spacing = np.asarray(current_spacing, dtype=float)
    target_spacing = np.asarray(target_spacing, dtype=float)
    zoom_factors = current_spacing / target_spacing
    return ndimage.zoom(scan_data, zoom_factors, order=order)


def apply_hu_window(scan_data, window_level=-600, window_width=1500):
    """Applies HU windowing and normalizes image values to [0, 1]."""
    min_value = window_level - (window_width / 2)
    max_value = window_level + (window_width / 2)
    windowed_data = np.clip(scan_data, min_value, max_value)
    normalized_data = (windowed_data - min_value) / (max_value - min_value)
    return normalized_data


def largest_connected_components(mask, num_components=2):
    """Keeps the largest connected components from a binary mask."""
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask

    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    num_components = max(1, num_components)
    largest_indices = np.argsort(sizes)[-num_components:]

    selected = np.zeros_like(mask, dtype=bool)
    for idx in largest_indices:
        selected |= labels == (idx + 1)

    return selected


def get_lung_mask(scan_data, threshold=-400, keep_components=2):
    """Generates a lung mask using thresholding and morphological cleanup."""
    binary_mask = scan_data < threshold
    cleaned_mask = ndimage.binary_opening(binary_mask, iterations=2)
    cleaned_mask = ndimage.binary_closing(cleaned_mask, iterations=2)
    cleaned_mask = ndimage.binary_fill_holes(cleaned_mask)
    cleaned_mask = largest_connected_components(cleaned_mask, num_components=keep_components)
    return cleaned_mask.astype(bool)


def detect_ggo(scan_data, lung_mask, lower_hu=-800, upper_hu=-300):
    """Detects Ground Glass Opacities inside the lung region."""
    ggo_threshold_mask = (scan_data >= lower_hu) & (scan_data <= upper_hu)
    ggo_candidates = ggo_threshold_mask & lung_mask
    cleaned_ggo = ndimage.binary_opening(ggo_candidates, iterations=1)
    cleaned_ggo = ndimage.binary_closing(cleaned_ggo, iterations=1)
    return cleaned_ggo.astype(bool)


def suppress_vessels(mask, iterations=2):
    """Applies morphological opening to suppress small tubular vessel-like structures."""
    struct = ndimage.generate_binary_structure(3, 1)
    return ndimage.binary_opening(mask, structure=struct, iterations=iterations).astype(bool)


def compute_dice(pred_mask, gt_mask):
    """Computes the Dice similarity coefficient between two binary masks."""
    pred = np.asarray(pred_mask, dtype=bool)
    gt = np.asarray(gt_mask, dtype=bool)
    intersection = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    if total == 0:
        return 1.0
    return 2.0 * intersection / total


def compute_jaccard(pred_mask, gt_mask):
    """Computes the Jaccard index between two binary masks."""
    pred = np.asarray(pred_mask, dtype=bool)
    gt = np.asarray(gt_mask, dtype=bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return intersection / union


def display_slice(slice_data, title="CT Slice", cmap="gray"):
    plt.figure(figsize=(8, 8))
    plt.imshow(slice_data, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()


def display_overlay(scan_slice, mask_slice, title="Overlay", alpha=0.5):
    plt.figure(figsize=(8, 8))
    plt.imshow(scan_slice, cmap="gray")
    plt.imshow(np.ma.masked_where(mask_slice == 0, mask_slice), cmap="Reds", alpha=alpha)
    plt.title(title)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    print("Pipeline module loaded successfully.")
