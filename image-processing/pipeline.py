import os
import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage


def load_nifti_file(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    img = nib.load(filepath)
    data = img.get_fdata().astype(np.float64)

    return data, img.affine, img.header


def get_voxel_spacing(header):
    return tuple(float(v) for v in header.get_zooms()[:3])


def apply_hu_window(ct_data, window_level=-600, window_width=1500):
    min_hu = window_level - window_width / 2
    max_hu = window_level + window_width / 2

    windowed = np.clip(ct_data, min_hu, max_hu)
    windowed = (windowed - min_hu) / (max_hu - min_hu)

    return windowed


def resample_volume(volume, current_spacing, target_spacing=(1.0, 1.0, 1.0), order=1):
    current_spacing = np.asarray(current_spacing, dtype=float)
    target_spacing = np.asarray(target_spacing, dtype=float)

    zoom_factors = current_spacing / target_spacing

    print("Original shape :", volume.shape)
    print("Current spacing:", current_spacing)
    print("Target spacing :", target_spacing)
    print("Zoom factors   :", zoom_factors)

    resampled = ndimage.zoom(volume, zoom_factors, order=order)

    print("New shape      :", resampled.shape)

    return resampled


def spacing_to_sigma_voxels(spacing, sigma_mm):
    spacing = np.asarray(spacing, dtype=float)
    sigma_mm = np.asarray(sigma_mm, dtype=float)

    if sigma_mm.ndim == 0:
        sigma_mm = np.repeat(sigma_mm, 3)

    sigma_voxels = sigma_mm / np.maximum(spacing, 1e-6)

    return tuple(float(v) for v in sigma_voxels)


def min_volume_to_voxels(min_volume_mm3, spacing):
    spacing = np.asarray(spacing, dtype=float)
    voxel_volume_mm3 = float(np.prod(spacing))

    return max(1, int(round(float(min_volume_mm3) / max(voxel_volume_mm3, 1e-6))))


def largest_connected_components(mask, num_components=2):
    labels, count = ndimage.label(mask)

    if count == 0:
        return mask.astype(bool)

    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    largest_indices = np.argsort(sizes)[-num_components:]

    output = np.zeros_like(mask, dtype=bool)

    for idx in largest_indices:
        output |= labels == (idx + 1)

    return output


def remove_small_components(mask, min_size=300):
    labels, count = ndimage.label(mask)

    if count == 0:
        return mask.astype(bool)

    sizes = ndimage.sum(mask, labels, range(1, count + 1))

    output = np.zeros_like(mask, dtype=bool)

    for label_id, size in enumerate(sizes, start=1):
        if size >= min_size:
            output |= labels == label_id

    return output


def get_body_mask(ct_data, body_threshold=-700):
    body = ct_data > body_threshold

    body = ndimage.binary_closing(body, iterations=2)
    body = ndimage.binary_fill_holes(body)

    body = largest_connected_components(body, num_components=1)

    return body.astype(bool)


def get_lung_mask(ct_data, lung_threshold=-400):
    body_mask = get_body_mask(ct_data)

    lung = (ct_data < lung_threshold) & body_mask

    lung = ndimage.binary_opening(lung, iterations=1)
    lung = ndimage.binary_closing(lung, iterations=2)
    lung = ndimage.binary_fill_holes(lung)

    lung = largest_connected_components(lung, num_components=2)

    return lung.astype(bool)


def get_case_adaptive_threshold(smoothed_ct, lung_mask, percentile, lower_bound, upper_bound):
    lung_values = smoothed_ct[lung_mask]

    if lung_values.size == 0:
        return lower_bound

    adaptive_threshold = float(np.percentile(lung_values, percentile))
    adaptive_threshold = max(float(lower_bound), adaptive_threshold)
    adaptive_threshold = min(float(upper_bound), adaptive_threshold)

    return adaptive_threshold


def detect_ggo(
    ct_data,
    lung_mask,
    spacing=(1.0, 1.0, 1.0),
    lower_hu=-750,
    upper_hu=-250,
    min_size=300,
    smoothing_sigma_mm=1.5,
    adaptive_percentile=92
):
    sigma_voxels = spacing_to_sigma_voxels(spacing, smoothing_sigma_mm)
    smoothed_ct = ndimage.gaussian_filter(ct_data, sigma=sigma_voxels)
    adaptive_lower_hu = get_case_adaptive_threshold(
        smoothed_ct,
        lung_mask,
        percentile=adaptive_percentile,
        lower_bound=lower_hu,
        upper_bound=upper_hu
    )

    ggo_candidates = (
        (smoothed_ct >= adaptive_lower_hu) &
        (smoothed_ct <= upper_hu) &
        lung_mask
    )

    ggo_candidates = ndimage.binary_opening(ggo_candidates, iterations=1)
    ggo_candidates = ndimage.binary_closing(ggo_candidates, iterations=1)
    ggo_candidates = remove_small_components(ggo_candidates, min_size=min_size)

    return ggo_candidates.astype(bool)


def detect_lung_opacity(
    ct_data,
    lung_mask,
    spacing=(1.0, 1.0, 1.0),
    lower_hu=-700,
    upper_hu=100,
    min_size=300,
    smoothing_sigma_mm=2.0,
    adaptive_percentile=95
):
    sigma_voxels = spacing_to_sigma_voxels(spacing, smoothing_sigma_mm)
    smoothed_ct = ndimage.gaussian_filter(ct_data, sigma=sigma_voxels)
    adaptive_lower_hu = get_case_adaptive_threshold(
        smoothed_ct,
        lung_mask,
        percentile=adaptive_percentile,
        lower_bound=lower_hu,
        upper_bound=upper_hu
    )

    opacity_candidates = (
        (smoothed_ct >= adaptive_lower_hu) &
        (smoothed_ct <= upper_hu) &
        lung_mask
    )

    opacity_candidates = ndimage.binary_opening(opacity_candidates, iterations=1)
    opacity_candidates = ndimage.binary_closing(opacity_candidates, iterations=1)
    opacity_candidates = remove_small_components(opacity_candidates, min_size=min_size)

    return opacity_candidates.astype(bool)


def load_ggo_gt_mask(seg_path, ggo_channels):
    seg_data, affine, header = load_nifti_file(seg_path)
    seg_data = np.asarray(seg_data)

    if seg_data.ndim == 3:
        return seg_data.astype(bool), affine, header

    if seg_data.ndim != 4:
        raise ValueError(f"Unsupported segmentation shape: {seg_data.shape}")

    ggo_channels = [int(ch) for ch in ggo_channels]

    if max(ggo_channels) < seg_data.shape[0]:
        channel_first = seg_data
    elif max(ggo_channels) < seg_data.shape[-1]:
        channel_first = np.moveaxis(seg_data, -1, 0)
    else:
        raise ValueError(
            f"Cannot map channels {ggo_channels} onto segmentation shape {seg_data.shape}"
        )

    gt_mask = np.zeros(channel_first.shape[1:], dtype=bool)

    for ch in ggo_channels:
        gt_mask |= channel_first[ch].astype(bool)

    return gt_mask, affine, header


def compute_dice(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()

    if total == 0:
        return 1.0

    return 2.0 * intersection / total


def compute_iou(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 1.0

    return intersection / union


def compute_sensitivity_precision(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    sensitivity = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)

    return sensitivity, precision


def save_nifti(mask, affine, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    img = nib.Nifti1Image(mask.astype(np.uint8), affine)
    nib.save(img, output_path)
