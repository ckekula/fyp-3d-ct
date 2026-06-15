import os
import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage


# ============================================================
# 1. Basic NIfTI I/O
# ============================================================

def load_nifti_file(filepath):
    """
    Load a NIfTI CT or segmentation file.

    Returns:
        data   : image array
        affine : spatial transform matrix
        header : NIfTI metadata
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    img = nib.load(filepath)
    data = img.get_fdata().astype(np.float32)

    return data, img.affine, img.header


def save_nifti(mask, affine, output_path):
    """
    Save a binary mask as a NIfTI file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img = nib.Nifti1Image(mask.astype(np.uint8), affine)
    nib.save(img, output_path)


def get_voxel_spacing(header):
    """
    Get voxel spacing in mm.
    Example: (0.71, 0.71, 0.75)
    """
    return tuple(float(v) for v in header.get_zooms()[:3])


# ============================================================
# 2. Visualization
# ============================================================

def apply_hu_window(ct_data, window_level=-600, window_width=1500):
    """
    Convert CT HU values to [0, 1] for display.
    This is only for visualization, not detection.
    """
    min_hu = window_level - window_width / 2
    max_hu = window_level + window_width / 2

    windowed = np.clip(ct_data, min_hu, max_hu)
    windowed = (windowed - min_hu) / (max_hu - min_hu)

    return windowed


# ============================================================
# 3. Spacing-aware helpers
# ============================================================

def spacing_to_sigma_voxels(spacing, sigma_mm):
    """
    Convert Gaussian smoothing sigma from millimeters to voxel units.

    This keeps smoothing physically consistent across CT scans
    with different voxel sizes.
    """
    spacing = np.asarray(spacing, dtype=float)
    sigma_mm = np.asarray(sigma_mm, dtype=float)

    if sigma_mm.ndim == 0:
        sigma_mm = np.repeat(sigma_mm, 3)

    sigma_voxels = sigma_mm / np.maximum(spacing, 1e-6)

    return tuple(float(v) for v in sigma_voxels)


def min_volume_to_voxels(min_volume_mm3, spacing):
    """
    Convert a physical lesion volume in mm^3 to voxel count.
    """
    spacing = np.asarray(spacing, dtype=float)
    voxel_volume = float(np.prod(spacing))

    return max(1, int(round(float(min_volume_mm3) / max(voxel_volume, 1e-6))))


# ============================================================
# 4. Morphology helpers
# ============================================================

def largest_connected_components(mask, num_components=2):
    """
    Keep the largest connected components from a binary mask.
    For lung segmentation, usually keep 2 components: left and right lung.
    """
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
    """
    Remove connected components smaller than min_size voxels.
    """
    labels, count = ndimage.label(mask)

    if count == 0:
        return mask.astype(bool)

    sizes = ndimage.sum(mask, labels, range(1, count + 1))

    output = np.zeros_like(mask, dtype=bool)

    for label_id, size in enumerate(sizes, start=1):
        if size >= min_size:
            output |= labels == label_id

    return output


# ============================================================
# 5. Body and lung masks
# ============================================================

def get_body_mask(ct_data, body_threshold=-700):
    """
    Extract patient body to remove outside air.
    Outside air is usually around -1000 HU.
    """
    body = ct_data > body_threshold

    body = ndimage.binary_closing(body, iterations=2)
    body = ndimage.binary_fill_holes(body)
    body = largest_connected_components(body, num_components=1)

    return body.astype(bool)


def get_lung_mask(ct_data, lung_threshold=-400):
    """
    Strict lung mask from low-HU air-like lung regions.

    This is useful for finding the approximate lung area,
    but it may miss dense abnormal lung regions.
    """
    body_mask = get_body_mask(ct_data)

    lung = (ct_data < lung_threshold) & body_mask

    lung = ndimage.binary_opening(lung, iterations=1)
    lung = ndimage.binary_closing(lung, iterations=2)
    lung = ndimage.binary_fill_holes(lung)
    lung = largest_connected_components(lung, num_components=2)

    return lung.astype(bool)


def get_analysis_mask(
    ct_data,
    lung_threshold=-400,
    upper_hu=200,
    dilation_iterations=4,
    closing_iterations=2
):
    """
    Expanded lung analysis mask.

    Why this exists:
        GGO and abnormal opacity may be denser than normal lung.
        A strict lung mask may exclude abnormal regions.
        This mask expands the lung area slightly and allows denser voxels.

    Returns:
        analysis_mask: region where abnormal lung regions are searched.
    """
    body_mask = get_body_mask(ct_data)
    lung_mask = get_lung_mask(ct_data, lung_threshold=lung_threshold)

    analysis_mask = ndimage.binary_dilation(
        lung_mask,
        iterations=dilation_iterations
    )

    analysis_mask &= body_mask
    analysis_mask &= ct_data < upper_hu

    analysis_mask = ndimage.binary_closing(
        analysis_mask,
        iterations=closing_iterations
    )
    analysis_mask = ndimage.binary_fill_holes(analysis_mask)

    analysis_mask = largest_connected_components(
        analysis_mask,
        num_components=2
    )

    return analysis_mask.astype(bool)


# ============================================================
# 6. GT mask loading
# ============================================================

def load_target_gt_mask(seg_path, target_channels):
    """
    Load ReXGroundingCT segmentation channels and merge selected channels.

    target_channels:
        Example [0] or [0, 1]

    Works with:
        channel-first masks: (C, X, Y, Z)
        channel-last masks : (X, Y, Z, C)
    """
    seg_data, affine, header = load_nifti_file(seg_path)
    seg_data = np.asarray(seg_data)

    if seg_data.ndim == 3:
        return seg_data.astype(bool), affine, header

    if seg_data.ndim != 4:
        raise ValueError(f"Unsupported segmentation shape: {seg_data.shape}")

    target_channels = [int(ch) for ch in target_channels]
    max_ch = max(target_channels)

    # Safer channel-axis detection.
    # ReXGroundingCT usually has small channel count, not 512.
    if seg_data.shape[0] <= 32 and max_ch < seg_data.shape[0]:
        channel_first = seg_data

    elif seg_data.shape[-1] <= 32 and max_ch < seg_data.shape[-1]:
        channel_first = np.moveaxis(seg_data, -1, 0)

    else:
        raise ValueError(
            f"Cannot identify channel axis for segmentation shape {seg_data.shape} "
            f"and channels {target_channels}"
        )

    gt_mask = np.zeros(channel_first.shape[1:], dtype=bool)

    for ch in target_channels:
        gt_mask |= channel_first[ch].astype(bool)

    return gt_mask, affine, header


def load_ggo_gt_mask(seg_path, ggo_channels):
    """
    Compatibility wrapper for your current GGO evaluation.
    """
    return load_target_gt_mask(seg_path, ggo_channels)


# ============================================================
# 7. Patient-specific abnormal lung detection
# ============================================================

def compute_lung_hu_statistics(ct_data, analysis_mask):
    """
    Compute HU statistics inside the analysis lung region.
    """
    values = ct_data[analysis_mask]
    values = values[np.isfinite(values)]

    if values.size == 0:
        raise ValueError("Analysis mask is empty. Cannot compute HU statistics.")

    stats = {
        "min": float(np.min(values)),
        "p1": float(np.percentile(values, 1)),
        "p5": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p85": float(np.percentile(values, 85)),
        "p90": float(np.percentile(values, 90)),
        "p92": float(np.percentile(values, 92)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }

    return stats


def get_adaptive_abnormal_threshold(
    ct_data,
    analysis_mask,
    percentile=88,
    lower_bound=-800,
    upper_bound=-200
):
    """
    Patient-specific abnormality threshold.

    Instead of saying fixed HU range only, this chooses a dense-lung threshold
    from the current scan's lung HU distribution.
    """
    values = ct_data[analysis_mask]
    values = values[np.isfinite(values)]

    if values.size == 0:
        return lower_bound

    threshold = float(np.percentile(values, percentile))

    threshold = max(float(lower_bound), threshold)
    threshold = min(float(upper_bound), threshold)

    return threshold


def suppress_high_density_vessels(
    pred_mask,
    ct_data,
    vessel_hu=-200,
    dilation_iterations=1
):
    """
    Simple high-HU vessel/bronchial wall suppression.

    This removes very dense structures from abnormal candidates.
    Use carefully, because true opacity near vessels can also be removed.
    """
    vessel_mask = ct_data > vessel_hu
    vessel_mask = ndimage.binary_dilation(
        vessel_mask,
        iterations=dilation_iterations
    )

    cleaned = pred_mask & ~vessel_mask

    return cleaned.astype(bool)


def detect_abnormal_lung_regions(
    ct_data,
    analysis_mask,
    spacing=(1.0, 1.0, 1.0),
    lower_hu=-800,
    upper_hu=100,
    adaptive_percentile=88,
    smoothing_sigma_mm=1.0,
    min_lesion_volume_mm3=100.0,
    use_vessel_suppression=True,
    vessel_hu=-200
):
    """
    Main abnormal lung region detector.

    This detects abnormal dense lung regions rather than claiming
    every detected region is pure GGO.

    Steps:
        1. Smooth CT using spacing-aware Gaussian filter.
        2. Compute patient-specific adaptive threshold.
        3. Select dense abnormal voxels inside analysis lung mask.
        4. Morphology cleanup.
        5. Remove tiny components using physical mm^3 volume.
        6. Optional high-density vessel suppression.
    """
    sigma_voxels = spacing_to_sigma_voxels(spacing, smoothing_sigma_mm)
    smoothed_ct = ndimage.gaussian_filter(ct_data, sigma=sigma_voxels)

    adaptive_threshold = get_adaptive_abnormal_threshold(
        smoothed_ct,
        analysis_mask,
        percentile=adaptive_percentile,
        lower_bound=lower_hu,
        upper_bound=upper_hu
    )

    abnormal_mask = (
        (smoothed_ct >= adaptive_threshold) &
        (smoothed_ct <= upper_hu) &
        analysis_mask
    )

    before_cleanup_voxels = int(abnormal_mask.sum())

    abnormal_mask = ndimage.binary_opening(abnormal_mask, iterations=1)
    abnormal_mask = ndimage.binary_closing(abnormal_mask, iterations=1)

    min_component_voxels = min_volume_to_voxels(
        min_lesion_volume_mm3,
        spacing=spacing
    )

    abnormal_mask = remove_small_components(
        abnormal_mask,
        min_size=min_component_voxels
    )

    if use_vessel_suppression:
        abnormal_mask = suppress_high_density_vessels(
            abnormal_mask,
            smoothed_ct,
            vessel_hu=vessel_hu,
            dilation_iterations=1
        )

        abnormal_mask = remove_small_components(
            abnormal_mask,
            min_size=min_component_voxels
        )

    debug_info = {
        "sigma_voxels": sigma_voxels,
        "adaptive_threshold": adaptive_threshold,
        "before_cleanup_voxels": before_cleanup_voxels,
        "after_cleanup_voxels": int(abnormal_mask.sum()),
        "min_component_voxels": min_component_voxels,
    }

    return abnormal_mask.astype(bool), debug_info


def detect_ggo_focused_abnormal_regions(
    ct_data,
    analysis_mask,
    spacing=(1.0, 1.0, 1.0),
    lower_hu=-800,
    upper_hu=-200,
    adaptive_percentile=85,
    smoothing_sigma_mm=1.0,
    min_lesion_volume_mm3=50.0
):
    """
    GGO-focused abnormal region detector.

    This is still image-processing based, but it should be reported as:
        'GGO-focused abnormal opacity localization'
    not guaranteed exact GGO segmentation.
    """
    pred_mask, debug_info = detect_abnormal_lung_regions(
        ct_data=ct_data,
        analysis_mask=analysis_mask,
        spacing=spacing,
        lower_hu=lower_hu,
        upper_hu=upper_hu,
        adaptive_percentile=adaptive_percentile,
        smoothing_sigma_mm=smoothing_sigma_mm,
        min_lesion_volume_mm3=min_lesion_volume_mm3,
        use_vessel_suppression=False,
        vessel_hu=-200
    )

    return pred_mask, debug_info


# ============================================================
# 8. Evaluation
# ============================================================

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


def compute_gt_coverage(mask, region_mask):
    """
    How much GT is inside a candidate search region.
    """
    mask = mask.astype(bool)
    region_mask = region_mask.astype(bool)

    return float((mask & region_mask).sum() / (mask.sum() + 1e-8))
