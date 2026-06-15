import os
import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage


def load_nifti_file(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    img = nib.load(filepath)
    data = img.get_fdata().astype(np.float32)

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


def build_lung_seed(ct_data, body_mask, threshold):
    
    seed = (ct_data < threshold) & body_mask

    seed = ndimage.binary_opening(seed, iterations=1)
    seed = ndimage.binary_closing(seed, iterations=2)
    seed = ndimage.binary_fill_holes(seed)
    seed = largest_connected_components(seed, num_components=2)

    return seed.astype(bool)


def get_analysis_mask(
    ct_data,
    lung_threshold=-400,
    permissive_threshold=-250,
    upper_hu=100,
    dilation_iterations=2,
    closing_iterations=2,
    min_component_size=0,
    fallback_threshold=100,
    min_seed_ratio=0.01
):
    body_mask = get_body_mask(ct_data)
    strict_lung_mask = build_lung_seed(
        ct_data,
        body_mask=body_mask,
        threshold=lung_threshold
    )
    permissive_lung_mask = build_lung_seed(
        ct_data,
        body_mask=body_mask,
        threshold=permissive_threshold
    )

    analysis_seed = strict_lung_mask | permissive_lung_mask
    seed_ratio = float(analysis_seed.sum()) / max(float(body_mask.sum()), 1.0)

    if seed_ratio < min_seed_ratio:
        fallback_seed = (ct_data < fallback_threshold) & body_mask
        fallback_seed = ndimage.binary_opening(fallback_seed, iterations=1)
        fallback_seed = ndimage.binary_closing(fallback_seed, iterations=2)
        fallback_seed = ndimage.binary_fill_holes(fallback_seed)
        fallback_seed = largest_connected_components(fallback_seed, num_components=2)
        analysis_seed = fallback_seed.astype(bool)

    analysis_mask = ndimage.binary_dilation(
        analysis_seed,
        iterations=dilation_iterations
    )
    analysis_mask &= body_mask
    analysis_mask &= ct_data < upper_hu

    analysis_mask = ndimage.binary_closing(
        analysis_mask,
        iterations=closing_iterations
    )
    analysis_mask = ndimage.binary_fill_holes(analysis_mask)

    if min_component_size > 0:
        analysis_mask = remove_small_components(
            analysis_mask,
            min_size=min_component_size
        )

    analysis_mask = largest_connected_components(analysis_mask, num_components=2)

    return analysis_mask.astype(bool)


def get_case_adaptive_threshold(
    smoothed_ct,
    lung_mask,
    percentile,
    lower_bound,
    upper_bound,
    prefilter_lower_hu=None,
    prefilter_upper_hu=None
):
    lung_values = smoothed_ct[lung_mask.astype(bool)]

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
    adaptive_percentile=92,
    min_band_width_hu=75
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
    adaptive_lower_hu = min(
        float(adaptive_lower_hu),
        float(upper_hu) - float(min_band_width_hu)
    )
    adaptive_lower_hu = max(float(lower_hu), float(adaptive_lower_hu))

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
        unique_values = np.unique(seg_data)
        if unique_values.size > 2:
            print(
                "Warning: 3D segmentation has more than two unique values:",
                unique_values[:10]
            )
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


def compute_case_hit(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    return int(np.logical_and(pred, gt).any())


def compute_slice_hit_rate(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    gt_slices = gt.sum(axis=(0, 1)) > 0
    pred_slices = pred.sum(axis=(0, 1)) > 0

    gt_count = int(gt_slices.sum())
    if gt_count == 0:
        return 1.0

    hit_count = int(np.logical_and(gt_slices, pred_slices).sum())

    return hit_count / gt_count


def get_lung_side_masks(lung_mask):
    lung_mask = lung_mask.astype(bool)
    labels, count = ndimage.label(lung_mask)

    if count >= 2:
        sizes = ndimage.sum(lung_mask, labels, range(1, count + 1))
        keep = np.argsort(sizes)[-2:] + 1
        component_masks = []
        for label_id in keep:
            component = labels == label_id
            coords = np.argwhere(component)
            x_center = float(coords[:, 0].mean()) if coords.size else 0.0
            component_masks.append((x_center, component))

        component_masks.sort(key=lambda item: item[0])
        left_mask = component_masks[0][1]
        right_mask = component_masks[1][1]
    else:
        x_mid = lung_mask.shape[0] // 2
        left_mask = np.zeros_like(lung_mask, dtype=bool)
        right_mask = np.zeros_like(lung_mask, dtype=bool)
        left_mask[:x_mid, :, :] = lung_mask[:x_mid, :, :]
        right_mask[x_mid:, :, :] = lung_mask[x_mid:, :, :]

    return left_mask.astype(bool), right_mask.astype(bool)


def get_lung_zone_masks(lung_mask):
    lung_mask = lung_mask.astype(bool)
    z_presence = np.where(lung_mask.sum(axis=(0, 1)) > 0)[0]

    if z_presence.size == 0:
        empty = np.zeros_like(lung_mask, dtype=bool)
        return empty, empty, empty

    z_min = int(z_presence[0])
    z_max = int(z_presence[-1]) + 1
    bounds = np.linspace(z_min, z_max, 4, dtype=int)

    upper_mask = np.zeros_like(lung_mask, dtype=bool)
    mid_mask = np.zeros_like(lung_mask, dtype=bool)
    lower_mask = np.zeros_like(lung_mask, dtype=bool)

    upper_mask[:, :, bounds[0]:bounds[1]] = lung_mask[:, :, bounds[0]:bounds[1]]
    mid_mask[:, :, bounds[1]:bounds[2]] = lung_mask[:, :, bounds[1]:bounds[2]]
    lower_mask[:, :, bounds[2]:bounds[3]] = lung_mask[:, :, bounds[2]:bounds[3]]

    return upper_mask, mid_mask, lower_mask


def compute_zone_hits(pred_mask, gt_mask, lung_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    lung = lung_mask.astype(bool)

    left_lung, right_lung = get_lung_side_masks(lung)
    upper_lung, mid_lung, lower_lung = get_lung_zone_masks(lung)

    def zone_hit(zone_mask):
        gt_zone = gt & zone_mask
        if not gt_zone.any():
            return None
        pred_zone = pred & zone_mask
        return int(np.logical_and(pred_zone, gt_zone).any())

    return {
        "left_hit": zone_hit(left_lung),
        "right_hit": zone_hit(right_lung),
        "upper_hit": zone_hit(upper_lung),
        "mid_hit": zone_hit(mid_lung),
        "lower_hit": zone_hit(lower_lung),
    }


def compute_centroid_distance_mm(pred_mask, gt_mask, spacing):
    pred_coords = np.argwhere(pred_mask.astype(bool))
    gt_coords = np.argwhere(gt_mask.astype(bool))

    if pred_coords.size == 0 or gt_coords.size == 0:
        return None

    pred_centroid = pred_coords.mean(axis=0)
    gt_centroid = gt_coords.mean(axis=0)
    spacing = np.asarray(spacing, dtype=float)

    return float(np.linalg.norm((pred_centroid - gt_centroid) * spacing))


def compute_volume_ratio(pred_mask, gt_mask):
    pred_volume = float(pred_mask.astype(bool).sum())
    gt_volume = float(gt_mask.astype(bool).sum())

    if gt_volume == 0:
        return None

    return pred_volume / gt_volume


def save_nifti(mask, affine, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    img = nib.Nifti1Image(mask.astype(np.uint8), affine)
    nib.save(img, output_path)
