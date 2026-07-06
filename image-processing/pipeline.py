import os
from pathlib import Path
import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage

# ---------------------------------------------------------------------------
# Project root — single source of truth (imported by main.py via config.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_data_path(filepath):
    """
    Resolve a filepath that may be an absolute path from a different machine.
    Falls back to looking inside PROJECT_ROOT/data using known path markers,
    then tries a bare-filename fallback.
    """
    path = Path(filepath)
    if path.exists():
        return path

    normalized = str(filepath).replace("\\", "/")

    volume_marker = "/data/data_volumes/dataset/train_fixed/"
    if volume_marker in normalized:
        relative_volume = normalized.split(volume_marker, maxsplit=1)[1]
        candidate = PROJECT_ROOT / "data" / "data_volumes" / Path(relative_volume)
        if candidate.exists():
            return candidate

    segmentation_marker = "/data/segmentations/segmentations/"
    if segmentation_marker in normalized:
        relative_segmentation = normalized.split(segmentation_marker, maxsplit=1)[1]
        candidate = PROJECT_ROOT / "data" / "segmentations" / Path(relative_segmentation)
        if candidate.exists():
            return candidate

    fallback = PROJECT_ROOT / "data" / path.name
    if fallback.exists():
        return fallback

    return path


# ---------------------------------------------------------------------------
# NIfTI I/O
# ---------------------------------------------------------------------------

def load_nifti_file(filepath):
    """Load a NIfTI file and return (data as float32, affine, header)."""
    resolved_path = resolve_data_path(filepath)
    if not resolved_path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    img = nib.load(str(resolved_path))
    data = img.get_fdata().astype(np.float32)

    return data, img.affine, img.header


def get_voxel_spacing(header):
    """Return (sx, sy, sz) voxel spacing in mm from a NIfTI header."""
    return tuple(float(v) for v in header.get_zooms()[:3])


def save_nifti(mask, affine, output_path):
    """
    Save a boolean/uint8 mask as a NIfTI file.

    FIX #6: os.path.dirname returns '' for bare filenames, which caused
    os.makedirs to raise an error.  Guard with an explicit check.
    """
    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    img = nib.Nifti1Image(mask.astype(np.uint8), affine)
    nib.save(img, output_path)


# ---------------------------------------------------------------------------
# HU windowing & resampling
# ---------------------------------------------------------------------------

def apply_hu_window(ct_data, window_level=-600, window_width=1500):
    """
    Clip CT data to a lung-appropriate HU window and normalise to [0, 1].
    Default: level=-600 HU, width=1500 HU  → [-1350, 150] HU.
    """
    min_hu = window_level - window_width / 2
    max_hu = window_level + window_width / 2

    windowed = np.clip(ct_data, min_hu, max_hu)
    windowed = (windowed - min_hu) / (max_hu - min_hu)

    return windowed


def resample_volume(volume, current_spacing, target_spacing=(1.0, 1.0, 1.0), order=1):
    """
    Resample *volume* from *current_spacing* to *target_spacing* (mm) using
    scipy.ndimage.zoom.  order=1 for CT, order=0 for binary masks.
    """
    current_spacing = np.asarray(current_spacing, dtype=float)
    target_spacing  = np.asarray(target_spacing,  dtype=float)

    zoom_factors = current_spacing / target_spacing

    print("Original shape :", volume.shape)
    print("Current spacing:", current_spacing)
    print("Target spacing :", target_spacing)
    print("Zoom factors   :", zoom_factors)

    resampled = ndimage.zoom(volume, zoom_factors, order=order)

    print("New shape      :", resampled.shape)

    return resampled


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def spacing_to_sigma_voxels(spacing, sigma_mm):
    """Convert an isotropic or per-axis Gaussian sigma (mm) to voxel units."""
    spacing  = np.asarray(spacing,  dtype=float)
    sigma_mm = np.asarray(sigma_mm, dtype=float)

    if sigma_mm.ndim == 0:
        sigma_mm = np.repeat(sigma_mm, 3)

    sigma_voxels = sigma_mm / np.maximum(spacing, 1e-6)

    return tuple(float(v) for v in sigma_voxels)


def min_volume_to_voxels(min_volume_mm3, spacing):
    """Convert a minimum lesion volume (mm³) to a voxel count threshold."""
    spacing = np.asarray(spacing, dtype=float)
    voxel_volume_mm3 = float(np.prod(spacing))

    return max(1, int(round(float(min_volume_mm3) / max(voxel_volume_mm3, 1e-6))))


def fill_holes_per_slice(mask):
    """
    FIX #1: Apply binary_fill_holes slice-by-slice along the axial (z) axis.

    scipy.ndimage.binary_fill_holes operates in 3-D and only fills cavities
    that are fully enclosed in 3-D.  The trachea/airways connect the lungs to
    the outside, so a 3-D fill rarely closes the intra-lung space correctly.
    Filling each axial slice independently is the standard clinical approach.
    """
    filled = np.zeros_like(mask, dtype=bool)
    for z in range(mask.shape[2]):
        filled[:, :, z] = ndimage.binary_fill_holes(mask[:, :, z])
    return filled


def largest_connected_components(mask, num_components=2):
    """
    Keep the *num_components* largest connected components of *mask*.
    Returns a boolean array of the same shape.
    """
    labels, count = ndimage.label(mask)

    if count == 0:
        return mask.astype(bool)

    sizes           = ndimage.sum(mask, labels, range(1, count + 1))
    largest_indices = np.argsort(sizes)[-num_components:]

    output = np.zeros_like(mask, dtype=bool)
    for idx in largest_indices:
        output |= labels == (idx + 1)

    return output


def remove_small_components(mask, min_size=300):
    """Remove connected components whose voxel count is below *min_size*."""
    labels, count = ndimage.label(mask)

    if count == 0:
        return mask.astype(bool)

    sizes  = ndimage.sum(mask, labels, range(1, count + 1))
    output = np.zeros_like(mask, dtype=bool)

    for label_id, size in enumerate(sizes, start=1):
        if size >= min_size:
            output |= labels == label_id

    return output


# ---------------------------------------------------------------------------
# Body / lung masks
# ---------------------------------------------------------------------------

def get_body_mask(ct_data, body_threshold=-900):
    """
    Coarse whole-body mask: everything denser than air (-900 HU).

    FIX #1: replaced ndimage.binary_fill_holes with fill_holes_per_slice so
    that body cavities are closed slice-by-slice.
    """
    body = ct_data > body_threshold
    body = ndimage.binary_closing(body, iterations=2)
    body = fill_holes_per_slice(body)           # FIX #1
    body = largest_connected_components(body, num_components=1)

    return body.astype(bool)


def get_lung_mask(ct_data, lung_threshold=-400):
    """
    Extract a binary lung parenchyma mask.

    FIX #1: fill holes per axial slice to reliably close intra-lung regions
    that are connected to the outside via the trachea in 3-D.
    """
    body_mask = get_body_mask(ct_data)

    lung = (ct_data < lung_threshold) & body_mask
    lung = ndimage.binary_opening(lung, iterations=1)
    lung = ndimage.binary_closing(lung, iterations=2)
    lung = fill_holes_per_slice(lung)           # FIX #1
    lung = largest_connected_components(lung, num_components=2)

    return lung.astype(bool)


def build_lung_seed(ct_data, body_mask, threshold):
    """
    Build a lung seed mask at a given HU threshold.

    FIX #1: per-slice hole filling applied here as well.
    """
    seed = (ct_data < threshold) & body_mask
    seed = ndimage.binary_opening(seed, iterations=1)
    seed = ndimage.binary_closing(seed, iterations=2)
    seed = fill_holes_per_slice(seed)           # FIX #1
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
    """
    Build a permissive analysis mask that covers the lung parenchyma plus a
    small dilation margin to capture perifissural / subpleural lesions.

    FIX #3: The original code unioned strict_lung_mask | permissive_lung_mask,
    but because permissive_threshold > lung_threshold, the permissive mask
    always fully contains the strict mask (every voxel < -400 is also < -250).
    We now build only the permissive seed directly, saving one redundant
    build_lung_seed call.  The strict mask is still computed separately so it
    can serve as the seed_ratio reference (it is the tighter, more reliable
    lung estimate).

    FIX #1: fill_holes_per_slice is used inside build_lung_seed.
    """
    body_mask = get_body_mask(ct_data)

    # Use the stricter threshold for the seed ratio check (more reliable lung estimate)
    strict_lung_mask     = build_lung_seed(ct_data, body_mask=body_mask, threshold=lung_threshold)
    # FIX #3: only build the permissive seed; it always ⊇ strict, so union is redundant
    permissive_lung_mask = build_lung_seed(ct_data, body_mask=body_mask, threshold=permissive_threshold)

    analysis_seed = permissive_lung_mask                               # FIX #3
    seed_ratio    = float(strict_lung_mask.sum()) / max(float(body_mask.sum()), 1.0)

    if seed_ratio < min_seed_ratio:
        print(
            f"Warning: seed_ratio={seed_ratio:.4f} < {min_seed_ratio}. "
            "Using fallback threshold."
        )
        fallback_seed = (ct_data < fallback_threshold) & body_mask
        fallback_seed = ndimage.binary_opening(fallback_seed, iterations=1)
        fallback_seed = ndimage.binary_closing(fallback_seed, iterations=2)
        fallback_seed = fill_holes_per_slice(fallback_seed)            # FIX #1
        fallback_seed = largest_connected_components(fallback_seed, num_components=2)
        analysis_seed = fallback_seed.astype(bool)

    analysis_mask  = ndimage.binary_dilation(analysis_seed, iterations=dilation_iterations)
    analysis_mask &= body_mask
    analysis_mask &= ct_data < upper_hu

    analysis_mask  = ndimage.binary_closing(analysis_mask, iterations=closing_iterations)
    analysis_mask  = fill_holes_per_slice(analysis_mask)               # FIX #1

    if min_component_size > 0:
        analysis_mask = remove_small_components(analysis_mask, min_size=min_component_size)

    analysis_mask = largest_connected_components(analysis_mask, num_components=2)

    return analysis_mask.astype(bool)


# ---------------------------------------------------------------------------
# Adaptive threshold
# ---------------------------------------------------------------------------

def get_case_adaptive_threshold(
    smoothed_ct,
    lung_mask,
    percentile,
    lower_bound,
    upper_bound,
    prefilter_lower_hu=None,   # reserved for future use
    prefilter_upper_hu=None
):
    """
    Compute a case-adaptive HU lower threshold from the intensity distribution
    inside the lung mask.  Clamps the result to [lower_bound, upper_bound].
    """
    lung_values = smoothed_ct[lung_mask.astype(bool)]

    if lung_values.size == 0:
        return lower_bound

    adaptive_threshold = float(np.percentile(lung_values, percentile))
    adaptive_threshold = max(float(lower_bound), adaptive_threshold)
    adaptive_threshold = min(float(upper_bound), adaptive_threshold)

    return adaptive_threshold


# ---------------------------------------------------------------------------
# Detection algorithms
# ---------------------------------------------------------------------------

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
    """
    Detect Ground-Glass Opacities (GGO) inside the lung mask.

    Strategy:
    1. Smooth the CT to suppress noise.
    2. Compute a per-case adaptive lower HU threshold (percentile-based).
    3. Enforce a minimum band width so the detection range never collapses.
    4. Threshold → morphological clean-up → remove small components.

    FIX #4: Added a guard to ensure the HU range is wide enough before
    applying the band-width clamp so the two clamp steps cannot fight each
    other silently.  A warning is printed when the configured range is too
    narrow.
    """
    if upper_hu - lower_hu < min_band_width_hu:
        print(
            f"Warning: HU range [{lower_hu}, {upper_hu}] is narrower than "
            f"min_band_width_hu={min_band_width_hu}. Detection band may be empty."
        )

    sigma_voxels = spacing_to_sigma_voxels(spacing, smoothing_sigma_mm)
    smoothed_ct  = ndimage.gaussian_filter(ct_data, sigma=sigma_voxels)

    adaptive_lower_hu = get_case_adaptive_threshold(
        smoothed_ct,
        lung_mask,
        percentile=adaptive_percentile,
        lower_bound=lower_hu,
        upper_bound=upper_hu
    )

    # Guarantee a minimum detection band width (FIX #4: clearer order)
    max_allowed_lower = float(upper_hu) - float(min_band_width_hu)
    adaptive_lower_hu = min(float(adaptive_lower_hu), max_allowed_lower)
    adaptive_lower_hu = max(float(lower_hu), adaptive_lower_hu)

    print(f"  GGO adaptive lower HU: {adaptive_lower_hu:.1f}  upper HU: {upper_hu}")

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
    """
    Detect consolidated lung opacities (higher HU than GGO, e.g. consolidation).

    Uses a wider HU window and stronger smoothing compared to detect_ggo.
    """
    sigma_voxels = spacing_to_sigma_voxels(spacing, smoothing_sigma_mm)
    smoothed_ct  = ndimage.gaussian_filter(ct_data, sigma=sigma_voxels)

    adaptive_lower_hu = get_case_adaptive_threshold(
        smoothed_ct,
        lung_mask,
        percentile=adaptive_percentile,
        lower_bound=lower_hu,
        upper_bound=upper_hu
    )

    print(f"  Opacity adaptive lower HU: {adaptive_lower_hu:.1f}  upper HU: {upper_hu}")

    opacity_candidates = (
        (smoothed_ct >= adaptive_lower_hu) &
        (smoothed_ct <= upper_hu) &
        lung_mask
    )

    opacity_candidates = ndimage.binary_opening(opacity_candidates, iterations=1)
    opacity_candidates = ndimage.binary_closing(opacity_candidates, iterations=1)
    opacity_candidates = remove_small_components(opacity_candidates, min_size=min_size)

    return opacity_candidates.astype(bool)


# ---------------------------------------------------------------------------
# Ground-truth mask loading
# ---------------------------------------------------------------------------

def load_ggo_gt_mask(seg_path, ggo_channels):
    """
    Load the GGO ground-truth segmentation from a NIfTI file.

    Handles:
    - 3-D volumes (single-channel binary mask)
    - 4-D volumes (multi-channel; channel axis detected robustly)

    FIX #5: The original channel-axis detection checked
        max(ggo_channels) < seg_data.shape[0]
    which is unreliable when shape[0] is a large spatial dimension (e.g. 512).
    The fix identifies the channel axis as the *smallest* dimension, which is
    the standard convention for multi-label NIfTI segmentations.
    """
    seg_data, affine, header = load_nifti_file(seg_path)
    seg_data = np.asarray(seg_data)

    # ---- 3-D: treat the whole volume as a single binary mask ----
    if seg_data.ndim == 3:
        unique_values = np.unique(seg_data)
        if unique_values.size > 2:
            print(
                "Warning: 3-D segmentation has more than two unique values:",
                unique_values[:10]
            )
        return seg_data.astype(bool), affine, header

    if seg_data.ndim != 4:
        raise ValueError(f"Unsupported segmentation shape: {seg_data.shape}")

    ggo_channels = [int(ch) for ch in ggo_channels]
    max_ch       = max(ggo_channels)

    # FIX #5: channel axis = axis with the smallest size, provided it can
    # accommodate all requested channel indices.
    axis_sizes      = list(seg_data.shape)
    candidate_axes  = [i for i, s in enumerate(axis_sizes) if s > max_ch]

    if not candidate_axes:
        raise ValueError(
            f"No axis in shape {seg_data.shape} can accommodate "
            f"channel indices {ggo_channels}."
        )

    # Among valid candidates pick the one with the smallest size
    # (that is the channel axis by NIfTI/MONAI convention).
    channel_axis = min(candidate_axes, key=lambda i: seg_data.shape[i])

    # Bring channel axis to front
    channel_first = np.moveaxis(seg_data, channel_axis, 0)

    print(
        f"  Segmentation shape {seg_data.shape}: "
        f"channel axis={channel_axis}, channel_first shape={channel_first.shape}"
    )

    gt_mask = np.zeros(channel_first.shape[1:], dtype=bool)
    for ch in ggo_channels:
        gt_mask |= channel_first[ch].astype(bool)

    return gt_mask, affine, header


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_dice(pred_mask, gt_mask):
    """
    Sørensen–Dice coefficient.
    Returns 1.0 if both masks are empty (trivially perfect agreement).
    """
    pred = pred_mask.astype(bool)
    gt   = gt_mask.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    total        = pred.sum() + gt.sum()

    if total == 0:
        return 1.0

    return float(2.0 * intersection / total)


def compute_iou(pred_mask, gt_mask):
    """
    Intersection-over-Union (Jaccard index).
    Returns 1.0 if both masks are empty.
    """
    pred = pred_mask.astype(bool)
    gt   = gt_mask.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    union        = np.logical_or(pred, gt).sum()

    if union == 0:
        return 1.0

    return float(intersection / union)


def compute_sensitivity_precision(pred_mask, gt_mask):
    """
    Returns (sensitivity/recall, precision).
    Uses ε=1e-8 to avoid division by zero; result is numerically 0 when
    the denominator is truly zero (no positives in GT or prediction).
    """
    pred = pred_mask.astype(bool)
    gt   = gt_mask.astype(bool)

    tp = np.logical_and(pred,  gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    sensitivity = float(tp / (tp + fn + 1e-8))
    precision   = float(tp / (tp + fp + 1e-8))

    return sensitivity, precision


def compute_case_hit(pred_mask, gt_mask):
    """Binary: 1 if prediction overlaps GT at all, else 0."""
    pred = pred_mask.astype(bool)
    gt   = gt_mask.astype(bool)

    return int(np.logical_and(pred, gt).any())


def compute_slice_hit_rate(pred_mask, gt_mask, slice_axis=2):
    """
    Fraction of GT-positive axial slices that contain at least one predicted voxel.

    FIX #7: *slice_axis* is now an explicit parameter (default 2 = z-axis,
    consistent with NIfTI storage order) so callers can override if needed.
    """
    pred = pred_mask.astype(bool)
    gt   = gt_mask.astype(bool)

    reduce_axes = tuple(i for i in range(gt.ndim) if i != slice_axis)

    gt_slices   = gt.sum(axis=reduce_axes)   > 0
    pred_slices = pred.sum(axis=reduce_axes) > 0

    gt_count = int(gt_slices.sum())
    if gt_count == 0:
        return 1.0

    hit_count = int(np.logical_and(gt_slices, pred_slices).sum())

    return float(hit_count / gt_count)


# ---------------------------------------------------------------------------
# Zone / side masks
# ---------------------------------------------------------------------------

def get_lung_side_masks(lung_mask, affine=None):
    """
    Split the lung mask into left and right lung components.

    FIX #2: The original code sorted components by voxel x-coordinate and
    called the lower-x component "left lung".  In RAS-oriented NIfTI files
    (the standard), decreasing x corresponds to the patient's RIGHT side, so
    the labels were anatomically swapped.

    Fix: use the affine to determine whether the x-axis is left-right
    flipped.  If the affine is unavailable we fall back to the original
    heuristic but emit a warning.

    Returns: (left_lung_mask, right_lung_mask) in patient anatomical space.
    """
    lung_mask = lung_mask.astype(bool)
    labels, count = ndimage.label(lung_mask)

    if count < 2:
        # Cannot separate — split at mid-x as fallback
        x_mid      = lung_mask.shape[0] // 2
        left_mask  = np.zeros_like(lung_mask, dtype=bool)
        right_mask = np.zeros_like(lung_mask, dtype=bool)
        left_mask[:x_mid,  :, :] = lung_mask[:x_mid,  :, :]
        right_mask[x_mid:, :, :] = lung_mask[x_mid:,  :, :]
        print("Warning: fewer than 2 lung components found; using midline split.")
        return left_mask.astype(bool), right_mask.astype(bool)

    sizes = ndimage.sum(lung_mask, labels, range(1, count + 1))
    keep  = np.argsort(sizes)[-2:] + 1          # two largest components

    component_masks = []
    for label_id in keep:
        component = labels == label_id
        coords    = np.argwhere(component)
        x_center  = float(coords[:, 0].mean()) if coords.size else 0.0
        component_masks.append((x_center, component))

    # Sort by voxel x-coordinate (ascending)
    component_masks.sort(key=lambda item: item[0])
    low_x_mask  = component_masks[0][1]   # lower  voxel-x index
    high_x_mask = component_masks[1][1]   # higher voxel-x index

    # FIX #2: Determine the physical direction of the voxel x-axis.
    # The affine maps voxel indices → RAS mm coordinates.
    # affine[0, 0] > 0  ⟹  increasing voxel-x  ⟹  patient LEFT
    # affine[0, 0] < 0  ⟹  increasing voxel-x  ⟹  patient RIGHT  (common LAS/RAS flip)
    if affine is not None:
        x_direction = float(affine[0, 0])
        if x_direction >= 0:
            # voxel-x increases toward patient LEFT
            left_mask  = high_x_mask
            right_mask = low_x_mask
        else:
            # voxel-x increases toward patient RIGHT
            left_mask  = low_x_mask
            right_mask = high_x_mask
    else:
        print(
            "Warning: no affine provided to get_lung_side_masks. "
            "Left/right assignment may be anatomically incorrect for RAS images."
        )
        # Original heuristic: lower voxel-x → left (correct for LAS, wrong for RAS)
        left_mask  = low_x_mask
        right_mask = high_x_mask

    return left_mask.astype(bool), right_mask.astype(bool)


def get_lung_zone_masks(lung_mask):
    """
    Divide the lung into upper / mid / lower thirds along the z-axis
    (cranio-caudal direction in standard NIfTI orientation).
    """
    lung_mask  = lung_mask.astype(bool)
    z_presence = np.where(lung_mask.sum(axis=(0, 1)) > 0)[0]

    if z_presence.size == 0:
        empty = np.zeros_like(lung_mask, dtype=bool)
        return empty, empty, empty

    z_min  = int(z_presence[0])
    z_max  = int(z_presence[-1]) + 1
    bounds = np.linspace(z_min, z_max, 4, dtype=int)

    upper_mask = np.zeros_like(lung_mask, dtype=bool)
    mid_mask   = np.zeros_like(lung_mask, dtype=bool)
    lower_mask = np.zeros_like(lung_mask, dtype=bool)

    upper_mask[:, :, bounds[0]:bounds[1]] = lung_mask[:, :, bounds[0]:bounds[1]]
    mid_mask  [:, :, bounds[1]:bounds[2]] = lung_mask[:, :, bounds[1]:bounds[2]]
    lower_mask[:, :, bounds[2]:bounds[3]] = lung_mask[:, :, bounds[2]:bounds[3]]

    return upper_mask, mid_mask, lower_mask


def compute_zone_hits(pred_mask, gt_mask, lung_mask, affine=None):
    """
    Compute per-zone binary hit flags.

    FIX #2: Pass *affine* through to get_lung_side_masks so that left/right
    anatomical labeling is correct for RAS-oriented NIfTI files.
    """
    pred = pred_mask.astype(bool)
    gt   = gt_mask.astype(bool)
    lung = lung_mask.astype(bool)

    left_lung, right_lung         = get_lung_side_masks(lung, affine=affine)   # FIX #2
    upper_lung, mid_lung, lower_lung = get_lung_zone_masks(lung)

    def zone_hit(zone_mask):
        gt_zone = gt & zone_mask
        if not gt_zone.any():
            return None
        pred_zone = pred & zone_mask
        return int(np.logical_and(pred_zone, gt_zone).any())

    return {
        "left_hit":  zone_hit(left_lung),
        "right_hit": zone_hit(right_lung),
        "upper_hit": zone_hit(upper_lung),
        "mid_hit":   zone_hit(mid_lung),
        "lower_hit": zone_hit(lower_lung),
    }


# ---------------------------------------------------------------------------
# Distance / volume metrics
# ---------------------------------------------------------------------------

def compute_centroid_distance_mm(pred_mask, gt_mask, spacing):
    """
    Euclidean distance between the centroids of pred and GT masks, in mm.
    Returns None if either mask is empty.
    """
    pred_coords = np.argwhere(pred_mask.astype(bool))
    gt_coords   = np.argwhere(gt_mask.astype(bool))

    if pred_coords.size == 0 or gt_coords.size == 0:
        return None

    pred_centroid = pred_coords.mean(axis=0)
    gt_centroid   = gt_coords.mean(axis=0)
    spacing       = np.asarray(spacing, dtype=float)

    return float(np.linalg.norm((pred_centroid - gt_centroid) * spacing))


def compute_volume_ratio(pred_mask, gt_mask):
    """
    Ratio of predicted voxel count to GT voxel count.
    Returns None if GT mask is empty.
    """
    pred_volume = float(pred_mask.astype(bool).sum())
    gt_volume   = float(gt_mask.astype(bool).sum())

    if gt_volume == 0:
        return None

    return pred_volume / gt_volume