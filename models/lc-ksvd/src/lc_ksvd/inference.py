"""
inference.py
Dense sliding-window inference + back-projection for a trained LC-KSVD2 model.

Pipeline (per volume):
  1. Load & preprocess the CT volume exactly as in training
     (resample → HU window → [0,1] normalise).
  2. Extract ALL patches on a dense grid with stride = INFERENCE_STRIDE.
  3. L2-normalise each patch column (same as training normalise_columns()).
     Zero patches are skipped; their contribution maps remain 0.
  4. Sparse-code every patch:   Gamma = model.transform(X_norm)   [K × N]
  5. Classify every patch:      scores = W @ Gamma                 [C × N]
  6. Detect which classes are present in this volume using a simple
     per-class score threshold (see _detect_classes()).
  7. For each detected abnormality class, build a back-projection map:
       a. Identify "discriminative" atoms for that class:
          atoms whose |W[c, :]| > DISCRIMINATIVE_ATOM_PERCENTILE-th percentile.
       b. For each patch, compute a scalar contribution:
          contrib[patch] = sum of |Gamma[discriminative_atoms, patch]| × |W[c, discriminative_atoms]|
          This is the weighted activation of class-relevant atoms in that patch.
       c. Splat each patch's contribution into a 3-D accumulation volume
          (same shape as the resampled CT). Overlapping patches are averaged.
  8. Normalise the contribution map to [0, 1], threshold it (Otsu or fixed),
     and save as a binary NIfTI mask — one file per detected class.

Output files (INFERENCE_DIR / <volume_name> /):
  <volume_name>_class_<cls>_mask.nii.gz   — binary segmentation mask
  <volume_name>_class_<cls>_contrib.nii.gz — soft contribution map (float32)

Usage:
  from lc_ksvd.inference import run_inference
  run_inference("train_1_a_1")

  # or from the CLI via infer.py
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from tqdm import tqdm

from lc_ksvd.config import (
    CONTRIB_FIXED_THRESHOLD,
    CONTRIB_THRESHOLD_MODE,
    DETECTION_SCORE_THRESHOLD,
    DISCRIMINATIVE_ATOM_PERCENTILE,
    INFERENCE_DIR,
    INFERENCE_STRIDE,
    MODELS_DIR,
    N_FEATURES,
    PATCH_SIZE,
    TARGET_SPACING_MM,
)
from lc_ksvd.data_loader import (
    load_volume,
    resample_volume,
    window_and_normalise,
)

logger = logging.getLogger(__name__)


# ─── Column normalisation (mirrors train.py exactly) ─────────────────────────

def _normalise_columns(
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    L2-normalise each column of X.
    Returns (X_norm, norms, zero_mask).
    zero_mask[j] is True when column j had near-zero norm and was left as-is.
    """
    norms = np.linalg.norm(X, axis=0)          # (N,)
    zero_mask = norms < 1e-10
    norms_safe = np.where(zero_mask, 1.0, norms)
    X_norm = X / norms_safe[np.newaxis, :]
    return X_norm, norms, zero_mask


# ─── Dense patch extraction ───────────────────────────────────────────────────

def _sliding_window_centres(
    volume_shape: Tuple[int, int, int],
    patch_size: int,
    stride: int,
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """
    Generate all valid patch centre coordinates on a dense grid.

    Returns:
        centres  : (N, 3) int array of (cx, cy, cz) coordinates
        grid_shape: (nx, ny, nz) — number of steps along each axis
    """
    h = patch_size // 2
    H, W, D = volume_shape

    xs = np.arange(h, H - h, stride)
    ys = np.arange(h, W - h, stride)
    zs = np.arange(h, D - h, stride)

    # Ensure the last valid position along each axis is always included
    def _append_last(coords: np.ndarray, limit: int) -> np.ndarray:
        last = limit - h - 1
        if coords[-1] < last:
            coords = np.append(coords, last)
        return coords

    xs = _append_last(xs, H)
    ys = _append_last(ys, W)
    zs = _append_last(zs, D)

    grid_shape = (len(xs), len(ys), len(zs))
    cx, cy, cz = np.meshgrid(xs, ys, zs, indexing="ij")
    centres = np.stack([cx.ravel(), cy.ravel(), cz.ravel()], axis=1)  # (N, 3)
    return centres, grid_shape


def _extract_patches_dense(
    volume: np.ndarray,
    centres: np.ndarray,
    patch_size: int,
) -> np.ndarray:
    """
    Extract patches at all centre coordinates and flatten them into columns.

    Returns X of shape (N_FEATURES, N_patches).
    Patches that fall outside the volume (shouldn't happen given _sliding_window_centres
    bounds) are filled with zeros.
    """
    h = patch_size // 2
    n = len(centres)
    X = np.zeros((N_FEATURES, n), dtype=np.float64)

    for j, (cx, cy, cz) in enumerate(centres):
        x0, x1 = cx - h, cx - h + patch_size
        y0, y1 = cy - h, cy - h + patch_size
        z0, z1 = cz - h, cz - h + patch_size
        patch = volume[x0:x1, y0:y1, z0:z1]
        if patch.shape == (patch_size, patch_size, patch_size):
            X[:, j] = patch.ravel(order="C")

    return X


# ─── Class detection ──────────────────────────────────────────────────────────

def _detect_classes(
    scores: np.ndarray,
    class_order: List[str],
    threshold_fraction: float = DETECTION_SCORE_THRESHOLD,
) -> List[str]:
    """
    Decide which *abnormality* classes are present in this volume.

    Strategy: for each class, take the mean of the top-10% patch scores
    (robust maximum) and compare against the global score range across all
    classes. A class is "detected" when its robust max exceeds
    `threshold_fraction` × (global_max − global_min) + global_min.

    "normal" (index 0) is never returned as a detection — it is the null class.

    Args:
        scores            : (n_classes, n_patches) float array
        class_order       : list of class names matching rows of scores
        threshold_fraction: fraction of score range above which a class fires

    Returns:
        List of detected abnormality class names (may be empty).
    """
    # Robust per-class maximum: mean of top 10% of patches for that class
    k = max(1, scores.shape[1] // 10)
    per_class_robust_max = np.array([
        float(np.mean(np.partition(scores[c], -k)[-k:]))
        for c in range(len(class_order))
    ])

    global_min = float(scores.min())
    global_max = float(scores.max())
    score_range = global_max - global_min
    if score_range < 1e-12:
        return []

    threshold = threshold_fraction * score_range + global_min

    detected = []
    for i, cls in enumerate(class_order):
        if cls == "normal":
            continue
        if per_class_robust_max[i] > threshold:
            detected.append(cls)

    return detected


# ─── Back-projection ──────────────────────────────────────────────────────────

def _backproject_class(
    class_idx: int,
    W: np.ndarray,          # (n_classes, n_components)
    Gamma: np.ndarray,      # (n_components, n_patches)
    centres: np.ndarray,    # (n_patches, 3)
    volume_shape: Tuple[int, int, int],
    patch_size: int,
) -> np.ndarray:
    """
    Build a 3-D float32 contribution map for one class.

    For each patch j:
      contrib[j] = Σ_{k ∈ discriminative atoms} |W[c, k]| × |Gamma[k, j]|

    This is the class-weighted sparse activation magnitude — a direct
    analogue of the training classifier score W[c, :] @ Gamma[:, j], but
    restricted to atoms that are genuinely discriminative for class c and
    expressed as a non-negative contribution so it accumulates sensibly.

    Overlapping patches contribute to the same voxels; each voxel's final
    value is the mean across all patches that cover it (tracked via a
    count map).
    """
    h = patch_size // 2
    H, W_vol, D = volume_shape

    # ── Select discriminative atoms for this class ────────────────────────────
    class_weights = np.abs(W[class_idx, :])              # (n_components,)
    percentile_val = np.percentile(class_weights, DISCRIMINATIVE_ATOM_PERCENTILE)
    disc_atoms = class_weights >= percentile_val         # boolean mask (n_components,)

    # ── Per-patch contribution scalar ─────────────────────────────────────────
    # weighted_activation[j] = Σ_k  |W[c,k]| * |Gamma[k,j]|  for k in disc_atoms
    w_disc   = class_weights[disc_atoms]                 # (n_disc,)
    G_disc   = np.abs(Gamma[disc_atoms, :])              # (n_disc, n_patches)
    contrib  = w_disc @ G_disc                           # (n_patches,)  — dot product

    # ── Splat into volume ─────────────────────────────────────────────────────
    accum = np.zeros(volume_shape, dtype=np.float64)
    count = np.zeros(volume_shape, dtype=np.float64)

    for j, (cx, cy, cz) in enumerate(centres):
        x0, x1 = cx - h, cx - h + patch_size
        y0, y1 = cy - h, cy - h + patch_size
        z0, z1 = cz - h, cz - h + patch_size
        accum[x0:x1, y0:y1, z0:z1] += contrib[j]
        count[x0:x1, y0:y1, z0:z1] += 1.0

    # Average over overlapping patches (avoid division by zero in padding areas)
    with np.errstate(invalid="ignore", divide="ignore"):
        contrib_map = np.where(count > 0, accum / count, 0.0)

    return contrib_map.astype(np.float32)


def _threshold_map(
    contrib_map: np.ndarray,
    mode: str = CONTRIB_THRESHOLD_MODE,
    fixed_threshold: float = CONTRIB_FIXED_THRESHOLD,
) -> np.ndarray:
    """
    Binarize a [0, 1]-normalised contribution map.

    mode="otsu"  : compute Otsu threshold from the non-zero voxels.
    mode="fixed" : use fixed_threshold directly.

    Returns a uint8 binary mask (1 = detected region, 0 = background).
    """
    if mode == "otsu":
        nonzero = contrib_map[contrib_map > 0]
        if len(nonzero) == 0:
            return np.zeros_like(contrib_map, dtype=np.uint8)
        threshold = _otsu_threshold(nonzero)
    else:
        threshold = fixed_threshold

    return (contrib_map >= threshold).astype(np.uint8)


def _otsu_threshold(values: np.ndarray, n_bins: int = 256) -> float:
    """
    Compute the Otsu threshold from a 1-D array of values in [0, 1].
    Pure-NumPy implementation (no sklearn / skimage dependency).
    """
    counts, bin_edges = np.histogram(values, bins=n_bins, range=(0.0, 1.0))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    total = counts.sum()
    if total == 0:
        return 0.5

    # Cumulative sums
    cumsum   = np.cumsum(counts)
    cumsum_w = np.cumsum(counts * bin_centers)

    w0 = cumsum / total
    w1 = 1.0 - w0

    mu0 = np.where(cumsum > 0, cumsum_w / cumsum, 0.0)
    mu1 = np.where((total - cumsum) > 0,
                   (cumsum_w[-1] - cumsum_w) / (total - cumsum),
                   0.0)

    sigma_b2 = w0 * w1 * (mu0 - mu1) ** 2

    best_idx = int(np.argmax(sigma_b2))
    return float(bin_centers[best_idx])


# ─── NIfTI saving ─────────────────────────────────────────────────────────────

def _save_nifti(
    data: np.ndarray,
    volume_name: str,
    suffix: str,
    out_dir: Path,
    spacing_mm: float = TARGET_SPACING_MM,
) -> Path:
    """
    Save a 3-D array as a NIfTI file.

    The affine encodes isotropic spacing at TARGET_SPACING_MM with no
    rotation — consistent with the resampled volume space used throughout
    the pipeline. If you need the mask in original scanner space, apply
    the inverse of the resampling transform externally.
    """
    affine = np.diag([spacing_mm, spacing_mm, spacing_mm, 1.0])
    img = nib.Nifti1Image(data, affine=affine)
    img.header.set_zooms((spacing_mm, spacing_mm, spacing_mm))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{volume_name}_{suffix}.nii.gz"
    nib.save(img, str(out_path))
    return out_path


# ─── Public API ───────────────────────────────────────────────────────────────

def run_inference(
    volume_name: str,
    model_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    stride: int = INFERENCE_STRIDE,
    save_contrib_maps: bool = True,
) -> Dict[str, Path]:
    """
    Run dense sliding-window inference on a single volume and save binary
    segmentation masks (+ optionally soft contribution maps) as NIfTI files.

    Args:
        volume_name      : scan ID, e.g. "train_1_a_1"
        model_path       : path to the .pkl saved by train.py
                           (defaults to MODELS_DIR / "unified_lcksvd2.pkl")
        out_dir          : directory for output masks
                           (defaults to INFERENCE_DIR / volume_name)
        stride           : sliding-window stride in voxels (default: INFERENCE_STRIDE)
        save_contrib_maps: whether to also save the soft float32 contribution maps

    Returns:
        dict mapping class name → Path of the binary mask NIfTI file
        (only for detected abnormality classes)
    """
    # ── Load model ────────────────────────────────────────────────────────────
    if model_path is None:
        model_path = MODELS_DIR / "unified_lcksvd2.pkl"

    logger.info(f"Loading model from {model_path}")
    with open(model_path, "rb") as f:
        payload = pickle.load(f)

    model       = payload["model"]
    class_order = payload["class_order"]   # must match CLASS_ORDER
    W           = model.W_                 # (n_classes, n_components)

    # ── Load & preprocess volume (mirrors ScanLoader.load() in data_loader.py) ─
    logger.info(f"Loading volume: {volume_name}")
    vol_hu, spacing = load_volume(volume_name)
    vol_rs = resample_volume(vol_hu, spacing)
    volume = window_and_normalise(vol_rs)          # float32 [H, W, D] in [0,1]
    logger.info(f"Volume shape after resampling: {volume.shape}")

    # ── Dense patch extraction ────────────────────────────────────────────────
    logger.info(f"Extracting patches (stride={stride})…")
    centres, grid_shape = _sliding_window_centres(volume.shape, PATCH_SIZE, stride)
    logger.info(f"Grid: {grid_shape} → {len(centres)} patches")

    X = _extract_patches_dense(volume, centres, PATCH_SIZE)   # (N_FEATURES, N)

    # ── Column normalisation (same as training) ───────────────────────────────
    X_norm, _, zero_mask = _normalise_columns(X)

    # Keep track of which patch indices are non-zero so we can zero-out their
    # contribution in the back-projection rather than removing them from the
    # centres array (which would break the centres ↔ column correspondence).
    valid_patch_mask = ~zero_mask   # (N,) bool

    # ── Sparse coding ─────────────────────────────────────────────────────────
    logger.info("Computing sparse codes…")
    Gamma = model.transform(X_norm)             # (n_components, N)

    # Zero out codes for near-zero patches (they carry no signal)
    Gamma[:, ~valid_patch_mask] = 0.0

    # ── Classification scores ─────────────────────────────────────────────────
    scores = W @ Gamma                          # (n_classes, N)

    # ── Detect present classes ────────────────────────────────────────────────
    detected_classes = _detect_classes(scores, class_order)
    logger.info(
        f"Detected classes: {detected_classes if detected_classes else ['(none — normal)']}"
    )

    # ── Back-projection + save ────────────────────────────────────────────────
    if out_dir is None:
        out_dir = INFERENCE_DIR / volume_name

    output_paths: Dict[str, Path] = {}

    for cls in detected_classes:
        class_idx = class_order.index(cls)
        logger.info(f"  Back-projecting class '{cls}' (idx={class_idx})…")

        contrib_map = _backproject_class(
            class_idx=class_idx,
            W=W,
            Gamma=Gamma,
            centres=centres,
            volume_shape=volume.shape,
            patch_size=PATCH_SIZE,
        )

        # Normalise to [0, 1] for thresholding
        cmin, cmax = float(contrib_map.min()), float(contrib_map.max())
        if cmax - cmin > 1e-12:
            contrib_norm = (contrib_map - cmin) / (cmax - cmin)
        else:
            contrib_norm = np.zeros_like(contrib_map)

        # Binary mask
        binary_mask = _threshold_map(contrib_norm)

        n_voxels = int(binary_mask.sum())
        logger.info(f"    Mask voxels: {n_voxels} "
                    f"({100.0 * n_voxels / binary_mask.size:.2f}% of volume)")

        # Save binary mask
        mask_path = _save_nifti(
            data=binary_mask,
            volume_name=volume_name,
            suffix=f"class_{cls}_mask",
            out_dir=out_dir,
        )
        output_paths[cls] = mask_path
        logger.info(f"    Saved mask → {mask_path}")

        # Optionally save soft contribution map
        if save_contrib_maps:
            contrib_path = _save_nifti(
                data=contrib_norm,
                volume_name=volume_name,
                suffix=f"class_{cls}_contrib",
                out_dir=out_dir,
            )
            logger.info(f"    Saved contrib → {contrib_path}")

    if not detected_classes:
        logger.info("  No abnormalities detected — volume classified as normal.")

    return output_paths


def run_inference_batch(
    volume_names: List[str],
    model_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    stride: int = INFERENCE_STRIDE,
    save_contrib_maps: bool = True,
) -> Dict[str, Dict[str, Path]]:
    """
    Run run_inference() over a list of volumes, skipping failures.

    Returns:
        {volume_name: {class_name: mask_path, ...}, ...}
    """
    results: Dict[str, Dict[str, Path]] = {}

    for volume_name in tqdm(volume_names, desc="inference"):
        try:
            paths = run_inference(
                volume_name=volume_name,
                model_path=model_path,
                out_dir=out_dir,
                stride=stride,
                save_contrib_maps=save_contrib_maps,
            )
            results[volume_name] = paths
        except Exception as e:
            logger.warning(f"Skipping {volume_name}: {e}")

    return results