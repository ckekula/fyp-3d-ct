"""
patch_extractor.py
Extracts 3D patches from CT volumes and builds the (n_features, n_patches) matrix X
and the (2, n_patches) one-hot label matrix H required by reppi's LCKSVD.

One separate (X, H) pair is built per abnormality (Approach B: 4 binary classifiers).

Patch sampling strategy:
  Positive patches: centres sampled from within the lesion mask for the target
                    abnormality, accepted only if overlap ratio >= MIN_OVERLAP_RATIO.
  Negative patches: centres sampled from regions with no lesion for the target
                    abnormality. Includes both background regions of positive scans
                    and patches from normal scans.

The resulting matrices are saved as compressed .npz files to PATCHES_DIR.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from lc_ksvd.config import (
    ABNORMALITY_CATEGORIES, MIN_OVERLAP_RATIO, N_FEATURES,
    N_POSITIVE_PATCHES_PER_SCAN, NEG_TO_POS_RATIO, PATCH_SIZE,
    PATCHES_DIR, RANDOM_SEED, CLASS_ORDER
)
from lc_ksvd.data_loader import LabelRegistry, MetadataRegistry, ScanLoader, resolve_volume_path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

rng = np.random.default_rng(RANDOM_SEED)


# ─── Core patch utilities ────────────────────────────────────────────────────

def _extract_patch(volume: np.ndarray, centre: Tuple[int, int, int]) -> Optional[np.ndarray]:
    """
    Extract a cubic patch of PATCH_SIZE centred at (cx, cy, cz).
    Returns None if the patch extends outside the volume boundary.
    """
    p = PATCH_SIZE
    h = p // 2
    cx, cy, cz = centre
    H, W, D = volume.shape

    x0, x1 = cx - h, cx - h + p
    y0, y1 = cy - h, cy - h + p
    z0, z1 = cz - h, cz - h + p

    if x0 < 0 or y0 < 0 or z0 < 0 or x1 > H or y1 > W or z1 > D:
        return None

    return volume[x0:x1, y0:y1, z0:z1].copy()


def _overlap_ratio(
    centre: Tuple[int, int, int],
    binary_mask: np.ndarray,
) -> float:
    """Fraction of patch voxels that are positive in the binary_mask."""
    p = PATCH_SIZE
    h = p // 2
    cx, cy, cz = centre
    H, W, D = binary_mask.shape

    x0, x1 = cx - h, cx - h + p
    y0, y1 = cy - h, cy - h + p
    z0, z1 = cz - h, cz - h + p

    if x0 < 0 or y0 < 0 or z0 < 0 or x1 > H or y1 > W or z1 > D:
        return 0.0

    patch_mask = binary_mask[x0:x1, y0:y1, z0:z1]
    return float(patch_mask.sum()) / (p ** 3)


def _valid_centre_range(volume_shape: Tuple[int, int, int]) -> Tuple:
    """Return the inclusive range of valid patch centres (avoids boundary)."""
    h = PATCH_SIZE // 2
    H, W, D = volume_shape
    return (h, H - h), (h, W - h), (h, D - h)


# ─── Positive patch sampling ─────────────────────────────────────────────────

def sample_positive_patches(
    volume: np.ndarray,
    binary_mask: np.ndarray,
    n_patches: int,
    max_attempts_multiplier: int = 10,
) -> List[np.ndarray]:
    """
    Sample up to n_patches positive patches from voxels within the lesion mask.

    Strategy:
      1. Find all foreground voxel coordinates (where binary_mask == 1).
      2. Randomly pick coordinates as candidate patch centres.
      3. Accept if overlap_ratio >= MIN_OVERLAP_RATIO and patch is in bounds.
    """
    foreground_coords = np.argwhere(binary_mask > 0)
    if len(foreground_coords) == 0:
        return []

    patches = []
    max_attempts = n_patches * max_attempts_multiplier
    attempts = 0

    # Stage 1: strict overlap threshold from config
    while len(patches) < n_patches and attempts < max_attempts:
        attempts += 1
        idx = rng.integers(0, len(foreground_coords))
        centre = tuple(foreground_coords[idx])

        if _overlap_ratio(centre, binary_mask) >= MIN_OVERLAP_RATIO:
            patch = _extract_patch(volume, centre)
            if patch is not None:
                patches.append(patch)

    # Stage 2 fallback: tiny lesions may never satisfy MIN_OVERLAP_RATIO for large patch sizes.
    # Fill the remainder with any in-bounds patch that still overlaps lesion (>0).
    if len(patches) < n_patches:
        while len(patches) < n_patches and attempts < (2 * max_attempts):
            attempts += 1
            idx = rng.integers(0, len(foreground_coords))
            centre = tuple(foreground_coords[idx])

            if _overlap_ratio(centre, binary_mask) <= 0.0:
                continue

            patch = _extract_patch(volume, centre)
            if patch is not None:
                patches.append(patch)

    if len(patches) < n_patches:
        logger.debug(
            f"Only sampled {len(patches)}/{n_patches} positive patches "
            f"(lesion may be small relative to patch size)."
        )
    return patches


# ─── Negative patch sampling ─────────────────────────────────────────────────

def sample_negative_patches(
    volume: np.ndarray,
    binary_mask: Optional[np.ndarray],
    n_patches: int,
    max_attempts_multiplier: int = 10,
) -> List[np.ndarray]:
    """
    Sample n_patches negative patches (no lesion overlap).

    If binary_mask is None (normal scan), all patches are valid negatives.
    If binary_mask is provided, reject any patch with overlap_ratio > 0.
    """
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = _valid_centre_range(volume.shape)

    patches = []
    max_attempts = n_patches * max_attempts_multiplier
    attempts = 0

    while len(patches) < n_patches and attempts < max_attempts:
        attempts += 1
        cx = int(rng.integers(xlo, xhi))
        cy = int(rng.integers(ylo, yhi))
        cz = int(rng.integers(zlo, zhi))
        centre = (cx, cy, cz)

        if binary_mask is not None and _overlap_ratio(centre, binary_mask) > 0:
            continue  # overlaps lesion — reject

        patch = _extract_patch(volume, centre)
        if patch is not None:
            patches.append(patch)

    return patches


# ─── Per-scan finding mask extraction ────────────────────────────────────────

def get_binary_mask_for_abnormality(
    mask_4d: Optional[np.ndarray],
    finding_map: Dict[int, str],
    abnormality: str,
) -> Optional[np.ndarray]:
    """
    Collapse the 4D mask [F, H, W, D] to a binary [H, W, D] mask for
    the target abnormality by OR-ing all F-slices labelled as that abnormality.

    Returns None if the abnormality is not present in the finding_map.
    """
    if mask_4d is None:
        return None

    relevant_slices = [
        f_idx for f_idx, ab in finding_map.items()
        if ab == abnormality and f_idx < mask_4d.shape[0]
    ]
    if not relevant_slices:
        return None

    binary = np.zeros(mask_4d.shape[1:], dtype=np.uint8)
    for f_idx in relevant_slices:
        binary = np.logical_or(binary, mask_4d[f_idx] > 0).astype(np.uint8)

    return binary


# ─── Majority-class label for a patch centre ─────────────────────────────────

def get_majority_class_label(
    centre: Tuple[int, int, int],
    masks_by_class: Dict[str, np.ndarray],
    class_to_row: Dict[str, int],
) -> Tuple[int, str, float]:
    """
    Returns:
      row_index  : int class index of the majority class (-1 if no overlap)
      winner_cls : class key (str)
      max_overlap: overlap ratio of the winning class (float)
    """
    best_cls     = None
    best_overlap = 0.0

    for cls, mask in masks_by_class.items():
        overlap = _overlap_ratio(centre, mask)
        if overlap > best_overlap:
            best_overlap = overlap
            best_cls     = cls

    row_index = class_to_row[best_cls] if best_cls is not None else -1
    return row_index, best_cls, best_overlap


# ─── Unified matrix builder ───────────────────────────────────────────────────

def build_unified_patch_matrix(
    scan_ids: Dict[str, List[str]],
    loader: ScanLoader,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a single X (n_features, n_patches) and H (n_classes, n_patches).

    Abnormal scans:
      - Collect all per-abnormality binary masks for the scan.
      - Build a union mask (any abnormality).
      - Sample patch centres from within the union mask only.
      - For each centre, assign a one-hot label to the majority class
        (the abnormality whose mask has the highest overlap with that patch).
      - Patches where every class has zero overlap are discarded.

    Normal scans:
      - Sample patches from anywhere in the volume (no mask).
      - Label as class "normal" (one-hot row 0).
    """
    n_classes    = len(CLASS_ORDER)
    class_to_row = {cls: i for i, cls in enumerate(CLASS_ORDER)}
    abnormality_keys = [k for k in CLASS_ORDER if k != "normal"]

    all_patches: List[np.ndarray] = []
    all_labels: List[int] = []

    # ── Abnormal scans ────────────────────────────────────────────────────────
    logger.info(f"Sampling masked patches from {len(scan_ids['positive'])} abnormal scans…")

    for scan_id in tqdm(scan_ids["positive"], desc="abnormal scans"):
        try:
            scan = loader.load(scan_id)
        except Exception as e:
            logger.warning(f"Skipping {scan_id}: {e}")
            continue

        if scan["mask"] is None:
            logger.debug(f"  {scan_id}: no mask available, skipping.")
            continue

        # Build per-class binary masks for every recognised abnormality present
        masks_by_class: Dict[str, np.ndarray] = {}
        for cls in abnormality_keys:
            m = get_binary_mask_for_abnormality(
                scan["mask"], scan["finding_map"], cls
            )
            if m is not None and m.sum() > 0:
                masks_by_class[cls] = m

        if not masks_by_class:
            logger.debug(f"  {scan_id}: no recognised abnormality masks found, skipping.")
            continue

        # Union mask — defines the candidate patch centre space
        union_mask = np.zeros(scan["volume"].shape, dtype=np.uint8)
        for m in masks_by_class.values():
            union_mask = np.logical_or(union_mask, m).astype(np.uint8)

        # Determine how many patches to sample (proportional to abnormalities present)
        n_target = N_POSITIVE_PATCHES_PER_SCAN * len(masks_by_class)

        # Sample centres from within the union mask foreground
        foreground_coords = np.argwhere(union_mask > 0)
        max_attempts      = n_target * 10
        attempts          = 0
        scan_patches      = 0

        while scan_patches < n_target and attempts < max_attempts:
            attempts += 1
            idx    = rng.integers(0, len(foreground_coords))
            centre = tuple(foreground_coords[idx])

            patch = _extract_patch(scan["volume"], centre)
            if patch is None:
                continue   # out of bounds

            row_index, winner_cls, overlap = get_majority_class_label(
                centre, masks_by_class, class_to_row
            )
            if winner_cls is None or overlap < MIN_OVERLAP_RATIO:
                continue

            all_patches.append(patch)
            all_labels.append(row_index)
            scan_patches += 1

        if scan_patches < n_target:
            logger.debug(
                f"  {scan_id}: collected {scan_patches}/{n_target} patches "
                f"(small or thin lesions)."
            )

    n_abnormal_patches = len(all_patches)
    logger.info(f"Collected {n_abnormal_patches} patches from abnormal scans.")

    if n_abnormal_patches == 0:
        raise RuntimeError(
            "No patches collected from abnormal scans. "
            "Check MASKS_DIR, METADATA_JSON, and MIN_OVERLAP_RATIO."
        )

    # ── Normal scans ──────────────────────────────────────────────────────────
    normal_label = np.zeros(n_classes, dtype=np.float64)
    normal_label[class_to_row["normal"]] = 1.0

    n_normal_target = int(n_abnormal_patches * NEG_TO_POS_RATIO)

    if n_normal_target > 0 and scan_ids["normal"]:
        n_per_normal = max(1, n_normal_target // len(scan_ids["normal"]))
        logger.info(
            f"Sampling ~{n_per_normal} patches from each of "
            f"{len(scan_ids['normal'])} normal scans…"
        )
        for scan_id in tqdm(scan_ids["normal"], desc="normal scans"):
            try:
                scan = loader.load(scan_id)
            except Exception as e:
                logger.warning(f"Skipping {scan_id}: {e}")
                continue

            patches = sample_negative_patches(
                scan["volume"], binary_mask=None, n_patches=n_per_normal
            )
            normal_row = class_to_row["normal"]
            for p in patches:
                all_patches.append(p)
                all_labels.append(normal_row)

    logger.info(f"Total patches: {len(all_patches)}  "
                f"(abnormal: {n_abnormal_patches}, "
                f"normal: {len(all_patches) - n_abnormal_patches})")

    # ── Shuffle ───────────────────────────────────────────────────────────────
    order      = rng.permutation(len(all_patches))
    all_patches = [all_patches[i] for i in order]
    all_labels  = [all_labels[i]  for i in order]

    # ── Assemble X and H ──────────────────────────────────────────────────────
    n_patches = len(all_patches)
    X = np.zeros((N_FEATURES, n_patches), dtype=np.float64)
    H = np.empty(n_patches, dtype=np.int64)   # (n_patches,) label vector

    for j, (patch, label) in enumerate(zip(all_patches, all_labels)):
        X[:, j] = patch.ravel()
        H[j]    = label

    return X, H

def extract_unified(split: str = "train") -> None:
    """
    Run unified patch extraction for all classes in one pass and save a
    single .npz file: patches/unified_{split}.npz
    """
    PATCHES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PATCHES_DIR / f"unified_{split}.npz"

    if out_path.exists():
        logger.info(f"Unified patch matrix already exists at {out_path}, skipping.")
        return

    metadata = MetadataRegistry(split=split)
    labels   = LabelRegistry(metadata, split=split)
    loader   = ScanLoader(metadata)

    def _filter_existing(ids):
        valid = []
        for vid in ids:
            try:
                resolve_volume_path(vid)
                valid.append(vid)
            except Exception:
                pass
        return valid

    # All volumes that have at least one recognised finding
    abnormality_keys = [k for k in CLASS_ORDER if k != "normal"]
    positive_ids = set()
    for ab in abnormality_keys:
        positive_ids.update(labels.get_positive_volume_names(ab))
    positive_ids = _filter_existing(list(positive_ids))

    normal_ids = _filter_existing(labels.get_normal_volume_names())

    logger.info(
        f"Unified extraction — split={split}\n"
        f"  Abnormal volumes : {len(positive_ids)}\n"
        f"  Normal   volumes : {len(normal_ids)}"
    )

    scan_ids = {"positive": positive_ids, "normal": normal_ids}
    X, H = build_unified_patch_matrix(scan_ids, loader)

    np.savez_compressed(out_path, X=X, H=H)
    logger.info(f"Saved unified patch matrix → {out_path}  (X: {X.shape}, H: {H.shape})")


def load_unified_patch_matrix(split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
    """Load the unified patch matrix from disk."""
    path = PATCHES_DIR / f"unified_{split}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Unified patch matrix not found: {path}. Run extract_unified() first."
        )
    data = np.load(path)
    return data["X"], data["H"]


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting patch extraction for all abnormalities (train split)…")
    extract_unified(split="train")
    logger.info("Done.")
    