"""
patch_extractor.py
Extracts 3D patches from CT volumes and builds the (n_features, n_patches) matrix X
and the (n_patches,) integer label vector H required for LC-KSVD2 training.

Phase 1 — Normal scans:
  Sample patches from anywhere in the volume (no mask constraint).
  All labelled as class index 0 ("normal").

Phase 2 — Abnormal scans:
  For each scan, load all finding masks from the 4D segmentation.
  For each finding, extract only patches that are covered by that finding's mask
  (overlap >= MIN_OVERLAP_RATIO). Each patch is labelled by its majority class
  across all findings in that scan (handles overlapping findings).

The resulting matrices are saved as compressed .npz files to PATCHES_DIR.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from lc_ksvd.config import (
    MIN_OVERLAP_RATIO, N_FEATURES,
    N_POSITIVE_PATCHES_PER_SCAN, PATCH_SIZE,
    PATCHES_DIR, RANDOM_SEED, CLASS_ORDER
)
from lc_ksvd.data_loader import (
    LabelRegistry, MetadataRegistry, ScanLoader, resolve_volume_path
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ─── Core patch utilities ─────────────────────────────────────────────────────

def _extract_patch(
    volume: np.ndarray,
    centre: Tuple[int, int, int],
) -> Optional[np.ndarray]:
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
    p = PATCH_SIZE
    h = p // 2
    cx, cy, cz = centre
    H, W, D = binary_mask.shape

    x0, x1 = cx - h, cx - h + p
    y0, y1 = cy - h, cy - h + p
    z0, z1 = cz - h, cz - h + p

    if x0 < 0 or y0 < 0 or z0 < 0 or x1 > H or y1 > W or z1 > D:
        return 0.0

    return float(binary_mask[x0:x1, y0:y1, z0:z1].sum()) / (p ** 3)


def _valid_centre_range(
    volume_shape: Tuple[int, int, int],
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    h = PATCH_SIZE // 2
    H, W, D = volume_shape
    return (h, H - h), (h, W - h), (h, D - h)


# ─── Phase 1: Normal patch sampling ──────────────────────────────────────────

def sample_normal_patches(
    volume: np.ndarray,
    n_patches: int,
    rng: np.random.Generator,
    max_attempts_multiplier: int = 10,
) -> List[np.ndarray]:
    """
    Sample n_patches from anywhere in the volume.
    No mask constraint — every in-bounds patch is a valid normal patch.
    """
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = _valid_centre_range(volume.shape)
    patches = []
    max_attempts = n_patches * max_attempts_multiplier

    for _ in range(max_attempts):
        if len(patches) >= n_patches:
            break
        centre = (
            int(rng.integers(xlo, xhi)),
            int(rng.integers(ylo, yhi)),
            int(rng.integers(zlo, zhi)),
        )
        patch = _extract_patch(volume, centre)
        if patch is not None:
            patches.append(patch)

    if len(patches) < n_patches:
        logger.debug(f"Normal scan: only sampled {len(patches)}/{n_patches} patches.")

    return patches


def collect_normal_patches(
    normal_ids: List[str],
    loader: ScanLoader,
    n_per_scan: int,
    rng: np.random.Generator,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Phase 1: collect patches from all normal scans.
    Returns flat lists of patches and their integer class labels (all 0 = "normal").
    """
    normal_class_idx = CLASS_ORDER.index("normal")
    all_patches: List[np.ndarray] = []
    all_labels: List[int] = []

    logger.info(f"Phase 1 — sampling {n_per_scan} patches from each of "
                f"{len(normal_ids)} normal scans…")

    for scan_id in tqdm(normal_ids, desc="normal scans"):
        try:
            scan = loader.load(scan_id)
        except Exception as e:
            logger.warning(f"Skipping {scan_id}: {e}")
            continue

        patches = sample_normal_patches(scan["volume"], n_per_scan, rng)
        all_patches.extend(patches)
        all_labels.extend([normal_class_idx] * len(patches))

    logger.info(f"  → {len(all_patches)} normal patches collected.")
    return all_patches, all_labels


# ─── Phase 2: Abnormal patch sampling ────────────────────────────────────────

def _build_finding_masks(
    mask_4d: np.ndarray,
    finding_map: Dict[int, str],
) -> Dict[str, np.ndarray]:
    """
    Collapse the 4D mask [F, H, W, D] into per-category binary masks.

    Each finding (F-index) maps to a category string via finding_map.
    Multiple findings with the same category are OR-ed together into one mask.
    Findings whose category is not in CLASS_ORDER are skipped.

    Returns:
        { category_str: binary_mask [H, W, D] }
    """
    category_masks: Dict[str, np.ndarray] = {}
    volume_shape = mask_4d.shape[1:]   # (H, W, D)

    for f_idx, category in finding_map.items():
        if category not in CLASS_ORDER:
            continue
        if f_idx >= mask_4d.shape[0]:
            logger.warning(f"  f_idx={f_idx} out of range for mask shape {mask_4d.shape}")
            continue

        finding_mask = (mask_4d[f_idx] > 0).astype(np.uint8)

        if category not in category_masks:
            category_masks[category] = np.zeros(volume_shape, dtype=np.uint8)
        category_masks[category] = np.logical_or(
            category_masks[category], finding_mask
        ).astype(np.uint8)

    return {cat: m for cat, m in category_masks.items() if m.sum() > 0}


def _majority_class(
    centre: Tuple[int, int, int],
    category_masks: Dict[str, np.ndarray],
) -> Tuple[Optional[str], float]:
    """
    Among all per-category masks, return the one with the highest overlap
    with the patch centred at `centre`, and its overlap ratio.
    """
    best_cat, best_overlap = None, 0.0

    for cat, mask in category_masks.items():
        overlap = _overlap_ratio(centre, mask)
        if overlap > best_overlap:
            best_overlap = overlap
            best_cat = cat

    return best_cat, best_overlap


def sample_patches_from_mask(
    volume: np.ndarray,
    binary_mask: np.ndarray,      # (H, W, D) — caller builds this
    n_patches: int,
    rng: np.random.Generator,
    max_attempts_multiplier: int = 10,
) -> List[np.ndarray]:
    """
    Sample n_patches whose centres fall within binary_mask foreground,
    each with overlap >= MIN_OVERLAP_RATIO.
    """
    foreground_coords = np.argwhere(binary_mask > 0)
    if len(foreground_coords) == 0:
        return []

    patches = []
    max_attempts = n_patches * max_attempts_multiplier

    for _ in range(max_attempts):
        if len(patches) >= n_patches:
            break
        idx    = rng.integers(0, len(foreground_coords))
        centre = tuple(foreground_coords[idx])

        patch = _extract_patch(volume, centre)
        if patch is None:
            continue

        if _overlap_ratio(centre, binary_mask) < MIN_OVERLAP_RATIO:
            continue

        patches.append(patch)

    if len(patches) < n_patches:
        logger.debug(f"Sampled {len(patches)}/{n_patches} patches from mask.")

    return patches


def collect_abnormal_patches(
    positive_ids: List[str],
    loader: ScanLoader,
    n_patches_per_scan: int,
    rng: np.random.Generator,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Phase 2: collect patches from all abnormal scans.
 
    For each scan:
      1. Build per-category masks via _build_finding_masks (OR-ing findings of
         the same category together).
      2. Build a union mask across all categories to use as the sampling region.
      3. Sample up to n_patches_per_scan patch candidates from the union mask.
      4. For each candidate, assign the label of the category mask with the
         highest overlap (_majority_class), discarding the patch if no category
         clears MIN_OVERLAP_RATIO.
 
    This correctly handles overlapping findings: a patch that sits in the
    overlap of a 2b and 2d region is labelled by whichever mask covers more
    of it, rather than by iteration order.
    """
    all_patches: List[np.ndarray] = []
    all_labels:  List[int]        = []
    class_to_idx = {cls: i for i, cls in enumerate(CLASS_ORDER)}
 
    for scan_id in tqdm(positive_ids, desc="abnormal scans"):
        try:
            scan = loader.load(scan_id)
        except Exception as e:
            logger.warning(f"Skipping {scan_id}: {e}")
            continue
 
        if scan["mask"] is None or not scan["finding_map"]:
            continue
 
        # Build per-category binary masks for this scan
        category_masks = _build_finding_masks(scan["mask"], scan["finding_map"])
        if not category_masks:
            continue
 
        # Union mask: sample candidates from anywhere an abnormality exists
        volume_shape = scan["volume"].shape
        union_mask = np.zeros(volume_shape, dtype=np.uint8)
        for m in category_masks.values():
            union_mask = np.logical_or(union_mask, m).astype(np.uint8)
 
        # Sample patch centres from the union mask
        foreground_coords = np.argwhere(union_mask > 0)
        if len(foreground_coords) == 0:
            continue
 
        max_attempts = n_patches_per_scan * 10
        collected = 0
 
        for _ in range(max_attempts):
            if collected >= n_patches_per_scan:
                break
 
            idx    = rng.integers(0, len(foreground_coords))
            centre = tuple(foreground_coords[idx])
 
            patch = _extract_patch(scan["volume"], centre)
            if patch is None:
                continue
 
            # Assign to the category with the highest overlap
            best_cat, best_overlap = _majority_class(centre, category_masks)
            if best_cat is None or best_overlap < MIN_OVERLAP_RATIO:
                continue
 
            all_patches.append(patch)
            all_labels.append(class_to_idx[best_cat])
            collected += 1
 
        if collected < n_patches_per_scan:
            logger.debug(
                f"{scan_id}: collected {collected}/{n_patches_per_scan} patches."
            )
 
    # Log final counts per category
    label_arr = np.array(all_labels) if all_labels else np.array([], dtype=np.int64)
    abnormality_keys = [k for k in CLASS_ORDER if k != "normal"]
    for category in abnormality_keys:
        idx   = class_to_idx[category]
        count = int((label_arr == idx).sum()) if len(label_arr) else 0
        logger.info(f"  → {count} patches for {category}")
 
    return all_patches, all_labels

# ─── Assembly ─────────────────────────────────────────────────────────────────

def build_unified_patch_matrix(
    normal_ids: List[str],
    positive_ids: List[str],
    loader: ScanLoader,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run both phases and assemble into X (n_features, n_patches) and H (n_patches,).
 
    Normal patches: N_POSITIVE_PATCHES_PER_SCAN per scan.
    Abnormal patches: N_POSITIVE_PATCHES_PER_SCAN per scan (same constant), so
    class balance is governed by the normal/abnormal scan counts in the dataset
    and the single shared budget per scan.
    The NEG_TO_POS_RATIO is logged for reference but is no longer used to shrink
    the per-scan budget to near zero on large datasets.
    """
    # ── Phase 1: normal patches ───────────────────────────────────────────────
    n_per_scan = max(1, N_POSITIVE_PATCHES_PER_SCAN)
    normal_patches, normal_labels = collect_normal_patches(
        normal_ids, loader, n_per_scan=n_per_scan, rng=rng
    )
 
    n_normal_collected = len(normal_patches)
    logger.info(
        f"Normal patches: {n_normal_collected}  "
        f"abnormal budget = {n_per_scan} patches/scan)"
    )
 
    # ── Phase 2: abnormal patches ─────────────────────────────────────────────
    # FIX (issue 2): use a flat per-scan budget instead of dividing by
    # len(positive_ids)*n_abnorm_cats, which produced ~1 patch/scan on real datasets.
    abnormal_patches, abnormal_labels = collect_abnormal_patches(
        positive_ids, loader, n_patches_per_scan=n_per_scan, rng=rng
    )
 
    if len(abnormal_patches) == 0:
        raise RuntimeError(
            "No abnormal patches collected. "
            "Check MASKS_DIR, METADATA_JSON, and MIN_OVERLAP_RATIO."
        )
 
    # ── Combine ───────────────────────────────────────────────────────────────
    all_patches = normal_patches + abnormal_patches
    all_labels  = normal_labels  + abnormal_labels
 
    # ── Assemble X and H ──────────────────────────────────────────────────────
    n_patches = len(all_patches)
    X = np.zeros((N_FEATURES, n_patches), dtype=np.float64)
    H = np.empty(n_patches, dtype=np.int64)
 
    for j, (patch, label) in enumerate(zip(all_patches, all_labels)):
        X[:, j] = patch.ravel(order='C')
        H[j]    = label
 
    logger.info(f"Final matrix: X={X.shape}, H={H.shape}")
    return X, H


# ─── Public API ───────────────────────────────────────────────────────────────

def extract_unified(split: str = "train") -> None:
    """
    Run both phases of patch extraction and save a single .npz:
        patches/unified_{split}.npz
    """
    PATCHES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PATCHES_DIR / f"unified_{split}.npz"

    if out_path.exists():
        logger.info(f"Already exists: {out_path}, skipping.")
        return

    rng      = np.random.default_rng(RANDOM_SEED)
    metadata = MetadataRegistry(split=split)
    labels   = LabelRegistry(metadata, split=split)
    loader   = ScanLoader(metadata)

    def _filter_existing(ids: List[str]) -> List[str]:
        valid = []
        missing = []
        for vid in ids:
            try:
                resolve_volume_path(vid)
                valid.append(vid)
            except Exception:
                missing.append(vid)

        if missing:
            logger.debug(
                f"_filter_existing: {len(missing)} missing volumes (sample up to 5): {missing[:5]}"
            )
        return valid

    # Abnormal: any scan with at least one recognised finding
    abnormality_keys = [k for k in CLASS_ORDER if k != "normal"]

    # Raw lists per category (before dedup & disk existence filtering)
    raw_by_cat = {ab: labels.get_positive_volume_names(ab) for ab in abnormality_keys}
    for ab, lst in raw_by_cat.items():
        logger.info(f"  category {ab}: {len(lst)} volumes (sample: {lst[:3]})")

    positive_ids: List[str] = list({
        vid
        for ab in abnormality_keys
        for vid in raw_by_cat.get(ab, [])
    })
    logger.info(f"Raw positive IDs deduped: {len(positive_ids)} (sample: {positive_ids[:5]})")

    positive_ids = _filter_existing(positive_ids)
    logger.info(f"After path resolution: abnormal={len(positive_ids)}")

    raw_normals = labels.get_normal_volume_names()
    logger.info(f"Raw normal IDs from metadata: {len(raw_normals)}, e.g. {raw_normals[:3]}")
    if len(raw_normals) == 0:
        logger.warning(
            "No normal volumes found in metadata — dataset may contain only abnormal scans."
        )
    normal_ids = _filter_existing(raw_normals)

    logger.info(
        f"Split={split} | abnormal={len(positive_ids)} | normal={len(normal_ids)}"
    )

    X, H = build_unified_patch_matrix(normal_ids, positive_ids, loader, rng)

    np.savez_compressed(out_path, X=X, H=H)
    logger.info(f"Saved → {out_path}  (X: {X.shape}, H: {H.shape})")


def load_unified_patch_matrix(split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
    path = PATCHES_DIR / f"unified_{split}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Patch matrix not found: {path}. Run extract_unified() first."
        )
    data = np.load(path)
    return data["X"], data["H"]


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    extract_unified(split="train")