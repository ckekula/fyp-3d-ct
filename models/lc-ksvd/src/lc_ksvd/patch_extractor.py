"""
patch_extractor.py
Extracts 3D patches from CT volumes and builds the (n_features, n_patches) matrix X,
the (n_patches,) integer label vector H, and the (n_patches,) scan-ID string array
scan_ids required for LC-KSVD2 training and scan-level evaluation.

Phase 1 — Normal scans:
  Sample patches from anywhere in the volume (no mask constraint).
  All labelled as class index 0 ("normal").

Phase 2 — Abnormal scans:
  For each scan, load all finding masks from the 4D segmentation.
  For each finding, extract patches whose centres fall within that finding's mask
  foreground. Each patch is labelled by its majority class across all findings in
  that scan (handles overlapping findings via _majority_class).
  The MIN_OVERLAP_RATIO gate has been removed: any patch whose centre lands on a
  foreground voxel is accepted. Diffuse findings such as ground-glass opacity (2c)
  produce patches that are predominantly normal tissue; the centre-in-foreground
  criterion is the meaningful positive signal for such classes.

Assembly:
  After both phases, minority classes are upsampled with replacement to match the
  patch count of the largest class, preventing the dictionary from being dominated
  by normal or majority-abnormality patches.

The resulting matrices are saved as compressed .npz files to PATCHES_DIR.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from lc_ksvd.config import (
    N_FEATURES,
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
    """
    Fraction of patch voxels that are foreground in binary_mask.
    Used only by _majority_class for tie-breaking between overlapping abnormalities;
    no longer used as an acceptance gate.
    """
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
) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """
    Phase 1: collect patches from all normal scans.
    Returns flat lists of patches, integer class labels (all 0), and scan IDs.
    """
    normal_class_idx = CLASS_ORDER.index("normal")
    all_patches: List[np.ndarray] = []
    all_labels: List[int] = []
    all_scan_ids: List[str] = []

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
        all_scan_ids.extend([scan_id] * len(patches))

    logger.info(f"  → {len(all_patches)} normal patches collected.")
    return all_patches, all_labels, all_scan_ids


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
    Among all per-category masks, return the one with the highest voxel overlap
    with the patch centred at `centre`, and its overlap ratio.

    The overlap ratio is used only for disambiguation between categories; it is
    no longer used as an acceptance gate (that gate has been removed).
    A centre that lands on foreground in exactly one category mask will have
    overlap > 0 for that category and 0 for all others.
    """
    best_cat, best_overlap = None, 0.0

    for cat, mask in category_masks.items():
        overlap = _overlap_ratio(centre, mask)
        if overlap > best_overlap:
            best_overlap = overlap
            best_cat = cat

    return best_cat, best_overlap


def collect_abnormal_patches(
    positive_ids: List[str],
    loader: ScanLoader,
    n_patches_per_scan: int,
    rng: np.random.Generator,
) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """
    Phase 2: collect patches from all abnormal scans.

    For each scan:
      1. Build per-category masks via _build_finding_masks.
      2. Build a union mask across all categories.
      3. Sample patch centres from foreground voxels of the union mask.
      4. Label each patch by the category with the highest overlap
         (_majority_class). No minimum overlap threshold is applied —
         any centre in a foreground voxel is accepted.

    Returns flat lists of patches, integer class labels, and scan IDs.
    """
    all_patches: List[np.ndarray] = []
    all_labels:  List[int]        = []
    all_scan_ids: List[str]       = []
    class_to_idx = {cls: i for i, cls in enumerate(CLASS_ORDER)}

    for scan_id in tqdm(positive_ids, desc="abnormal scans"):
        try:
            scan = loader.load(scan_id)
        except Exception as e:
            logger.warning(f"Skipping {scan_id}: {e}")
            continue

        if scan["mask"] is None or not scan["finding_map"]:
            continue

        category_masks = _build_finding_masks(scan["mask"], scan["finding_map"])
        if not category_masks:
            continue

        volume_shape = scan["volume"].shape
        union_mask = np.zeros(volume_shape, dtype=np.uint8)
        for m in category_masks.values():
            union_mask = np.logical_or(union_mask, m).astype(np.uint8)

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

            # Label by highest-overlap category; no minimum overlap gate.
            best_cat, _ = _majority_class(centre, category_masks)
            if best_cat is None:
                continue

            all_patches.append(patch)
            all_labels.append(class_to_idx[best_cat])
            all_scan_ids.append(scan_id)
            collected += 1

        if collected < n_patches_per_scan:
            logger.debug(
                f"{scan_id}: collected {collected}/{n_patches_per_scan} patches."
            )

    # Log per-category counts before upsampling
    label_arr = np.array(all_labels) if all_labels else np.array([], dtype=np.int64)
    for category in [k for k in CLASS_ORDER if k != "normal"]:
        idx   = class_to_idx[category]
        count = int((label_arr == idx).sum()) if len(label_arr) else 0
        logger.info(f"  → {count} patches for {category} (before upsampling)")

    return all_patches, all_labels, all_scan_ids


# ─── Class-balance upsampling ─────────────────────────────────────────────────

def upsample_to_balance(
    patches: List[np.ndarray],
    labels: List[int],
    scan_ids: List[str],
    rng: np.random.Generator,
) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """
    Upsample minority classes with replacement so every class has the same
    number of patches as the largest class.

    Upsampling (rather than downsampling) is used to avoid discarding
    hard-won abnormal patches, which are already scarce.

    scan_ids are carried through so the balanced matrix remains compatible
    with scan-level evaluation.
    """
    label_arr = np.array(labels, dtype=np.int64)
    classes   = sorted(set(labels))
    counts    = {c: int((label_arr == c).sum()) for c in classes}
    target    = max(counts.values())

    logger.info(f"Upsampling to {target} patches per class. Before: {counts}")

    balanced_patches:  List[np.ndarray] = list(patches)
    balanced_labels:   List[int]        = list(labels)
    balanced_scan_ids: List[str]        = list(scan_ids)

    for cls in classes:
        deficit = target - counts[cls]
        if deficit <= 0:
            continue

        cls_indices = np.where(label_arr == cls)[0]
        chosen = rng.choice(cls_indices, size=deficit, replace=True)

        for i in chosen:
            balanced_patches.append(patches[i])
            balanced_labels.append(labels[i])
            balanced_scan_ids.append(scan_ids[i])

    new_counts = {c: int((np.array(balanced_labels) == c).sum()) for c in classes}
    logger.info(f"After upsampling: {new_counts}")

    return balanced_patches, balanced_labels, balanced_scan_ids


# ─── Assembly ─────────────────────────────────────────────────────────────────

def build_unified_patch_matrix(
    normal_ids: List[str],
    positive_ids: List[str],
    loader: ScanLoader,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run both phases, apply class-balance upsampling, and assemble into:
      X        — (n_features, n_patches)    float64
      H        — (n_patches,)               int64
      scan_ids — (n_patches,)               str  (for scan-level evaluation)
    """
    n_per_scan = max(1, N_POSITIVE_PATCHES_PER_SCAN)

    # Phase 1
    normal_patches, normal_labels, normal_scan_ids = collect_normal_patches(
        normal_ids, loader, n_per_scan=n_per_scan, rng=rng
    )

    # Phase 2
    abnormal_patches, abnormal_labels, abnormal_scan_ids = collect_abnormal_patches(
        positive_ids, loader, n_patches_per_scan=n_per_scan, rng=rng
    )

    if len(abnormal_patches) == 0:
        raise RuntimeError(
            "No abnormal patches collected. "
            "Check MASKS_DIR, METADATA_JSON, and the foreground mask contents."
        )

    # Combine before balancing so normal is included in the target count
    all_patches  = normal_patches  + abnormal_patches
    all_labels   = normal_labels   + abnormal_labels
    all_scan_ids = normal_scan_ids + abnormal_scan_ids

    # Balance classes by upsampling minorities
    all_patches, all_labels, all_scan_ids = upsample_to_balance(
        all_patches, all_labels, all_scan_ids, rng
    )

    # Assemble X and H
    n_patches = len(all_patches)
    X        = np.zeros((N_FEATURES, n_patches), dtype=np.float64)
    H        = np.empty(n_patches, dtype=np.int64)
    scan_ids = np.empty(n_patches, dtype=object)

    for j, (patch, label, sid) in enumerate(zip(all_patches, all_labels, all_scan_ids)):
        X[:, j]        = patch.ravel(order='C')
        H[j]           = label
        scan_ids[j]    = sid

    logger.info(f"Final matrix: X={X.shape}, H={H.shape}, scan_ids={scan_ids.shape}")
    return X, H, scan_ids


# ─── Public API ───────────────────────────────────────────────────────────────

def extract_unified(split: str = "train") -> None:
    """
    Run both phases of patch extraction and save a single .npz:
        patches/unified_{split}.npz
    Stores X, H, and scan_ids.
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
        valid, missing = [], []
        for vid in ids:
            try:
                resolve_volume_path(vid)
                valid.append(vid)
            except Exception:
                missing.append(vid)
        if missing:
            logger.debug(
                f"_filter_existing: {len(missing)} missing volumes "
                f"(sample up to 5): {missing[:5]}"
            )
        return valid

    abnormality_keys = [k for k in CLASS_ORDER if k != "normal"]

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

    X, H, scan_ids = build_unified_patch_matrix(normal_ids, positive_ids, loader, rng)

    np.savez_compressed(out_path, X=X, H=H, scan_ids=scan_ids)
    logger.info(f"Saved → {out_path}  (X: {X.shape}, H: {H.shape})")


def load_unified_patch_matrix(
    split: str = "train",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns X (n_features, n_patches), H (n_patches,), scan_ids (n_patches,).
    """
    path = PATCHES_DIR / f"unified_{split}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Patch matrix not found: {path}. Run extract_unified() first."
        )
    data = np.load(path, allow_pickle=True)
    return data["X"], data["H"], data["scan_ids"]


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    extract_unified(split="train")