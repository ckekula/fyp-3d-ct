"""
patch_extractor.py
Extracts 3D patches from CT volumes and builds the (n_features, n_patches) matrix X,
the (n_patches,) integer label vector H, and the (n_patches,) scan-ID string array
scan_ids required for LC-KSVD2 training and scan-level evaluation.

Phase 1 — Normal scans:
  Extract a non-overlapping grid of patches (stride = PATCH_SIZE) across the entire
  volume. Patches where more than 50% of voxels are zero (background) are discarded.
  All accepted patches are labelled as class index 0 ("normal").

Phase 2 — Abnormal scans:
  For each scan and each category present in its finding map, compute the tight
  bounding box of the category mask's foreground, then slide a patch window over
  that bounding box with stride=2 in every axis direction. Every in-bounds patch
  within the bounding box is included regardless of whether its centre overlaps
  foreground. Patches with >50% zero voxels are discarded. Each patch is directly
  labelled with the category being iterated.

The resulting matrices are saved as compressed .npz files to PATCHES_DIR.

Memory note: patches are streamed to a temporary on-disk binary file as they are
produced (see _PatchStore) rather than accumulated in Python lists, so peak RAM
stays bounded (roughly one patch at a time) regardless of dataset size. Only the
much smaller H (labels) and scan_ids arrays are kept fully in memory.
"""

import logging
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from lc_ksvd.config import (
    CLASS_ORDER, N_FEATURES, PATCH_SIZE, PATCHES_DIR, ABNORMAL_STRIDE, ZERO_FRACTION_THRESHOLD
)
from lc_ksvd.data_loader import (
    LabelRegistry,
    MetadataRegistry,
    ScanLoader,
    resolve_volume_path,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ─── Low-level patch utilities ────────────────────────────────────────────────

def _extract_patch(
    volume: np.ndarray,
    x0: int,
    y0: int,
    z0: int,
) -> Optional[np.ndarray]:
    """
    Extract a PATCH_SIZE³ patch with its top-left-front corner at (x0, y0, z0).
    Returns None if the patch would exceed volume bounds.
    """
    p = PATCH_SIZE
    H, W, D = volume.shape
    x1, y1, z1 = x0 + p, y0 + p, z0 + p

    if x1 > H or y1 > W or z1 > D:
        return None

    return volume[x0:x1, y0:y1, z0:z1].copy()


def _is_background(patch: np.ndarray) -> bool:
    """Return True if more than 50% of voxels are zero (background)."""
    return (patch < 1e-6).mean() > ZERO_FRACTION_THRESHOLD


# ─── Disk-backed patch accumulation ──────────────────────────────────────────

class _PatchStore:
    """
    Streams patch feature vectors to a temporary binary file on disk instead of
    holding them in a Python list, so peak memory stays roughly O(1 patch)
    regardless of how many patches the full dataset produces.

    Labels and scan IDs are kept in ordinary Python lists — these are tiny
    (one int / one short string per patch) compared to the patch data itself
    (N_FEATURES floats per patch), so keeping them in RAM is not a concern.
    """

    def __init__(self, path: Path, n_features: int):
        self.path = path
        self.n_features = n_features
        self._fh = open(path, "wb")
        self.labels: List[int] = []
        self.scan_ids: List[str] = []

    def add(self, patch: np.ndarray, label: int, scan_id: str) -> None:
        flat = np.ascontiguousarray(patch.ravel(order="C"), dtype=np.float64)
        if flat.size != self.n_features:
            raise ValueError(
                f"Patch has {flat.size} voxels, expected N_FEATURES={self.n_features}"
            )
        self._fh.write(flat.tobytes())
        self.labels.append(label)
        self.scan_ids.append(scan_id)

    def __len__(self) -> int:
        return len(self.labels)

    def close(self) -> int:
        """Flush and close the backing file. Returns the number of patches written."""
        self._fh.flush()
        self._fh.close()
        return len(self.labels)


# ─── Phase 1: Normal grid sampling ───────────────────────────────────────────

def _grid_origins(
    volume_shape: Tuple[int, int, int],
    stride: int,
) -> Generator[Tuple[int, int, int], None, None]:
    """Yield (x0, y0, z0) top-left-front corners on a regular grid."""
    H, W, D = volume_shape
    p = PATCH_SIZE
    for x0 in range(0, H - p + 1, stride):
        for y0 in range(0, W - p + 1, stride):
            for z0 in range(0, D - p + 1, stride):
                yield x0, y0, z0


def sample_normal_patches(
    volume: np.ndarray,
    store: "_PatchStore",
    label: int,
    scan_id: str,
) -> int:
    """
    Extract a non-overlapping grid of patches (stride = PATCH_SIZE) over the
    full volume, discarding patches where >50% of voxels are zero. Accepted
    patches are written directly to `store`. Returns the count added.
    """
    n_added = 0
    for x0, y0, z0 in _grid_origins(volume.shape, stride=PATCH_SIZE):
        patch = _extract_patch(volume, x0, y0, z0)
        if patch is None:
            continue
        if _is_background(patch):
            continue
        store.add(patch, label, scan_id)
        n_added += 1
    return n_added


def collect_normal_patches(
    normal_ids: List[str],
    loader: ScanLoader,
    store: "_PatchStore",
) -> int:
    """
    Phase 1: grid-sample all normal scans, streaming patches into `store` and
    labelling them all as CLASS_ORDER.index("normal"). Returns total count added.
    """
    normal_class_idx = CLASS_ORDER.index("normal")
    total_added = 0

    logger.info(f"Phase 1 — grid-sampling {len(normal_ids)} normal scans…")

    for scan_id in tqdm(normal_ids, desc="normal scans"):
        try:
            scan = loader.load(scan_id)
        except Exception as exc:
            logger.warning(f"Skipping {scan_id}: {exc}")
            continue

        n_added = sample_normal_patches(scan["volume"], store, normal_class_idx, scan_id)
        total_added += n_added

        logger.debug(f"  {scan_id}: {n_added} normal patches")

    logger.info(f"  → {total_added} total normal patches collected.")
    return total_added


# ─── Phase 2: Abnormal bounding-box sampling ─────────────────────────────────

def _build_category_masks(
    mask_4d: np.ndarray,
    finding_map: Dict[int, str],
) -> Dict[str, np.ndarray]:
    """
    Collapse the 4D mask [F, H, W, D] into per-category binary masks by OR-ing
    all finding slices that share the same category. Categories absent from
    CLASS_ORDER are skipped. Returns only masks with at least one foreground voxel.
    """
    volume_shape = mask_4d.shape[1:]          # (H, W, D)
    category_masks: Dict[str, np.ndarray] = {}

    for f_idx, category in finding_map.items():
        if category not in CLASS_ORDER:
            continue
        if f_idx >= mask_4d.shape[0]:
            logger.warning(
                f"f_idx={f_idx} out of range for mask shape {mask_4d.shape}; skipping."
            )
            continue

        finding_bin = (mask_4d[f_idx] > 0).astype(np.uint8)
        if category not in category_masks:
            category_masks[category] = np.zeros(volume_shape, dtype=np.uint8)
        np.logical_or(category_masks[category], finding_bin, out=category_masks[category])

    return {cat: m for cat, m in category_masks.items() if m.sum() > 0}


def _foreground_bbox(
    binary_mask: np.ndarray,
) -> Tuple[int, int, int, int, int, int]:
    """
    Return the tight axis-aligned bounding box of foreground voxels as
    (x_min, x_max, y_min, y_max, z_min, z_max) — all inclusive.
    Assumes binary_mask has at least one foreground voxel.
    """
    coords = np.argwhere(binary_mask > 0)
    x_min, y_min, z_min = coords.min(axis=0).tolist()
    x_max, y_max, z_max = coords.max(axis=0).tolist()
    return x_min, x_max, y_min, y_max, z_min, z_max


def _bbox_origins(
    bbox: Tuple[int, int, int, int, int, int],
    volume_shape: Tuple[int, int, int],
    stride: int,
) -> Generator[Tuple[int, int, int], None, None]:
    """
    Yield all patch top-left-front corners whose patch window overlaps the
    bounding box and remains fully within the volume.

    To cover the entire bbox, the starting x0 ranges from
    max(0, x_min - PATCH_SIZE + 1) to min(H - PATCH_SIZE, x_max), and
    analogously for y and z, stepped by `stride`.
    """
    p = PATCH_SIZE
    H, W, D = volume_shape
    x_min, x_max, y_min, y_max, z_min, z_max = bbox

    x_start = max(0,        x_min - p + 1)
    x_stop  = min(H - p,    x_max)
    y_start = max(0,        y_min - p + 1)
    y_stop  = min(W - p,    y_max)
    z_start = max(0,        z_min - p + 1)
    z_stop  = min(D - p,    z_max)

    for x0 in range(x_start, x_stop + 1, stride):
        for y0 in range(y_start, y_stop + 1, stride):
            for z0 in range(z_start, z_stop + 1, stride):
                yield x0, y0, z0


def sample_abnormal_patches(
    volume: np.ndarray,
    category_mask: np.ndarray,
    category: str,
    class_to_idx: Dict[str, int],
    store: "_PatchStore",
    scan_id: str,
) -> int:
    """
    Slide a patch window with stride=ABNORMAL_STRIDE over the bounding box of
    `category_mask`. All in-bounds patches are included regardless of foreground
    overlap; patches with >50% zero voxels are discarded. Label is `category`.
    Accepted patches are written directly to `store`. Returns the count added.
    """
    label_idx = class_to_idx[category]
    bbox = _foreground_bbox(category_mask)

    n_added = 0
    for x0, y0, z0 in _bbox_origins(bbox, volume.shape, stride=ABNORMAL_STRIDE):
        patch = _extract_patch(volume, x0, y0, z0)
        if patch is None:
            continue
        if _is_background(patch):
            continue
        store.add(patch, label_idx, scan_id)
        n_added += 1

    return n_added


def collect_abnormal_patches(
    positive_ids: List[str],
    loader: ScanLoader,
    store: "_PatchStore",
) -> int:
    """
    Phase 2: for every abnormal scan, iterate over each category present,
    extract all bbox patches with stride=ABNORMAL_STRIDE, and label directly
    with that category. Patches are streamed into `store`. Returns total count added.
    """
    class_to_idx = {cls: i for i, cls in enumerate(CLASS_ORDER)}
    total_added = 0
    per_category_counts: Dict[str, int] = {cls: 0 for cls in CLASS_ORDER if cls != "normal"}

    logger.info(f"Phase 2 — bbox-sampling {len(positive_ids)} abnormal scans…")

    for scan_id in tqdm(positive_ids, desc="abnormal scans"):
        try:
            scan = loader.load(scan_id)
        except Exception as exc:
            logger.warning(f"Skipping {scan_id}: {exc}")
            continue

        if scan["mask"] is None or not scan["finding_map"]:
            logger.debug(f"  {scan_id}: no mask or finding_map, skipping.")
            continue

        category_masks = _build_category_masks(scan["mask"], scan["finding_map"])
        if not category_masks:
            logger.debug(f"  {scan_id}: no valid category masks, skipping.")
            continue

        scan_patch_count = 0
        for category, cat_mask in category_masks.items():
            n_added = sample_abnormal_patches(
                scan["volume"], cat_mask, category, class_to_idx, store, scan_id
            )
            total_added += n_added
            per_category_counts[category] += n_added
            scan_patch_count += n_added
            logger.debug(
                f"  {scan_id} [{category}]: {n_added} patches from bbox"
            )

        logger.debug(f"  {scan_id}: {scan_patch_count} total patches across all categories")

    # Per-category summary
    for category, count in per_category_counts.items():
        logger.info(f"  → {count} patches for '{category}'")

    logger.info(f"  → {total_added} total abnormal patches collected.")
    return total_added


# ─── Assembly ─────────────────────────────────────────────────────────────────

def build_unified_patch_matrix(
    normal_ids: List[str],
    positive_ids: List[str],
    loader: ScanLoader,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    """
    Run both phases, streaming patches to a temporary on-disk file, and assemble:
      X        — (n_features, n_patches)  float64, a read-only memmap over the
                  temp file (no full in-RAM copy of the patch data)
      H        — (n_patches,)             int64
      scan_ids — (n_patches,)             object (str)

    Also returns the temp file path so the caller can delete it once X has been
    persisted to its final destination.
    """
    PATCHES_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = PATCHES_DIR / ".patch_store.tmp.bin"

    store = _PatchStore(tmp_path, N_FEATURES)
    try:
        collect_normal_patches(normal_ids, loader, store)
        n_abnormal = collect_abnormal_patches(positive_ids, loader, store)
    finally:
        n_patches = store.close()

    if n_abnormal == 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            "No abnormal patches collected. "
            "Check MASKS_DIR, METADATA_JSON, and foreground mask contents."
        )

    # Patches were appended row-by-row: memmap as (n_patches, n_features), then
    # transpose (a strided view, no copy) to the expected (n_features, n_patches)
    # orientation.
    X = np.memmap(
        tmp_path, dtype=np.float64, mode="r", shape=(n_patches, N_FEATURES)
    ).T
    H = np.array(store.labels, dtype=np.int64)
    scan_ids = np.array(store.scan_ids, dtype=object)

    logger.info(
        f"Final matrix: X={X.shape}, H={H.shape}, scan_ids={scan_ids.shape}"
    )
    return X, H, scan_ids, tmp_path


# ─── Public API ───────────────────────────────────────────────────────────────

def extract_unified(split: str = "train") -> None:
    """
    Run both phases of patch extraction and save a single compressed .npz:
        patches/unified_{split}.npz
    Stores X, H, and scan_ids.
    """
    PATCHES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PATCHES_DIR / f"unified_{split}.npz"

    if out_path.exists():
        logger.info(f"Already exists: {out_path} — skipping.")
        return

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
                f"(sample ≤5): {missing[:5]}"
            )
        return valid

    abnormality_keys = [k for k in CLASS_ORDER if k != "normal"]

    raw_by_cat = {ab: labels.get_positive_volume_names(ab) for ab in abnormality_keys}
    for ab, lst in raw_by_cat.items():
        logger.info(f"  category '{ab}': {len(lst)} volumes")

    positive_ids: List[str] = list({
        vid
        for ab in abnormality_keys
        for vid in raw_by_cat.get(ab, [])
    })
    logger.info(f"Raw positive IDs (deduped): {len(positive_ids)}")
    positive_ids = _filter_existing(positive_ids)
    logger.info(f"After path resolution: abnormal={len(positive_ids)}")

    raw_normals = labels.get_normal_volume_names()
    logger.info(f"Raw normal IDs from metadata: {len(raw_normals)}")
    if not raw_normals:
        logger.warning(
            "No normal volumes found in metadata — "
            "dataset may contain only abnormal scans."
        )
    normal_ids = _filter_existing(raw_normals)

    logger.info(
        f"Split={split!r} | normal={len(normal_ids)} | abnormal={len(positive_ids)}"
    )

    X, H, scan_ids, tmp_path = build_unified_patch_matrix(normal_ids, positive_ids, loader)

    try:
        np.savez_compressed(out_path, X=X, H=H, scan_ids=scan_ids)
    finally:
        # X is a memmap view over tmp_path; drop the reference before deleting
        # the backing file so no process holds it open.
        del X
        tmp_path.unlink(missing_ok=True)

    logger.info(f"Saved → {out_path}  (H: {H.shape})")


def load_unified_patch_matrix(
    split: str = "train",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and return X (n_features, n_patches), H (n_patches,), scan_ids (n_patches,).
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