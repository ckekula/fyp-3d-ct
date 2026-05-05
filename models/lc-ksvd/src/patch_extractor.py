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

import config
from data_loader import LabelRegistry, MetadataRegistry, ScanLoader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

rng = np.random.default_rng(config.RANDOM_SEED)


# ─── Core patch utilities ────────────────────────────────────────────────────

def _extract_patch(volume: np.ndarray, centre: Tuple[int, int, int]) -> Optional[np.ndarray]:
    """
    Extract a cubic patch of PATCH_SIZE centred at (cx, cy, cz).
    Returns None if the patch extends outside the volume boundary.
    """
    p = config.PATCH_SIZE
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
    p = config.PATCH_SIZE
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
    h = config.PATCH_SIZE // 2
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

    while len(patches) < n_patches and attempts < max_attempts:
        attempts += 1
        idx = rng.integers(0, len(foreground_coords))
        centre = tuple(foreground_coords[idx])

        if _overlap_ratio(centre, binary_mask) >= config.MIN_OVERLAP_RATIO:
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


# ─── Matrix builder for one abnormality ──────────────────────────────────────

def build_patch_matrix(
    abnormality: str,
    scan_ids: Dict[str, List[str]],   # {"positive": [...], "negative": [...]}
    loader: ScanLoader,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build X (n_features, n_patches) and H (2, n_patches) for one binary model.

    H convention (matching reppi's one-hot expectation):
      row 0 = negative class (no abnormality)
      row 1 = positive class (abnormality present)

    Returns (X, H) as float64 arrays (reppi uses float64 internally).
    """
    positive_patches = []
    negative_patches = []

    # ── Positive scans ────────────────────────────────────────────────────────
    logger.info(f"[{abnormality}] Sampling positive patches from {len(scan_ids['positive'])} scans…")

    for scan_id in tqdm(scan_ids["positive"], desc=f"{abnormality} positive scans"):
        try:
            scan = loader.load(scan_id)
        except Exception as e:
            logger.warning(f"Skipping {scan_id}: {e}")
            continue

        binary_mask = get_binary_mask_for_abnormality(
            scan["mask"], scan["finding_map"], abnormality
        )

        if binary_mask is None or binary_mask.sum() == 0:
            # Volume labelled positive in CSV but no matching F-slice in metadata/mask.
            # Use as a negative-only source to avoid wasting the scan.
            negs = sample_negative_patches(
                scan["volume"], binary_mask=None,
                n_patches=config.N_POSITIVE_PATCHES_PER_SCAN,
            )
            negative_patches.extend(negs)
            logger.debug(f"  {scan_id}: no matching mask slice — used as negative source.")
            continue

        # Positive patches from the lesion region
        pos = sample_positive_patches(
            scan["volume"], binary_mask,
            n_patches=config.N_POSITIVE_PATCHES_PER_SCAN,
        )
        positive_patches.extend(pos)

        # Background negatives from the same scan (same number as positives)
        negs = sample_negative_patches(
            scan["volume"], binary_mask,
            n_patches=len(pos),
        )
        negative_patches.extend(negs)

    # ── Normal scans (pure negatives) ────────────────────────────────────────
    n_extra_neg_needed = max(
        0,
        int(len(positive_patches) * config.NEG_TO_POS_RATIO) - len(negative_patches)
    )

    if n_extra_neg_needed > 0 and scan_ids["normal"]:
        n_per_normal = max(1, n_extra_neg_needed // len(scan_ids["normal"]))
        logger.info(
            f"[{abnormality}] Sampling {n_per_normal} negative patches from "
            f"{len(scan_ids['normal'])} normal scans to balance…"
        )
        for scan_id in tqdm(scan_ids["normal"], desc=f"{abnormality} normal scans"):
            if len(negative_patches) >= int(len(positive_patches) * config.NEG_TO_POS_RATIO):
                break
            try:
                scan = loader.load(scan_id)
            except Exception as e:
                logger.warning(f"Skipping {scan_id}: {e}")
                continue
            negs = sample_negative_patches(
                scan["volume"], binary_mask=None, n_patches=n_per_normal
            )
            negative_patches.extend(negs)

    # ── Balance to NEG_TO_POS_RATIO ──────────────────────────────────────────
    n_pos = len(positive_patches)
    n_neg_target = int(n_pos * config.NEG_TO_POS_RATIO)
    if len(negative_patches) > n_neg_target:
        neg_idx = rng.choice(len(negative_patches), n_neg_target, replace=False)
        negative_patches = [negative_patches[i] for i in neg_idx]

    logger.info(
        f"[{abnormality}] Patch counts — positive: {len(positive_patches)}, "
        f"negative: {len(negative_patches)}"
    )

    if len(positive_patches) == 0:
        raise RuntimeError(
            f"No positive patches collected for {abnormality!r}. "
            "Check METADATA_JSON, LABELS_CSV, and ABNORMALITY_ALIASES in config.py."
        )

    # ── Assemble matrices ─────────────────────────────────────────────────────
    all_patches  = positive_patches + negative_patches
    labels       = [1] * len(positive_patches) + [0] * len(negative_patches)

    # Shuffle together
    order = rng.permutation(len(all_patches))
    all_patches  = [all_patches[i] for i in order]
    labels       = [labels[i] for i in order]

    n_patches  = len(all_patches)
    n_features = config.N_FEATURES

    # X: (n_features, n_patches) — column-major, matches reppi convention
    X = np.zeros((n_features, n_patches), dtype=np.float64)
    for j, patch in enumerate(all_patches):
        X[:, j] = patch.ravel()

    # H: (2, n_patches) — one-hot
    H = np.zeros((2, n_patches), dtype=np.float64)
    for j, lbl in enumerate(labels):
        H[lbl, j] = 1.0

    return X, H


# ─── Top-level extraction runner ─────────────────────────────────────────────

def extract_all_abnormalities(split: str = "train") -> None:
    """
    Run patch extraction for all 4 abnormalities and save .npz files.

    split: "train", "val", or "test" — used only for output naming;
           caller is responsible for passing the correct scan_id lists.
    """
    config.PATCHES_DIR.mkdir(parents=True, exist_ok=True)

    metadata = MetadataRegistry()
    labels   = LabelRegistry()
    loader   = ScanLoader(metadata)

    normal_ids = labels.get_normal_scan_ids()

    for abnormality in config.ABNORMALITIES:
        out_path = config.PATCHES_DIR / f"{abnormality}_{split}.npz"
        if out_path.exists():
            logger.info(f"[{abnormality}] Patch matrix already exists at {out_path}, skipping.")
            continue

        positive_ids = labels.get_positive_scan_ids(abnormality)
        logger.info(
            f"\n{'='*60}\n"
            f"Abnormality : {abnormality}\n"
            f"Positive scans: {len(positive_ids)}  |  Normal scans: {len(normal_ids)}\n"
            f"{'='*60}"
        )

        scan_ids = {"positive": positive_ids, "normal": normal_ids}
        X, H = build_patch_matrix(abnormality, scan_ids, loader)

        np.savez_compressed(out_path, X=X, H=H)
        logger.info(f"[{abnormality}] Saved {out_path}  (X shape: {X.shape}, H shape: {H.shape})")


def load_patch_matrix(abnormality: str, split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
    """Load a previously saved patch matrix from disk."""
    path = config.PATCHES_DIR / f"{abnormality}_{split}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Patch matrix not found: {path}. Run extract_all_abnormalities() first."
        )
    data = np.load(path)
    return data["X"], data["H"]


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting patch extraction for all abnormalities (train split)…")
    extract_all_abnormalities(split="train")
    logger.info("Done.")