"""
patch_sampling_abnormal.py
Phase 2 — Abnormal scans:
  For each scan and each category present in its finding map, compute the tight
  bounding box of the category mask's foreground, then slide a patch window
  over that bounding box with stride=2 in every axis direction. Every in-bounds
  patch within the bounding box is included regardless of whether its centre
  overlaps foreground. Patches with >50% zero voxels are discarded. Each patch
  is directly labelled with the category being iterated.
"""

import logging
from typing import Dict, Generator, List, Tuple

import numpy as np
from tqdm import tqdm

from lc_ksvd.config import ABNORMAL_PATCH_STRIDE, CLASS_ORDER, PATCH_SIZE
from lc_ksvd.data_loader.scan_loader import ScanLoader
from lc_ksvd.patch_extractor.patch_io import _PatchStreamWriter, extract_patch, is_background

logger = logging.getLogger(__name__)


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
    writer: _PatchStreamWriter,
) -> List[int]:
    labels: List[int] = []
    label_idx = class_to_idx[category]
    bbox = _foreground_bbox(category_mask)

    for x0, y0, z0 in _bbox_origins(bbox, volume.shape, stride=ABNORMAL_PATCH_STRIDE):
        patch = extract_patch(volume, x0, y0, z0)
        if patch is None or is_background(patch):
            continue
        writer.write(patch)
        labels.append(label_idx)

    return labels


def collect_abnormal_patches(
    positive_ids: List[str],
    loader: ScanLoader,
    writer: _PatchStreamWriter,
) -> Tuple[List[int], List[str]]:
    all_labels: List[int] = []
    all_scan_ids: List[str] = []
    class_to_idx = {cls: i for i, cls in enumerate(CLASS_ORDER)}

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
            labels = sample_abnormal_patches(
                scan["volume"], cat_mask, category, class_to_idx, writer
            )
            all_labels.extend(labels)
            all_scan_ids.extend([scan_id] * len(labels))
            scan_patch_count += len(labels)
            logger.debug(f"  {scan_id} [{category}]: {len(labels)} patches from bbox")

        logger.debug(f"  {scan_id}: {scan_patch_count} total patches across all categories")

    label_arr = np.array(all_labels, dtype=np.int64) if all_labels else np.array([], dtype=np.int64)
    for category in [k for k in CLASS_ORDER if k != "normal"]:
        idx = class_to_idx[category]
        count = int((label_arr == idx).sum())
        logger.info(f"  → {count} patches for '{category}'")

    logger.info(f"  → {len(all_labels)} total abnormal patches collected.")
    return all_labels, all_scan_ids