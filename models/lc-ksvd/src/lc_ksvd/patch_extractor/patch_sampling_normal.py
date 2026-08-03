"""
patch_sampling_normal.py
Phase 1 — Normal scans:
  Extract a non-overlapping grid of patches (stride = PATCH_SIZE) across the
  entire volume. Patches where more than 50% of voxels are zero (background)
  are discarded. All accepted patches are labelled as class index 0 ("normal").
"""

import logging
from collections.abc import Generator

import numpy as np
from tqdm import tqdm

from lc_ksvd.config import (
    CLASS_ORDER,
    PATCH_SIZE,
    ZERO_FRACTION_THRESHOLD,
)
from lc_ksvd.data_loader.scan_loader import ScanLoader
from lc_ksvd.patch_extractor.patch_io import _PatchStreamWriter, extract_patch

logger = logging.getLogger(__name__)


def is_background(patch: np.ndarray) -> bool:
    """Return True if more than 50% of voxels are zero (background)."""
    return np.isclose(patch, -1.0, atol=1e-6).mean() > ZERO_FRACTION_THRESHOLD


def _grid_origins(
    volume_shape: tuple[int, int, int],
    stride: int,
) -> Generator[tuple[int, int, int], None, None]:
    """Yield (x0, y0, z0) top-left-front corners on a regular grid."""
    H, W, D = volume_shape
    p = PATCH_SIZE
    for x0 in range(0, H - p + 1, stride):
        for y0 in range(0, W - p + 1, stride):
            for z0 in range(0, D - p + 1, stride):
                yield x0, y0, z0


def sample_normal_patches(volume: np.ndarray, writer: _PatchStreamWriter) -> int:
    """Grid-sample the volume, writing accepted patches to `writer`. Returns count."""
    n = 0
    for x0, y0, z0 in _grid_origins(volume.shape, stride=PATCH_SIZE):
        patch = extract_patch(volume, x0, y0, z0)
        if patch is None or is_background(patch):
            continue
        writer.write(patch, (x0, y0, z0))
        n += 1
    return n


def collect_normal_patches(
    normal_ids: list[str],
    loader: ScanLoader,
    writer: _PatchStreamWriter,
) -> tuple[list[int], list[str], list[tuple[int, int, int]]]:
    normal_class_idx = CLASS_ORDER.index("normal")
    all_labels: list[int] = []
    all_scan_ids: list[str] = []

    logger.info(f"Phase 1 — grid-sampling {len(normal_ids)} normal scans…")

    for scan_id in tqdm(normal_ids, desc="normal scans"):
        try:
            scan = loader.load(scan_id)
        except Exception as exc:
            logger.warning(f"Skipping {scan_id}: {exc}")
            continue

        n = sample_normal_patches(scan["volume"], writer)
        all_labels.extend([normal_class_idx] * n)
        all_scan_ids.extend([scan_id] * n)
        logger.info(f"  {scan_id}: {n} normal patches")

    logger.info(f"  → {len(all_labels)} total normal patches collected.")
    return all_labels, all_scan_ids, writer.coords
