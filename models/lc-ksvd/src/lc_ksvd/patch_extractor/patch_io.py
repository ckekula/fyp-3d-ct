"""
patch_io.py
Low-level patch extraction primitives and streaming writer shared by both the
normal-grid and abnormal-bbox sampling phases.
"""

from typing import Optional

import numpy as np

from lc_ksvd.config import PATCH_SIZE


class _PatchStreamWriter:
    """Appends raveled patches (float32) to a scratch file, avoiding
    in-memory accumulation of individual patch arrays during extraction."""

    def __init__(self, path):
        self.path = path
        self._fh = open(path, "wb")
        self.count = 0
        self._closed = False

    def write(self, patch: np.ndarray) -> None:
        self._fh.write(np.ascontiguousarray(patch, dtype=np.float32).tobytes())
        self.count += 1

    def close(self) -> None:
        if not self._closed:
            self._fh.close()
            self._closed = True


def extract_patch(
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

