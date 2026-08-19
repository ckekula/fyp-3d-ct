"""
patch_io.py
Low-level patch extraction primitives and streaming writer shared by both the
normal-grid and abnormal-bbox sampling phases.
"""

import os

import numpy as np

from lc_ksvd.config import N_FEATURES, PATCH_SIZE

RECORD_BYTES = N_FEATURES * np.dtype(np.float32).itemsize


class _PatchStreamWriter:
    """Appends raveled patches (float32) to a scratch file, avoiding
    in-memory accumulation of individual patch arrays during extraction.
    Also accumulates the (x0, y0, z0) origin of each written patch.

    When `resume_count` > 0, the scratch file is truncated to exactly
    `resume_count` records before appending, so a partially-flushed final
    record left over from a crash mid-write can't misalign later reads."""

    def __init__(self, path, resume_count: int = 0):
        self.path = path
        if resume_count and path.exists():
            with open(path, "r+b") as f:
                f.truncate(resume_count * RECORD_BYTES)
            self._fh = open(path, "ab")
        else:
            self._fh = open(path, "wb")
        self.count = 0
        self._closed = False
        self.coords: list[tuple[int, int, int]] = []

    def write(self, patch: np.ndarray, coord: tuple[int, int, int]) -> None:
        self._fh.write(np.ascontiguousarray(patch, dtype=np.float32).tobytes())
        self.coords.append(coord)
        self.count += 1

    def flush_scan(self) -> None:
        """Durably flush all writes so far to disk. Call at scan boundaries
        so a crash can't leave a partially-written record on disk that a
        checkpoint doesn't know about."""
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        if not self._closed:
            self._fh.close()
            self._closed = True

def extract_patch(
    volume: np.ndarray,
    x0: int,
    y0: int,
    z0: int,
) -> np.ndarray | None:
    """
    Extract a PATCH_SIZE (x, y, z) patch with its top-left-front corner at (x0, y0, z0).
    Returns None if the patch would exceed volume bounds.
    """
    px, py, pz = PATCH_SIZE
    H, W, D = volume.shape
    x1, y1, z1 = x0 + px, y0 + py, z0 + pz

    if x1 > H or y1 > W or z1 > D:
        return None

    return volume[x0:x1, y0:y1, z0:z1].copy()

