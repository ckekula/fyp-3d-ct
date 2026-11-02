"""
Resample and align a ground-truth segmentation mask to the same voxel grid
used by the extracted patches.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from src import config


def _crop_or_pad_1d(size: int, target: int) -> tuple[slice, tuple[int, int]]:
    """Return (source_slice, pad_width) for a single axis.

    - If size >= target: a centered crop slice of length `target`, no padding.
    - If size < target: the full source slice, plus symmetric pad widths
      that bring it up to `target`.
    """
    if size >= target:
        start = (size - target) // 2
        return slice(start, start + target), (0, 0)
    total_pad = target - size
    before = total_pad // 2
    after = total_pad - before
    return slice(0, size), (before, after)


def crop_or_pad(
    volume: np.ndarray, target_shape: tuple[int, int, int], pad_value: float
) -> np.ndarray:
    """Center-crop and/or pad a 3D array to target_shape."""
    if volume.shape == tuple(target_shape):
        return volume

    slices, pad_widths = [], []
    for size, target in zip(volume.shape, target_shape):
        sl, pw = _crop_or_pad_1d(size, target)
        slices.append(sl)
        pad_widths.append(pw)

    cropped = volume[tuple(slices)]
    if any(pw != (0, 0) for pw in pad_widths):
        cropped = np.pad(cropped, pad_widths, mode="constant", constant_values=pad_value)
    return cropped


def resample_and_align_mask(raw_mask: np.ndarray, scan_id: str) -> np.ndarray:
    """Resample a raw segmentation mask to TARGET_SPACING_MM, then center
    crop/pad it to TARGET_SHAPE so it aligns with the patch coordinate space.

    Parameters
    ----------
    raw_mask : np.ndarray, shape (H, W, D)
        Binary/label mask straight from the NIfTI file, in its native spacing.
    scan_id : str
        Used to look up the scan's original voxel spacing.
    """
    original_spacing = config.get_scan_spacing(scan_id)
    zoom_factors = np.asarray(original_spacing, dtype=np.float32) / np.asarray(
        config.TARGET_SPACING_MM, dtype=np.float32
    )

    resampled = ndimage.zoom(
        raw_mask,
        zoom_factors,
        order=0,  # nearest-neighbour: preserve integer labels
        mode="nearest",
    )
    return crop_or_pad(resampled, config.TARGET_SHAPE, pad_value=0).astype(np.uint8)
