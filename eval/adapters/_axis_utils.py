# eval/adapters/_axis_utils.py

from __future__ import annotations

import numpy as np


def xyz_to_zyx(mask: np.ndarray, spacing_xyz: tuple[float, float, float]) -> tuple[np.ndarray, tuple[float, float, float]]:
    """
    Convert a (X, Y, Z)-ordered mask + (x, y, z) mm spacing to the (Z, Y, X)
    array order + (z, y, x) spacing convention used by
    BiomedParseLocalizationAdapter / MerlinLocalizationAdapter.

    Dice/IoU are axis-order-agnostic (elementwise set operations), so this
    mismatch was previously silent. It matters once physically-aware metrics
    are computed (e.g. surface distance / centroid distance in mm), which
    multiply voxel offsets by per-axis spacing -- if array axis order and
    spacing order don't agree, those distances are silently wrong. Every
    localization adapter must produce masks/spacing in ZYX before
    constructing a LocalizationSample.
    """
    mask = np.asarray(mask)
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape: {mask.shape}")

    mask_zyx = np.transpose(mask, (2, 1, 0))
    spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
    return mask_zyx, spacing_zyx
