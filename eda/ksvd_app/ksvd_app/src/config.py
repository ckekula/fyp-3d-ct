"""
Central configuration for the K-SVD patch explorer.

All filesystem paths and dataset-specific constants live here so the rest
of the codebase never hardcodes a path or a magic number.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------
# Filesystem layout
# --------------------------------------------------------------------------
# Expected layout, relative to the app root:
#
#   data/
#     ksvd_checkpoint.npz          -> D, Gamma, n_nonzero_coefs, ...
#     unified_train_2d.npz         -> X, scan_ids, coords
#     dataset_filtered.json        -> per-scan finding metadata
#     segmentations/
#       <scan_id>.nii.gz           -> ground-truth mask for each scan
#
APP_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = APP_ROOT / "data"

KSVD_CHECKPOINT_PATH: Final[Path] = DATA_DIR / "ksvd_checkpoint.npz"
PATCHES_2D_PATH: Final[Path] = DATA_DIR / "unified_train_2d.npz"
DATASET_JSON_PATH: Final[Path] = DATA_DIR / "dataset_filtered.json"
SEGMENTATIONS_DIR: Final[Path] = DATA_DIR / "segmentations"


def segmentation_path(scan_id: str) -> Path:
    """Path to the ground-truth segmentation file for a given scan."""
    return SEGMENTATIONS_DIR / f"{scan_id}.nii.gz"


# --------------------------------------------------------------------------
# Patch geometry
# --------------------------------------------------------------------------
# Flattened patch vectors are (X, Y, Z) = (12, 12, 6) voxels, in that order.
PATCH_SHAPE_XYZ: Final[tuple[int, int, int]] = (12, 12, 6)

# --------------------------------------------------------------------------
# Volume resampling / cropping targets used when preparing ground-truth
# segmentation masks so they align with the patch coordinate space.
# --------------------------------------------------------------------------
TARGET_SPACING_MM: Final[tuple[float, float, float]] = (0.75, 0.75, 1.5)
TARGET_SHAPE: Final[tuple[int, int, int]] = (480, 480, 240)

# Original voxel spacing (mm) per scan, as stored in the source NIfTI header.
# Extend this table as more scans are added to `data/segmentations/`.
SCAN_ORIGINAL_SPACING_MM: Final[dict[str, tuple[float, float, float]]] = {
    "train_7880_a_1": (0.6738281, 0.6738281, 1.5),
    "train_59_a_2": (0.5761719, 0.5761719, 1.5),
    
}


def get_scan_spacing(scan_id: str) -> tuple[float, float, float]:
    """Look up the original voxel spacing for a scan.

    Raises
    ------
    KeyError
        If the scan isn't registered in SCAN_ORIGINAL_SPACING_MM. Callers
        should surface this as an actionable message rather than guessing
        a spacing, since an incorrect value silently corrupts the
        resampled mask alignment.
    """
    try:
        return SCAN_ORIGINAL_SPACING_MM[scan_id]
    except KeyError as exc:
        raise KeyError(
            f"No original spacing registered for scan '{scan_id}'. "
            "Add it to SCAN_ORIGINAL_SPACING_MM in src/config.py."
        ) from exc
