"""
nifti_io.py
Low-level I/O for the ReXGroundingCT dataset:
  - Resolves the nested volume/mask path from a scan ID
  - Loads NIfTI CT volumes and 4D segmentation masks
  - Resamples to isotropic target spacing
  - Applies HU windowing and lung-mask preprocessing to [0, 1] normalisation
"""

import warnings
from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage
import SimpleITK as sitk
from lungmask import LMInferer

from lc_ksvd.config import (
    BACKGROUND_HU, LOWER_HU, MASKS_DIR, TARGET_SPACING_MM,
    UPPER_HU, VOLUMES_DIR,
)

lung_inferer = LMInferer(tqdm_disable=True)

# ─── Path resolution ──────────────────────────────────────────────────────────

def _stem(filename: str) -> str:
    """Return the scan ID (filename without .nii.gz or .nii suffix)."""
    return filename.replace(".nii.gz", "").replace(".nii", "")


def resolve_volume_path(volume_name: str) -> Path:
    """Volumes are flat in VOLUMES_DIR."""
    path = VOLUMES_DIR / f"{volume_name}.nii.gz"
    if not path.exists():
        raise FileNotFoundError(f"Volume not found: {path}")
    return path


def resolve_mask_path(volume_name: str) -> Path:
    """Masks are flat in MASKS_DIR with the same filename as the volume."""
    path = MASKS_DIR / f"{volume_name}.nii.gz"
    if not path.exists():
        raise FileNotFoundError(f"Mask not found: {path}")
    return path


# ─── NIfTI I/O ────────────────────────────────────────────────────────────────

def load_volume(volume_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a CT volume and return:
      volume_hu   : float32 array [H, W, D] in raw HU values
      voxel_spacing: float array [3] (mm per voxel, x/y/z)
    """
    path = resolve_volume_path(volume_name)
    img  = nib.load(str(path))
    vol  = np.asarray(img.dataobj, dtype=np.float32)

    # nibabel loads in (x, y, z); we keep (H=x, W=y, D=z) convention
    zooms = np.abs(np.array(img.header.get_zooms()[:3], dtype=np.float32))
    return vol, zooms


def load_mask(volume_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the 4D segmentation mask and return:
      mask        : uint8 array [F, H, W, D]
      voxel_spacing: float array [3]
    """
    path = resolve_mask_path(volume_name)
    img  = nib.load(str(path))
    mask = np.asarray(img.dataobj, dtype=np.uint8)

    zooms = np.abs(np.array(img.header.get_zooms()[:3], dtype=np.float32))
    return mask, zooms


# ─── Resampling ───────────────────────────────────────────────────────────────

def _zoom_factors(current_spacing: np.ndarray, target_spacing: float) -> np.ndarray:
    return current_spacing / target_spacing


def resample_volume(vol: np.ndarray, current_spacing: np.ndarray) -> np.ndarray:
    """Resample volume to TARGET_SPACING_MM isotropic using trilinear interpolation."""
    factors = _zoom_factors(current_spacing, TARGET_SPACING_MM)
    if np.allclose(factors, 1.0, atol=0.01):
        return vol
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resampled = ndimage.zoom(vol, factors, order=1, mode='nearest')   # order=1 → trilinear, mode='nearest' to avoid blending edge voxels with 0
    return resampled.astype(np.float32)


def resample_mask(mask: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Resample each finding slice of the 4D mask to match target_shape exactly,
    using nearest-neighbour to preserve integer labels.
    target_shape should be the spatial shape of the already-resampled volume.
    """
    current_shape = np.array(mask.shape[1:], dtype=float)  # (H, W, D)
    target        = np.array(target_shape,   dtype=float)
    factors       = target / current_shape

    if np.allclose(factors, 1.0, atol=0.01):
        return mask

    resampled_slices = []
    for f in range(mask.shape[0]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rs = ndimage.zoom(mask[f], factors, order=0, mode='nearest')
        resampled_slices.append(rs.astype(np.uint8))
    return np.stack(resampled_slices, axis=0)



def preprocess(vol: np.ndarray) -> np.ndarray:
    """
    Lung preprocessing using lungmask.

    Steps:
        1. Convert numpy CT volume to SimpleITK image
        2. Predict lung segmentation using lungmask
        3. Apply lung mask to CT
        4. Set outside-lung voxels to -1000 HU
        5. Clip HU range
        6. Rescale to [0,1]

    Input:
        vol: float32 CT volume in HU
              Shape expected: (H, W, D)

    Output:
        float32 volume in [0,1]
    """

    # SimpleITK expects (z,y,x)
    vol_sitk = sitk.GetImageFromArray(np.transpose(vol, (2, 0, 1)))
    segmentation = lung_inferer.apply(vol_sitk)

    # lungmask returns:
    # 0 = background
    # 1 = left lung
    # 2 = right lung

    lung_mask = segmentation > 0


    # Convert back to (H,W,D)
    lung_mask = np.transpose(
        lung_mask,
        (1,2,0)
    )

    vol = vol.copy()
    vol[~lung_mask] = BACKGROUND_HU
    vol = np.clip(vol, LOWER_HU, UPPER_HU)
    vol = (vol - LOWER_HU) / (UPPER_HU - LOWER_HU)

    return vol.astype(np.float32)