"""
data_loader.py
Handles all I/O for the ReXGroundingCT dataset:
  - Resolves the nested volume path from a scan ID
  - Loads NIfTI CT volumes and 4D segmentation masks
  - Resamples to isotropic target spacing
  - Applies HU windowing and [0,1] normalisation
  - Parses the JSON metadata to identify which F-slice maps to which abnormality
  - Parses the CSV to get volume-level one-hot labels
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

from lc_ksvd.config import ABNORMALITY_CATEGORIES, HU_MAX, HU_MIN, MASKS_DIR, METADATA_JSON, TARGET_SPACING_MM, VOLUMES_DIR


# ─── Path resolution ──────────────────────────────────────────────────────────

def _stem(filename: str) -> str:
    """Return the scan ID (filename without .nii.gz or .nii suffix)."""
    return filename.replace(".nii.gz", "").replace(".nii", "")


def resolve_volume_path(volume_name: str) -> Path:
    """
    Reconstruct the nested volume path from a scan ID.

    Naming convention: train_<study>_<series>_<number>
    e.g. "train_1_a_1"  →  volumes/dataset/train/train_1/train_1_a/train_1_a_1.nii.gz

    The nesting is: VOLUMES_DIR / train_<study> / train_<study>_<series> / <scan_id>.nii.gz
    """
    parts = volume_name.split("_")          # ["train", "1", "a", "1"]
    if len(parts) < 4 or parts[0] != "train":
        raise ValueError(f"Unexpected volume_name format: {volume_name!r}")

    study  = parts[1]                   # "1"
    series = parts[2]                   # "a"

    study_dir  = f"train_{study}"                   # "train_1"
    series_dir = f"train_{study}_{series}"          # "train_1_a"

    candidate = (
        VOLUMES_DIR
        / study_dir
        / series_dir
        / f"{volume_name}.nii.gz"
    )
    if candidate.exists():
        return candidate

    # Fallback: some series use numeric identifiers ("train_1_1")
    series_dir_num = f"train_{study}_{series}"
    candidate2 = (
        VOLUMES_DIR
        / study_dir
        / series_dir_num
        / f"{volume_name}.nii.gz"
    )
    if candidate2.exists():
        return candidate2

    raise FileNotFoundError(
        f"Could not find volume for volume_name={volume_name!r}. "
        f"Tried:\n  {candidate}\n  {candidate2}"
    )


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
        resampled = zoom(vol, factors, order=1)   # order=1 → trilinear
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
            rs = zoom(mask[f], factors, order=0)
        resampled_slices.append(rs.astype(np.uint8))
    return np.stack(resampled_slices, axis=0)


# ─── HU windowing and normalisation ──────────────────────────────────────────

def window_and_normalise(vol: np.ndarray) -> np.ndarray:
    """
    Apply lung window [HU_MIN, HU_MAX] and rescale to [0, 1].
    Input:  float32 array in HU
    Output: float32 array in [0, 1]
    """
    vol = np.clip(vol, HU_MIN,  HU_MAX)
    vol = (vol - HU_MIN) / (HU_MAX - HU_MIN)
    return vol.astype(np.float32)


# ─── Metadata parsing ────────────────────────────────────────────────────────

class MetadataRegistry:
    """
    Loads the JSON metadata file once and provides fast lookup:
      get_finding_map(volume_name) → dict mapping F-index (int) → abnormality category (str)
    
    Expected JSON format:
    {
        "train": [
            {
                "name": "train_1935_a_1.nii.gz",
                "findings": {"0": "description", ...},
                "categories": {"0": "2a", "1": "2c", ...},  # F-index → category
                ...
            },
            ...
        ],
        "val": [...],
        "test": [...]
    }
    """

    def __init__(self, split: Optional[str] = None):
        with open(METADATA_JSON, "r") as f:
            self._raw: Dict = json.load(f)
        self._volume_index: Dict[str, Dict[int, str]] = {}

        split_names = [split] if split else ["train", "val", "test"]

        # Index all volumes from all splits (train, val, test)
        for split_name in split_names:
            if split_name not in self._raw:
                continue
            split_list = self._raw[split_name]
            if not isinstance(split_list, list):
                continue

            for item in split_list:
                if not isinstance(item, dict):
                    continue

                # Extract volume name and strip extension
                filename = item.get("name", "")
                volume_name = _stem(filename)
                if not volume_name:
                    continue

                # Map F-indices to categories directly from metadata
                categories_dict = item.get("categories", {})
                finding_map = {}
                for f_idx_str, category in categories_dict.items():
                    try:
                        f_idx = int(f_idx_str)
                        finding_map[f_idx] = str(category)
                    except (TypeError, ValueError):
                        continue

                if finding_map:
                    self._volume_index[volume_name] = finding_map

    def get_finding_map(self, volume_name: str) -> Dict[int, str]:
        """
        Returns {0: "2a", 1: "2c", ...} for a given volume.
        Maps F-index (int) to abnormality category (str).
        Returns empty dict if volume not found or has no findings.
        """
        volume_name = _stem(volume_name)
        return self._volume_index.get(volume_name, {})


# ─── Label inference from metadata ────────────────────────────────────────────

class LabelRegistry:
    """
    Infers volume-level labels from the metadata JSON.
    A volume is positive for an abnormality category if it has any findings 
    with that category.
    
    Provides high-level queries:
      get_labels(volume_name) → dict {abnormality_key: 0 or 1}
      get_all_volume_names()  → list of all volume names
      get_positive_volume_names(category) → list of volumes with that category
      get_normal_volume_names() → list of volumes with no findings
    """

    def __init__(self, metadata: MetadataRegistry, split: Optional[str] = None):
        self.metadata = metadata
        self.split = split
        self._build_label_index()

    def _build_label_index(self):
        """Build a lookup table of volume names and their labels."""
        self._volume_names: List[str] = []
        self._volume_labels: Dict[str, Dict[str, int]] = {}
        abnormalities = list(ABNORMALITY_CATEGORIES.keys())

        with open(METADATA_JSON, "r") as f:
            raw = json.load(f)

        # Index all volumes from all splits
        split_names = [self.split] if self.split else ["train", "val", "test"]
        for split_name in split_names:
            if split_name not in raw or not isinstance(raw[split_name], list):
                continue

            for item in raw[split_name]:
                if not isinstance(item, dict):
                    continue

                filename = item.get("name", "")
                volume_name = _stem(filename)
                if not volume_name or volume_name in self._volume_names:
                    continue

                # Get categories present in this volume
                categories_present: Dict[str, int] = {ab: 0 for ab in abnormalities}
                categories_dict = item.get("categories", {})

                for category in categories_dict.values():
                    if str(category) in categories_present:
                        categories_present[str(category)] = 1

                self._volume_names.append(volume_name)
                self._volume_labels[volume_name] = categories_present

        # --- debug summary ---
        logger = logging.getLogger(__name__)
        total = len(self._volume_names)
        per_cat_counts = {ab: sum(self._volume_labels[v].get(ab, 0) for v in self._volume_names)
                          for ab in abnormalities}
        normal_count = len(self.get_normal_volume_names())
        # sample positives per category (up to 5)
        sample_pos = {ab: self.get_positive_volume_names(ab)[:5] for ab in abnormalities}
        # sample normal (empty-category) volumes
        sample_normals = self.get_normal_volume_names()[:5]

        logger.info(
            f"LabelRegistry built split={self.split!r} total_volumes={total} "
            f"normal={normal_count} per_category_counts={per_cat_counts}"
        )
        logger.debug(f"Sample positives per category (up to 5): {sample_pos}")
        logger.debug(f"Sample volumes with no categories (normals, up to 5): {sample_normals}")
        
    def get_labels(self, scan_id: str) -> Dict[str, int]:
        """Return binary labels {category: 0 or 1} for a volume."""
        scan_id = _stem(scan_id)
        abnormalities = list(ABNORMALITY_CATEGORIES.keys())
        return self._volume_labels.get(scan_id, {ab: 0 for ab in abnormalities})

    def get_all_volume_names(self) -> List[str]:
        """Return all volume names in the metadata."""
        return list(self._volume_names)

    def get_positive_volume_names(self, category: str) -> List[str]:
        """Return volume names where the given category is present."""
        return [vol for vol in self._volume_names 
                if self._volume_labels.get(vol, {}).get(category, 0) == 1]

    def get_normal_volume_names(self) -> List[str]:
        """Return volume names with no findings in any category."""
        abnormalities = list(ABNORMALITY_CATEGORIES.keys())
        return [vol for vol in self._volume_names
                if all(self._volume_labels.get(vol, {}).get(ab, 0) == 0
                   for ab in abnormalities)]


# ─── Full scan loader (convenience wrapper) ───────────────────────────────────

class ScanLoader:
    """
    Combines volume + mask loading, resampling, and windowing into one call.
    Caches nothing — caller is responsible for not reloading unnecessarily.
    """

    def __init__(self, metadata: MetadataRegistry):
        self.metadata = metadata

    def load(self, volume_name: str) -> Dict:
        """
        Returns a dict with:
          "volume"      : float32 [H, W, D] in [0, 1] after windowing
          "mask"        : uint8   [F, H, W, D] resampled (may be None if no mask)
          "finding_map" : {f_index: canonical_abnormality_key}
          "volume_name" : str
        """
        # Load and resample volume
        vol_hu, spacing = load_volume(volume_name)
        vol_rs = resample_volume(vol_hu, spacing)
        vol    = window_and_normalise(vol_rs)

        # Load and resample mask to match the resampled volume shape exactly,
        # ignoring the mask's own header spacing to avoid shape mismatches.
        try:
            mask_raw, _ = load_mask(volume_name)
            mask = resample_mask(mask_raw, target_shape=vol_rs.shape)
        except FileNotFoundError:
            mask = None

        finding_map = self.metadata.get_finding_map(volume_name)

        return {
            "volume_name": volume_name,
            "volume":      vol,
            "mask":        mask,
            "finding_map": finding_map,
        }