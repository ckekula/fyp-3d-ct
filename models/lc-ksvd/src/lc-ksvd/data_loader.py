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
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom

import config


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
        config.VOLUMES_DIR
        / study_dir
        / series_dir
        / f"{volume_name}.nii.gz"
    )
    if candidate.exists():
        return candidate

    # Fallback: some series use numeric identifiers ("train_1_1")
    series_dir_num = f"train_{study}_{series}"
    candidate2 = (
        config.VOLUMES_DIR
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
    path = config.MASKS_DIR / f"{volume_name}.nii.gz"
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

    # Mask may be stored as [H, W, D, F] in some NIfTI conventions;
    # detect and transpose if the last axis is small (number of findings)
    if mask.ndim == 4 and mask.shape[-1] < mask.shape[0]:
        mask = np.moveaxis(mask, -1, 0)   # → [F, H, W, D]

    zooms = np.abs(np.array(img.header.get_zooms()[:3], dtype=np.float32))
    return mask, zooms


# ─── Resampling ───────────────────────────────────────────────────────────────

def _zoom_factors(current_spacing: np.ndarray, target_spacing: float) -> np.ndarray:
    return current_spacing / target_spacing


def resample_volume(vol: np.ndarray, current_spacing: np.ndarray) -> np.ndarray:
    """Resample volume to TARGET_SPACING_MM isotropic using trilinear interpolation."""
    factors = _zoom_factors(current_spacing, config.TARGET_SPACING_MM)
    if np.allclose(factors, 1.0, atol=0.01):
        return vol
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resampled = zoom(vol, factors, order=1)   # order=1 → trilinear
    return resampled.astype(np.float32)


def resample_mask(mask: np.ndarray, current_spacing: np.ndarray) -> np.ndarray:
    """
    Resample each finding slice of the 4D mask using nearest-neighbour
    to preserve integer entity labels.
    """
    factors = _zoom_factors(current_spacing, config.TARGET_SPACING_MM)
    if np.allclose(factors, 1.0, atol=0.01):
        return mask

    resampled_slices = []
    for f in range(mask.shape[0]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rs = zoom(mask[f], factors, order=0)  # order=0 → nearest neighbour
        resampled_slices.append(rs.astype(np.uint8))

    return np.stack(resampled_slices, axis=0)


# ─── HU windowing and normalisation ──────────────────────────────────────────

def window_and_normalise(vol: np.ndarray) -> np.ndarray:
    """
    Apply lung window [HU_MIN, HU_MAX] and rescale to [0, 1].
    Input:  float32 array in HU
    Output: float32 array in [0, 1]
    """
    vol = np.clip(vol, config.HU_MIN, config.HU_MAX)
    vol = (vol - config.HU_MIN) / (config.HU_MAX - config.HU_MIN)
    return vol.astype(np.float32)


# ─── Metadata parsing ────────────────────────────────────────────────────────

def _normalise_finding_label(label: str) -> Optional[str]:
    """
    Map a free-text finding label from the JSON to a canonical abnormality key.
    Returns None if the label does not match any known abnormality.
    """
    label_lower = label.strip().lower()
    for canonical, aliases in config.ABNORMALITY_ALIASES.items():
        if any(alias in label_lower for alias in aliases):
            return canonical
    return None


class MetadataRegistry:
    """
    Loads the JSON metadata file once and provides fast lookup:
      get_finding_map(volume_name) → dict mapping F-index (int) → canonical abnormality key
    """

    def __init__(self):
        with open(config.METADATA_JSON, "r") as f:
            self._raw: Dict = json.load(f)
        self._list_schema_index: Dict[str, Dict[int, str]] = {}

        # Support MMSEG-style list schema:
        # {
        #   "training": [{"prompt": ..., "image": "images_flat/<id>.nii.gz", "key": "0", ...}],
        #   "testing":  [...]
        # }
        if isinstance(self._raw, dict) and isinstance(self._raw.get("training"), list):
            for item in self._raw["training"]:
                if not isinstance(item, dict):
                    continue

                image_name = str(item.get("image", "")).split("/")[-1]
                volume_name = _stem(image_name)
                if not volume_name:
                    continue

                key_raw = item.get("key")
                prompt = str(item.get("prompt", ""))
                canonical = _normalise_finding_label(prompt)
                if canonical is None:
                    continue

                try:
                    f_idx = int(key_raw)
                except (TypeError, ValueError):
                    continue

                if volume_name not in self._list_schema_index:
                    self._list_schema_index[volume_name] = {}
                self._list_schema_index[volume_name][f_idx] = canonical

    def get_finding_map(self, volume_name: str) -> Dict[int, str]:
        """
        Returns {0: "lung_nodule", 1: "consolidation", ...} for a given volume.
        F-indices with no recognised label are omitted.
        """
        volume_name = _stem(volume_name)

        # Fast path for list-based schema parsed in __init__
        if volume_name in self._list_schema_index:
            return self._list_schema_index[volume_name]

        entry = self._raw.get(volume_name, self._raw.get(f"{volume_name}.nii.gz", {}))
        result = {}
        for idx_str, label in entry.items():
            canonical = _normalise_finding_label(label)
            if canonical is not None:
                result[int(idx_str)] = canonical
        return result


# ─── CSV label loading ────────────────────────────────────────────────────────

class LabelRegistry:
    """
    Loads the labels CSV and provides fast lookup:
      get_labels(volume_name) → dict {abnormality_key: 0 or 1}
      get_all_volume_names()  → list of all volume names in the CSV
      get_normal_volume_names() → list of volumes with no abnormality present
    """

    def __init__(self):
        df = pd.read_csv(config.LABELS_CSV)
        # Normalise filename column to VolumeName (strip extension)
        if "filename" in df.columns:
            df["VolumeName"] = df["filename"].apply(_stem)
        elif "VolumeName" not in df.columns:
            raise ValueError("CSV must have a 'filename' or 'VolumeName' column.")
        else:
            df["VolumeName"] = df["VolumeName"].astype(str).apply(_stem)

        # Ensure all abnormality columns are present
        for col in config.ABNORMALITIES:
            if col not in df.columns:
                raise ValueError(f"CSV missing expected column: {col!r}")

        df = df.set_index("VolumeName")
        self._df = df

    def get_labels(self, scan_id: str) -> Dict[str, int]:
        row = self._df.loc[scan_id]
        return {ab: int(row[ab]) for ab in config.ABNORMALITIES}

    def get_all_volume_names(self) -> List[str]:
        return list(self._df.index)

    def get_positive_volume_names(self, abnormality: str) -> List[str]:
        """Volume names where the given abnormality is labelled 1."""
        return list(self._df[self._df[abnormality] == 1].index)

    def get_normal_volume_names(self) -> List[str]:
        """Volume names with all abnormality labels == 0."""
        mask = (self._df[config.ABNORMALITIES] == 0).all(axis=1)
        return list(self._df[mask].index)


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

        # Load and resample mask (normal scans may not have a mask)
        try:
            mask_raw, mask_spacing = load_mask(volume_name)
            mask = resample_mask(mask_raw, mask_spacing)
        except FileNotFoundError:
            mask = None

        finding_map = self.metadata.get_finding_map(volume_name)

        return {
            "volume_name": volume_name,
            "volume":      vol,
            "mask":        mask,        # [F, H, W, D] or None
            "finding_map": finding_map, # {int: str}
        }