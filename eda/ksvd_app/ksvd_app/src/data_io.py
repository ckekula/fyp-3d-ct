"""
Data loading utilities.

Every loader is Streamlit-cached so the (large) source files are read from
disk once per server process, not once per interaction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import streamlit as st

from src import config


@dataclass(frozen=True)
class KSVDCheckpoint:
    """Trained K-SVD dictionary and associated metadata."""

    dictionary: np.ndarray  # D, shape (n_features, n_components)
    gamma: np.ndarray  # Gamma, shape (n_components, n_samples)
    n_components: int
    n_nonzero_coefs: int
    n_features: int


@dataclass(frozen=True)
class PatchDataset:
    """Flattened 2D patches extracted from CT volumes."""

    X: np.ndarray  # shape (n_features, n_patches)
    scan_ids: np.ndarray  # shape (n_patches,)
    coords: np.ndarray  # shape (n_patches, 3), global (x, y, z) origin per patch


def _require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found at '{path}'. "
            f"Place it there before using the app (see README)."
        )


@st.cache_resource(show_spinner="Loading K-SVD checkpoint...")
def load_ksvd_checkpoint() -> KSVDCheckpoint:
    _require_file(config.KSVD_CHECKPOINT_PATH, "K-SVD checkpoint")
    with np.load(config.KSVD_CHECKPOINT_PATH, allow_pickle=True) as npz:
        return KSVDCheckpoint(
            dictionary=npz["D"].astype(np.float32),
            gamma=npz["Gamma"].astype(np.float32),
            n_components=int(npz["n_components"]),
            n_nonzero_coefs=int(npz["n_nonzero_coefs"]),
            n_features=int(npz["n_features"]),
        )


@st.cache_resource(show_spinner="Loading patch dataset...")
def load_patch_dataset() -> PatchDataset:
    _require_file(config.PATCHES_2D_PATH, "Patch dataset")
    with np.load(config.PATCHES_2D_PATH, allow_pickle=True) as npz:
        return PatchDataset(
            X=npz["X"].astype(np.float32),
            scan_ids=npz["scan_ids"],
            coords=npz["coords"],
        )


@st.cache_resource(show_spinner=False)
def load_dataset_metadata() -> dict[str, Any] | None:
    """Load dataset_filtered.json if present; optional, so return None if missing."""
    if not config.DATASET_JSON_PATH.exists():
        return None
    with open(config.DATASET_JSON_PATH, "r") as f:
        return json.load(f)


def get_finding_info(scan_id: str) -> dict[str, Any] | None:
    """Return the finding/entity metadata for a scan from dataset_filtered.json, if any."""
    metadata = load_dataset_metadata()
    if metadata is None:
        return None
    for item in metadata.get("train", []):
        name = item.get("name", "")
        if name.removesuffix(".nii.gz") == scan_id:
            return item
    return None


def available_scan_ids(patches: PatchDataset) -> list[str]:
    """Scan IDs present in the patch dataset, sorted, deduplicated."""
    return sorted(str(s) for s in np.unique(patches.scan_ids))


def has_segmentation(scan_id: str) -> bool:
    return config.segmentation_path(scan_id).exists()


@st.cache_data(show_spinner="Loading ground-truth segmentation...")
def load_raw_segmentation_mask(scan_id: str) -> np.ndarray:
    """Load the raw segmentation mask for a scan, collapsed to a single 3D volume.

    Source masks are stored as [F, H, W, D] where F indexes individual
    findings. When a scan has multiple findings, they are combined with a
    logical OR into one binary volume for visualization purposes.
    """
    path = config.segmentation_path(scan_id)
    _require_file(path, f"Segmentation mask for scan '{scan_id}'")
    mask = nib.load(str(path)).get_fdata()

    if mask.ndim == 4:
        mask = (mask > 0).any(axis=0).astype(np.uint8)
    elif mask.ndim == 3:
        mask = (mask > 0).astype(np.uint8)
    else:
        raise ValueError(
            f"Unexpected segmentation shape {mask.shape} for scan '{scan_id}' "
            "(expected 3 or 4 dimensions)."
        )
    return mask
