"""
scan_loader.py
High-level convenience wrapper that combines volume + mask loading,
resampling, and HU windowing into one call per scan.
"""

import numpy as np

from lc_ksvd.config import LOWER_HU, TARGET_SHAPE
from lc_ksvd.data_loader.metadata_registry import MetadataRegistry
from lc_ksvd.data_loader.nifti_io import (
    crop_or_pad,
    crop_or_pad_mask,
    load_mask,
    load_volume,
    preprocess,
    resample_mask,
    resample_volume,
)


class ScanLoader:
    """
    Combines volume + mask loading, resampling, and windowing into one call.
    Caches nothing — caller is responsible for not reloading unnecessarily.
    """

    def __init__(self, metadata: MetadataRegistry):
        self.metadata = metadata

    def load(self, volume_name: str) -> dict:
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
        vol_cp = crop_or_pad(vol_rs, TARGET_SHAPE, pad_value=LOWER_HU)
        vol = preprocess(vol_cp)

        # Load and resample mask to match the resampled volume shape exactly,
        # ignoring the mask's own header spacing to avoid shape mismatches.
        try:
            mask_raw, _ = load_mask(volume_name)
            mask = resample_mask(mask_raw, target_shape=vol_rs.shape)
            mask = crop_or_pad_mask(mask, TARGET_SHAPE, pad_value=0)
        except FileNotFoundError:
            mask = None

        finding_map = self.metadata.get_finding_map(volume_name)

        return {
            "volume_name": volume_name,
            "volume":      vol,
            "mask":        mask,
            "finding_map": finding_map,
        }
