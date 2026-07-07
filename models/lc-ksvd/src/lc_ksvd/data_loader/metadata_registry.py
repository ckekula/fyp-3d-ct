"""
metadata_registry.py
Parses the ReXGroundingCT JSON metadata to:
  - Map each F-slice index to its abnormality category (MetadataRegistry)
  - Infer volume-level one-hot labels and expose volume-name queries (LabelRegistry)
"""

import json
import logging
from typing import Dict, List, Optional

from lc_ksvd.config import ABNORMALITY_CATEGORIES, METADATA_JSON
from lc_ksvd.data_loader.nifti_io import _stem

logger = logging.getLogger(__name__)


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
        "test": [...]
    }
    """

    def __init__(self, split: Optional[str] = None):
        with open(METADATA_JSON, "r") as f:
            self._raw: Dict = json.load(f)
        self._volume_index: Dict[str, Dict[int, str]] = {}

        split_names = [split] if split else ["train", "test"]

        # Index all volumes from all splits
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
        split_names = [self.split] if self.split else ["train", "test"]
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