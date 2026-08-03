"""
metadata_registry.py
Parses the ReXGroundingCT JSON metadata to:
  - Map each F-slice index to its abnormality category (MetadataRegistry)
  - Infer volume-level one-hot labels and expose volume-name queries (LabelRegistry)
"""

import json
import logging

from lc_ksvd.config import ABNORMALITY_CATEGORIES, METADATA_JSON
from lc_ksvd.data_loader.nifti_io import _stem

logger = logging.getLogger(__name__)


# ─── Metadata parsing ────────────────────────────────────────────────────────

class MetadataRegistry:
    def __init__(self, split: str | None = None):
        self.split = split
        with open(METADATA_JSON, "r") as f:
            self._raw: dict = json.load(f)
        self._volume_index: dict[str, dict[int, str]] = {}
        self._volume_names: list[str] = []

        if split is None:
            raise ValueError("Split cannot be None")

        if split not in self._raw:
            raise ValueError(f"'{split}' split not found in metadata JSON")
        split_list = self._raw[split]
        if not isinstance(split_list, list):
            raise ValueError(f"'{split}' split is not a list in metadata JSON")

        for item in split_list:
            if not isinstance(item, dict):
                continue

            filename = item.get("name", "")
            volume_name = _stem(filename)
            if not volume_name:
                continue

            self._volume_names.append(volume_name)

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

    def get_finding_map(self, volume_name: str) -> dict[int, str]:
        volume_name = _stem(volume_name)
        return self._volume_index.get(volume_name, {})

    def get_all_volume_names(self) -> list[str]:        # NEW
        """Return all volume names present in the metadata (including normals)."""
        return list(self._volume_names)

# ─── Label inference from metadata ────────────────────────────────────────────

class LabelRegistry:
    """
    Infers volume-level labels from MetadataRegistry.
    A volume is positive for an abnormality category if it has any findings
    with that category.
    """
    def __init__(self, metadata: MetadataRegistry):
        self.metadata = metadata
        self.split = metadata.split
        self._build_label_index()

    def _build_label_index(self):
        """Build a lookup table of volume names and their labels, derived from MetadataRegistry."""
        abnormalities = list(ABNORMALITY_CATEGORIES.keys())
        self._volume_names: list[str] = self.metadata.get_all_volume_names()
        self._volume_labels: dict[str, dict[str, int]] = {}

        for volume_name in self._volume_names:
            finding_map = self.metadata.get_finding_map(volume_name)
            categories_present = {ab: 0 for ab in abnormalities}
            for category in finding_map.values():
                if category in categories_present:
                    categories_present[category] = 1
            self._volume_labels[volume_name] = categories_present

        # --- debug summary (unchanged) ---
        total = len(self._volume_names)
        per_cat_counts = {ab: sum(self._volume_labels[v].get(ab, 0) for v in self._volume_names)
                          for ab in abnormalities}
        normal_count = len(self.get_normal_volume_names())
        sample_pos = {ab: self.get_positive_volume_names(ab)[:5] for ab in abnormalities}
        sample_normals = self.get_normal_volume_names()[:5]

        logger.info(
            f"LabelRegistry built split={self.split!r} total_volumes={total} "
            f"normal={normal_count} per_category_counts={per_cat_counts}"
        )
        logger.debug(f"Sample positives per category (up to 5): {sample_pos}")
        logger.debug(f"Sample volumes with no categories (normals, up to 5): {sample_normals}")

    def get_labels(self, scan_id: str) -> dict[str, int]:
        """Return binary labels {category: 0 or 1} for a volume."""
        scan_id = _stem(scan_id)
        abnormalities = list(ABNORMALITY_CATEGORIES.keys())
        return self._volume_labels.get(scan_id, {ab: 0 for ab in abnormalities})

    def get_all_volume_names(self) -> list[str]:
        """Return all volume names in the metadata."""
        return list(self._volume_names)

    def get_positive_volume_names(self, category: str) -> list[str]:
        """Return volume names where the given category is present."""
        return [vol for vol in self._volume_names
                if self._volume_labels.get(vol, {}).get(category, 0) == 1]

    def get_normal_volume_names(self) -> list[str]:
        """Return volume names with no findings in any category."""
        abnormalities = list(ABNORMALITY_CATEGORIES.keys())
        return [vol for vol in self._volume_names
                if all(self._volume_labels.get(vol, {}).get(ab, 0) == 0
                   for ab in abnormalities)]