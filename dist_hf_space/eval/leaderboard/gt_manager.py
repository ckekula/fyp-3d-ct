"""
leaderboard/gt_manager.py
Private Ground Truth resource manager & safety guards for ThorAxis Leaderboard.
Handles secure token-authenticated downloading from private Hugging Face datasets,
local secure caching, and public-directory leakage prevention.
"""

from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("gt_manager")


class GroundTruthSecurityError(Exception):
    """Raised when ground truth data is detected in a public or insecure directory."""
    pass


class GroundTruthManager:
    """Manages ground-truth labels and volumetric masks securely."""

    def __init__(
        self,
        hf_dataset_repo: Optional[str] = None,
        token_env_var: str = "HF_GT_TOKEN",
        local_override_dir: Optional[Union[str, Path]] = None,
        sample_gt_path: Optional[Union[str, Path]] = None
    ) -> None:
        self.hf_dataset_repo = hf_dataset_repo or os.environ.get("HF_GT_REPO", "LCKSVD/cad-eval-groundtruth")
        self.token_env_var = token_env_var
        self.local_override_dir = Path(local_override_dir) if local_override_dir else None
        self.sample_gt_path = Path(sample_gt_path) if sample_gt_path else None
        self._gt_class_dict: Optional[Dict[str, Dict[str, int]]] = None
        self._gt_masks_dict: Optional[Dict[str, Dict[str, Any]]] = None

    def assert_no_public_leakage(self, root_dir: Union[str, Path]) -> None:
        """
        Verify that full ground-truth directories (e.g. data/segmentations/)
        are NOT present inside the public git-tracked Space tree.
        """
        root = Path(root_dir).resolve()
        
        # Check for forbidden sensitive directories
        forbidden_patterns = [
            root / "data" / "segmentations",
            root / "data" / "data_volumes" / "dataset",
            root / "data" / "rexgrounding-ct" / "gt_masks",
        ]
        
        for p in forbidden_patterns:
            if p.exists() and any(p.iterdir()):
                logger.warning(
                    f"⚠️ Ground Truth Security Notice: Found private data directory at '{p}'. "
                    f"Ensure this directory is excluded from public Git staging (.gitignore)."
                )

    def load_ground_truth_labels(self) -> Dict[str, Dict[str, int]]:
        """Load classification ground truth dictionary."""
        if self._gt_class_dict is not None:
            return self._gt_class_dict

        token = os.environ.get(self.token_env_var)

        # 1. Attempt HF Hub Private Dataset download if token is configured
        if token and self.hf_dataset_repo:
            try:
                from huggingface_hub import snapshot_download
                cache_dir = Path("/tmp/thoraxis_gt_cache")
                cache_dir.mkdir(parents=True, exist_ok=True)
                download_path = snapshot_download(
                    repo_id=self.hf_dataset_repo,
                    repo_type="dataset",
                    token=token,
                    local_dir=str(cache_dir)
                )
                gt_json = Path(download_path) / "ground_truth_labels.json"
                if gt_json.exists():
                    with open(gt_json, "r") as f:
                        self._gt_class_dict = json.load(f)
                    return self._gt_class_dict
            except Exception as e:
                logger.warning(f"Could not load GT from HF Hub: {str(e)}. Falling back to local storage.")

        # 2. Local override directory
        if self.local_override_dir and self.local_override_dir.exists():
            gt_json = self.local_override_dir / "ground_truth_labels.json"
            if gt_json.exists():
                with open(gt_json, "r") as f:
                    self._gt_class_dict = json.load(f)
                return self._gt_class_dict

        # 3. Fallback sample ground truth (for web demo / smoke tests)
        if self.sample_gt_path and self.sample_gt_path.exists():
            with open(self.sample_gt_path, "r") as f:
                self._gt_class_dict = json.load(f)
            return self._gt_class_dict

        return {}
