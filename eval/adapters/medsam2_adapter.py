# eval/adapters/medsam2_adapter.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np

from eval.core.schemas import LocalizationSample

CATEGORY_TO_CLASS = {
    "2d": "lung_nodule",
    "2c": "lung_opacity",
}

FOCAL_CLASSES = {"lung_nodule"}


class MedSAM2LocalizationAdapter:
    """
    Adapter for the MedSAM2 RexGroundingCT lung-nodule (2d) / GGO (2c) pipeline
    (models/med-sam2/notebooks/MedSAM2_RexGroundingCT_Inference.ipynb).

    Expected output directory:
      <output_dir>/
        reports.json
        masks/<case_id>.npz   # keys: lung_nodule, lung_opacity -> (X, Y, Z) uint8

    GT masks are read from the unified 4D volume (gt_mask_root/<case_id>.nii.gz,
    shape (F, X, Y, Z)) and indexed directly by the integer finding index from
    `metadata_json`'s "categories" dict -- NOT by positional enumeration, since
    dataset_2_last.json finding keys are frequently non-contiguous (e.g. {"0",
    "2"}), which would misalign channels under a naive enumerate().
    """

    def __init__(
        self,
        output_dir: str | Path,
        gt_mask_root: str | Path,
        metadata_json: str | Path,
        model_name: str = "medsam2",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.gt_mask_root = Path(gt_mask_root)
        self.metadata_json = Path(metadata_json)
        self.model_name = model_name

        self.reports_path = self.output_dir / "reports.json"
        self.masks_dir = self.output_dir / "masks"

        self._categories_by_case = self._load_categories_index()

    def load(self) -> List[LocalizationSample]:
        if not self.reports_path.exists():
            raise FileNotFoundError(f"Could not find reports.json at: {self.reports_path}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Could not find masks directory at: {self.masks_dir}")

        reports = json.loads(self.reports_path.read_text(encoding="utf-8"))
        samples: List[LocalizationSample] = []

        for report in reports:
            case_id = str(report.get("case_id") or self._stem(report.get("volume_name", "")))
            mask_file = self.masks_dir / f"{case_id}.npz"
            if not mask_file.exists():
                print(f"[WARN] Prediction mask file not found for case: {case_id}")
                continue

            pred_npz = np.load(mask_file)
            gt_by_class = self._load_gt_masks(case_id)

            for class_name, gt_mask in gt_by_class.items():
                if class_name not in pred_npz.files:
                    pred_mask = np.zeros_like(gt_mask, dtype=np.uint8)
                else:
                    pred_mask = np.asarray(pred_npz[class_name])

                if pred_mask.shape != gt_mask.shape:
                    print(
                        f"[WARN] Shape mismatch. case={case_id}, class={class_name}, "
                        f"pred={pred_mask.shape}, gt={gt_mask.shape}. Skipping."
                    )
                    continue

                item = report.get("predictions", {}).get(class_name, {})
                existence_score = item.get("existence_score") if isinstance(item, dict) else None

                samples.append(
                    LocalizationSample(
                        case_id=case_id,
                        model_name=self.model_name,
                        class_name=class_name,
                        pred_mask=pred_mask,
                        gt_mask=gt_mask,
                        spacing=self._get_spacing(report),
                        pred_score_map=pred_mask.astype(np.float32),
                        existence_score=existence_score,
                        morphology="focal" if class_name in FOCAL_CLASSES else "non_focal",
                        dataset="rexgroundingct",
                    )
                )

        return samples

    def _load_gt_masks(self, case_id: str) -> Dict[str, np.ndarray]:
        finding_map = self._categories_by_case.get(case_id, {})
        if not finding_map:
            return {}

        gt_path = self.gt_mask_root / f"{case_id}.nii.gz"
        if not gt_path.exists():
            print(f"[WARN] GT mask missing for case: {case_id} (looked for {gt_path})")
            return {}

        mask = np.asarray(nib.load(str(gt_path)).dataobj, dtype=np.uint8)  # (F, X, Y, Z)

        by_class: Dict[str, np.ndarray] = {}
        for f_idx, class_name in finding_map.items():
            if f_idx >= mask.shape[0]:
                continue
            channel = (mask[f_idx] > 0).astype(np.uint8)
            if class_name not in by_class:
                by_class[class_name] = np.zeros_like(channel)
            by_class[class_name] = np.logical_or(by_class[class_name], channel).astype(np.uint8)

        return by_class

    def _load_categories_index(self) -> Dict[str, Dict[int, str]]:
        raw = json.loads(self.metadata_json.read_text(encoding="utf-8"))
        index: Dict[str, Dict[int, str]] = {}

        for split_entries in raw.values():
            if not isinstance(split_entries, list):
                continue
            for entry in split_entries:
                if not isinstance(entry, dict):
                    continue
                categories = entry.get("categories", {})
                if not categories:
                    continue

                finding_map: Dict[int, str] = {}
                for key, category in categories.items():
                    class_name = CATEGORY_TO_CLASS.get(str(category))
                    if class_name is None:
                        continue
                    try:
                        finding_map[int(key)] = class_name
                    except (TypeError, ValueError):
                        continue

                if finding_map:
                    index[self._stem(entry.get("name", ""))] = finding_map

        return index

    def _stem(self, name: str) -> str:
        if name.endswith(".nii.gz"):
            return name[: -len(".nii.gz")]
        if name.endswith(".nii"):
            return name[: -len(".nii")]
        return name

    def _get_spacing(self, report: Dict) -> tuple[float, float, float]:
        artifacts = report.get("artifacts", {})
        if isinstance(artifacts, dict):
            volume_metadata = artifacts.get("volume_metadata", {})
            if isinstance(volume_metadata, dict):
                spacing_xyz = volume_metadata.get("spacing_xyz")
                if spacing_xyz and len(spacing_xyz) >= 3:
                    return (float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2]))
        return (1.0, 1.0, 1.0)
