# eval/adapters/biomedparse_adapter.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np

from eval.core.schemas import LocalizationSample


class BiomedParseLocalizationAdapter:
    """
    Adapter for BiomedParse ReXGroundingCT outputs.

    Expected BiomedParse output directory:
      outputs/biomedparse_rexgroundingct/
        reports.json
        masks/
          <case>.npz

    Expected GT mask root examples:
      data/rexgrounding-ct/gt_masks/
        case001_lung_nodule.npy
        case001_lung_opacity.npy

    or:
      data/rexgrounding-ct/gt_masks/
        lung_nodule/
          case001.npy
        lung_opacity/
          case001.npy
    """

    def __init__(
        self,
        output_dir: str | Path,
        gt_mask_root: str | Path,
        metadata_json: str | Path | None = None,
        model_name: str = "biomed_parse",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.gt_mask_root = Path(gt_mask_root)
        self.metadata_json = Path(metadata_json) if metadata_json else None
        self.model_name = model_name

        self.reports_path = self.output_dir / "summary.json"
        self.masks_dir = self.output_dir / "masks"

    def load(self) -> List[LocalizationSample]:
        if not self.reports_path.exists():
            raise FileNotFoundError(
                f"Could not find summary.json at: {self.reports_path}"
            )

        if not self.masks_dir.exists():
            raise FileNotFoundError(
                f"Could not find masks directory at: {self.masks_dir}"
            )

        reports = json.loads(self.reports_path.read_text(encoding="utf-8"))
        samples: List[LocalizationSample] = []

        for report in reports:
            case_id = self._get_case_id(report)
            pred_mask_file = self._find_prediction_mask_file(report, case_id)

            if pred_mask_file is None:
                print(f"[WARN] Prediction mask file not found for case: {case_id}")
                continue

            pred_npz = np.load(pred_mask_file)

            prediction_items = report.get("predictions", {})

            for raw_class_name in prediction_items.keys():
                class_name = self._normalize_class_name(raw_class_name)

                pred_mask = self._load_pred_mask_from_npz(
                    pred_npz=pred_npz,
                    class_name=class_name,
                    raw_class_name=raw_class_name,
                )

                if pred_mask is None:
                    print(
                        f"[WARN] Prediction mask missing. "
                        f"case={case_id}, class={class_name}, file={pred_mask_file}"
                    )
                    continue

                gt_mask = self._load_gt_mask(case_id, class_name)

                if gt_mask is None:
                    print(
                        f"[WARN] GT mask missing. "
                        f"case={case_id}, class={class_name}, root={self.gt_mask_root}"
                    )
                    continue

                pred_mask = self._ensure_zyx(pred_mask)
                gt_mask = self._ensure_zyx(gt_mask)

                if pred_mask.shape != gt_mask.shape:
                    print(
                        f"[WARN] Shape mismatch. "
                        f"case={case_id}, class={class_name}, "
                        f"pred={pred_mask.shape}, gt={gt_mask.shape}. Skipping."
                    )
                    continue

                spacing = self._get_spacing(report)

                item = prediction_items[raw_class_name]
                existence_score = None

                if isinstance(item, dict):
                    existence_score = item.get("existence_score", None)

                samples.append(
                    LocalizationSample(
                        case_id=case_id,
                        model_name=self.model_name,
                        class_name=class_name,
                        pred_mask=pred_mask,
                        gt_mask=gt_mask,
                        spacing=spacing,
                        pred_score_map=pred_mask,
                        existence_score=existence_score,
                        morphology=self._get_morphology(class_name),
                        dataset="rexgroundingct",
                    )
                )

        return samples

    def _get_case_id(self, report: Dict) -> str:
        if "case_id" in report:
            return str(report["case_id"])

        volume_name = str(report.get("volume_name", ""))

        if volume_name.endswith(".nii.gz"):
            return volume_name.replace(".nii.gz", "")

        if volume_name.endswith(".nii"):
            return volume_name.replace(".nii", "")

        return Path(volume_name).stem

    def _find_prediction_mask_file(
        self,
        report: Dict,
        case_id: str,
    ) -> Optional[Path]:
        artifacts = report.get("artifacts", {})

        if isinstance(artifacts, dict):
            mask_file = artifacts.get("mask_file")

            if mask_file:
                path = Path(mask_file)

                if path.exists():
                    return path

                path_from_output = self.output_dir / mask_file

                if path_from_output.exists():
                    return path_from_output

        volume_name = str(report.get("volume_name", ""))

        candidates = [
            self.masks_dir / f"{case_id}.npz",
            self.masks_dir / f"{volume_name}.npz",
            self.masks_dir / f"{Path(volume_name).stem}.npz",
        ]

        if volume_name.endswith(".nii.gz"):
            candidates.append(
                self.masks_dir / f"{volume_name.replace('.nii.gz', '')}.npz"
            )

        for path in candidates:
            if path.exists():
                return path

        return None

    def _load_pred_mask_from_npz(
        self,
        pred_npz,
        class_name: str,
        raw_class_name: str,
    ) -> Optional[np.ndarray]:
        possible_keys = [
            class_name,
            f"{class_name}_soft",
            f"{class_name}_binary",
            raw_class_name,
            raw_class_name.lower().replace(" ", "_"),
            raw_class_name.lower().replace("-", "_").replace(" ", "_"),
        ]

        for key in possible_keys:
            if key in pred_npz.files:
                return np.asarray(pred_npz[key])

        # fallback: fuzzy match
        for key in pred_npz.files:
            normalized_key = self._normalize_class_name(key)
            if normalized_key == class_name:
                return np.asarray(pred_npz[key])

            if normalized_key == f"{class_name}_soft":
                return np.asarray(pred_npz[key])

            if normalized_key == f"{class_name}_binary":
                return np.asarray(pred_npz[key])

        return None

    def _load_gt_mask(
        self,
        case_id: str,
        class_name: str,
    ) -> Optional[np.ndarray]:
        candidates = [
            self.gt_mask_root / f"{case_id}_{class_name}.npy",
            self.gt_mask_root / f"{case_id}_{class_name}.npz",
            self.gt_mask_root / f"{case_id}_{class_name}.nii",
            self.gt_mask_root / f"{case_id}_{class_name}.nii.gz",

            self.gt_mask_root / class_name / f"{case_id}.npy",
            self.gt_mask_root / class_name / f"{case_id}.npz",
            self.gt_mask_root / class_name / f"{case_id}.nii",
            self.gt_mask_root / class_name / f"{case_id}.nii.gz",

            self.gt_mask_root / case_id / f"{class_name}.npy",
            self.gt_mask_root / case_id / f"{class_name}.npz",
            self.gt_mask_root / case_id / f"{class_name}.nii",
            self.gt_mask_root / case_id / f"{class_name}.nii.gz",
        ]

        for path in candidates:
            if path.exists():
                return self._load_mask_file(path)

        return None

    def _load_mask_file(self, path: Path) -> np.ndarray:
        suffixes = "".join(path.suffixes)

        if suffixes.endswith(".npy"):
            return np.load(path)

        if suffixes.endswith(".npz"):
            data = np.load(path)

            if len(data.files) == 1:
                return np.asarray(data[data.files[0]])

            for preferred_key in ["mask", "gt_mask", "label", "arr_0"]:
                if preferred_key in data.files:
                    return np.asarray(data[preferred_key])

            return np.asarray(data[data.files[0]])

        if suffixes.endswith(".nii") or suffixes.endswith(".nii.gz"):
            image = nib.load(str(path))
            mask = image.get_fdata(dtype=np.float32)

            # Convert XYZ -> ZYX to match your pipeline output convention
            if mask.ndim == 3:
                mask = np.transpose(mask, (2, 0, 1))

            return mask

        raise ValueError(f"Unsupported mask file format: {path}")

    def _get_spacing(self, report: Dict) -> tuple[float, float, float]:
        artifacts = report.get("artifacts", {})

        if isinstance(artifacts, dict):
            volume_metadata = artifacts.get("volume_metadata", {})

            if isinstance(volume_metadata, dict):
                spacing_xyz = volume_metadata.get("spacing_xyz")

                if spacing_xyz and len(spacing_xyz) >= 3:
                    # pipeline uses ZYX arrays, so convert XYZ spacing to ZYX
                    return (
                        float(spacing_xyz[2]),
                        float(spacing_xyz[1]),
                        float(spacing_xyz[0]),
                    )

        return (1.0, 1.0, 1.0)

    def _ensure_zyx(self, mask: np.ndarray) -> np.ndarray:
        mask = np.asarray(mask)

        if mask.ndim != 3:
            raise ValueError(f"Expected 3D mask, got shape: {mask.shape}")

        return mask

    def _normalize_class_name(self, name: str) -> str:
        name = str(name)

        name = name.replace("_soft", "")
        name = name.replace("_binary", "")

        return (
            name.strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

    def _get_morphology(self, class_name: str) -> str:
        focal_classes = {
            "lung_nodule",
            "nodule",
            "pulmonary_nodule",
        }

        if class_name in focal_classes:
            return "focal"

        return "non_focal"