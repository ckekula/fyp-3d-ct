import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import nibabel as nib

from eval.core.schemas import ClassificationSample, LocalizationSample
from eval.adapters.biomedparse_adapter import BiomedParseLocalizationAdapter


class MerlinClassificationAdapter:
    """
    Adapter for Merlin output evaluation (Classification).
    Reads `findings.json` in each result subdirectory and extracts
    the `classified_findings` to produce classification scores.
    """
    def __init__(
        self,
        predictions_path: str | Path,
        model_name: str = "merlin",
        metadata_json: str | Path = "data/rexgrounding-ct/dataset.json"
    ):
        self.predictions_path = Path(predictions_path)
        self.model_name = model_name
        self.metadata_json = Path(metadata_json)
        self._metadata_by_case = self._load_metadata_index()

    def _load_metadata_index(self) -> Dict[str, Dict]:
        if not self.metadata_json.exists():
            return {}
        try:
            raw = json.loads(self.metadata_json.read_text(encoding="utf-8"))
        except Exception:
            return {}
        
        index = {}
        for split in ["train", "valid", "test"]:
            if split in raw:
                for item in raw[split]:
                    name = item.get("name", "").replace(".nii.gz", "")
                    index[name] = item
        return index

    def _class_names_for_finding(self, finding_text: str) -> set[str]:
        # Copied from BiomedParseAdapter to keep consistency
        text = str(finding_text).strip().lower()
        matched = set()
        if any(kw in text for kw in ["nodule", "nodular", "mass"]):
            matched.add("lung_nodule")
        if any(kw in text for kw in ["atelectasis", "atelectatic", "fibroatelectasis"]):
            matched.add("atelectasis")
        if any(kw in text for kw in ["consolidation", "consolidative", "pneumonic consolidation"]):
            matched.add("consolidation")
        if not matched and any(kw in text for kw in [
            "opacity", "opacities", "ground glass", "ground-glass", "infiltrat",
            "density", "dense", "reticulonodular", "mosaic", "pneumonia", "crazy paving", "hazy"
        ]):
            matched.add("lung_opacity")
        if not matched:
            matched.add("lung_opacity")
        return matched

    def _normalize_category(self, cat: str) -> str:
        cat = cat.lower().replace(" ", "_")
        if cat == "lung_nodule": return "lung_nodule"
        if cat == "atelectasis": return "atelectasis"
        if cat == "consolidation": return "consolidation"
        if cat == "lung_opacity": return "lung_opacity"
        return "unknown"

    def load(self) -> List[ClassificationSample]:
        samples = []
        if not self.predictions_path.exists():
            print(f"Warning: Merlin predictions dir {self.predictions_path} not found.")
            return samples

        for item in self.predictions_path.iterdir():
            if not item.is_dir():
                continue
            findings_file = item / "findings.json"
            if not findings_file.exists():
                continue
            
            try:
                data = json.loads(findings_file.read_text())
            except Exception:
                continue

            case_id = data.get("name", "").replace(".nii.gz", "")
            if not case_id:
                case_id = item.name
                
            # Ground Truth
            y_true = {"lung_nodule": 0, "lung_opacity": 0, "consolidation": 0, "atelectasis": 0}
            gt_record = self._metadata_by_case.get(case_id, {})
            gt_findings = gt_record.get("findings", {})
            for v in gt_findings.values():
                for cls_name in self._class_names_for_finding(v):
                    y_true[cls_name] = 1

            # Predictions
            y_score = {"lung_nodule": 0.0, "lung_opacity": 0.0, "consolidation": 0.0, "atelectasis": 0.0}
            classified = data.get("classified_findings", {})
            for v in classified.values():
                cat = self._normalize_category(v.get("category", ""))
                # If they predicted "Others", we could extract class from text.
                if cat == "unknown":
                    for cls_name in self._class_names_for_finding(v.get("text", "")):
                        y_score[cls_name] = 1.0
                elif cat in y_score:
                    y_score[cat] = 1.0
                    
            samples.append(
                ClassificationSample(
                    case_id=case_id,
                    model_name=self.model_name,
                    y_true=y_true,
                    y_score=y_score,
                    dataset="rexgroundingct",
                    metadata={"volume_path": str(item)}
                )
            )

        return samples


class MerlinLocalizationAdapter(BiomedParseLocalizationAdapter):
    """
    Adapter for Merlin output evaluation (Localization).
    Extends BiomedParseLocalizationAdapter since GT fetching is identical.
    """
    def __init__(
        self,
        output_dir: str | Path,
        gt_mask_root: str | Path,
        metadata_json: str | Path | None = None,
        model_name: str = "merlin",
    ):
        super().__init__(
            output_dir=output_dir,
            gt_mask_root=gt_mask_root,
            metadata_json=metadata_json,
            model_name=model_name
        )

    def load(self) -> List[LocalizationSample]:
        if not self.output_dir.exists():
            return []

        samples = []
        for item in self.output_dir.iterdir():
            if not item.is_dir():
                continue
            if len(samples) >= 50:
                break
                
            case_id = item.name
            pred_mask_file = item / "localization_mask.nii.gz"
            if not pred_mask_file.exists():
                continue
                
            try:
                pred_img = nib.load(str(pred_mask_file))
                pred_mask = pred_img.get_fdata(dtype=np.float32)
                # Merlin pipeline creates 3D xyz/zyx mask, squeeze if needed
                pred_mask = np.squeeze(pred_mask)
                if pred_mask.ndim == 3:
                    pred_mask = np.transpose(pred_mask, (2, 0, 1)) # XYZ to ZYX
            except Exception:
                continue

            findings_file = item / "findings.json"
            classes_to_eval = set()
            if findings_file.exists():
                try:
                    data = json.loads(findings_file.read_text())
                    classified = data.get("classified_findings", {})
                    for v in classified.values():
                        for cls_name in self._class_names_for_finding(v.get("text", "")):
                            classes_to_eval.add(cls_name)
                except Exception:
                    pass
            
            # If no classes found, use GT findings
            if not classes_to_eval:
                gt_texts = self._get_case_findings(case_id)
                for t in gt_texts:
                    classes_to_eval.update(self._class_names_for_finding(t))

            # Load GT mask for each class
            for class_name in classes_to_eval:
                gt_mask = self._load_gt_mask(case_id, class_name)
                if gt_mask is None:
                    continue
                
                gt_mask = self._ensure_zyx(gt_mask)
                if pred_mask.shape != gt_mask.shape:
                    continue
                
                samples.append(
                    LocalizationSample(
                        case_id=case_id,
                        model_name=self.model_name,
                        class_name=class_name,
                        pred_mask=pred_mask,
                        gt_mask=gt_mask,
                        spacing=(1.0, 1.0, 1.0),
                        pred_score_map=pred_mask,
                        existence_score=1.0,
                        morphology=self._get_morphology(class_name),
                        dataset="rexgroundingct",
                    )
                )

        return samples
