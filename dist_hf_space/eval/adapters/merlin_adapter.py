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
        metadata_json: str | Path = "data/rexgrounding-ct/dataset_4.json"
    ):
        self.predictions_path = Path(predictions_path)
        self.model_name = model_name
        self.metadata_json = Path(metadata_json)
        self._metadata_by_case = self._load_metadata_index()

    def _load_metadata_index(self) -> Dict[str, Dict]:
        if not self.metadata_json.exists():
            # Every sample's y_true will silently come out all-zero without
            # this file (no ground truth to match case_id against) -- warn
            # loudly rather than let that look like a real "no findings"
            # result.
            print(
                f"[WARN] Merlin metadata_json not found at {self.metadata_json}: "
                "ground truth cannot be loaded, all y_true will be 0 for every class/case."
            )
            return {}
        try:
            raw = json.loads(self.metadata_json.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[WARN] Failed to parse Merlin metadata_json {self.metadata_json}: {exc}")
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

    def _class_names_for_gt_category(self, category_code: str) -> set[str]:
        code = str(category_code).strip().lower()
        if code == "2d":
            return {"lung_nodule"}
        if code == "2c":
            return {"lung_opacity"}
        if code == "2b":
            return {"consolidation", "atelectasis"}
        return set()

    def _normalize_category(self, cat: str) -> str:
        cat = cat.lower().replace(" ", "_")
        if cat == "lung_nodule": return "lung_nodule"
        if cat == "atelectasis": return "atelectasis"
        if cat == "consolidation": return "consolidation"
        if cat == "lung_opacity": return "lung_opacity"
        return "unknown"

    def load(self) -> List[ClassificationSample]:
        samples = []
        n_no_findings_file = 0
        n_parse_failed = 0

        if not self.predictions_path.exists():
            print(f"Warning: Merlin predictions dir {self.predictions_path} not found.")
            return samples

        for item in self.predictions_path.iterdir():
            if not item.is_dir():
                continue
            findings_file = item / "findings.json"
            if not findings_file.exists():
                n_no_findings_file += 1
                continue

            try:
                data = json.loads(findings_file.read_text())
            except Exception as exc:
                n_parse_failed += 1
                print(f"[WARN] Failed to parse {findings_file}: {exc}")
                continue

            case_id = data.get("name", "").replace(".nii.gz", "")
            if not case_id:
                case_id = item.name
                
            # Ground Truth
            y_true = {"lung_nodule": 0, "lung_opacity": 0, "consolidation": 0, "atelectasis": 0}
            gt_record = self._metadata_by_case.get(case_id, {})
            gt_categories = gt_record.get("categories", {})
            if isinstance(gt_categories, dict):
                category_values = [str(gt_categories[key]) for key in sorted(gt_categories.keys(), key=str)]
            elif isinstance(gt_categories, list):
                category_values = [str(item) for item in gt_categories]
            else:
                category_values = []

            for category_code in category_values:
                for cls_name in self._class_names_for_gt_category(category_code):
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
                    metadata={"volume_path": str(item)},
                    # Merlin's findings.json only carries a category label per
                    # finding (see classified_findings[*].category), no numeric
                    # confidence -- y_score above is a 0/1 decision, not a
                    # probability, so rank-based metrics don't apply to it.
                    score_type="hard_label",
                )
            )

        if n_no_findings_file or n_parse_failed:
            print(
                f"[INFO] Merlin classification: loaded {len(samples)} samples, "
                f"skipped {n_no_findings_file} dirs with no findings.json, "
                f"{n_parse_failed} with unparseable findings.json."
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
        n_no_mask_file = 0
        n_mask_load_failed = 0
        n_findings_parse_failed = 0

        for item in self.output_dir.iterdir():
            if not item.is_dir():
                continue

            case_id = item.name
            pred_mask_file = item / "localization_masks.npz"
            if not pred_mask_file.exists():
                n_no_mask_file += 1
                continue

            try:
                pred_npz = np.load(str(pred_mask_file))
            except Exception as exc:
                n_mask_load_failed += 1
                print(f"[WARN] Failed to load {pred_mask_file}: {exc}")
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
                except Exception as exc:
                    n_findings_parse_failed += 1
                    print(f"[WARN] Failed to parse {findings_file}: {exc}")

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
                
                # Retrieve soft mask from npz
                pred_mask = self._load_pred_mask_from_npz(
                    pred_npz=pred_npz,
                    class_name=class_name,
                    raw_class_name=class_name,
                )
                
                if pred_mask is None:
                    continue
                
                # ZYX conversion if needed
                if pred_mask.ndim == 3 and pred_mask.shape != gt_mask.shape:
                    if np.transpose(pred_mask, (2, 0, 1)).shape == gt_mask.shape:
                        pred_mask = np.transpose(pred_mask, (2, 0, 1))
                
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
                        is_soft_mask=False,
                    )
                )

        if n_no_mask_file or n_mask_load_failed or n_findings_parse_failed:
            print(
                f"[INFO] Merlin localization: loaded {len(samples)} samples, "
                f"skipped {n_no_mask_file} dirs with no localization_masks.npz, "
                f"{n_mask_load_failed} with unloadable npz, "
                f"{n_findings_parse_failed} with unparseable findings.json (fell back to GT findings)."
            )

        return samples
