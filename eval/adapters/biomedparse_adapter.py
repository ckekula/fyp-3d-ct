import json
from pathlib import Path
import numpy as np
from eval.core.schemas import LocalizationSample


class BiomedParseLocalizationAdapter:
    def __init__(
        self,
        output_dir: str | Path,
        gt_mask_loader,
        spacing_loader,
        model_name="biomed-parse",
    ):
        self.output_dir = Path(output_dir)
        self.gt_mask_loader = gt_mask_loader
        self.spacing_loader = spacing_loader
        self.model_name = model_name

    def load(self):
        reports_path = self.output_dir / "reports.json"
        masks_dir = self.output_dir / "masks"

        reports = json.loads(reports_path.read_text(encoding="utf-8"))
        samples = []

        for report in reports:
            case_id = report["volume_name"]
            mask_path = masks_dir / f"{case_id}.npz"

            if not mask_path.exists():
                continue

            pred_masks = np.load(mask_path)
            spacing = self.spacing_loader(case_id)

            for mask_key in pred_masks.files:
                class_name = self._mask_key_to_class_name(mask_key)

                pred_mask = pred_masks[mask_key]
                gt_mask = self.gt_mask_loader(case_id, class_name)

                samples.append(
                    LocalizationSample(
                        case_id=case_id,
                        model_name=self.model_name,
                        class_name=class_name,
                        pred_mask=pred_mask,
                        gt_mask=gt_mask,
                        spacing=spacing,
                        existence_score=self._get_existence_score(report, class_name),
                        morphology=self._get_morphology(class_name),
                    )
                )

        return samples

    def _mask_key_to_class_name(self, mask_key):
        return mask_key.replace("_", " ")

    def _get_existence_score(self, report, class_name):
        predictions = report.get("predictions", {})
        item = predictions.get(class_name)

        if item is None:
            return None

        return item.get("existence_score")

    def _get_morphology(self, class_name):
        focal = {"lung nodule"}
        return "focal" if class_name.lower() in focal else "non_focal"