# eval/adapters/nnunet_adapter.py

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import nibabel as nib
import numpy as np

from eval.core.schemas import LocalizationSample
from eval.adapters.biomedparse_adapter import BiomedParseLocalizationAdapter


class NNUNetLocalizationAdapter(BiomedParseLocalizationAdapter):
    """
    Adapter for nnU-Net binary segmentation outputs.

    Each nnU-Net model under models/nnu-net/storage/nnUNet_results/ is
    trained as a single binary foreground-vs-background segmenter for one
    finding class (e.g. Dataset100_GGO -> lung_opacity,
    Dataset104_Nodules -> lung_nodule) -- the predictions directory carries
    no per-case class label, so the class this adapter instance evaluates
    must be supplied explicitly via `class_name`.

    Expected predictions directory
      (nnUNet_results/<Dataset>/predictions[_pp]/):
        <case_id>.nii.gz   -- binary volume, foreground label == 1

    GT mask lookup/spacing conventions are inherited from
    BiomedParseLocalizationAdapter (same gt_mask_root / unified per-case
    volume convention as the other localization adapters).
    """

    def __init__(
        self,
        output_dir: str | Path,
        gt_mask_root: str | Path,
        class_name: str,
        metadata_json: str | Path | None = None,
        model_name: str = "nnunet",
    ) -> None:
        super().__init__(
            output_dir=output_dir,
            gt_mask_root=gt_mask_root,
            metadata_json=metadata_json,
            model_name=model_name,
        )
        self.class_name = self._normalize_class_name(class_name)

    def load(self) -> List[LocalizationSample]:
        if not self.output_dir.exists():
            return []

        samples: List[LocalizationSample] = []
        n_no_gt = 0
        n_shape_mismatch = 0

        for pred_path in sorted(self.output_dir.glob("*.nii.gz")):
            case_id = pred_path.name[: -len(".nii.gz")]

            pred_mask, spacing = self._load_prediction(pred_path)

            gt_mask = self._load_gt_mask(case_id, self.class_name)
            if gt_mask is None:
                n_no_gt += 1
                continue

            gt_mask = self._ensure_zyx(gt_mask)

            if pred_mask.shape != gt_mask.shape:
                n_shape_mismatch += 1
                continue

            samples.append(
                LocalizationSample(
                    case_id=case_id,
                    model_name=self.model_name,
                    class_name=self.class_name,
                    pred_mask=pred_mask,
                    gt_mask=gt_mask,
                    spacing=spacing,
                    pred_score_map=pred_mask,
                    existence_score=1.0 if pred_mask.any() else 0.0,
                    morphology=self._get_morphology(self.class_name),
                    dataset="rexgroundingct",
                    # nnU-Net emits a hard argmax segmentation, not a soft
                    # probability map.
                    is_soft_mask=False,
                )
            )

        if n_no_gt or n_shape_mismatch:
            print(
                f"[INFO] nnU-Net localization ({self.class_name}): "
                f"loaded {len(samples)} samples, "
                f"skipped {n_no_gt} with no GT mask, "
                f"{n_shape_mismatch} with pred/GT shape mismatch."
            )

        return samples

    def _load_prediction(self, path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        mask = self._load_mask_file(path)  # ZYX, per base class's XYZ->ZYX transpose
        mask = (mask > 0).astype(np.float32)

        spacing_xyz = nib.load(str(path)).header.get_zooms()[:3]
        spacing = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))

        return mask, spacing
