# eval/adapters/lcksvd_adapter.py

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from eval.adapters._axis_utils import xyz_to_zyx
from eval.core.schemas import ClassificationSample, LocalizationSample

# RexGroundingCT category code -> canonical class name used across all adapters.
# LC-KSVD's SVM test set (models/lc-ksvd/src/lc_ksvd/outputs/results/*/scan_level.csv)
# only contains scans labeled "2c" (lung_opacity), "2d" (lung_nodule), or "normal".
# There is no "2b" (consolidation/atelectasis) representation in this split, so
# those two classes will always have zero positive ground truth here -- they are
# reported as not_evaluated (see compute_classification_metrics) rather than
# scored, the same way MedSAM2's localization coverage gap is handled.
CATEGORY_TO_CLASS = {
    "2c": "lung_opacity",
    "2d": "lung_nodule",
}

ALL_CLASSES = ["lung_nodule", "lung_opacity", "consolidation", "atelectasis"]


class LCKSVDClassificationAdapter:
    """
    Adapter for LC-KSVD scan-level classification evaluation (Axis A).

    Reads a `scan_level.csv` produced by the LC-KSVD SVM test pipeline
    (models/lc-ksvd/src/lc_ksvd/outputs/results/<run>/scan_level.csv), which
    has columns: scan_id, true_label, pred_label, n_patches, true_class,
    pred_class, correct. true_class/pred_class are one of {"2c", "2d",
    "normal"}.

    IMPORTANT: pred_class is a hard multiclass decision from an SVM, not a
    calibrated probability, so samples are marked score_type="hard_label"
    (see ClassificationSample) -- AUROC/AP/ECE will correctly be reported as
    N/A for this model rather than computed on a degenerate 0/1 score.
    """

    def __init__(
        self,
        predictions_path: str | Path,
        model_name: str = "lc_ksvd",
    ) -> None:
        self.predictions_path = Path(predictions_path)
        self.model_name = model_name

    def _resolve_csv_path(self) -> Path:
        if self.predictions_path.is_file():
            return self.predictions_path
        candidate = self.predictions_path / "scan_level.csv"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"Could not find scan_level.csv at or under: {self.predictions_path}"
        )

    def load(self) -> List[ClassificationSample]:
        csv_path = self._resolve_csv_path()
        samples: List[ClassificationSample] = []

        with csv_path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                case_id = row["scan_id"]

                y_true = {cls: 0 for cls in ALL_CLASSES}
                true_class = CATEGORY_TO_CLASS.get(row["true_class"])
                if true_class is not None:
                    y_true[true_class] = 1

                y_score = {cls: 0.0 for cls in ALL_CLASSES}
                pred_class = CATEGORY_TO_CLASS.get(row["pred_class"])
                if pred_class is not None:
                    y_score[pred_class] = 1.0

                samples.append(
                    ClassificationSample(
                        case_id=case_id,
                        model_name=self.model_name,
                        y_true=y_true,
                        y_score=y_score,
                        dataset="rexgroundingct",
                        metadata={
                            "n_patches": row.get("n_patches"),
                            "true_class_code": row["true_class"],
                            "pred_class_code": row["pred_class"],
                        },
                        score_type="hard_label",
                    )
                )

        return samples


FOCAL_CLASSES = {"lung_nodule"}


class LCKSVDLocalizationAdapter:
    """
    Adapter for LC-KSVD grounded localization evaluation (Axis B).

    Consumes the output of
    models/lc-ksvd/src/lc_ksvd/inference/segmentation.py::evaluate_split_segmentation(save_masks=True):
      <output_dir>/
        manifest.json          # {"scan_ids": [...], "spacing_xyz": [x,y,z], "class_order": ["2c","2d"]}
        masks/<scan_id>.npz    # keys "pred_2c"/"gt_2c", "pred_2d"/"gt_2d" (uint8, boolean coverage)

    Raw pred/gt masks (not LC-KSVD's own precomputed segmentation_metrics())
    are loaded so Dice/IoU are computed by eval/core/localization_metrics.py
    -- the exact same code path used for every other model -- rather than by
    LC-KSVD's own metric function, which uses different empty-mask edge-case
    conventions and would not be a like-for-like comparison.

    LC-KSVD's dictionary is only trained on "2c" (lung_opacity) and "2d"
    (lung_nodule) -- see models/lc-ksvd/src/lc_ksvd/config.py
    ABNORMALITY_CATEGORIES / CLASS_ORDER. There is no "2b" (consolidation /
    atelectasis) block in the dictionary at all, so this adapter structurally
    can never produce those two classes -- this is a permanent limitation of
    the trained model, not a missing-test-data gap like MedSAM2's.

    NOTE: This adapter could not be exercised against real data in the
    development environment -- evaluate_split_segmentation requires the full
    RexGroundingCT CT volumes + trained dictionary/SVM models, which live on
    the training machine (see the hardcoded DATASET_ROOT =
    "/home/chest_ct/code/data" in config.py), not in this checkout. Run
    `evaluate_split_segmentation(save_masks=True)` there first to produce the
    manifest.json/masks/ this adapter expects.
    """

    CATEGORY_TO_CLASS = CATEGORY_TO_CLASS

    def __init__(
        self,
        output_dir: str | Path,
        gt_mask_root: str | Path | None = None,
        metadata_json: str | Path | None = None,
        model_name: str = "lc_ksvd",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        # gt_mask_root/metadata_json accepted for interface parity with the
        # other localization adapters (see evaluate_localization.py's shared
        # call signature); unused here since GT masks are already baked into
        # each scan's npz by evaluate_split_segmentation.

    def load(self) -> List[LocalizationSample]:
        manifest_path = self.output_dir / "manifest.json"
        masks_dir = self.output_dir / "masks"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Could not find manifest.json at: {manifest_path}")
        if not masks_dir.exists():
            raise FileNotFoundError(f"Could not find masks directory at: {masks_dir}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        spacing_xyz = tuple(manifest.get("spacing_xyz", (1.0, 1.0, 1.0)))

        samples: List[LocalizationSample] = []
        for scan_id in manifest.get("scan_ids", []):
            mask_file = masks_dir / f"{scan_id}.npz"
            if not mask_file.exists():
                continue

            npz = np.load(mask_file)
            for category_code, class_name in self.CATEGORY_TO_CLASS.items():
                pred_key, gt_key = f"pred_{category_code}", f"gt_{category_code}"
                if pred_key not in npz.files or gt_key not in npz.files:
                    continue

                # masks/spacing are (H, W, D)=(x, y, z)-ordered per LC-KSVD's
                # config.py (D=axial last); convert to the (Z, Y, X) + zyx
                # spacing convention every other localization adapter uses.
                pred_mask_zyx, spacing_zyx = xyz_to_zyx(np.asarray(npz[pred_key]), spacing_xyz)
                gt_mask_zyx, _ = xyz_to_zyx(np.asarray(npz[gt_key]), spacing_xyz)

                samples.append(
                    LocalizationSample(
                        case_id=scan_id,
                        model_name=self.model_name,
                        class_name=class_name,
                        pred_mask=pred_mask_zyx,
                        gt_mask=gt_mask_zyx,
                        spacing=spacing_zyx,
                        pred_score_map=None,
                        existence_score=None,
                        morphology="focal" if class_name in FOCAL_CLASSES else "non_focal",
                        dataset="rexgroundingct",
                        is_soft_mask=False,
                    )
                )

        return samples
