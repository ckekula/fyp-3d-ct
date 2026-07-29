import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval import choose_operating_point  # noqa: E402  (reuse the original CT-CLIP eval logic as-is)

ROOT = Path(__file__).resolve().parents[1]
PRED_CSV = ROOT / "rexground_predictions/full/predictions_full.csv"
DATA_ROOT = Path("/home/chest_ct/code/data")
TRAIN_LABELS = DATA_ROOT / "ct-rate/train_labels.csv"
VALID_LABELS = DATA_ROOT / "ct-rate/valid_labels.csv"
OUT_CSV = ROOT / "rexground_predictions/full/operating_points.csv"

PATHOLOGY_TO_PRED_COL = {
    "Lung nodule": "prob_lung_nodule",
    "Lung opacity": "prob_lung_opacity",
}


def choose_operating_threshold(fpr, tpr, thresholds):
    """Same Youden's-J scan as eval.choose_operating_point, but also
    surfaces the threshold value (the original function only returns
    sens/spec, not the cutoff needed to actually binarize predictions)."""
    best_j, best_thr, sens, spec = -1, 0.5, 0.0, 0.0
    for _fpr, _tpr, _thr in zip(fpr, tpr, thresholds):
        j = _tpr - _fpr
        if j > best_j:
            best_j, best_thr, sens, spec = j, _thr, _tpr, 1 - _fpr
    return best_thr, sens, spec


def main():
    preds = pd.read_csv(PRED_CSV)
    preds = preds[preds["error"].isna()] if "error" in preds.columns else preds
    preds["name"] = preds["name"]

    labels = pd.concat(
        [pd.read_csv(TRAIN_LABELS), pd.read_csv(VALID_LABELS)]
    ).drop_duplicates("VolumeName").set_index("VolumeName")

    rows = []
    for pathology, pred_col in PATHOLOGY_TO_PRED_COL.items():
        merged = preds.merge(labels[[pathology]], left_on="name", right_index=True)
        y_true = merged[pathology].values
        y_pred = merged[pred_col].values

        if len(np.unique(y_true)) < 2:
            print(f"{pathology}: skipped, only one class present in matched ground truth")
            continue

        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        # sens/spec via the ORIGINAL, unmodified eval.py function
        sens, spec = choose_operating_point(fpr, tpr, thresholds)
        # + the threshold that produced them (not returned by the original)
        best_thr, sens_check, spec_check = choose_operating_threshold(fpr, tpr, thresholds)

        y_pred_bin = (y_pred >= best_thr).astype(int)
        accuracy = (y_pred_bin == y_true).mean()

        print(f"{pathology:20s} n={len(merged):5d}  AUROC={roc_auc:.4f}  "
              f"operating_threshold={best_thr:.4f}  sens={sens:.3f}  spec={spec:.3f}  "
              f"accuracy@threshold={accuracy:.3f}")

        rows.append({
            "pathology": pathology, "n": len(merged), "auroc": roc_auc,
            "operating_threshold": best_thr, "sensitivity": sens, "specificity": spec,
            "accuracy_at_threshold": accuracy,
        })

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nSaved operating points to {OUT_CSV}")


if __name__ == "__main__":
    main()
