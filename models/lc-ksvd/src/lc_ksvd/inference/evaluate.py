import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from lc_ksvd.config import CLASS_ORDER, RESULTS_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def evaluate(predictions, labels):
    acc = accuracy_score(labels, predictions)

    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, predictions))

    print("\nConfusion Matrix:")
    print(confusion_matrix(labels, predictions))


def _scan_level(patch_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate patch-level predictions to one row per scan via majority vote."""

    def majority_vote(s: pd.Series):
        return s.mode().iloc[0]

    scan_df = (
        patch_df.groupby("scan_id")
        .agg(
            true_label=("true_label", majority_vote),
            pred_label=("pred_label", majority_vote),
            n_patches=("pred_label", "size"),
        )
        .reset_index()
    )
    scan_df["true_class"] = scan_df["true_label"].map(lambda i: CLASS_ORDER[i])
    scan_df["pred_class"] = scan_df["pred_label"].map(lambda i: CLASS_ORDER[i])
    scan_df["correct"] = scan_df["true_label"] == scan_df["pred_label"]
    return scan_df


def save_classification_results(
    predictions: np.ndarray,
    labels: np.ndarray,
    scan_ids: np.ndarray,
    model_name: str,
    model_config: dict,
    split: str,
    results_dir: Path = RESULTS_DIR,
) -> Path:
    """Save patch-level and scan-level predictions plus summary stats to CSV.

    Writes into results_dir/<model_name>_<split>_<timestamp>/:
      - patch_level.csv
      - scan_level.csv
      - summary.csv (metrics, one row)
      - config.json (model configuration used for this run)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_dir / f"{model_name}_{split}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    patch_df = pd.DataFrame(
        {
            "scan_id": scan_ids,
            "true_label": labels,
            "pred_label": predictions,
        }
    )
    patch_df["true_class"] = patch_df["true_label"].map(lambda i: CLASS_ORDER[i])
    patch_df["pred_class"] = patch_df["pred_label"].map(lambda i: CLASS_ORDER[i])
    patch_df["correct"] = patch_df["true_label"] == patch_df["pred_label"]
    patch_df.to_csv(run_dir / "patch_level.csv", index=False)

    scan_df = _scan_level(patch_df)
    scan_df.to_csv(run_dir / "scan_level.csv", index=False)

    summary_rows = []
    for level, df in (("patch", patch_df), ("scan", scan_df)):
        y_true, y_pred = df["true_label"], df["pred_label"]
        summary_rows.append(
            {
                "level": level,
                "n_samples": len(df),
                "accuracy": accuracy_score(y_true, y_pred),
                "f1_macro": f1_score(y_true, y_pred, average="macro"),
                "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(run_dir / "summary.csv", index=False)

    report = {
        "model_name": model_name,
        "split": split,
        "timestamp": timestamp,
        "class_order": CLASS_ORDER,
        "model_config": model_config,
        "patch_classification_report": classification_report(
            patch_df["true_label"], patch_df["pred_label"], target_names=CLASS_ORDER, output_dict=True
        ),
        "patch_confusion_matrix": confusion_matrix(patch_df["true_label"], patch_df["pred_label"]).tolist(),
        "scan_classification_report": classification_report(
            scan_df["true_label"], scan_df["pred_label"], target_names=CLASS_ORDER, output_dict=True
        ),
        "scan_confusion_matrix": confusion_matrix(scan_df["true_label"], scan_df["pred_label"]).tolist(),
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Saved classification results -> {run_dir}")
    return run_dir
