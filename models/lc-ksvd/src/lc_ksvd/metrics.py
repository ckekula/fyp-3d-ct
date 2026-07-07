"""
metrics.py
Column normalisation, per-class distribution logging, and scan-level
evaluation (mean-pooling patch scores across each scan's patches).
"""

import logging
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from lc_ksvd.config import CLASS_ORDER

logger = logging.getLogger(__name__)


def normalise_columns(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    norms = np.linalg.norm(X, axis=0)
    zero_mask = norms < 1e-10
    norms_safe = np.where(zero_mask, 1.0, norms)
    X_norm = X / norms_safe[np.newaxis, :]
    return X_norm, norms, zero_mask


def log_class_distribution(H: np.ndarray, prefix: str) -> None:
    """Log per-class patch counts for the given label vector."""
    for i, cls in enumerate(CLASS_ORDER):
        count = int((H == i).sum())
        logger.info(f"  {prefix} class {cls}: {count} patches")


def evaluate(
    model,
    X_norm: np.ndarray,
    H: np.ndarray,           # (n_patches,)  int64 - patch-level class indices
    scan_ids: np.ndarray,    # (n_patches,)  str   - one scan ID per patch
    split_name: str,
) -> Dict[str, float]:
    """
    Compute scan-level classification metrics by mean-pooling patch scores
    across all patches belonging to the same scan.

    Works for any model exposing `.transform(X) -> Gamma` and `.W_`
    (n_classes x n_components) - both LCKSVD and IncrementalFrozenDictionary
    satisfy this.

    For each scan:
      - Score vector = mean of (W @ Gamma) over its patches  ->  (n_classes,)
      - Predicted class = argmax of the mean score vector
      - Ground-truth class = class label

    AUROC and AP are computed one-vs-rest using the mean score for each class
    as the continuous ranking signal.
    """
    assert H.ndim == 1, (
        f"evaluate() expects a 1D integer label vector; got shape {H.shape}."
    )

    Gamma  = model.transform(X_norm)   # (n_components, n_patches)
    W      = model.W_                  # (n_classes, n_components)
    scores = W @ Gamma                 # (n_classes, n_patches)

    # -- Aggregate to scan level --------------------------------------------
    unique_scans = np.unique(scan_ids)
    n_scans      = len(unique_scans)
    n_classes    = len(CLASS_ORDER)

    scan_scores   = np.zeros((n_classes, n_scans), dtype=np.float64)
    scan_gt       = np.zeros(n_scans, dtype=np.int64)

    for j, sid in enumerate(unique_scans):
        mask = scan_ids == sid
        # Mean-pool patch scores for this scan
        scan_scores[:, j] = scores[:, mask].mean(axis=1)
        # class label for this scan's patches as ground truth
        patch_labels = H[mask]
        counts = np.bincount(patch_labels, minlength=n_classes)
        scan_gt[j] = int(np.argmax(counts))

    scan_pred = np.argmax(scan_scores, axis=0)   # (n_scans,)

    # -- Metrics --------------------------------------------------------------
    metrics: Dict[str, float] = {}
    aurocs, aps = [], []

    for c in range(n_classes):
        gt_bin  = (scan_gt == c).astype(int)
        score_c = scan_scores[c, :]

        try:
            aurocs.append(roc_auc_score(gt_bin, score_c))
        except ValueError:
            aurocs.append(float("nan"))

        try:
            aps.append(average_precision_score(gt_bin, score_c))
        except ValueError:
            aps.append(float("nan"))

    metrics["auroc_macro"] = float(np.nanmean(aurocs))
    metrics["ap_macro"]    = float(np.nanmean(aps))
    metrics["f1_macro"]    = f1_score(
        scan_gt, scan_pred, average="macro", zero_division=0
    )
    metrics["n_scans"]     = n_scans

    for i, cls in enumerate(CLASS_ORDER):
        metrics[f"auroc_{cls}"] = aurocs[i]
        metrics[f"ap_{cls}"]    = aps[i]

    logger.info(
        f"  [{split_name}] {n_scans} scans - "
        f"AUROC(macro)={metrics['auroc_macro']:.4f}  "
        f"F1(macro)={metrics['f1_macro']:.4f}  "
        f"AP(macro)={metrics['ap_macro']:.4f}"
    )
    for cls in CLASS_ORDER:
        logger.info(
            f"    {cls}: AUROC={metrics[f'auroc_{cls}']:.4f}  "
            f"AP={metrics[f'ap_{cls}']:.4f}"
        )

    return metrics