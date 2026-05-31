"""
train.py
Trains one LC-KSVD2 multi-class model over all abnormalities + normal.

H rows (CLASS_ORDER):
  0 → normal
  1 → 2b  (atelectasis / consolidation)
  2 → 2c  (groundglass opacity)
  3 → 2d  (pulmonary nodules/masses)

Usage:
  python train.py                   # extract patches then train
  python train.py --skip-extraction # use existing unified .npz
"""

import argparse
import logging
import pickle
import time
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from lc_ksvd.config import (
    CLASS_ORDER, HU_MAX, HU_MIN, LCKSVD_CONFIG,
    MODELS_DIR, PATCH_SIZE, TARGET_SPACING_MM,
)
from lc_ksvd.patch_extractor import extract_unified, load_unified_patch_matrix
from reppi import LCKSVD

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ─── Column normalisation ─────────────────────────────────────────────────────

def normalise_columns(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    norms = np.linalg.norm(X, axis=0)
    zero_mask = norms < 1e-10
    norms_safe = np.where(zero_mask, 1.0, norms)
    X_norm = X / norms_safe[np.newaxis, :]
    return X_norm, norms, zero_mask


def _log_class_distribution(H: np.ndarray, prefix: str) -> None:
    """Log per-class patch counts for the given label vector."""
    for i, cls in enumerate(CLASS_ORDER):
        count = int((H == i).sum())
        logger.info(f"  {prefix} class {cls}: {count} patches")


# ─── Scan-level evaluation ────────────────────────────────────────────────────

def evaluate(
    model: LCKSVD,
    X_norm: np.ndarray,
    H: np.ndarray,           # (n_patches,)  int64 — patch-level class indices
    scan_ids: np.ndarray,    # (n_patches,)  str   — one scan ID per patch
    split_name: str,
) -> Dict[str, float]:
    """
    Compute scan-level classification metrics by mean-pooling patch scores
    across all patches belonging to the same scan.

    For each scan:
      - Score vector = mean of (W @ Gamma) over its patches  →  (n_classes,)
      - Predicted class = argmax of the mean score vector
      - Ground-truth class = majority class label among the scan's patches
        (all patches from a normal scan have label 0; abnormal scans may have
         patches from multiple classes, but the scan-level GT is the class that
         appears most, which in practice is always the single abnormality class
         assigned during extraction)

    AUROC and AP are computed one-vs-rest using the mean score for each class
    as the continuous ranking signal.
    """
    assert H.ndim == 1, (
        f"evaluate() expects a 1D integer label vector; got shape {H.shape}."
    )

    Gamma  = model.transform(X_norm)   # (n_components, n_patches)
    W      = model.W_                  # (n_classes, n_components)
    scores = W @ Gamma                 # (n_classes, n_patches)

    # ── Aggregate to scan level ───────────────────────────────────────────────
    unique_scans = np.unique(scan_ids)
    n_scans      = len(unique_scans)
    n_classes    = len(CLASS_ORDER)

    scan_scores   = np.zeros((n_classes, n_scans), dtype=np.float64)
    scan_gt       = np.zeros(n_scans, dtype=np.int64)

    for j, sid in enumerate(unique_scans):
        mask = scan_ids == sid
        # Mean-pool patch scores for this scan
        scan_scores[:, j] = scores[:, mask].mean(axis=1)
        # Majority class among this scan's patches as ground truth
        patch_labels = H[mask]
        counts = np.bincount(patch_labels, minlength=n_classes)
        scan_gt[j] = int(np.argmax(counts))

    scan_pred = np.argmax(scan_scores, axis=0)   # (n_scans,)

    # ── Metrics ───────────────────────────────────────────────────────────────
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
        f"  [{split_name}] {n_scans} scans — "
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


# ─── Training ─────────────────────────────────────────────────────────────────

def train() -> Dict:
    logger.info(f"\n{'='*60}\nTraining unified LC-KSVD2 model\n{'='*60}")

    # ── Load patches ──────────────────────────────────────────────────────────
    X, H, scan_ids = load_unified_patch_matrix(split="train")
    logger.info(f"Train — X: {X.shape}, H: {H.shape}")
    _log_class_distribution(H, prefix="train (raw)")

    # ── Normalise + drop zero patches ─────────────────────────────────────────
    X_norm, _, zero_mask = normalise_columns(X)
    keep   = ~zero_mask
    X_norm   = X_norm[:, keep]
    H        = H[keep]
    scan_ids = scan_ids[keep]

    n_dropped = int(zero_mask.sum())
    logger.info(f"Dropped {n_dropped} zero-norm patches; {keep.sum()} remaining.")
    _log_class_distribution(H, prefix="train (after zero-norm drop)")

    # ── Validation set ────────────────────────────────────────────────────────
    X_val, H_val, scan_ids_val = load_unified_patch_matrix(split="val")
    X_val_norm, _, val_zero = normalise_columns(X_val)
    keep_val       = ~val_zero
    X_val_norm     = X_val_norm[:, keep_val]
    H_val          = H_val[keep_val]
    scan_ids_val   = scan_ids_val[keep_val]

    n_dropped_val = int(val_zero.sum())
    logger.info(f"Val: dropped {n_dropped_val} zero-norm patches; {keep_val.sum()} remaining.")
    _log_class_distribution(H_val, prefix="val (after zero-norm drop)")

    # ── Adapt dictionary size for small datasets ──────────────────────────────
    cfg = dict(LCKSVD_CONFIG)
    max_atoms = max(8, X_norm.shape[1] // 2)
    cfg["n_components"]    = min(cfg["n_components"], max_atoms)
    cfg["n_nonzero_coefs"] = min(cfg["n_nonzero_coefs"],
                                 max(1, cfg["n_components"] // 2))

    # ── Train ─────────────────────────────────────────────────────────────────
    model = LCKSVD(**cfg)
    logger.info("Starting LC-KSVD2 training…")
    t0 = time.time()

    classes  = list(range(len(CLASS_ORDER)))
    H_onehot = label_binarize(H, classes=classes).T  # (n_classes, n_patches)

    model.fit(X_norm, H_onehot)
    elapsed = time.time() - t0
    logger.info(f"Training complete in {elapsed:.1f}s")

    # ── Evaluate at scan level — pass integer H and scan_ids ──────────────────
    train_metrics = evaluate(model, X_norm,     H,     scan_ids,     split_name="train")
    val_metrics   = evaluate(model, X_val_norm, H_val, scan_ids_val, split_name="val")

    # ── Save ──────────────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "unified_lcksvd2.pkl"

    payload = {
        "model":           model,
        "class_order":     CLASS_ORDER,
        "train_metrics":   train_metrics,
        "val_metrics":     val_metrics,
        "lcksvd_config":   cfg,
        "patch_size":      PATCH_SIZE,
        "target_spacing":  TARGET_SPACING_MM,
        "hu_window":       (HU_MIN, HU_MAX),
        "training_time_s": elapsed,
    }

    with open(model_path, "wb") as f:
        pickle.dump(payload, f)

    logger.info(f"Model saved → {model_path}")
    return payload


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train a single unified LC-KSVD2 model (normal + 3 abnormalities)"
    )
    parser.add_argument(
        "--skip-extraction", action="store_true",
        help="Skip patch extraction and use existing unified .npz files."
    )
    args = parser.parse_args()

    if not args.skip_extraction:
        logger.info("Running unified patch extraction (train + val)…")
        extract_unified(split="train")
        extract_unified(split="val")

    result = train()

    vm = result["val_metrics"]
    logger.info(
        f"\nFinal val ({int(vm['n_scans'])} scans) — "
        f"AUROC(macro)={vm['auroc_macro']:.4f}  "
        f"F1(macro)={vm['f1_macro']:.4f}  AP(macro)={vm['ap_macro']:.4f}"
    )


if __name__ == "__main__":
    main()