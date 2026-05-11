"""
train.py
Trains one LC-KSVD2 multi-class model over all abnormalities + normal.

H rows (CLASS_ORDER):
  0 → normal
  1 → 2a  (linear / atelectasis / scarring)
  2 → 2b  (atelectasis / consolidation)
  3 → 2c  (groundglass opacity)
  4 → 2d  (pulmonary nodules/masses)

Usage:
  python train.py                   # extract patches then train
  python train.py --skip-extraction # use existing unified .npz
"""

import argparse
import logging
import pickle
import time
from pathlib import Path
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


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(
    model: LCKSVD,
    X_norm: np.ndarray,
    H: np.ndarray,           # (n_patches,) int64
    split_name: str,
) -> Dict[str, float]:
    Gamma  = model.transform(X_norm)   # (n_components, n_patches)
    W      = model.W_                  # (n_classes, n_components)
    scores = W @ Gamma                 # (n_classes, n_patches)

    y_pred = np.argmax(scores, axis=0) # (n_patches,)
    y_true = H                         # already integer class indices

    n_classes = len(CLASS_ORDER)
    metrics: Dict[str, float] = {}

    aurocs, aps = [], []
    for c in range(n_classes):
        y_true_bin = (y_true == c).astype(int)   # binarise for one-vs-rest
        score_c    = scores[c, :]

        try:
            aurocs.append(roc_auc_score(y_true_bin, score_c))
        except ValueError:
            aurocs.append(float("nan"))

        try:
            aps.append(average_precision_score(y_true_bin, score_c))
        except ValueError:
            aps.append(float("nan"))

    metrics["auroc_macro"] = float(np.nanmean(aurocs))
    metrics["ap_macro"]    = float(np.nanmean(aps))
    metrics["f1_macro"]    = f1_score(y_true, y_pred, average="macro", zero_division=0)

    for i, cls in enumerate(CLASS_ORDER):
        metrics[f"auroc_{cls}"] = aurocs[i]
        metrics[f"ap_{cls}"]    = aps[i]

    logger.info(
        f"  [{split_name}] AUROC(macro)={metrics['auroc_macro']:.4f}  "
        f"F1(macro)={metrics['f1_macro']:.4f}  AP(macro)={metrics['ap_macro']:.4f}"
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
    X, H = load_unified_patch_matrix(split="train")
    logger.info(f"Train — X: {X.shape}, H: {H.shape}")
    for i, cls in enumerate(CLASS_ORDER):
        logger.info(f"  class {cls}: {int(H[i].sum())} patches")

    # ── Normalise + drop zero patches ─────────────────────────────────────────
    X_norm, _, zero_mask = normalise_columns(X)
    keep   = ~zero_mask
    X_norm = X_norm[:, keep]
    H      = H[keep]
    logger.info(f"After zero-patch removal: {X_norm.shape[1]} patches")

    # ── Validation set ────────────────────────────────────────────────────────
    X_val, H_val = load_unified_patch_matrix(split="val")
    X_val_norm, _, val_zero = normalise_columns(X_val)
    X_val_norm = X_val_norm[:, ~val_zero]
    H_val      = H_val[:, ~val_zero]
    logger.info(f"Val   — {X_val_norm.shape[1]} patches")

    # ── Normalise validation columns ───────────────────────────────────────────────
    X_val_norm, _, val_zero_mask = normalise_columns(X_val)
    X_val_norm = X_val_norm[:, ~val_zero_mask]
    H_val = H_val[~val_zero_mask]
    logger.info(f"Validation after zero-patch removal: {X_val_norm.shape[1]} patches")

    # ── Final train/val split ──────────────────────────────────────────────────────
    X_tr, H_tr = X_norm, H
    X_val, H_val = X_val_norm, H_val
    logger.info(f"Train: {X_tr.shape[1]} patches  |  Val: {X_val.shape[1]} patches")

    # ── Adapt LC-KSVD dictionary size for small training sets ────────────────
    # reppi's K-SVD initialisation requires enough training signals to seed the
    # dictionary. For very small patch matrices (e.g. opacity/consolidation),
    # the default 128 atoms can exceed the available training columns.
    base_cfg = dict(LCKSVD_CONFIG)
    max_atoms = max(8, X_tr.shape[1] // 2)
    effective_n_components = min(base_cfg["n_components"], max_atoms)
    effective_n_nonzero = min(base_cfg["n_nonzero_coefs"], max(1, effective_n_components // 2))

    if effective_n_components != base_cfg["n_components"]:
        logger.info(
            f"Adjusting n_components from {base_cfg['n_components']} to {effective_n_components} "
            f"for {abnormality} (train patches={X_tr.shape[1]})"
        )

    base_cfg["n_components"] = effective_n_components
    base_cfg["n_nonzero_coefs"] = effective_n_nonzero

    # ── Train LC-KSVD2 ────────────────────────────────────────────────────────
    model = LCKSVD(**base_cfg)

    # ── Train ─────────────────────────────────────────────────────────────────
    model = LCKSVD(**cfg)
    logger.info("Starting LC-KSVD2 training…")
    t0 = time.time()
    # reppi expects one-hot
    H_onehot = label_binarize(H, classes=list(range(len(CLASS_ORDER)))).T  # (n_classes, n_patches)
    H_val_onehot = label_binarize(H_val, classes=list(range(len(CLASS_ORDER)))).T
    model.fit(X_norm, H_onehot)
    elapsed = time.time() - t0
    logger.info(f"Training complete in {elapsed:.1f}s")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    train_metrics = evaluate(model, X_norm,     H_onehot,     split_name="train")
    val_metrics   = evaluate(model, X_val_norm, H_val_onehot, split_name="val")

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
        description="Train a single unified LC-KSVD2 model (normal + 4 abnormalities)"
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
        f"\nFinal val — AUROC(macro)={vm['auroc_macro']:.4f}  "
        f"F1(macro)={vm['f1_macro']:.4f}  AP(macro)={vm['ap_macro']:.4f}"
    )


if __name__ == "__main__":
    main()