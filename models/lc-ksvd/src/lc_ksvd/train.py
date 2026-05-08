"""
train.py
Trains one LC-KSVD2 binary classifier per abnormality using reppi.

Pipeline per abnormality:
  1. Load patch matrix (X, H) from PATCHES_DIR
  2. Train LCKSVD (variant="lcksvd2") using hyperparameters from config
  3. Evaluate on validation split
  4. Save model to MODELS_DIR as a pickle file

Usage:
  python train.py                        # train all 4 abnormalities
  python train.py --abnormality lung_nodule  # train one
  python train.py --skip-extraction      # use existing .npz files
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

from lc_ksvd.config import ABNORMALITY_CATEGORIES, HU_MAX, HU_MIN, LCKSVD_CONFIG, MODELS_DIR, PATCH_SIZE, TARGET_SPACING_MM
from lc_ksvd.patch_extractor import extract_all_abnormalities, load_patch_matrix

try:
    from reppi import LCKSVD
except ImportError:
    raise ImportError(
        "reppi is not installed. Run: pip install reppi"
    )

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ─── Column normalisation ─────────────────────────────────────────────────────

def normalise_columns(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Unit-normalise each column of X (each patch vector).
    LC-KSVD and OMP require unit-norm signals for correct sparse coding.

    Returns:
      X_norm : (n_features, n_patches) with unit-norm columns
      norms  : (n_patches,) original L2 norms — needed to invert for back-projection
      zero_mask: (n_patches,) bool — True where norm was 0 (all-zero patch)
    """
    norms = np.linalg.norm(X, axis=0)          # (n_patches,)
    zero_mask = norms < 1e-10
    norms_safe = np.where(zero_mask, 1.0, norms)
    X_norm = X / norms_safe[np.newaxis, :]
    return X_norm, norms, zero_mask





# ─── Evaluation helpers ───────────────────────────────────────────────────────

def evaluate(
    model: LCKSVD,
    X_norm: np.ndarray,
    H: np.ndarray,
    split_name: str,
) -> Dict[str, float]:
    """
    Compute classification metrics on a (normalised) patch matrix.

    Metrics:
      AUROC, F1 (threshold=0.5 on W·Gamma), Average Precision
    """
    # Sparse codes
    Gamma = model.transform(X_norm)                # (n_components, n_patches)

    # Raw scores: project onto classifier weight for positive class (row 1 of W)
    # W is (n_classes, n_components) = (2, K)
    W = model.W_
    scores = W[1, :] @ Gamma                       # (n_patches,)

    # Ground-truth binary labels
    y_true = H[1, :].astype(int)                   # 1 = positive, 0 = negative

    # Predictions at threshold 0.5 (after sigmoid-like normalisation)
    # W·Gamma scores are unbounded; we use sign for hard predictions
    y_pred = (scores > 0).astype(int)

    metrics = {}
    try:
        metrics["auroc"] = roc_auc_score(y_true, scores)
    except ValueError:
        metrics["auroc"] = float("nan")

    metrics["f1"]   = f1_score(y_true, y_pred, zero_division=0)

    try:
        metrics["ap"] = average_precision_score(y_true, scores)
    except ValueError:
        metrics["ap"] = float("nan")

    logger.info(
        f"  [{split_name}] AUROC={metrics['auroc']:.4f}  "
        f"F1={metrics['f1']:.4f}  AP={metrics['ap']:.4f}"
    )
    return metrics


# ─── Single-abnormality training ─────────────────────────────────────────────

def train_one(abnormality: str) -> Dict:
    """
    Full training pipeline for one binary LC-KSVD2 model.
    Returns a results dict with trained model, metrics, and norms.
    """
    logger.info(f"\n{'='*60}\nTraining: {abnormality}\n{'='*60}")

    # ── Load patches ──────────────────────────────────────────────────────────
    X, H = load_patch_matrix(abnormality, split="train")
    logger.info(f"Loaded X {X.shape}, H {H.shape}  "
                f"(pos={int(H[1].sum())}, neg={int(H[0].sum())})")

    # ── Normalise columns ─────────────────────────────────────────────────────
    X_norm, col_norms, zero_mask = normalise_columns(X)
    logger.info(f"Column-normalised X.  Zero patches: {zero_mask.sum()}")

    # Remove all-zero patches (uninformative)
    keep = ~zero_mask
    X_norm = X_norm[:, keep]
    H      = H[:, keep]
    logger.info(f"After zero-patch removal: {X_norm.shape[1]} patches")

    # ── Load validation patches ───────────────────────────────────────────────────
    X_val, H_val = load_patch_matrix(abnormality, split="val")
    logger.info(f"Loaded validation X {X_val.shape}, H {H_val.shape}  "
                f"(pos={int(H_val[1].sum())}, neg={int(H_val[0].sum())})")

    # ── Normalise validation columns ───────────────────────────────────────────────
    X_val_norm, _, val_zero_mask = normalise_columns(X_val)
    X_val_norm = X_val_norm[:, ~val_zero_mask]
    H_val = H_val[:, ~val_zero_mask]
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

    logger.info("Starting LC-KSVD2 training…")
    t0 = time.time()
    model.fit(X_tr, H_tr)
    elapsed = time.time() - t0
    logger.info(f"Training complete in {elapsed:.1f}s")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    train_metrics = evaluate(model, X_tr,  H_tr,  split_name="train")
    val_metrics   = evaluate(model, X_val, H_val, split_name="val")

    # ── Save model ────────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{abnormality}_lcksvd2.pkl"

    payload = {
        "model":           model,
        "abnormality":     abnormality,
        "train_metrics":   train_metrics,
        "val_metrics":     val_metrics,
        "lcksvd_config":   base_cfg,
        "patch_size":      PATCH_SIZE,
        "target_spacing":  TARGET_SPACING_MM,
        "hu_window":       (HU_MIN, HU_MAX),
        "training_time_s": elapsed,
    }

    with open(model_path, "wb") as f:
        pickle.dump(payload, f)

    logger.info(f"Model saved to {model_path}")
    return payload


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    abnormalities = list(ABNORMALITY_CATEGORIES.keys())

    parser = argparse.ArgumentParser(description="Train LC-KSVD2 models for chest CT abnormalities")
    parser.add_argument(
        "--abnormality", type=str, default=None,
        choices=abnormalities + [None],
        help="Train a single abnormality. Omit to train all 4."
    )
    parser.add_argument(
        "--skip-extraction", action="store_true",
        help="Skip patch extraction and use existing .npz files."
    )
    args = parser.parse_args()

    # ── Patch extraction ──────────────────────────────────────────────────────
    if not args.skip_extraction:
        logger.info("Running patch extraction…")
        extract_all_abnormalities(split="train")

    # ── Training ──────────────────────────────────────────────────────────────
    targets = [args.abnormality] if args.abnormality else abnormalities
    all_results = {}

    for abnormality in targets:
        try:
            result = train_one(abnormality)
            all_results[abnormality] = result
        except Exception as e:
            logger.error(f"Failed to train {abnormality}: {e}", exc_info=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}\nTraining Summary\n{'='*60}")
    for ab, res in all_results.items():
        vm = res["val_metrics"]
        logger.info(
            f"  {ab:<20}  Val AUROC={vm['auroc']:.4f}  "
            f"F1={vm['f1']:.4f}  AP={vm['ap']:.4f}"
        )


if __name__ == "__main__":
    main()