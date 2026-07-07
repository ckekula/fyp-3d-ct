"""
train.py
Training pipeline: load the unified patch matrix, normalise columns, fit the
chosen algorithm (frozen or lcksvd), evaluate at scan level, and persist the
resulting model + metadata to disk.
"""

import logging
import pickle
import time
from typing import Dict

from lc_ksvd.config import (
    CLASS_ORDER, HU_MAX, HU_MIN, LCKSVD_CONFIG,
    MODELS_DIR, PATCH_SIZE, TARGET_SPACING_MM,
)
from lc_ksvd.metrics import evaluate, log_class_distribution, normalise_columns
from lc_ksvd.model_fitting import _fit_frozen, _fit_lcksvd
from lc_ksvd.patch_extractor.patch_extraction import load_unified_patch_matrix

logger = logging.getLogger(__name__)


def train(algorithm: str) -> Dict:
    logger.info(f"\n{'='*60}\nTraining unified model (algorithm={algorithm})\n{'='*60}")

    # -- Load patches ---------------------------------------------------------
    X, H, scan_ids = load_unified_patch_matrix(split="train")
    logger.info(f"Train - X: {X.shape}, H: {H.shape}")
    log_class_distribution(H, prefix="train (raw)")

    # -- Normalise + drop zero patches ----------------------------------------
    X_norm, _, zero_mask = normalise_columns(X)
    keep   = ~zero_mask
    X_norm   = X_norm[:, keep]
    H        = H[keep]
    scan_ids = scan_ids[keep]

    n_dropped = int(zero_mask.sum())
    logger.info(f"Dropped {n_dropped} zero-norm patches; {keep.sum()} remaining.")
    log_class_distribution(H, prefix="train (after zero-norm drop)")

    # -- Train ------------------------------------------------------------------
    cfg = dict(LCKSVD_CONFIG)
    t0 = time.time()

    if algorithm == "frozen":
        model = _fit_frozen(X_norm, H, cfg)
    elif algorithm == "lcksvd":
        model = _fit_lcksvd(X_norm, H, cfg)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm!r}")

    elapsed = time.time() - t0
    logger.info(f"Training complete in {elapsed:.1f}s")

    # -- Evaluate at scan level - pass integer H and scan_ids --------------------
    train_metrics = evaluate(model, X_norm,     H,     scan_ids,     split_name="train")

    # -- Save ---------------------------------------------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_filename = "unified_frozen_lcksvd.pkl" if algorithm == "frozen" else "unified_lcksvd2.pkl"
    model_path = MODELS_DIR / model_filename

    payload = {
        "model":           model,
        "algorithm":       algorithm,
        "class_order":     CLASS_ORDER,
        "train_metrics":   train_metrics,
        "lcksvd_config":   cfg,
        "patch_size":      PATCH_SIZE,
        "target_spacing":  TARGET_SPACING_MM,
        "hu_window":       (HU_MIN, HU_MAX),
        "training_time_s": elapsed,
    }

    with open(model_path, "wb") as f:
        pickle.dump(payload, f)

    logger.info(f"Model saved -> {model_path}")
    return payload