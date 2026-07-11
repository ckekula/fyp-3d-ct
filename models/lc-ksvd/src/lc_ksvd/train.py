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
    CLASS_ORDER, HU_MAX, HU_MIN, KSVD_CONFIG, LCKSVD_CONFIG, N_FEATURES,
    NORMAL_CLASS_IDX, MODELS_DIR, PATCH_SIZE, TARGET_SPACING_MM,
)
from lc_ksvd.metrics import evaluate, log_class_distribution, normalise_columns
from lc_ksvd.model_fitting import _fit_frozen, _fit_lcksvd
from lc_ksvd.patch_extractor.patch_extraction import load_unified_patch_matrix

logger = logging.getLogger(__name__)


def _build_residual_n_components_by_class() -> Dict[str, int]:
    """
    N_FEATURES*4 for the base (set directly on KSVD_CONFIG, not here),
    N_FEATURES*2 for the first abnormality class in CLASS_ORDER (excluding
    the normal class), N_FEATURES for the second.

    Relies on dict insertion order matching CLASS_ORDER's order of
    abnormality classes (Python 3.7+ dicts preserve insertion order, and
    CLASS_ORDER is a fixed, ordered sequence) — if CLASS_ORDER's ordering
    ever changes, this mapping tracks it automatically since it's built
    from CLASS_ORDER directly rather than hardcoding class names.
    """
    abnormal_classes = [c for c in CLASS_ORDER if c != CLASS_ORDER[NORMAL_CLASS_IDX]]
    if len(abnormal_classes) != 2:
        raise ValueError(
            f"Expected exactly 2 abnormality classes in CLASS_ORDER, got "
            f"{len(abnormal_classes)}: {abnormal_classes}. Update "
            "_build_residual_n_components_by_class to match."
        )
    return {
        abnormal_classes[0]: N_FEATURES * 2,
        abnormal_classes[1]: N_FEATURES,
    }


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
    frozen_cfg = dict(KSVD_CONFIG)
    frozen_cfg["n_components"] = N_FEATURES * 4  # base dictionary size
    lcksvd_cfg = dict(LCKSVD_CONFIG)
    t0 = time.time()

    if algorithm == "frozen":
        cfg = frozen_cfg
        model = _fit_frozen(
            X_norm, H, frozen_cfg,
            residual_n_components_by_class=_build_residual_n_components_by_class(),
        )
    elif algorithm == "lcksvd":
        cfg = lcksvd_cfg
        model = _fit_lcksvd(X_norm, H, lcksvd_cfg)
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
        "config":          cfg,
        "patch_size":      PATCH_SIZE,
        "target_spacing":  TARGET_SPACING_MM,
        "hu_window":       (HU_MIN, HU_MAX),
        "training_time_s": elapsed,
    }

    with open(model_path, "wb") as f:
        pickle.dump(payload, f)

    logger.info(f"Model saved -> {model_path}")
    return payload