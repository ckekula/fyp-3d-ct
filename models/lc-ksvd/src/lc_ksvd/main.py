"""
main.py
Trains one multi-class model over all abnormalities + normal.

Two algorithms are supported via --algorithm:

  frozen (default) - IncrementalFrozenDictionary (LC-KSVD-based Frozen
      Dictionary Learning). A base LC-KSVD2 dictionary is learned on
      "normal" patches only; abnormality classes are then added one at a
      time via add_class(), each learning a residual dictionary on top of
      the previously learned (frozen) atoms.

  lcksvd - the original single-shot LC-KSVD2 model trained jointly over
      all classes at once.

H rows (CLASS_ORDER):
  0 -> normal
  1 -> 2c  (groundglass opacity)
  2 -> 2d  (pulmonary nodules/masses)

Usage:
  python main.py                          # frozen (default), extract + train
  python main.py --algorithm lcksvd        # original joint LC-KSVD2
  python main.py --skip-extraction         # use existing unified .npz
"""

import argparse
import logging
import pickle
import time
from typing import Dict, List, Tuple

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
from reppi.dictionary.frozen import IncrementalFrozenDictionary

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

NORMAL_CLASS_IDX = 0  # CLASS_ORDER[0] == "normal"


# --- Column normalisation ---------------------------------------------------

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


def _adapt_lcksvd_kwargs(base_cfg: Dict, n_samples: int) -> Dict:
    """
    Shrink n_components / n_nonzero_coefs to fit a (possibly small) stage's
    sample count, mirroring the original small-dataset safeguard.
    """
    cfg = dict(base_cfg)
    max_atoms = max(8, n_samples // 2)
    cfg["n_components"] = min(cfg["n_components"], max_atoms)
    cfg["n_nonzero_coefs"] = min(cfg["n_nonzero_coefs"], max(1, cfg["n_components"] // 2))
    return cfg


# --- Scan-level evaluation ---------------------------------------------------

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


# --- Joint LC-KSVD2 training (original behaviour) ---------------------------

def _fit_lcksvd(X_norm: np.ndarray, H: np.ndarray, cfg: Dict) -> LCKSVD:
    """Train a single LC-KSVD2 model jointly over all classes."""
    stage_cfg = _adapt_lcksvd_kwargs(cfg, X_norm.shape[1])
    model = LCKSVD(**stage_cfg)
    logger.info("Starting LC-KSVD2 training (joint, all classes)...")

    classes  = list(range(len(CLASS_ORDER)))
    H_onehot = label_binarize(H, classes=classes).T  # (n_classes, n_patches)

    model.fit(X_norm, H_onehot)
    return model


# --- Incremental Frozen Dictionary (LC-KSVD-based) training -----------------

def _onehot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Build a (n_classes, n_samples) one-hot matrix.

    sklearn's label_binarize collapses the n_classes == 2 case to a single
    column, which IncrementalFrozenDictionary's class-counting logic does
    not expect, so that case is handled explicitly.
    """
    if n_classes == 2:
        row1 = labels.astype(np.float64)
        return np.vstack([1.0 - row1, row1])
    return label_binarize(labels, classes=list(range(n_classes))).T.astype(np.float64)


def _fit_frozen(X_norm: np.ndarray, H: np.ndarray, cfg: Dict) -> IncrementalFrozenDictionary:
    """
    Train an IncrementalFrozenDictionary:
      stage 0 (base)        - LCKSVD on "normal" patches only.
      stage 1..k (residual) - LCKSVD residual dictionary added one
                               abnormality class at a time, in CLASS_ORDER.

    All prior atoms (base + previously added classes) are frozen at each
    add_class() step; the classifier W is refit jointly over all
    accumulated data after every stage.
    """
    abnormal_indices: List[int] = [i for i in range(len(CLASS_ORDER)) if i != NORMAL_CLASS_IDX]

    # -- Base stage: normal patches only -------------------------------------
    base_mask = H == NORMAL_CLASS_IDX
    X_base = X_norm[:, base_mask]
    H_base = np.ones((1, X_base.shape[1]), dtype=np.float64)  # single "normal" row

    base_kwargs = _adapt_lcksvd_kwargs(cfg, X_base.shape[1])

    inc = IncrementalFrozenDictionary(
        base_learner_class=LCKSVD,
        base_learner_kwargs=base_kwargs,
        residual_learner_class=LCKSVD,
        residual_learner_kwargs=dict(cfg),  # overridden per-stage below
        n_nonzero_coefs=base_kwargs["n_nonzero_coefs"],
        learn_on_residual=True,
        refit_classifier=True,
        freeze_classifier=False,
    )

    logger.info(f"Fitting base dictionary on {X_base.shape[1]} normal patches...")
    inc.fit_base(X_base, H_base)

    # -- Residual stages: add each abnormality class in CLASS_ORDER order ----
    # add_class() refits W against ALL accumulated data, so the H passed in
    # each call must be one-hot over every class seen so far (including the
    # new one), with columns in the exact order the data was accumulated:
    # [base, class_1, class_2, ...].
    accumulated_label_chunks: List[np.ndarray] = [
        np.zeros(X_base.shape[1], dtype=np.int64)  # base = remapped label 0
    ]

    for stage, class_idx in enumerate(abnormal_indices, start=1):
        class_mask = H == class_idx
        X_class = X_norm[:, class_mask]
        n_class = X_class.shape[1]

        accumulated_label_chunks.append(np.full(n_class, stage, dtype=np.int64))
        n_classes_so_far = stage + 1

        all_labels = np.concatenate(accumulated_label_chunks)
        H_full = _onehot(all_labels, n_classes_so_far)

        stage_kwargs = _adapt_lcksvd_kwargs(cfg, n_class)
        cls_name = CLASS_ORDER[class_idx]
        logger.info(
            f"Adding residual dictionary for class '{cls_name}' "
            f"({n_class} patches, n_components={stage_kwargs['n_components']})..."
        )
        inc.add_class(
            X_class,
            H_full,
            class_label=class_idx,
            learner_kwargs_override=stage_kwargs,
        )

    return inc


# --- Training ----------------------------------------------------------------

def train(algorithm: str) -> Dict:
    logger.info(f"\n{'='*60}\nTraining unified model (algorithm={algorithm})\n{'='*60}")

    # -- Load patches ---------------------------------------------------------
    X, H, scan_ids = load_unified_patch_matrix(split="train")
    logger.info(f"Train - X: {X.shape}, H: {H.shape}")
    _log_class_distribution(H, prefix="train (raw)")

    # -- Normalise + drop zero patches ----------------------------------------
    X_norm, _, zero_mask = normalise_columns(X)
    keep   = ~zero_mask
    X_norm   = X_norm[:, keep]
    H        = H[keep]
    scan_ids = scan_ids[keep]

    n_dropped = int(zero_mask.sum())
    logger.info(f"Dropped {n_dropped} zero-norm patches; {keep.sum()} remaining.")
    _log_class_distribution(H, prefix="train (after zero-norm drop)")

    # -- Validation set ---------------------------------------------------------
    X_val, H_val, scan_ids_val = load_unified_patch_matrix(split="val")
    X_val_norm, _, val_zero = normalise_columns(X_val)
    keep_val       = ~val_zero
    X_val_norm     = X_val_norm[:, keep_val]
    H_val          = H_val[keep_val]
    scan_ids_val   = scan_ids_val[keep_val]

    n_dropped_val = int(val_zero.sum())
    logger.info(f"Val: dropped {n_dropped_val} zero-norm patches; {keep_val.sum()} remaining.")
    _log_class_distribution(H_val, prefix="val (after zero-norm drop)")

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
    val_metrics   = evaluate(model, X_val_norm, H_val, scan_ids_val, split_name="val")

    # -- Save ---------------------------------------------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_filename = "unified_frozen_lcksvd.pkl" if algorithm == "frozen" else "unified_lcksvd2.pkl"
    model_path = MODELS_DIR / model_filename

    payload = {
        "model":           model,
        "algorithm":       algorithm,
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

    logger.info(f"Model saved -> {model_path}")
    return payload


# --- Entry point ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train a unified model (normal + 3 abnormalities) using "
                     "either Frozen Dictionary Learning with LC-KSVD or plain LC-KSVD2."
    )
    parser.add_argument(
        "--skip-extraction", action="store_true",
        help="Skip patch extraction and use existing unified .npz files."
    )
    parser.add_argument(
        "--algorithm", choices=["frozen", "lcksvd"], default="frozen",
        help="'frozen' = IncrementalFrozenDictionary (LC-KSVD-based Frozen "
             "Dictionary Learning, default). 'lcksvd' = original joint LC-KSVD2."
    )
    args = parser.parse_args()

    if not args.skip_extraction:
        logger.info("Running unified patch extraction (train + val)...")
        extract_unified(split="train")
        extract_unified(split="val")

    result = train(algorithm=args.algorithm)

    vm = result["val_metrics"]
    logger.info(
        f"\nFinal val ({int(vm['n_scans'])} scans) - algorithm={args.algorithm} - "
        f"AUROC(macro)={vm['auroc_macro']:.4f}  "
        f"F1(macro)={vm['f1_macro']:.4f}  AP(macro)={vm['ap_macro']:.4f}"
    )


if __name__ == "__main__":
    main()