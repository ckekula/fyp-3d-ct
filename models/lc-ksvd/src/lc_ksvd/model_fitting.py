"""
model_fitting.py
Model-fitting logic for the two supported algorithms:
  - _fit_lcksvd  - original single-shot LC-KSVD2 trained jointly over all classes.
  - _fit_frozen  - IncrementalFrozenDictionary: a base LC-KSVD2 dictionary learned
                   on "normal" patches only, then one residual dictionary added per
                   abnormality class (in CLASS_ORDER), freezing prior atoms each step.
"""

import logging
from typing import Dict, List

import numpy as np
from sklearn.preprocessing import label_binarize

from lc_ksvd.config import CLASS_ORDER, NORMAL_CLASS_IDX, CHECKPOINT_DIR, CHECKPOINT_RESUME
from reppi import LCKSVD
from reppi.dictionary.frozen import IncrementalFrozenDictionary

logger = logging.getLogger(__name__)


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
    inc.fit_base(
        X_base,
        H_base,
        checkpoint_dir=str(CHECKPOINT_DIR),
        resume=CHECKPOINT_RESUME,
    )

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
            checkpoint_dir=str(CHECKPOINT_DIR),
            resume=CHECKPOINT_RESUME,
        )

    return inc