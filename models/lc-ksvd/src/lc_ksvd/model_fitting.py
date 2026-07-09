"""
model_fitting.py
Model-fitting logic for the two supported algorithms:
  - _fit_lcksvd  - original single-shot LC-KSVD2 trained jointly over all classes.
  - _fit_frozen  - IncrementalFrozenDictionary: a base K-SVD dictionary learned
                   on "normal" patches only, then one residual dictionary added per
                   abnormality class (in CLASS_ORDER), freezing prior atoms each step.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
from sklearn.preprocessing import label_binarize

from lc_ksvd.config import CLASS_ORDER, NORMAL_CLASS_IDX, CHECKPOINT_DIR, CHECKPOINT_RESUME
from reppi import KSVD, LCKSVD
from reppi.dictionary.frozen import IncrementalFrozenDictionary

logger = logging.getLogger(__name__)


def _adapt_ksvd_kwargs(base_cfg: Dict, n_samples: int) -> Dict:
    """
    Shrink n_components / n_nonzero_coefs to fit a (possibly small) stage's
    sample count, mirroring the original small-dataset safeguard.
    """
    cfg = dict(base_cfg)
    max_atoms = max(8, n_samples // 2)
    cfg["n_components"] = min(cfg["n_components"], max_atoms)
    cfg["n_nonzero_coefs"] = min(cfg["n_nonzero_coefs"], max(1, cfg["n_components"] // 2))
    return cfg


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


# --- Incremental Frozen Dictionary (KSVD-based) training ---------------------

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


def _fit_frozen(
    X_norm: np.ndarray,
    H: np.ndarray,
    base_cfg: Dict,
    residual_n_components_by_class: Optional[Dict[str, int]] = None,
) -> IncrementalFrozenDictionary:
    """
    Train an IncrementalFrozenDictionary:
      stage 0 (base)        - KSVD on "normal" patches only, sized per
                               base_cfg["n_components"].
      stage 1..k (residual) - KSVD residual dictionary added one
                               abnormality class at a time, in CLASS_ORDER,
                               each sized independently via
                               residual_n_components_by_class.

    All prior atoms (base + previously added classes) are frozen at each
    add_class() step; the classifier W is refit jointly over all
    accumulated data after every stage.

    Parameters
    ----------
    base_cfg : Dict
        Full KSVD config (n_components, n_nonzero_coefs, n_iter, ...).
        Its n_components sizes the base dictionary.
    residual_n_components_by_class : Dict[str, int] or None
        Maps a CLASS_ORDER class name -> desired n_components for that
        class's residual dictionary. A class not present here falls back
        to base_cfg["n_components"] (subject to the same per-stage
        small-dataset clamping as everything else). Pass None to size
        every residual stage identically to the base.
    """
    abnormal_indices: List[int] = [i for i in range(len(CLASS_ORDER)) if i != NORMAL_CLASS_IDX]
    residual_n_components_by_class = residual_n_components_by_class or {}

    # -- Base stage: normal patches only -------------------------------------
    base_mask = H == NORMAL_CLASS_IDX
    X_base = X_norm[:, base_mask]
    H_base = np.ones((1, X_base.shape[1]), dtype=np.float64)

    base_kwargs = _adapt_ksvd_kwargs(base_cfg, X_base.shape[1])

    # Constructor-level default for residual_learner_kwargs: always
    # overridden per-stage below via learner_kwargs_override, but must
    # still be a sane, correctly-shaped default in case add_class is ever
    # called directly without an override.
    inc = IncrementalFrozenDictionary(
        base_learner_class=KSVD,
        base_learner_kwargs=base_kwargs,
        residual_learner_class=KSVD,
        residual_learner_kwargs=dict(base_kwargs),
        n_nonzero_coefs=base_kwargs["n_nonzero_coefs"],
        refit_classifier=True,
        freeze_classifier=False,
    )

    logger.info(
        f"Fitting base dictionary on {X_base.shape[1]} normal patches "
        f"(n_components={base_kwargs['n_components']})..."
    )
    inc.fit_base(
        X_base,
        H_base,
        class_label=NORMAL_CLASS_IDX,
        checkpoint_dir=str(CHECKPOINT_DIR),
        resume=CHECKPOINT_RESUME,
    )

    accumulated_label_chunks: List[np.ndarray] = [
        np.zeros(X_base.shape[1], dtype=np.int64)
    ]

    for stage, class_idx in enumerate(abnormal_indices, start=1):
        class_mask = H == class_idx
        X_class = X_norm[:, class_mask]
        n_class = X_class.shape[1]

        accumulated_label_chunks.append(np.full(n_class, stage, dtype=np.int64))
        n_classes_so_far = stage + 1

        all_labels = np.concatenate(accumulated_label_chunks)
        H_full = _onehot(all_labels, n_classes_so_far)

        cls_name = CLASS_ORDER[class_idx]

        # Per-class residual atom count, falling back to the base's if this
        # class wasn't given its own entry.
        stage_cfg = dict(base_cfg)
        stage_cfg["n_components"] = residual_n_components_by_class.get(
            cls_name, base_cfg["n_components"]
        )
        stage_kwargs = _adapt_ksvd_kwargs(stage_cfg, n_class)

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