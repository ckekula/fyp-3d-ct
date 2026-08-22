"""
model_fitting.py
Model-fitting logic for the two supported algorithms:
  - _fit_lcksvd  - original single-shot LC-KSVD2 trained jointly over all classes.
  - _fit_frozen  - IncrementalFrozenDictionary: a base K-SVD dictionary learned
                   on "normal" patches only, then one residual dictionary added per
                   abnormality class (in CLASS_ORDER), freezing prior atoms each step.
"""

import logging

import numpy as np
from reppi import FDDL, KSVD, LCKSVD, IncrementalFrozenDictionary
from sklearn.preprocessing import label_binarize

from lc_ksvd.config import (
    CHECKPOINT_DIR,
    CHECKPOINT_RESUME,
    CLASS_ORDER,
    NORMAL_CLASS_IDX,
)

logger = logging.getLogger(__name__)


# --- Joint LC-KSVD2 training (original behaviour) ---------------------------

def _fit_ksvd(
    X_norm: np.ndarray,
    H: np.ndarray,
    base_cfg: dict,
    n_components_by_class: dict[str, int] | None = None,
) -> dict[str, KSVD]:
    n_components_by_class = n_components_by_class or {}
    models: dict[str, KSVD] = {}

    for class_idx, cls_name in enumerate(CLASS_ORDER):
        class_mask = H == class_idx
        X_class = X_norm[:, class_mask]
        n_class = X_class.shape[1]
 
        stage_cfg = dict(base_cfg)
        stage_cfg["n_components"] = n_components_by_class.get(
            cls_name, base_cfg["n_components"]
        )

        logger.info(
            f"Starting K-SVD training for class '{cls_name}' "
            f"({n_class} patches, n_components={stage_cfg['n_components']})..."
        )
        model = KSVD(**stage_cfg)
        model.fit(
            X_class,
            checkpoint_dir=str(CHECKPOINT_DIR / cls_name),
            resume=CHECKPOINT_RESUME,
        )
        models[cls_name] = model

    return models

def _fit_lcksvd(X_norm: np.ndarray, H: np.ndarray, cfg: dict) -> LCKSVD:
    """Train a single LC-KSVD2 model jointly over all classes."""
    model = LCKSVD(**cfg)
    logger.info("Starting LC-KSVD2 training (joint, all classes)...")

    classes  = list(range(len(CLASS_ORDER)))
    H_onehot = label_binarize(H, classes=classes).T  # (n_classes, n_patches)

    model.fit(X_norm, H_onehot,
        checkpoint_dir=str(CHECKPOINT_DIR),
        resume=CHECKPOINT_RESUME,
    )
    return model

def _fit_fddl(X_norm: np.ndarray, H: np.ndarray, cfg: dict) -> FDDL:
    """Train a single FDDL model jointly over all classes."""
    model = FDDL(**cfg)
    logger.info("Starting FDDL training (joint, all classes)...")
    model.fit(
        X_norm, H,
        checkpoint_dir=str(CHECKPOINT_DIR),
        resume=CHECKPOINT_RESUME,
    )
    return model

def _fit_frozen(
    X_norm: np.ndarray,
    H: np.ndarray,
    base_cfg: dict,
    residual_n_components_by_class: dict[str, int] | None = None,
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
    add_class() step.

    Parameters
    ----------
    base_cfg : dict
        Full KSVD config (n_components, n_nonzero_coefs, n_iter, ...).
        Its n_components sizes the base dictionary.
    residual_n_components_by_class : Dict[str, int] or None
        Maps a CLASS_ORDER class name -> desired n_components for that
        class's residual dictionary. A class not present here falls back
        to base_cfg["n_components"] (subject to the same per-stage
        small-dataset clamping as everything else). Pass None to size
        every residual stage identically to the base.
    """
    abnormal_indices: list[int] = [i for i in range(len(CLASS_ORDER)) if i != NORMAL_CLASS_IDX]
    residual_n_components_by_class = residual_n_components_by_class or {}

    # -- Base stage: normal patches only -------------------------------------
    base_mask = H == NORMAL_CLASS_IDX
    X_base = X_norm[:, base_mask]
    # Constructor-level default for residual_learner_kwargs: always
    # overridden per-stage below via learner_kwargs_override, but must
    # still be a sane, correctly-shaped default in case add_class is ever
    # called directly without an override.
    inc = IncrementalFrozenDictionary(
        base_learner_class=KSVD,
        base_learner_kwargs=base_cfg,
        residual_learner_class=KSVD,
        residual_learner_kwargs=dict(base_cfg),
        n_nonzero_coefs=base_cfg["n_nonzero_coefs"],
    )

    inc.fit_base(
        X_base,
        class_label=NORMAL_CLASS_IDX,
        checkpoint_dir=str(CHECKPOINT_DIR),
        resume=CHECKPOINT_RESUME,
    )
 
    for class_idx in abnormal_indices:
        class_mask = H == class_idx
        X_class = X_norm[:, class_mask]
        n_class = X_class.shape[1]
 
        cls_name = CLASS_ORDER[class_idx]
 
        # Per-class residual atom count, falling back to the base's if this
        # class wasn't given its own entry.
        stage_cfg = dict(base_cfg)
        stage_cfg["n_components"] = residual_n_components_by_class.get(
            cls_name, base_cfg["n_components"]
        )
 
        logger.info(
            f"Adding residual dictionary for class '{cls_name}' "
            f"({n_class} patches, n_components={stage_cfg['n_components']})..."
        )
        inc.add_class(
            X_class,
            class_label=class_idx,
            learner_kwargs_override=stage_cfg,
            checkpoint_dir=str(CHECKPOINT_DIR),
            resume=CHECKPOINT_RESUME,
        )
 
    return inc