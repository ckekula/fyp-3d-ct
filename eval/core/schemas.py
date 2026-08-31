from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import numpy as np


@dataclass
class ClassificationSample:
    case_id: str
    model_name: str
    y_true: Dict[str, int]
    y_score: Dict[str, float]
    dataset: str
    metadata: Optional[dict] = None
    # "probabilistic": y_score is a real continuous confidence in [0, 1].
    # "hard_label": y_score is a 0/1 decision with no underlying confidence
    # (e.g. an LLM/VLM category classifier with no numeric score). Rank-based
    # metrics (AUROC, AP, ECE) are not meaningful for hard_label samples and
    # must be reported as N/A rather than computed.
    score_type: str = "probabilistic"


@dataclass
class LocalizationSample:
    case_id: str
    model_name: str
    class_name: str
    pred_mask: np.ndarray
    gt_mask: np.ndarray
    spacing: tuple[float, float, float]
    pred_score_map: Optional[np.ndarray] = None
    existence_score: Optional[float] = None
    morphology: Optional[str] = None
    dataset: str = "rexgroundingct"
    # False when pred_mask is already a hard 0/1 decision (e.g. MedSAM2,
    # Merlin, LC-KSVD) rather than a continuous confidence map (BiomedParse).
    # Binarizing a hard mask at different thresholds always yields the same
    # result, so threshold-sweep evaluation must skip these samples instead
    # of reporting a fake "sensitivity" that's really just the same number
    # repeated at every threshold.
    is_soft_mask: bool = True


@dataclass
class ExplainabilitySample:
    case_id: str
    model_name: str
    class_name: str
    attribution_map: np.ndarray
    gt_mask: np.ndarray
    y_true: int
    y_pred: int
    y_score: float
    spacing: tuple[float, float, float]
    atom_ids: Optional[list[int]] = None
    sparse_codes: Optional[np.ndarray] = None
    metadata: Optional[dict] = None