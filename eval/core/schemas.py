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