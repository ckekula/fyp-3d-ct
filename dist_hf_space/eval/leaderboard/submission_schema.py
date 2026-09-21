"""
leaderboard/submission_schema.py
Pydantic models and JSON Schema definitions for Track A, Track B, and Track C submissions.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple, Union
from pydantic import BaseModel, Field, field_validator


CANONICAL_CLASSES = ["lung_nodule", "lung_opacity", "consolidation", "atelectasis"]


class TrackAClassificationRow(BaseModel):
    case_id: str = Field(..., description="Evaluation scan identifier")
    lung_nodule: float = Field(..., ge=0.0, le=1.0, description="Predicted score for lung nodule")
    lung_opacity: float = Field(..., ge=0.0, le=1.0, description="Predicted score for lung opacity")
    consolidation: float = Field(..., ge=0.0, le=1.0, description="Predicted score for consolidation")
    atelectasis: float = Field(..., ge=0.0, le=1.0, description="Predicted score for atelectasis")
    score_type: Literal["probabilistic", "hard_label"] = Field(
        default="probabilistic",
        description="Whether predictions are continuous probabilities or discrete 0/1 hard decisions"
    )

    @field_validator("lung_nodule", "lung_opacity", "consolidation", "atelectasis")
    @classmethod
    def validate_score_range(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError(f"Score must be between 0.0 and 1.0, got {v}")
        return float(v)


class ClassManifestEntry(BaseModel):
    morphology: Literal["focal", "non_focal"] = Field(
        ...,
        description="'focal' uses centroid distance; 'non_focal' uses surface distance (ASSD)"
    )
    is_soft_mask: bool = Field(
        default=True,
        description="True for continuous probabilities [0, 1]; False for hard 0/1 binary masks"
    )


class TrackBManifest(BaseModel):
    model_name: str = Field(..., min_length=1, description="Unique model name")
    axis_order: Literal["ZYX", "XYZ"] = Field(
        ...,
        description="Axis order of 3D arrays: 'ZYX' (depth, height, width) or 'XYZ' (width, height, depth)"
    )
    spacing_mm: Tuple[float, float, float] = Field(
        ...,
        description="Physical voxel spacing in millimeters as a 3-tuple (e.g. [1.25, 0.75, 0.75])"
    )
    is_soft_mask: bool = Field(
        default=True,
        description="Global default soft-mask indicator across all finding classes"
    )
    classes: Optional[Dict[str, ClassManifestEntry]] = Field(
        default=None,
        description="Per-class morphology and soft mask overrides"
    )

    @field_validator("spacing_mm")
    @classmethod
    def validate_spacing(cls, v: Tuple[float, float, float]) -> Tuple[float, float, float]:
        if len(v) != 3 or any(s <= 0 for s in v):
            raise ValueError(f"Voxel spacing must be 3 positive floats, got {v}")
        return (float(v[0]), float(v[1]), float(v[2]))


class TrackCManifest(TrackBManifest):
    predictions_csv_path: Optional[str] = Field(
        default="predictions.csv",
        description="Relative path to Track A classification predictions CSV"
    )
    masks_dir: Optional[str] = Field(
        default="masks",
        description="Relative path to Track B segmentation masks directory"
    )


def get_json_schema() -> Dict[str, dict]:
    """Return JSON Schema for all submission tracks."""
    return {
        "track_a_row": TrackAClassificationRow.model_json_schema(),
        "track_b_manifest": TrackBManifest.model_json_schema(),
        "track_c_manifest": TrackCManifest.model_json_schema(),
    }
