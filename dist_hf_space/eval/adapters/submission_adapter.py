"""
eval/adapters/submission_adapter.py
Generic submission adapter converting validated user uploads (Tracks A, B, C)
into canonical ClassificationSample and LocalizationSample dataclass instances.
Enforces strict spatial axis order handling (XYZ -> ZYX) via eval.adapters._axis_utils.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from eval.core.schemas import ClassificationSample, LocalizationSample
from eval.adapters._axis_utils import xyz_to_zyx

CANONICAL_CLASSES = ["lung_nodule", "lung_opacity", "consolidation", "atelectasis"]


def load_submission_classification_samples(
    parsed_data: Dict[str, Any],
    gt_dict: Dict[str, Dict[str, int]],
    model_name: str = "Submitted Model",
    dataset: str = "benchmark_test"
) -> List[ClassificationSample]:
    """
    Convert validated Track A parsed rows into ClassificationSample objects.
    Matches cases against ground truth dictionary.
    """
    rows = parsed_data.get("rows", [])
    samples: List[ClassificationSample] = []

    for row in rows:
        case_id = str(row["case_id"]).strip()

        # Match with GT key
        matched_gt = None
        if case_id in gt_dict:
            matched_gt = gt_dict[case_id]
        elif f"{case_id}.nii.gz" in gt_dict:
            matched_gt = gt_dict[f"{case_id}.nii.gz"]
        else:
            for k, v in gt_dict.items():
                if case_id == k or case_id in k or k in case_id:
                    matched_gt = v
                    break

        if matched_gt is None:
            # Skip cases without ground truth
            continue

        y_true = {c: int(matched_gt.get(c, 0)) for c in CANONICAL_CLASSES}
        y_score = {c: float(row["scores"].get(c, 0.0)) for c in CANONICAL_CLASSES}
        score_type = row.get("score_type", "probabilistic")

        samples.append(
            ClassificationSample(
                case_id=case_id,
                model_name=model_name,
                y_true=y_true,
                y_score=y_score,
                dataset=dataset,
                score_type=score_type,
                metadata={"raw_row": row}
            )
        )

    return samples


def _load_mask_array_from_zip(
    zf: zipfile.ZipFile,
    mask_path: str
) -> np.ndarray:
    """Load 3D array from ZIP (.npy or .npz)."""
    raw_bytes = zf.read(mask_path)
    bio = io.BytesIO(raw_bytes)
    
    if mask_path.endswith('.npy'):
        arr = np.load(bio)
    elif mask_path.endswith('.npz'):
        npz = np.load(bio)
        keys = list(npz.keys())
        arr = npz[keys[0]]
    else:
        # Generic numpy loader
        arr = np.load(bio)
        
    return np.asarray(arr)


def load_submission_localization_samples(
    parsed_data: Dict[str, Any],
    gt_masks_dict: Dict[str, Dict[str, np.ndarray]],
    model_name: Optional[str] = None,
    dataset: str = "benchmark_test"
) -> List[LocalizationSample]:
    """
    Convert validated Track B parsed masks into LocalizationSample objects.
    Enforces axis order transformation to ZYX and attaches physical voxel spacing.
    """
    manifest = parsed_data.get("manifest", {})
    masks_index = parsed_data.get("masks", {})
    zf = parsed_data.get("zip_file")

    declared_model_name = model_name or manifest.get("model_name", "Submitted Model")
    declared_axis_order = manifest.get("axis_order", "ZYX").upper()
    declared_spacing = tuple(float(s) for s in manifest.get("spacing_mm", [1.0, 1.0, 1.0]))
    global_is_soft = bool(manifest.get("is_soft_mask", True))
    classes_config = manifest.get("classes", {})

    samples: List[LocalizationSample] = []

    for case_id, class_map in masks_index.items():
        # Match case in GT
        matched_gt_case = None
        if case_id in gt_masks_dict:
            matched_gt_case = gt_masks_dict[case_id]
        else:
            for k in gt_masks_dict:
                if case_id == k or case_id in k or k in case_id:
                    matched_gt_case = gt_masks_dict[k]
                    break

        if matched_gt_case is None:
            continue

        for class_name, mask_rel_path in class_map.items():
            if class_name not in matched_gt_case:
                continue

            gt_mask = np.asarray(matched_gt_case[class_name])
            
            # Load predicted mask from zip
            pred_mask = _load_mask_array_from_zip(zf, mask_rel_path)

            # Spatial axis alignment: NEVER assume array is ZYX if manifest states XYZ
            current_spacing = declared_spacing
            if declared_axis_order == "XYZ":
                pred_mask, current_spacing = xyz_to_zyx(pred_mask, declared_spacing)

            # Read class specific properties from manifest
            class_cfg = classes_config.get(class_name, {})
            morphology = class_cfg.get("morphology", "focal" if class_name == "lung_nodule" else "non_focal")
            is_soft_mask = class_cfg.get("is_soft_mask", global_is_soft)

            samples.append(
                LocalizationSample(
                    case_id=case_id,
                    model_name=declared_model_name,
                    class_name=class_name,
                    pred_mask=pred_mask,
                    gt_mask=gt_mask,
                    spacing=current_spacing,
                    morphology=morphology,
                    dataset=dataset,
                    is_soft_mask=is_soft_mask
                )
            )

    return samples
