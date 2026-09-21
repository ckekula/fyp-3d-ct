"""
leaderboard/validation.py
Pure format validator for Track A, Track B, and Track C submissions.
Validates schemas, data ranges, manifest properties, axis order, and spatial shapes.
"""

from __future__ import annotations

import io
import os
import csv
import json
import math
import struct
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

CANONICAL_CLASSES = ["lung_nodule", "lung_opacity", "consolidation", "atelectasis"]
VALID_SCORE_TYPES = {"probabilistic", "hard_label"}
VALID_AXIS_ORDERS = {"ZYX", "XYZ"}
VALID_MORPHOLOGIES = {"focal", "non_focal"}


@dataclass
class ValidationResult:
    is_valid: bool
    track: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    parsed_data: Optional[Dict[str, Any]] = None
    summary: Optional[Dict[str, Any]] = None

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


def _read_npy_header_and_shape(data_bytes: bytes) -> Tuple[Tuple[int, ...], str, bool]:
    """Parse .npy header from raw bytes in pure python."""
    if len(data_bytes) < 10 or data_bytes[:8] != b'\x93NUMPY\x01\x00':
        raise ValueError("Invalid .npy magic header")
    
    header_len = struct.unpack('<H', data_bytes[8:10])[0]
    header_str = data_bytes[10:10 + header_len].decode('latin1').strip()
    
    # Extract shape
    shape_start = header_str.find("'shape': (")
    if shape_start == -1:
        shape_start = header_str.find('"shape": (')
    if shape_start == -1:
        raise ValueError("Could not parse shape from .npy header")
    
    shape_end = header_str.find(')', shape_start)
    shape_tuple_str = header_str[shape_start + 9:shape_end + 1]
    
    # Safe eval tuple of ints
    shape_parts = [int(s.strip()) for s in shape_tuple_str.strip('()').split(',') if s.strip()]
    shape = tuple(shape_parts)
    
    # Extract descr
    descr_start = header_str.find("'descr': '")
    if descr_start == -1:
        descr_start = header_str.find('"descr": "')
    descr = header_str[descr_start + 10:descr_start + 13] if descr_start != -1 else "<f4"
    
    # Fortran order
    fortran = "fortran_order': True" in header_str or 'fortran_order": True' in header_str
    return shape, descr, fortran


def validate_track_a(
    input_file: Union[str, Path, bytes, io.BytesIO],
    known_cases: Optional[Set[str]] = None
) -> ValidationResult:
    """Validate a Track A (Classification-Only) submission."""
    result = ValidationResult(is_valid=True, track="Track A")
    
    try:
        # Read content
        if isinstance(input_file, (str, Path)):
            with open(input_file, "r", encoding="utf-8") as f:
                content = f.read()
        elif isinstance(input_file, bytes):
            content = input_file.decode("utf-8", errors="replace")
        elif isinstance(input_file, io.BytesIO):
            input_file.seek(0)
            content = input_file.read().decode("utf-8", errors="replace")
        else:
            result.add_error("Unsupported input type for Track A.")
            return result

        reader = csv.DictReader(io.StringIO(content))
        if not reader.fieldnames:
            result.add_error("CSV file is empty or missing headers.")
            return result

        # Check required columns
        fieldnames = [f.strip() for f in reader.fieldnames if f]
        
        # Determine case identifier column
        case_col = None
        for candidate in ["case_id", "volume_name", "scan_id", "name"]:
            if candidate in fieldnames:
                case_col = candidate
                break
        
        if not case_col:
            result.add_error(
                f"Missing required case identifier column. Expected 'case_id' or 'volume_name', found: {fieldnames}"
            )
            return result

        # Verify class columns
        missing_classes = [c for c in CANONICAL_CLASSES if c not in fieldnames]
        if missing_classes:
            result.add_error(f"Missing required pathology class columns: {missing_classes}")
            return result

        parsed_rows: List[Dict[str, Any]] = []
        seen_cases: Set[str] = set()

        for line_idx, row in enumerate(reader, start=2):
            raw_case = row.get(case_col, "").strip()
            if not raw_case:
                result.add_error(f"Line {line_idx}: empty case identifier.")
                continue

            if raw_case in seen_cases:
                result.add_error(f"Line {line_idx}: duplicate case_id '{raw_case}'.")
                continue
            seen_cases.add(raw_case)

            if known_cases is not None and raw_case not in known_cases:
                result.add_warning(f"Line {line_idx}: case_id '{raw_case}' not found in official test split.")

            score_type = row.get("score_type", "probabilistic").strip()
            if score_type not in VALID_SCORE_TYPES:
                result.add_error(
                    f"Line {line_idx} (case '{raw_case}'): invalid score_type '{score_type}'. "
                    f"Must be one of {list(VALID_SCORE_TYPES)}."
                )
                continue

            class_scores: Dict[str, float] = {}
            row_has_error = False

            for c in CANONICAL_CLASSES:
                raw_val = row.get(c, "").strip()
                if raw_val == "":
                    result.add_error(f"Line {line_idx} (case '{raw_case}'): missing value for '{c}'.")
                    row_has_error = True
                    break

                try:
                    val = float(raw_val)
                except ValueError:
                    result.add_error(f"Line {line_idx} (case '{raw_case}'): non-numeric value '{raw_val}' for '{c}'.")
                    row_has_error = True
                    break

                if math.isnan(val) or math.isinf(val):
                    result.add_error(f"Line {line_idx} (case '{raw_case}'): NaN/Inf score for '{c}'.")
                    row_has_error = True
                    break

                if val < 0.0 or val > 1.0:
                    result.add_error(f"Line {line_idx} (case '{raw_case}'): score {val} for '{c}' is outside [0.0, 1.0].")
                    row_has_error = True
                    break

                if score_type == "hard_label" and val not in (0.0, 1.0):
                    result.add_error(
                        f"Line {line_idx} (case '{raw_case}'): score_type is 'hard_label' but '{c}' has non-binary score {val}. "
                        f"Must be exactly 0.0 or 1.0."
                    )
                    row_has_error = True
                    break

                class_scores[c] = val

            if not row_has_error:
                parsed_rows.append({
                    "case_id": raw_case,
                    "scores": class_scores,
                    "score_type": score_type
                })

        if not parsed_rows and result.is_valid:
            result.add_error("No valid data rows found in CSV.")

        if result.is_valid:
            score_types_present = list({r["score_type"] for r in parsed_rows})
            result.parsed_data = {
                "track": "Track A",
                "rows": parsed_rows,
                "score_types": score_types_present
            }
            result.summary = {
                "num_cases": len(parsed_rows),
                "classes": CANONICAL_CLASSES,
                "score_types": score_types_present
            }

    except Exception as e:
        result.add_error(f"Failed to parse Track A file: {str(e)}")

    return result


def validate_track_b(
    input_file: Union[str, Path, bytes, io.BytesIO],
    known_cases: Optional[Set[str]] = None
) -> ValidationResult:
    """Validate a Track B (Localization-Only) submission zip package."""
    result = ValidationResult(is_valid=True, track="Track B")

    try:
        if isinstance(input_file, (str, Path)):
            zf = zipfile.ZipFile(input_file, "r")
        elif isinstance(input_file, (bytes, io.BytesIO)):
            file_bytes = input_file if isinstance(input_file, io.BytesIO) else io.BytesIO(input_file)
            zf = zipfile.ZipFile(file_bytes, "r")
        else:
            result.add_error("Unsupported input type for Track B.")
            return result

        with zf:
            file_names = zf.namelist()
            
            # 1. Validate manifest.json
            manifest_candidates = [f for f in file_names if f == "manifest.json" or f.endswith("/manifest.json")]
            if not manifest_candidates:
                result.add_error(
                    "Missing 'manifest.json' at the root of the ZIP archive. "
                    "A valid manifest declaring 'axis_order', 'spacing_mm', and 'is_soft_mask' is strictly required."
                )
                return result

            manifest_name = sorted(manifest_candidates, key=len)[0]
            try:
                manifest_data = json.loads(zf.read(manifest_name).decode("utf-8"))
            except Exception as e:
                result.add_error(f"Malformed 'manifest.json': {str(e)}")
                return result

            # Validate manifest fields
            if "axis_order" not in manifest_data:
                result.add_error("manifest.json missing required 'axis_order' field ('ZYX' or 'XYZ').")
            elif manifest_data["axis_order"] not in VALID_AXIS_ORDERS:
                result.add_error(
                    f"manifest.json invalid 'axis_order': '{manifest_data['axis_order']}'. "
                    f"Must be 'ZYX' or 'XYZ'."
                )

            if "spacing_mm" not in manifest_data:
                result.add_error("manifest.json missing required 'spacing_mm' 3-tuple.")
            else:
                spacing = manifest_data["spacing_mm"]
                if not isinstance(spacing, (list, tuple)) or len(spacing) != 3 or any(not isinstance(s, (int, float)) or s <= 0 for s in spacing):
                    result.add_error(
                        f"manifest.json 'spacing_mm' must be a list of 3 positive numbers, got: {spacing}"
                    )

            if "is_soft_mask" not in manifest_data:
                result.add_error("manifest.json missing required 'is_soft_mask' boolean flag.")
            elif not isinstance(manifest_data["is_soft_mask"], bool):
                result.add_error(f"manifest.json 'is_soft_mask' must be a boolean, got: {type(manifest_data['is_soft_mask']).__name__}")

            # Validate per-class configuration if present
            classes_config = manifest_data.get("classes", {})
            for c_name, c_cfg in classes_config.items():
                if c_name not in CANONICAL_CLASSES:
                    result.add_warning(f"manifest.json mentions non-canonical class '{c_name}'.")
                morph = c_cfg.get("morphology")
                if morph and morph not in VALID_MORPHOLOGIES:
                    result.add_error(f"Class '{c_name}': invalid morphology '{morph}'. Must be 'focal' or 'non_focal'.")

            # 2. Check mask files
            mask_files = [f for f in file_names if not f.endswith("/") and not f.endswith("manifest.json") and not f.startswith("__MACOSX")]
            if not mask_files:
                result.add_error("ZIP archive contains no mask files.")
                return result

            parsed_masks: Dict[str, Dict[str, str]] = {}
            for m_path in mask_files:
                # Check extension
                if not m_path.endswith(('.npy', '.npz', '.nii.gz', '.nii')):
                    result.add_warning(f"Skipping non-mask file: '{m_path}'")
                    continue

                base_name = Path(m_path).stem
                if base_name.endswith('.nii'):
                    base_name = Path(base_name).stem

                # Identify class and case_id
                matched_class = None
                for c in CANONICAL_CLASSES:
                    if f"_{c}" in base_name or f"/{c}/" in m_path or base_name.endswith(c):
                        matched_class = c
                        break

                if not matched_class:
                    result.add_warning(f"Could not deduce canonical class for mask: '{m_path}'.")
                    continue

                case_id = base_name.replace(f"_{matched_class}", "").replace(matched_class, "").strip("_-")
                if not case_id:
                    case_id = Path(m_path).parent.name

                if known_cases is not None and case_id not in known_cases:
                    result.add_warning(f"Mask '{m_path}' references unknown case_id '{case_id}'.")

                if case_id not in parsed_masks:
                    parsed_masks[case_id] = {}
                parsed_masks[case_id][matched_class] = m_path

                # For .npy, quickly inspect shape
                if m_path.endswith('.npy'):
                    try:
                        raw_bytes = zf.read(m_path)
                        shape, dtype, _ = _read_npy_header_and_shape(raw_bytes)
                        if len(shape) != 3:
                            result.add_error(f"Mask '{m_path}': expected 3D array, found shape {shape}.")
                    except Exception as e:
                        result.add_error(f"Failed to read .npy header for '{m_path}': {str(e)}")

            if not parsed_masks and result.is_valid:
                result.add_error("Could not parse any valid 3D class mask files from ZIP archive.")

            if result.is_valid:
                result.parsed_data = {
                    "track": "Track B",
                    "manifest": manifest_data,
                    "masks": parsed_masks,
                    "zip_file": zf
                }
                result.summary = {
                    "model_name": manifest_data.get("model_name", "Anonymous"),
                    "axis_order": manifest_data.get("axis_order"),
                    "spacing_mm": manifest_data.get("spacing_mm"),
                    "num_cases": len(parsed_masks),
                    "total_masks": sum(len(v) for v in parsed_masks.values())
                }

    except Exception as e:
        result.add_error(f"Failed to validate Track B zip: {str(e)}")

    return result


def validate_track_c(
    input_file: Union[str, Path, bytes, io.BytesIO],
    known_cases: Optional[Set[str]] = None
) -> ValidationResult:
    """Validate a Track C (Unified Classification + Localization) submission."""
    result = ValidationResult(is_valid=True, track="Track C")

    try:
        if isinstance(input_file, (str, Path)):
            zf = zipfile.ZipFile(input_file, "r")
        elif isinstance(input_file, (bytes, io.BytesIO)):
            file_bytes = input_file if isinstance(input_file, io.BytesIO) else io.BytesIO(input_file)
            zf = zipfile.ZipFile(file_bytes, "r")
        else:
            result.add_error("Unsupported input type for Track C.")
            return result

        with zf:
            file_names = zf.namelist()

            # 1. Locate and validate CSV
            csv_candidates = [f for f in file_names if f.endswith('.csv') and not f.startswith('__MACOSX')]
            if not csv_candidates:
                result.add_error("Track C submission missing classification predictions CSV table.")
                return result

            csv_name = sorted(csv_candidates, key=len)[0]
            csv_bytes = zf.read(csv_name)
            res_a = validate_track_a(csv_bytes, known_cases=known_cases)
            if not res_a.is_valid:
                for err in res_a.errors:
                    result.add_error(f"Classification CSV error: {err}")

            # 2. Validate Track B masks & manifest
            res_b = validate_track_b(input_file, known_cases=known_cases)
            if not res_b.is_valid:
                for err in res_b.errors:
                    result.add_error(f"Localization archive error: {err}")

            if result.is_valid:
                result.parsed_data = {
                    "track": "Track C",
                    "classification": res_a.parsed_data,
                    "localization": res_b.parsed_data,
                    "manifest": res_b.parsed_data.get("manifest") if res_b.parsed_data else {}
                }
                result.summary = {
                    "classification_cases": res_a.summary.get("num_cases") if res_a.summary else 0,
                    "localization_cases": res_b.summary.get("num_cases") if res_b.summary else 0,
                    "manifest": res_b.parsed_data.get("manifest") if res_b.parsed_data else {}
                }

    except Exception as e:
        result.add_error(f"Failed to validate Track C submission: {str(e)}")

    return result
