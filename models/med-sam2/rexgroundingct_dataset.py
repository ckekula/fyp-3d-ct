"""
rexgroundingct_dataset.py

Loads the RexGroundingCT 163-case test subset whose findings are exclusively
lung-nodule (`2d`) or ground-glass opacity (`2c`) -- the same subset
models/lc-ksvd/src/lc_ksvd/config.py treats as canonical
(data/rexgrounding-ct/dataset_2_last.json, "test" split).

Used by notebooks/MedSAM2_RexGroundingCT_Inference.ipynb.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_METADATA_JSON = ROOT / "data" / "rexgrounding-ct" / "dataset_2_last.json"
DEFAULT_VOLUME_ROOT = ROOT / "data" / "data_volumes" / "dataset" / "train_fixed"
DEFAULT_MASK_ROOT = ROOT / "data" / "segmentations" / "segmentations"

CATEGORY_TO_CLASS = {
    "2d": "lung_nodule",
    "2c": "lung_opacity",
}


@dataclass
class Finding:
    index: int
    category: str
    class_name: str
    text: str


@dataclass
class RexCase:
    case_id: str
    volume_name: str
    volume_path: Path
    mask_path: Path
    findings: List[Finding]
    protocol: Optional[str] = None


def _stem(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return name


def load_cases(
    metadata_json: str | Path = DEFAULT_METADATA_JSON,
    volume_root: str | Path = DEFAULT_VOLUME_ROOT,
    mask_root: str | Path = DEFAULT_MASK_ROOT,
    split: str = "test",
    categories: tuple[str, ...] = ("2c", "2d"),
    require_pure: bool = True,
) -> List[RexCase]:
    """
    `require_pure=True` keeps a case only if every finding on it falls in
    `categories` (no mixed consolidation/atelectasis cases) -- this is what
    produces the 163-case subset.
    """
    metadata_json = Path(metadata_json)
    volume_root = Path(volume_root)
    mask_root = Path(mask_root)
    allowed = set(categories)

    payload = json.loads(metadata_json.read_text(encoding="utf-8"))
    entries = payload.get(split, [])

    cases: List[RexCase] = []
    for entry in entries:
        raw_categories: Dict[str, str] = entry.get("categories", {})
        raw_findings: Dict[str, str] = entry.get("findings", {})

        if not raw_categories:
            continue

        cat_values = set(raw_categories.values())
        if require_pure and not cat_values <= allowed:
            continue
        if not require_pure and not (cat_values & allowed):
            continue

        findings: List[Finding] = []
        for key, category in raw_categories.items():
            if category not in allowed:
                continue
            findings.append(
                Finding(
                    index=int(key),
                    category=category,
                    class_name=CATEGORY_TO_CLASS[category],
                    text=raw_findings.get(key, ""),
                )
            )
        if not findings:
            continue

        volume_name = entry["name"]
        case_id = _stem(volume_name)

        volume_path = volume_root / volume_name
        mask_path = mask_root / volume_name
        if not volume_path.exists() or not mask_path.exists():
            continue

        cases.append(
            RexCase(
                case_id=case_id,
                volume_name=volume_name,
                volume_path=volume_path,
                mask_path=mask_path,
                findings=sorted(findings, key=lambda f: f.index),
                protocol=entry.get("protocol"),
            )
        )

    return cases
