from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .prompts import DEFAULT_DISEASES


DISEASE_KEYWORDS: Dict[str, List[str]] = {
    "Lung nodule": [
        "lung nodule",
        "lung nodules",
        "pulmonary nodule",
        "pulmonary nodules",
        "subcentimeter pulmonary nodule",
        "subcentimeter pulmonary nodules",
        "small lung nodule",
        "small pulmonary nodule",
        "nodule",
        "nodules",
        "nodular lesion",
        "nodular lesions",
        "nonspecific pulmonary nodule",
        "nonspecific pulmonary nodules",
        "micronodule",
        "micronodules",
        "reticulonodular density",
        "reticulonodular density increase",
    ],
    "Lung opacity": [
        "lung opacity",
        "lung opacities",
        "pulmonary opacity",
        "pulmonary opacities",
        "opacity",
        "opacities",
        "ground glass opacity",
        "ground glass opacities",
        "ground-glass opacity",
        "ground-glass opacities",
        "ggo",
        "mosaic attenuation",
        "parenchymal opacity",
        "parenchymal opacities",
        "density increase",
        "density increases",
    ],
    "Consolidation": [
        "consolidation",
        "consolidations",
        "pulmonary consolidation",
        "lobar consolidation",
        "segmental consolidation",
        "airspace consolidation",
        "consolidation area",
        "consolidation areas",
        "consolidative opacity",
        "consolidative opacities",
        "nodular consolidation",
    ],
    "Atelectasis": [
        "atelectasis",
        "linear atelectasis",
        "segmental atelectasis",
        "subsegmental atelectasis",
        "atelectatic change",
        "atelectatic changes",
        "fibroatelectatic change",
        "fibroatelectatic changes",
        "fibroatelectatic",
        "lung collapse",
    ],
}

@dataclass
class RexCase:
    volume_name: str
    volume_path: Path
    findings: List[str]
    matched_diseases: List[str]
    protocol: str | None = None


def normalize_text(text: str) -> str:
    """
    Normalize text for keyword matching.

    This converts:
    - uppercase to lowercase
    - hyphenated words to spaced words
    - punctuation to spaces
    - repeated spaces to a single space
    """
    text = str(text).lower()
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_disease_name(name: str) -> str:
    """Map common disease aliases to your canonical project labels."""
    normalized = normalize_text(name)

    aliases = {
        "lung nodule": "Lung nodule",
        "lung nodules": "Lung nodule",
        "pulmonary nodule": "Lung nodule",
        "pulmonary nodules": "Lung nodule",
        "nodule": "Lung nodule",
        "nodules": "Lung nodule",
        "nodular lesion": "Lung nodule",
        "nodular lesions": "Lung nodule",

        "lung opacity": "Lung opacity",
        "pulmonary opacity": "Lung opacity",
        "pulmonary opacities": "Lung opacity",
        "opacity": "Lung opacity",
        "opacities": "Lung opacity",
        "ground glass opacity": "Lung opacity",
        "ground glass opacities": "Lung opacity",
        "ground-glass opacity": "Lung opacity",
        "ground-glass opacities": "Lung opacity",
        "ggo": "Lung opacity",

        "consolidation": "Consolidation",
        "consolidations": "Consolidation",
        "pulmonary consolidation": "Consolidation",
        "lobar consolidation": "Consolidation",
        "segmental consolidation": "Consolidation",

        "atelectasis": "Atelectasis",
        "atelectatic": "Atelectasis",
        "linear atelectasis": "Atelectasis",
        "segmental atelectasis": "Atelectasis",
        "subsegmental atelectasis": "Atelectasis",
        "fibroatelectatic": "Atelectasis",
    }

    return aliases.get(normalized, name)


def match_diseases_from_text(text: str) -> List[str]:
    """
    Return all target diseases found in a finding sentence.

    A single finding can mention multiple concepts.
    Example:
        "Nodular lesions with ground glass opacity"
    should match:
        ["Lung nodule", "Lung opacity"]
    """
    normalized = normalize_text(text)
    matched: List[str] = []

    for disease, keywords in DISEASE_KEYWORDS.items():
        for keyword in keywords:
            keyword_norm = normalize_text(keyword)
            if keyword_norm and keyword_norm in normalized:
                matched.append(disease)
                break

    return matched


def _join_findings(findings: Dict[str, str]) -> List[str]:
    ordered_keys = sorted(findings.keys(), key=lambda value: int(value))
    return [findings[key] for key in ordered_keys]


def load_rexgroundingct_cases(
    metadata_json: str | Path,
    volume_root: str | Path,
) -> List[RexCase]:
    """
    Load ReXGroundingCT cases from dataset.json and resolve matching NIfTI volumes.

    The function supports nested volume folders because the dataset may be stored as:
        data_volumes/*/*/*.nii.gz
    """
    metadata_json = Path(metadata_json)
    volume_root = Path(volume_root)

    file_index: Dict[str, Path] = {}

    if volume_root.exists():
        file_index = {path.name: path for path in volume_root.rglob("*.nii.gz")}

    with metadata_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    entries = (
        payload.get("train", [])
        + payload.get("val", [])
        + payload.get("valid", [])
        + payload.get("test", [])
    )

    def _resolve_volume_path(name: str) -> Path:
        candidates = [name]

        if not name.endswith(".nii.gz"):
            candidates.append(f"{name}.nii.gz")

        if name.endswith(".nii.gz"):
            candidates.append(name[: -len(".nii.gz")])

        for candidate in candidates:
            if candidate in file_index:
                return file_index[candidate]

        return volume_root / name

    cases: List[RexCase] = []

    for entry in entries:
        findings = _join_findings(entry.get("findings", {}))

        matched: List[str] = []

        for finding in findings:
            diseases = match_diseases_from_text(finding)

            for disease in diseases:
                if disease not in matched:
                    matched.append(disease)

        volume_path = _resolve_volume_path(entry["name"])

        if not volume_path.exists():
            # Skip metadata entries whose volume file is not available locally.
            continue

        cases.append(
            RexCase(
                volume_name=entry["name"],
                volume_path=volume_path,
                findings=findings,
                matched_diseases=matched,
                protocol=entry.get("protocol"),
            )
        )

    return cases


def target_case_filter(
    case: RexCase,
    diseases: Optional[List[str]] = None,
) -> bool:
    selected = diseases if diseases else list(DEFAULT_DISEASES)
    selected = [normalize_disease_name(disease) for disease in selected]

    return any(disease in case.matched_diseases for disease in selected)


def iter_target_cases(
    cases: Iterable[RexCase],
    diseases: Optional[List[str]] = None,
) -> Iterable[RexCase]:
    for case in cases:
        if target_case_filter(case, diseases=diseases):
            yield case
