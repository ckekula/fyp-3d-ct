from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from prompts import TARGET_DISEASES, normalize_disease_name


@dataclass
class RexCase:
    volume_name: str
    volume_path: Path
    findings: List[str]
    matched_diseases: List[str]
    protocol: str | None = None


def _join_findings(findings: Dict[str, str]) -> List[str]:
    ordered_keys = sorted(findings.keys(), key=lambda value: int(value))
    return [findings[key] for key in ordered_keys]


def load_rexgroundingct_cases(metadata_json: str | Path, volume_root: str | Path) -> List[RexCase]:
    metadata_json = Path(metadata_json)
    volume_root = Path(volume_root)

    file_index: Dict[str, Path] = {}
    if volume_root.exists():
        # Build a lookup for nested volume layouts (e.g. data_volumes/*/*/*.nii.gz)
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
            disease = normalize_disease_name(finding)
            if disease and disease not in matched:
                matched.append(disease)

        volume_path = _resolve_volume_path(entry["name"])
        if not volume_path.exists():
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


def target_case_filter(case: RexCase, diseases: Optional[List[str]] = None) -> bool:
    selected = diseases if diseases else list(TARGET_DISEASES.keys())
    return any(disease in case.matched_diseases for disease in selected)


def iter_target_cases(cases: Iterable[RexCase], diseases: Optional[List[str]] = None) -> Iterable[RexCase]:
    for case in cases:
        if target_case_filter(case, diseases=diseases):
            yield case