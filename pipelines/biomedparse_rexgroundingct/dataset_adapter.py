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

    with metadata_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    cases: List[RexCase] = []
    for entry in payload.get("train", []) + payload.get("valid", []):
        findings = _join_findings(entry.get("findings", {}))
        matched: List[str] = []
        for finding in findings:
            disease = normalize_disease_name(finding)
            if disease and disease not in matched:
                matched.append(disease)

        cases.append(
            RexCase(
                volume_name=entry["name"],
                volume_path=volume_root / entry["name"],
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