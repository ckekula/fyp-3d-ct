# eval/runners/common.py

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_CLASSES = [
    "lung_nodule",
    "lung_opacity",
    "consolidation",
    "atelectasis",
]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def flatten_dict(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}

    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)

        if isinstance(value, dict):
            flat.update(flatten_dict(value, full_key))
        else:
            flat[full_key] = value

    return flat


def save_csv(rows: List[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_classes(classes: Iterable[str] | None) -> List[str]:
    if not classes:
        return DEFAULT_CLASSES

    return [normalize_class_name(item) for item in classes]


def normalize_class_name(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def safe_mean(values: List[float | None]) -> float | None:
    clean = [v for v in values if v is not None]

    if not clean:
        return None

    return float(sum(clean) / len(clean))