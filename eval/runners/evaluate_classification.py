# eval/runners/evaluate_classification.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from eval.core.classification_metrics import compute_classification_metrics
from eval.runners.common import (
    DEFAULT_CLASSES,
    ensure_dir,
    flatten_dict,
    normalize_class_name,
    parse_classes,
    save_csv,
    save_json,
)


def load_classification_samples(
    model: str,
    predictions_path: Path,
    model_name: str,
):
    model = normalize_class_name(model)

    if model == "ct_clip":
        from eval.adapters.ctclip_adapter import CTClipClassificationAdapter

        adapter = CTClipClassificationAdapter(
            predictions_csv=predictions_path,
            model_name=model_name,
        )
        return adapter.load()

    if model == "merlin":
        from eval.adapters.merlin_adapter import MerlinClassificationAdapter

        adapter = MerlinClassificationAdapter(
            predictions_path=predictions_path,
            model_name=model_name,
        )
        return adapter.load()

    if model == "biomed_parse":
        from eval.adapters.biomedparse_adapter import BiomedParseClassificationAdapter

        adapter = BiomedParseClassificationAdapter(
            predictions_path=predictions_path,
            model_name=model_name,
        )
        return adapter.load()

    if model == "lc_ksvd":
        from eval.adapters.lcksvd_adapter import LCKSVDClassificationAdapter

        adapter = LCKSVDClassificationAdapter(
            predictions_path=predictions_path,
            model_name=model_name,
        )
        return adapter.load()

    raise ValueError(
        f"Unsupported classification model: {model}. "
        "Supported: ct_clip, merlin, biomed_parse, lc_ksvd"
    )


def build_per_class_rows(results: dict, model_name: str, dataset: str) -> List[dict]:
    rows = []

    for class_name, metrics in results.items():
        if not isinstance(metrics, dict):
            continue

        row = {
            "model_name": model_name,
            "dataset": dataset,
            "class_name": class_name,
        }
        row.update(flatten_dict(metrics))
        rows.append(row)

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate classification predictions for Axis A."
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=["ct_clip", "merlin", "biomed_parse", "lc_ksvd"],
        help="Model adapter to use.",
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to model prediction file, for example predictions.csv.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eval/classification"),
        help="Directory to save classification evaluation results.",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Display name for the model in reports.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="unknown",
        help="Dataset name, for example ctrate, rexgroundingct, omniabnorm.",
    )

    parser.add_argument(
        "--classes",
        nargs="*",
        default=DEFAULT_CLASSES,
        help="Target abnormality classes.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for F1, precision, recall, and accuracy.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_name = args.model_name or args.model
    class_names = parse_classes(args.classes)

    output_dir = ensure_dir(args.output_dir)

    samples = load_classification_samples(
        model=args.model,
        predictions_path=args.predictions,
        model_name=model_name,
    )

    if not samples:
        raise RuntimeError("No classification samples were loaded by the adapter.")

    results = compute_classification_metrics(
        samples=samples,
        class_names=class_names,
        threshold=args.threshold,
    )

    payload = {
        "task": "classification_accuracy",
        "axis": "Axis A",
        "model_name": model_name,
        "dataset": args.dataset,
        "num_samples": len(samples),
        "classes": class_names,
        "threshold": args.threshold,
        "results": results,
    }

    save_json(payload, output_dir / f"{model_name}_classification_metrics.json")

    rows = build_per_class_rows(
        results=results,
        model_name=model_name,
        dataset=args.dataset,
    )
    save_csv(rows, output_dir / f"{model_name}_classification_metrics.csv")

    print(f"Saved classification metrics to: {output_dir}")


if __name__ == "__main__":
    main()