# eval/runners/evaluate_localization.py

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import numpy as np

from eval.core.localization_metrics import compute_localization_metrics
from eval.runners.common import (
    ensure_dir,
    flatten_dict,
    normalize_class_name,
    save_csv,
    save_json,
)


def normalize_array(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)

    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))

    if x_max - x_min <= eps:
        return np.zeros_like(x, dtype=np.float32)

    return (x - x_min) / (x_max - x_min)


def parse_thresholds(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def load_localization_samples(
    model: str,
    predictions_dir: Path,
    gt_mask_root: Path,
    metadata_json: Path | None,
    model_name: str,
):
    model = normalize_class_name(model)

    if model == "biomed_parse":
        from eval.adapters.biomedparse_adapter import BiomedParseLocalizationAdapter

        adapter = BiomedParseLocalizationAdapter(
            output_dir=predictions_dir,
            gt_mask_root=gt_mask_root,
            metadata_json=metadata_json,
            model_name=model_name,
        )
        return adapter.load()

    if model == "lc_ksvd":
        from eval.adapters.lcksvd_adapter import LCKSVDLocalizationAdapter

        adapter = LCKSVDLocalizationAdapter(
            output_dir=predictions_dir,
            gt_mask_root=gt_mask_root,
            metadata_json=metadata_json,
            model_name=model_name,
        )
        return adapter.load()

    if model == "merlin":
        from eval.adapters.merlin_adapter import MerlinLocalizationAdapter

        adapter = MerlinLocalizationAdapter(
            output_dir=predictions_dir,
            gt_mask_root=gt_mask_root,
            metadata_json=metadata_json,
            model_name=model_name,
        )
        return adapter.load()

    if model == "nnunet":
        from eval.adapters.nnunet_adapter import NNUNetLocalizationAdapter  # type: ignore

        adapter = NNUNetLocalizationAdapter(
            output_dir=predictions_dir,
            gt_mask_root=gt_mask_root,
            metadata_json=metadata_json,
            model_name=model_name,
        )
        return adapter.load()

    raise ValueError(
        f"Unsupported localization model: {model}. "
        "Supported now: biomed_parse, lc_ksvd. "
        "Add more adapters for medsam2, segvol, nnunet, swin_unetr."
    )


def compute_grouped_metrics(samples: list, group_field: str) -> Dict[str, dict]:
    grouped: Dict[str, list] = {}

    for sample in samples:
        value = getattr(sample, group_field, None)
        if value is None:
            value = "unknown"

        grouped.setdefault(str(value), []).append(sample)

    results = {}

    for group_name, group_samples in grouped.items():
        results[group_name] = compute_localization_metrics(group_samples)

    return results


def compute_threshold_sweep(
    samples: list,
    thresholds: List[float],
    normalize_masks: bool,
) -> Dict[str, dict]:
    sweep_results = {}

    for threshold in thresholds:
        thresholded_samples = []

        for sample in samples:
            pred = np.asarray(sample.pred_mask)

            if normalize_masks:
                pred = normalize_array(pred)

            pred_binary = (pred >= threshold).astype(np.uint8)

            thresholded_samples.append(
                replace(sample, pred_mask=pred_binary)
            )

        sweep_results[str(threshold)] = compute_localization_metrics(thresholded_samples)

    return sweep_results


def build_per_case_rows(metrics_payload: dict) -> List[dict]:
    rows = []

    per_case = metrics_payload.get("per_case", [])

    for item in per_case:
        rows.append(flatten_dict(item))

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate grounded localization predictions for Axis B."
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=["biomed_parse", "lc_ksvd", "merlin", "medsam2", "segvol", "nnunet", "swin_unetr"],
        help="Model adapter to use.",
    )

    parser.add_argument(
        "--predictions-dir",
        type=Path,
        required=True,
        help="Directory containing model localization outputs.",
    )

    parser.add_argument(
        "--gt-mask-root",
        type=Path,
        required=True,
        help="Directory containing ground-truth masks.",
    )

    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=None,
        help="Optional dataset metadata JSON.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eval/localization"),
        help="Directory to save localization evaluation results.",
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
        default="rexgroundingct",
        help="Dataset name.",
    )

    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.5",
        help="Comma-separated thresholds for mask binarization. Example: 0.1,0.2,0.3,0.4,0.5",
    )

    parser.add_argument(
        "--normalize-masks",
        action="store_true",
        help="Normalize soft masks to [0, 1] before thresholding.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_name = args.model_name or args.model
    output_dir = ensure_dir(args.output_dir)
    thresholds = parse_thresholds(args.thresholds)

    samples = load_localization_samples(
        model=args.model,
        predictions_dir=args.predictions_dir,
        gt_mask_root=args.gt_mask_root,
        metadata_json=args.metadata_json,
        model_name=model_name,
    )

    if not samples:
        raise RuntimeError("No localization samples were loaded by the adapter.")

    # Main result uses the first threshold supplied.
    primary_threshold = thresholds[0]

    primary_samples = []
    for sample in samples:
        pred = np.asarray(sample.pred_mask)

        if args.normalize_masks:
            pred = normalize_array(pred)

        pred_binary = (pred >= primary_threshold).astype(np.uint8)
        primary_samples.append(replace(sample, pred_mask=pred_binary))

    overall_metrics = compute_localization_metrics(primary_samples)

    by_class = compute_grouped_metrics(primary_samples, "class_name")
    by_morphology = compute_grouped_metrics(primary_samples, "morphology")

    threshold_sweep = compute_threshold_sweep(
        samples=samples,
        thresholds=thresholds,
        normalize_masks=args.normalize_masks,
    )

    payload = {
        "task": "grounded_localization",
        "axis": "Axis B",
        "model_name": model_name,
        "dataset": args.dataset,
        "num_samples": len(samples),
        "primary_threshold": primary_threshold,
        "thresholds": thresholds,
        "normalize_masks": args.normalize_masks,
        "overall": overall_metrics,
        "by_class": by_class,
        "by_morphology": by_morphology,
        "threshold_sweep": threshold_sweep,
    }

    save_json(payload, output_dir / f"{model_name}_localization_metrics.json")

    per_case_rows = build_per_case_rows(overall_metrics)
    save_csv(per_case_rows, output_dir / f"{model_name}_localization_per_case.csv")

    summary_rows = []
    summary_rows.append({
        "model_name": model_name,
        "dataset": args.dataset,
        "group_type": "overall",
        "group_name": "all",
        **flatten_dict(overall_metrics.get("summary", {})),
    })

    for class_name, result in by_class.items():
        summary_rows.append({
            "model_name": model_name,
            "dataset": args.dataset,
            "group_type": "class",
            "group_name": class_name,
            **flatten_dict(result.get("summary", {})),
        })

    for morphology, result in by_morphology.items():
        summary_rows.append({
            "model_name": model_name,
            "dataset": args.dataset,
            "group_type": "morphology",
            "group_name": morphology,
            **flatten_dict(result.get("summary", {})),
        })

    save_csv(summary_rows, output_dir / f"{model_name}_localization_summary.csv")

    print(f"Saved localization metrics to: {output_dir}")


if __name__ == "__main__":
    main()