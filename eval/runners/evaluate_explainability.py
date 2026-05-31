# eval/runners/evaluate_explainability.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from eval.core.explainability_metrics import (
    attribution_mask_iou,
    energy_inside_mask,
    grounded_accuracy,
    pointing_game,
)
from eval.runners.common import (
    ensure_dir,
    flatten_dict,
    normalize_class_name,
    safe_mean,
    save_csv,
    save_json,
)


def load_explainability_samples(
    model: str,
    predictions_dir: Path,
    gt_mask_root: Path,
    metadata_json: Path | None,
    model_name: str,
):
    model = normalize_class_name(model)

    if model == "lc_ksvd":
        from eval.adapters.lcksvd_adapter import LCKSVDExplainabilityAdapter

        adapter = LCKSVDExplainabilityAdapter(
            output_dir=predictions_dir,
            gt_mask_root=gt_mask_root,
            metadata_json=metadata_json,
            model_name=model_name,
        )
        return adapter.load()

    if model == "ct_clip_gradcam":
        from eval.adapters.ctclip_adapter import CTClipGradCAMExplainabilityAdapter

        adapter = CTClipGradCAMExplainabilityAdapter(
            output_dir=predictions_dir,
            gt_mask_root=gt_mask_root,
            metadata_json=metadata_json,
            model_name=model_name,
        )
        return adapter.load()

    if model == "merlin_gradcam":
        from eval.adapters.merlin_adapter import MerlinGradCAMExplainabilityAdapter

        adapter = MerlinGradCAMExplainabilityAdapter(
            output_dir=predictions_dir,
            gt_mask_root=gt_mask_root,
            metadata_json=metadata_json,
            model_name=model_name,
        )
        return adapter.load()

    raise ValueError(
        f"Unsupported explainability model: {model}. "
        "Supported: lc_ksvd, ct_clip_gradcam, merlin_gradcam"
    )


def compute_explainability_rows(
    samples: list,
    attribution_threshold: float,
) -> List[dict]:
    rows = []

    for sample in samples:
        attr_iou = attribution_mask_iou(
            attribution_map=sample.attribution_map,
            gt_mask=sample.gt_mask,
            threshold=attribution_threshold,
        )

        point_acc = pointing_game(
            attribution_map=sample.attribution_map,
            gt_mask=sample.gt_mask,
        )

        energy_ratio = energy_inside_mask(
            attribution_map=sample.attribution_map,
            gt_mask=sample.gt_mask,
        )

        correct = int(sample.y_true == sample.y_pred)

        rows.append({
            "case_id": sample.case_id,
            "model_name": sample.model_name,
            "class_name": sample.class_name,
            "y_true": int(sample.y_true),
            "y_pred": int(sample.y_pred),
            "y_score": float(sample.y_score),
            "correct": correct,
            "attribution_mask_iou": attr_iou,
            "pointing_game": point_acc,
            "energy_inside_mask": energy_ratio,
            "num_gt_voxels": int(np.asarray(sample.gt_mask).astype(bool).sum()),
            "atom_ids": getattr(sample, "atom_ids", None),
        })

    return rows


def summarize_rows(rows: List[dict]) -> Dict[str, object]:
    return {
        "num_samples": len(rows),
        "classification_accuracy": safe_mean([row["correct"] for row in rows]),
        "mean_attribution_mask_iou": safe_mean([row["attribution_mask_iou"] for row in rows]),
        "mean_pointing_game": safe_mean([row["pointing_game"] for row in rows]),
        "mean_energy_inside_mask": safe_mean([row["energy_inside_mask"] for row in rows]),
    }


def summarize_by_class(rows: List[dict]) -> Dict[str, dict]:
    grouped: Dict[str, List[dict]] = {}

    for row in rows:
        grouped.setdefault(row["class_name"], []).append(row)

    return {
        class_name: summarize_rows(class_rows)
        for class_name, class_rows in grouped.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate explainability / attribution metrics for Axis C."
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=["lc_ksvd", "ct_clip_gradcam", "merlin_gradcam"],
        help="Explainability adapter to use.",
    )

    parser.add_argument(
        "--predictions-dir",
        type=Path,
        required=True,
        help="Directory containing attribution maps / LC-KSVD atom maps.",
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
        default=Path("outputs/eval/explainability"),
        help="Directory to save explainability evaluation results.",
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
        "--attribution-threshold",
        type=float,
        default=0.5,
        help="Threshold for binarizing attribution maps.",
    )

    parser.add_argument(
        "--grounded-dice-threshold",
        type=float,
        default=0.10,
        help="Dice threshold for grounded accuracy.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_name = args.model_name or args.model
    output_dir = ensure_dir(args.output_dir)

    samples = load_explainability_samples(
        model=args.model,
        predictions_dir=args.predictions_dir,
        gt_mask_root=args.gt_mask_root,
        metadata_json=args.metadata_json,
        model_name=model_name,
    )

    if not samples:
        raise RuntimeError("No explainability samples were loaded by the adapter.")

    rows = compute_explainability_rows(
        samples=samples,
        attribution_threshold=args.attribution_threshold,
    )

    overall_summary = summarize_rows(rows)
    by_class = summarize_by_class(rows)

    ga = grounded_accuracy(
        samples=samples,
        dice_threshold=args.grounded_dice_threshold,
    )

    payload = {
        "task": "inherent_explainability",
        "axis": "Axis C",
        "model_name": model_name,
        "dataset": args.dataset,
        "num_samples": len(samples),
        "attribution_threshold": args.attribution_threshold,
        "grounded_dice_threshold": args.grounded_dice_threshold,
        "overall": {
            **overall_summary,
            "grounded_accuracy": ga,
        },
        "by_class": by_class,
        "per_case": rows,
    }

    save_json(payload, output_dir / f"{model_name}_explainability_metrics.json")
    save_csv(rows, output_dir / f"{model_name}_explainability_per_case.csv")

    summary_rows = [
        {
            "model_name": model_name,
            "dataset": args.dataset,
            "group_type": "overall",
            "group_name": "all",
            **flatten_dict(payload["overall"]),
        }
    ]

    for class_name, summary in by_class.items():
        summary_rows.append({
            "model_name": model_name,
            "dataset": args.dataset,
            "group_type": "class",
            "group_name": class_name,
            **flatten_dict(summary),
        })

    save_csv(summary_rows, output_dir / f"{model_name}_explainability_summary.csv")

    print(f"Saved explainability metrics to: {output_dir}")


if __name__ == "__main__":
    main()