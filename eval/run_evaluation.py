#!/usr/bin/env python
"""
Run all evaluation tasks (classification, localization, explainability)
across every model this framework has an adapter for.

Each entry below is skipped (not failed) if its --predictions/--predictions-dir
path doesn't exist locally -- this repo only carries small smoke-test
prediction artifacts (a handful of cases) for most models; the full
RexGroundingCT-scale runs that produced the checked-in eval/outputs/*.json
files were generated on the training machine and are not reproducible from
this checkout.

IMPORTANT: results are written to eval/outputs/smoke_test/, NOT
eval/outputs/classification|localization/ -- the local prediction paths
below are tiny non-representative smoke-test subsets (e.g. 1-50 cases vs.
the real 800-3000 case runs), so this must never silently overwrite the
curated full-dataset results that live in eval/outputs/. Point the paths
below at your own full prediction output, and --output-dir at
eval/outputs/classification|localization, to regenerate those for real.

Usage:
    python run_evaluation.py
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SMOKE_TEST_OUTPUT_DIR = "eval/outputs/smoke_test"

CLASSIFICATION_RUNS = [
    {
        "model": "ct_clip",
        "predictions": "outputs/ctclip_rex_eval/predictions.csv",
        "model_name": "ct_clip_lipro",
        "dataset": "rexgroundingct",
    },
    {
        "model": "merlin",
        "predictions": "models/merlin/results_full",
        "model_name": "merlin",
        "dataset": "rexgroundingct",
    },
    {
        "model": "biomed_parse",
        "predictions": "outputs/biomedparse_rexgroundingct_fixed_cuda",
        "model_name": "biomed_parse",
        "dataset": "rexgroundingct",
    },
    {
        "model": "lc_ksvd",
        "predictions": "models/lc-ksvd/src/lc_ksvd/outputs/results/lcksvd2_svm_test_20260803_103648",
        "model_name": "lc_ksvd",
        "dataset": "rexgroundingct",
    },
]

LOCALIZATION_RUNS = [
    {
        "model": "biomed_parse",
        "predictions_dir": "outputs/biomedparse_rexgroundingct_fixed_cuda",
        "gt_mask_root": "data/segmentations/segmentations",
        "metadata_json": "data/rexgrounding-ct/dataset_anthima.json",
        "model_name": "biomed_parse",
    },
    {
        "model": "merlin",
        "predictions_dir": "models/merlin/results_full",
        "gt_mask_root": "data/segmentations/segmentations",
        "metadata_json": "data/rexgrounding-ct/dataset_anthima.json",
        "model_name": "merlin",
    },
    {
        # No predictions have been generated for MedSAM2 yet -- see
        # models/med-sam2/notebooks/MedSAM2_RexGroundingCT_Inference.ipynb.
        # Path is a placeholder; this run is skipped until that exists.
        "model": "medsam2",
        "predictions_dir": "outputs/medsam2_rexgroundingct",
        "gt_mask_root": "data/segmentations/segmentations",
        "metadata_json": "data/rexgrounding-ct/dataset_anthima.json",
        "model_name": "medsam2",
    },
    {
        # Requires models/lc-ksvd/src/lc_ksvd/inference/segmentation.py
        # ::evaluate_split_segmentation(save_masks=True) to have been run on
        # a machine with the full CT volumes + trained dictionary/SVM models
        # (see LCKSVDLocalizationAdapter's docstring). Path is a placeholder.
        "model": "lc_ksvd",
        "predictions_dir": "models/lc-ksvd/src/lc_ksvd/outputs/results/segmentation_test",
        "gt_mask_root": "data/segmentations/segmentations",
        "metadata_json": "data/rexgrounding-ct/dataset_anthima.json",
        "model_name": "lc_ksvd",
    },
]


def _run(cmd: list[str], label: str) -> bool:
    print("=" * 80)
    print(f"Running {label}...")
    print("=" * 80)
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.stdout:
        print(result.stdout)
    return result.returncode == 0


def run_classification(cfg: dict) -> bool | None:
    predictions_path = REPO_ROOT / cfg["predictions"]
    if not predictions_path.exists():
        print(f"[SKIP] classification/{cfg['model']}: {cfg['predictions']} not found locally.")
        return None

    cmd = [
        sys.executable, "-m", "eval.runners.evaluate_classification",
        "--model", cfg["model"],
        "--predictions", cfg["predictions"],
        "--output-dir", SMOKE_TEST_OUTPUT_DIR,
        "--dataset", cfg["dataset"],
        "--model-name", cfg["model_name"],
    ]
    return _run(cmd, f"Classification Evaluation ({cfg['model_name']})")


def run_localization(cfg: dict) -> bool | None:
    predictions_dir = REPO_ROOT / cfg["predictions_dir"]
    if not predictions_dir.exists():
        print(f"[SKIP] localization/{cfg['model']}: {cfg['predictions_dir']} not found locally.")
        return None

    cmd = [
        sys.executable, "-m", "eval.runners.evaluate_localization",
        "--model", cfg["model"],
        "--predictions-dir", cfg["predictions_dir"],
        "--gt-mask-root", cfg["gt_mask_root"],
        "--output-dir", SMOKE_TEST_OUTPUT_DIR,
        "--dataset", "rexgroundingct",
        "--model-name", cfg["model_name"],
    ]
    if cfg.get("metadata_json"):
        cmd += ["--metadata-json", cfg["metadata_json"]]
    return _run(cmd, f"Localization Evaluation ({cfg['model_name']})")


def main() -> int:
    results = {}

    for cfg in CLASSIFICATION_RUNS:
        results[f"classification/{cfg['model']}"] = run_classification(cfg)

    for cfg in LOCALIZATION_RUNS:
        results[f"localization/{cfg['model']}"] = run_localization(cfg)

    ran = {k: v for k, v in results.items() if v is not None}
    skipped = [k for k, v in results.items() if v is None]
    failed = [k for k, v in ran.items() if not v]

    print("\n" + "=" * 80)
    print(f"Ran: {len(ran)}  Skipped (no local data): {len(skipped)}  Failed: {len(failed)}")
    if skipped:
        print(f"Skipped: {', '.join(skipped)}")
    if failed:
        print(f"Failed:  {', '.join(failed)}")

    if failed:
        print("\n[FAIL] Some evaluations failed. Check the output above.")
        return 1

    print("\n[OK] All available evaluations completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
