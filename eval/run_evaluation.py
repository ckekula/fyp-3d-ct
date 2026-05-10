#!/usr/bin/env python
"""
Run all evaluation tasks (classification and localization).

Usage:
    python run_evaluation.py
"""

import subprocess
import sys
from pathlib import Path

def run_classification():
    """Run classification evaluation."""
    cmd = [
        sys.executable,
        "-m",
        "eval.runners.evaluate_classification",
        "--model", "ct_clip",
        "--predictions", "outputs/ctclip_rex_eval/predictions.csv",
        "--output-dir", "eval/outputs/classification",
        "--dataset", "ctrate",
        "--model-name", "ct_clip_zero_shot",
    ]
    
    print("=" * 80)
    print("Running Classification Evaluation...")
    print("=" * 80)
    result = subprocess.run(
        cmd,
        cwd=Path(__file__).parent.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    if result.stdout:
        print(result.stdout)
    return result.returncode == 0

def run_localization():
    """Run localization evaluation."""
    cmd = [
        sys.executable,
        "-m",
        "eval.runners.evaluate_localization",
        "--model", "biomed_parse",
        "--predictions-dir", "outputs/biomedparse",
        "--gt-mask-root", "data/segmentations/segmentations",
        "--metadata-json", "data/rexgrounding-ct/dataset_4.json",
        "--output-dir", "eval/outputs/localization",
        "--dataset", "rexgroundingct",
        "--model-name", "biomed_parse",
    ]
    
    print("=" * 80)
    print("Running Localization Evaluation...")
    print("=" * 80)
    result = subprocess.run(
        cmd,
        cwd=Path(__file__).parent.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    if result.stdout:
        print(result.stdout)
    return result.returncode == 0

def main():
    success = True
    
    success = run_classification() and success
    success = run_localization() and success
    
    if success:
        print("\n✅ All evaluations completed successfully!")
        return 0
    else:
        print("\n❌ Some evaluations failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
