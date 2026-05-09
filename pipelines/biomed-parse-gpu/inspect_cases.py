from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

try:
    from .dataset_adapter import load_rexgroundingct_cases, iter_target_cases
    from .prompts import DEFAULT_DISEASES
except ImportError:
    from dataset_adapter import load_rexgroundingct_cases, iter_target_cases
    from prompts import DEFAULT_DISEASES


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect ReXGroundingCT matched target cases before running BiomedParse."
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=ROOT / "data" / "rexgrounding-ct" / "dataset.json",
    )
    parser.add_argument(
        "--volume-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--diseases",
        nargs="*",
        default=DEFAULT_DISEASES,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cases = load_rexgroundingct_cases(args.metadata_json, args.volume_root)
    target_cases = list(iter_target_cases(cases, diseases=args.diseases))

    counter = Counter()

    for case in target_cases:
        for disease in case.matched_diseases:
            counter[disease] += 1

    print(f"Total available volumes loaded: {len(cases)}")
    print(f"Target cases selected: {len(target_cases)}")
    print("\nMatched disease counts:")
    for disease, count in counter.items():
        print(f"  {disease}: {count}")

    print("\nSample matched cases:")
    for case in target_cases[: max(0, args.limit)]:
        print("=" * 80)
        print(f"Volume: {case.volume_name}")
        print(f"Path: {case.volume_path}")
        print(f"Matched: {case.matched_diseases}")
        print("Findings:")
        for finding in case.findings:
            print(f"  - {finding}")


if __name__ == "__main__":
    main()
