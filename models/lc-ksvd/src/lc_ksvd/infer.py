"""
infer.py
CLI entry point for LC-KSVD2 inference and back-projection.

Runs dense sliding-window inference on one or more CT volumes and saves:
  outputs/inference/<volume_name>/<volume_name>_class_<cls>_mask.nii.gz
  outputs/inference/<volume_name>/<volume_name>_class_<cls>_contrib.nii.gz  (optional)

Usage:
  # Single volume
  python infer.py --volume train_1_a_1

  # All volumes in a split (reads IDs from the dataset metadata)
  python infer.py --split val

  # Custom model / output location
  python infer.py --volume train_1_a_1 \\
                  --model outputs/models/unified_lcksvd2.pkl \\
                  --out-dir outputs/inference/custom

  # Faster (no overlap) / finer (quarter-patch stride)
  python infer.py --split val --stride 32
  python infer.py --split val --stride 8

  # Skip saving soft contribution maps
  python infer.py --volume train_1_a_1 --no-contrib
"""

import argparse
import logging
from pathlib import Path

from lc_ksvd.config import INFERENCE_STRIDE, MODELS_DIR
from lc_ksvd.data_loader import LabelRegistry, MetadataRegistry
from lc_ksvd.inference import run_inference, run_inference_batch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _volume_names_for_split(split: str):
    """Return all volume IDs registered in the metadata for a given split."""
    metadata = MetadataRegistry(split=split)
    labels   = LabelRegistry(metadata, split=split)
    return labels.get_all_volume_names()


def main():
    parser = argparse.ArgumentParser(
        description="LC-KSVD2 inference: sliding-window classification + back-projection"
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--volume", metavar="VOLUME_NAME",
        help="Single volume ID to run inference on, e.g. train_1_a_1",
    )
    source.add_argument(
        "--split", choices=["train", "val", "test"],
        help="Run inference on all volumes in this metadata split.",
    )

    parser.add_argument(
        "--model", metavar="PATH", default=None,
        help=f"Path to the trained .pkl file (default: {MODELS_DIR}/unified_lcksvd2.pkl)",
    )
    parser.add_argument(
        "--out-dir", metavar="DIR", default=None,
        help="Root output directory (default: outputs/inference/<volume_name>)",
    )
    parser.add_argument(
        "--stride", type=int, default=INFERENCE_STRIDE,
        help=f"Sliding-window stride in voxels (default: {INFERENCE_STRIDE})",
    )
    parser.add_argument(
        "--no-contrib", action="store_true",
        help="Skip saving soft contribution maps (saves only binary masks).",
    )

    args = parser.parse_args()

    model_path = Path(args.model) if args.model else None
    out_dir    = Path(args.out_dir) if args.out_dir else None

    if args.volume:
        # ── Single volume ─────────────────────────────────────────────────────
        paths = run_inference(
            volume_name=args.volume,
            model_path=model_path,
            out_dir=out_dir,
            stride=args.stride,
            save_contrib_maps=not args.no_contrib,
        )

        if paths:
            logger.info(f"\nDetected {len(paths)} abnormality class(es):")
            for cls, path in paths.items():
                logger.info(f"  {cls}: {path}")
        else:
            logger.info("\nNo abnormalities detected — volume classified as normal.")

    else:
        # ── Full split ────────────────────────────────────────────────────────
        volume_names = _volume_names_for_split(args.split)
        logger.info(
            f"Running inference on {len(volume_names)} volumes from split='{args.split}'"
        )

        all_results = run_inference_batch(
            volume_names=volume_names,
            model_path=model_path,
            out_dir=out_dir,
            stride=args.stride,
            save_contrib_maps=not args.no_contrib,
        )

        # ── Summary ────────────────────────────────────────────────────────────
        n_abnormal = sum(1 for r in all_results.values() if r)
        n_normal   = len(all_results) - n_abnormal
        logger.info(
            f"\nDone — {len(all_results)} volumes processed: "
            f"{n_abnormal} with detections, {n_normal} classified as normal."
        )

        from lc_ksvd.config import CLASS_ORDER
        for cls in CLASS_ORDER:
            if cls == "normal":
                continue
            count = sum(1 for r in all_results.values() if cls in r)
            logger.info(f"  {cls}: {count} volumes")


if __name__ == "__main__":
    main()