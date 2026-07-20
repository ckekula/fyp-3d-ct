"""
main.py
Trains one multi-class model over all abnormalities + normal.

Two algorithms are supported via --algorithm:

  frozen (default) - IncrementalFrozenDictionary (LC-KSVD-based Frozen
      Dictionary Learning). A base LC-KSVD2 dictionary is learned on
      "normal" patches only; abnormality classes are then added one at a
      time via add_class(), each learning a residual dictionary on top of
      the previously learned (frozen) atoms.

  lcksvd - the original single-shot LC-KSVD2 model trained jointly over
      all classes at once.

H rows (CLASS_ORDER):
  0 -> normal
  1 -> 2c  (groundglass opacity)
  2 -> 2d  (pulmonary nodules/masses)

Usage:
  python main.py                          # frozen (default), extract + train
  python main.py --algorithm lcksvd        # original joint LC-KSVD2
  python main.py --skip-extraction         # use existing unified .npz
"""

import argparse
import logging

from lc_ksvd.patch_extractor.patch_extraction import extract_unified
from lc_ksvd.train import train

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Train a unified model (normal + 3 abnormalities) using "
                     "either Frozen Dictionary Learning with LC-KSVD or plain LC-KSVD2."
    )
    parser.add_argument(
        "--skip-extraction", action="store_true",
        help="Skip patch extraction and use existing unified .npz files."
    )
    parser.add_argument(
        "--algorithm", choices=["frozen", "lcksvd"], default="frozen",
        help="'frozen' = IncrementalFrozenDictionary. 'lcksvd' = original joint LC-KSVD2."
    )
    args = parser.parse_args()

    if not args.skip_extraction:
        logger.info("Running unified patch extraction for train split...")
        extract_unified(split="train")

    train(algorithm=args.algorithm)


if __name__ == "__main__":
    main()