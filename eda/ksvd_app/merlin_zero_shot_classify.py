#!/usr/bin/env python3
"""
Zero-shot abnormality classification on a chest CT using Stanford's Merlin
vision-language model.

IMPORTANT SCOPE NOTE
---------------------
Merlin (StanfordMIMI/Merlin) does NOT ship an abnormality-localization or
abnormality-segmentation head. The publicly released model supports:
  - contrastive image/text embeddings (zero-shot classification)
  - phenotype classification, five-year disease risk
  - radiology report generation
  - segmentation of 20 *normal* abdominal organs (via a separate nnU-Net
    integration: https://github.com/ashwinkumargb/Merlin-nnUNet)

This script performs zero-shot *classification* only: it reports a
probability that each queried finding is present, using paired
positive/negative text prompts compared against the image embedding. It does
not return a location, bounding box, or segmentation mask. Treat the output
as a screening signal, not a diagnosis or a localization.

Pipeline (matches merlin.data.monai_transforms.ImageTransforms exactly):
    LoadImaged -> EnsureChannelFirstd -> Orientationd(RAS)
    -> Spacingd(1.5, 1.5, 3.0, bilinear)
    -> ScaleIntensityRanged(-1000, 1000 -> 0, 1, clip=True)
    -> SpatialPadd/CenterSpatialCropd([224, 224, 160])

Usage:
    python merlin_zero_shot_classify.py --ct /path/to/chest_ct.nii.gz
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("merlin_zero_shot")


@dataclass(frozen=True)
class FindingPrompt:
    """A finding evaluated via a positive/negative text prompt pair."""

    name: str
    positive: str
    negative: str


# Paired prompts: Merlin's text encoder was trained on radiology-report
# language, so short clinical phrases work better than casual terms.
DEFAULT_FINDINGS: tuple[FindingPrompt, ...] = (
    FindingPrompt(
        name="Pulmonary nodule",
        positive="pulmonary nodule",
        negative="no pulmonary nodule",
    ),
    FindingPrompt(
        name="Ground-glass opacity",
        positive="ground-glass opacity",
        negative="no ground-glass opacity",
    ),
)


def validate_input(ct_path: Path) -> None:
    if not ct_path.exists():
        raise FileNotFoundError(f"CT file not found: {ct_path}")
    valid_suffixes = (".nii", ".gz")
    if ct_path.suffix not in valid_suffixes and not ct_path.name.endswith(".nii.gz"):
        raise ValueError(
            f"Expected a .nii or .nii.gz file, got: {ct_path.name}. "
            "Convert DICOM series first (e.g. with dicom2nifti or SimpleITK)."
        )


def load_model(device: torch.device) -> torch.nn.Module:
    """Load Merlin with combined image+text embeddings enabled."""
    from merlin import Merlin

    logger.info("Loading Merlin (weights download from Hugging Face on first run)...")
    model = Merlin()  # default: image + text embeddings, for zero-shot use
    model = model.to(device)
    model.eval()
    return model


def preprocess_ct(ct_path: Path, cache_dir: str) -> torch.Tensor:
    """Run Merlin's exact preprocessing pipeline and return a batched tensor."""
    from merlin.data import DataLoader

    datalist = [{"image": str(ct_path)}]
    loader = DataLoader(
        datalist=datalist,
        cache_dir=cache_dir,
        batchsize=1,
        shuffle=False,
        num_workers=0,
    )
    batch = next(iter(loader))
    image = batch["image"]
    if not torch.is_tensor(image):
        raise RuntimeError("Unexpected preprocessing output: 'image' is not a tensor.")
    return image


@torch.inference_mode()
def run_zero_shot(
    model: torch.nn.Module,
    image: torch.Tensor,
    findings: tuple[FindingPrompt, ...],
    device: torch.device,
) -> dict[str, float]:
    """
    Score each finding by softmax-ing the image/text similarity over its
    (positive, negative) prompt pair, following the model's CLIP-style
    contrastive design.
    """
    image = image.to(device)
    prompts: list[str] = []
    for f in findings:
        prompts.extend([f.positive, f.negative])

    image_features, _ehr_features, text_features = model(image, prompts)

    logit_scale = model.model.logit_scale.exp()
    logits = logit_scale * image_features @ text_features.t()  # [1, 2 * n_findings]
    logits = logits.squeeze(0)

    results: dict[str, float] = {}
    for i, finding in enumerate(findings):
        pos_logit, neg_logit = logits[2 * i], logits[2 * i + 1]
        prob_present = torch.softmax(torch.stack([pos_logit, neg_logit]), dim=0)[0]
        results[finding.name] = float(prob_present.item())
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct", required=True, type=Path, help="Path to chest CT (.nii/.nii.gz)")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory to cache preprocessed tensors (default: temp dir)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for flagging a finding as present (default: 0.5)",
    )
    args = parser.parse_args()

    try:
        validate_input(args.ct)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        return 1

    if not torch.cuda.is_available():
        logger.warning("CUDA not available; falling back to CPU (will be slow).")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    try:
        model = load_model(device)
    except Exception:
        logger.exception("Failed to load Merlin model.")
        return 1

    cache_ctx = (
        TemporaryDirectory() if args.cache_dir is None else None
    )
    cache_dir = str(args.cache_dir) if args.cache_dir else cache_ctx.name  # type: ignore[union-attr]

    try:
        logger.info("Preprocessing CT: %s", args.ct)
        image = preprocess_ct(args.ct, cache_dir)
        logger.info("Preprocessed tensor shape: %s", tuple(image.shape))

        logger.info("Running zero-shot classification...")
        scores = run_zero_shot(model, image, DEFAULT_FINDINGS, device)
    except Exception:
        logger.exception("Inference failed.")
        return 1
    finally:
        if cache_ctx is not None:
            cache_ctx.cleanup()

    print("\n=== Zero-shot finding probabilities (presence) ===")
    for name, prob in scores.items():
        flag = "FLAGGED" if prob >= args.threshold else "-"
        print(f"  {name:<28} {prob:6.1%}   {flag}")
    print(
        "\nNote: these are whole-volume presence probabilities, not a "
        "location or segmentation mask. Merlin's released model has no "
        "abnormality-localization/segmentation head."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())