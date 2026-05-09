from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "biomed-parse"

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from inference import merge_multiclass_masks, postprocess  # type: ignore
from utils import process_input, process_output  # type: ignore

# Supports both:
#   python pipelines/biomedparse_rexgroundingct/pipeline_fix.py
# and:
#   python -m pipelines.biomedparse_rexgroundingct.pipeline_fix
try:
    from .dataset_adapter import RexCase, iter_target_cases, load_rexgroundingct_cases
    from .prompts import DEFAULT_DISEASES, default_prompt_bundles
    from .visualize import save_overlay_png
except ImportError:
    from dataset_adapter import RexCase, iter_target_cases, load_rexgroundingct_cases
    from prompts import DEFAULT_DISEASES, default_prompt_bundles
    from visualize import save_overlay_png


def safe_key(name: str) -> str:
    """Create stable npz/json keys."""
    return " ".join(str(name).strip().lower().split()).replace(" ", "_").replace("/", "_")


def volume_stem(name: str) -> str:
    """Return a clean case stem for .nii, .nii.gz, .npz, etc."""
    text = Path(name).name
    for suffix in (".nii.gz", ".nii", ".npz", ".npy"):
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return Path(text).stem


def window_ct(volume: np.ndarray, width: float = 1500.0, level: float = -160.0) -> np.ndarray:
    """Apply lung-window CT preprocessing and rescale to uint8 [0, 255]."""
    lower = level - width / 2.0
    upper = level + width / 2.0

    clipped = np.clip(volume.astype(np.float32), lower, upper)
    scaled = (clipped - lower) / max(width, 1e-6)

    return np.clip(scaled * 255.0, 0.0, 255.0).astype(np.uint8)


def load_volume(volume_path: str | Path) -> np.ndarray:
    """
    Load a NIfTI CT volume and return it as (D, H, W).

    ReXGroundingCT volumes are commonly loaded as (H, W, D), so this transposes
    them into depth-first format for easier overlay/mask handling.
    """
    image = nib.load(str(volume_path))
    volume = image.get_fdata(dtype=np.float32)

    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D CT volume, got shape {volume.shape} from {volume_path}")

    return np.transpose(volume, (2, 0, 1))


def load_model(checkpoint_path: str | None, device: torch.device):
    """Load the official BiomedParse v2 3D model."""
    GlobalHydra.instance().clear()

    initialize_config_dir(
        config_dir=str(MODEL_DIR / "configs" / "model"),
        job_name="rexgroundingct_pipeline",
        version_base=None,
    )

    cfg = compose(config_name="biomedparse_3D")
    model = hydra.utils.instantiate(cfg, _convert_="object")

    if checkpoint_path is None:
        checkpoint_path = hf_hub_download(
            repo_id="microsoft/BiomedParse",
            filename="biomedparse_v2.ckpt",
        )

    model.load_pretrained(str(checkpoint_path))
    return model.to(device).eval()


def call_biomedparse_postprocess(
    mask_preds: torch.Tensor,
    object_existence: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """
    Call BiomedParse postprocess safely.

    Some BiomedParse versions accept threshold=..., while some older checked-out
    code only accepts two arguments.
    """
    try:
        return postprocess(mask_preds, object_existence, threshold=threshold)
    except TypeError:
        return postprocess(mask_preds, object_existence)


def to_numpy(array_like) -> np.ndarray:
    """Convert torch.Tensor or numpy-like output into numpy.ndarray."""
    if isinstance(array_like, torch.Tensor):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def _remove_padding_3d(vol: np.ndarray, pad_width) -> np.ndarray:
    """Remove BiomedParse square padding from a (D, H, W) array."""
    if pad_width is None:
        return vol

    h0 = int(pad_width[1][0])
    h1 = int(vol.shape[1] - pad_width[1][1])

    w0 = int(pad_width[2][0])
    w1 = int(vol.shape[2] - pad_width[2][1])

    return vol[:, h0:h1, w0:w1]


def process_probability_output(
    vol: torch.Tensor,
    pad_width,
    padded_size: int,
    valid_axis: int,
) -> np.ndarray:
    """
    Map a float probability volume back to original volume space.

    Official process_output casts to int, which is correct for final class masks
    but not correct for probability maps. This keeps float values for threshold
    sweeps and qualitative heatmap inspection.
    """
    if vol.ndim != 3:
        raise ValueError(f"Expected probability volume with shape (D, H, W), got {tuple(vol.shape)}")

    if vol.shape[-1] != padded_size or vol.shape[-2] != padded_size:
        vol = F.interpolate(
            vol.unsqueeze(0).float(),
            size=(padded_size, padded_size),
            mode="nearest",
        ).squeeze(0)

    vol_np = vol.detach().cpu().numpy().astype(np.float32)
    vol_np = _remove_padding_3d(vol_np, pad_width)
    vol_np = np.moveaxis(vol_np, 0, valid_axis)

    return vol_np


def build_case_report(case: RexCase, predictions: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    predicted_present = [
        disease_name
        for disease_name, item in predictions.items()
        if bool(item.get("present"))
    ]

    return {
        "volume_name": case.volume_name,
        "matched_diseases_in_finding_text": case.matched_diseases,
        "findings": case.findings,
        "predicted_present_diseases": predicted_present,
        "predictions": predictions,
        "protocol": case.protocol,
    }


def run_disease_prompt(
    model,
    image: torch.Tensor,
    bundle_text: str,
    pad_width,
    padded_size: int,
    valid_axis: int,
    device: torch.device,
    *,
    slice_batch_size: int,
    postprocess_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Run one disease prompt bundle.

    Returns:
        binary_mask:
            uint8 final foreground mask in original volume space.
        prob_map:
            float32 probability map in original volume space.
        existence_score:
            max object-existence score after sigmoid.
        prompt_count:
            number of prompt masks produced by BiomedParse.
    """
    input_tensor = {
        "image": image.unsqueeze(0),
        "text": [bundle_text],
    }

    with torch.inference_mode():
        output = model(input_tensor, mode="eval", slice_batch_size=slice_batch_size)

    mask_preds = output["predictions"]["pred_gmasks"]
    object_existence = output["predictions"]["object_existence"]

    mask_preds = F.interpolate(
        mask_preds,
        size=(512, 512),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )

    # Official-style BiomedParse post-processing.
    mask_preds = call_biomedparse_postprocess(
        mask_preds,
        object_existence,
        threshold=postprocess_threshold,
    )

    if mask_preds.ndim == 3:
        mask_preds = mask_preds.unsqueeze(0)

    if mask_preds.ndim != 4:
        raise ValueError(
            f"Expected postprocessed masks with shape (N, D, H, W), got {tuple(mask_preds.shape)}"
        )

    prompt_count = int(mask_preds.shape[0])
    prompt_ids = list(range(1, prompt_count + 1))

    # Official BiomedParse v2 merge step.
    merged_class_mask = merge_multiclass_masks(mask_preds, prompt_ids)

    # For ReXGroundingCT baseline, all synonym prompts become one disease foreground mask.
    binary_tensor = (merged_class_mask > 0).to(torch.uint8)

    # Important: process_output is okay here because this is already binary/integer.
    binary_mask = to_numpy(process_output(binary_tensor, pad_width, padded_size, valid_axis)).astype(np.uint8)

    # Save probability map separately for threshold experiments.
    prob_tensor = mask_preds.max(dim=0).values
    prob_map = process_probability_output(prob_tensor, pad_width, padded_size, valid_axis)

    existence_score = float(object_existence.sigmoid().max().detach().cpu().item())

    return binary_mask, prob_map, existence_score, prompt_count


def run_case(
    model,
    case: RexCase,
    device: torch.device,
    output_dir: Path,
    diseases: List[str],
    *,
    existence_threshold: float,
    min_mask_voxels: int,
    slice_batch_size: int,
    postprocess_threshold: float,
) -> Dict[str, Dict[str, object]]:
    print(f"Processing case: {case.volume_name}", flush=True)

    volume = load_volume(case.volume_path)
    volume = window_ct(volume)

    # Preprocess once per case. Only the text prompt changes per disease.
    image, pad_width, padded_size, valid_axis = process_input(volume, 512)
    image = image.to(device).int()

    case_id = volume_stem(case.volume_name)

    mask_file = output_dir / "masks" / f"{case_id}.npz"
    prob_file = output_dir / "prob_maps" / f"{case_id}.npz"

    mask_file.parent.mkdir(parents=True, exist_ok=True)
    prob_file.parent.mkdir(parents=True, exist_ok=True)

    selected_diseases = {" ".join(str(d).strip().lower().split()) for d in diseases}

    disease_scores: Dict[str, Dict[str, object]] = {}
    binary_outputs: Dict[str, np.ndarray] = {}
    prob_outputs: Dict[str, np.ndarray] = {}

    for bundle in default_prompt_bundles():
        if " ".join(bundle.disease.strip().lower().split()) not in selected_diseases:
            continue

        print(f"  Running disease prompt: {bundle.disease}", flush=True)

        binary_mask, prob_map, existence_score, prompt_count = run_disease_prompt(
            model,
            image,
            bundle.text,
            pad_width,
            padded_size,
            valid_axis,
            device,
            slice_batch_size=slice_batch_size,
            postprocess_threshold=postprocess_threshold,
        )

        mask_voxels = int((binary_mask > 0).sum())
        present = bool(
            existence_score >= existence_threshold
            and mask_voxels >= min_mask_voxels
        )

        key = safe_key(bundle.disease)

        binary_outputs[key] = binary_mask.astype(np.uint8)
        prob_outputs[key] = prob_map.astype(np.float32)

        overlay_path = output_dir / "overlays" / key / f"{case_id}.png"
        save_overlay_png(volume, binary_outputs[key], overlay_path)

        disease_scores[bundle.disease] = {
            "existence_score": existence_score,
            "mask_voxels": mask_voxels,
            "present": present,
            "prompt_count": prompt_count,
            "mask_key": key,
            "decision": {
                "existence_threshold": float(existence_threshold),
                "min_mask_voxels": int(min_mask_voxels),
                "postprocess_threshold": float(postprocess_threshold),
            },
        }

        print(
            f"  Saved {bundle.disease}: "
            f"score={existence_score:.4f}, "
            f"voxels={mask_voxels}, "
            f"present={present}",
            flush=True,
        )

    np.savez_compressed(mask_file, **binary_outputs)
    np.savez_compressed(prob_file, **prob_outputs)

    print(f"Finished case: {case.volume_name}", flush=True)

    return disease_scores


def save_reports_atomic(report_path: Path, reports: List[Dict[str, object]]) -> None:
    """
    Save reports safely.

    First writes to reports.tmp.json, then replaces reports.json.
    This reduces the chance of corrupting reports.json if the process stops.
    """
    temp_report_path = report_path.with_name("reports.tmp.json")
    temp_report_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    temp_report_path.replace(report_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BiomedParse v2 pipeline for ReXGroundingCT"
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
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "biomedparse_rexgroundingct",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of cases to process. Use 0 to process all selected cases.",
    )

    parser.add_argument(
        "--diseases",
        nargs="*",
        default=DEFAULT_DISEASES,
    )

    parser.add_argument(
        "--existence-threshold",
        type=float,
        default=0.30,
        help="Presence threshold on BiomedParse object_existence after sigmoid.",
    )

    parser.add_argument(
        "--postprocess-threshold",
        type=float,
        default=0.30,
        help="Threshold passed into BiomedParse postprocess when supported.",
    )

    parser.add_argument(
        "--min-mask-voxels",
        type=int,
        default=1,
        help="Minimum foreground voxels required for present=True.",
    )

    parser.add_argument(
        "--slice-batch-size",
        type=int,
        default=4,
        help="BiomedParse slice batch size. Reduce to 1 on CPU or low-memory GPUs.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = load_rexgroundingct_cases(args.metadata_json, args.volume_root)
    cases = list(iter_target_cases(cases, diseases=args.diseases))

    if args.limit > 0:
        cases = cases[: args.limit]

    print(f"Using device: {device}", flush=True)
    print(f"Selected cases: {len(cases)}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)

    model = load_model(
        str(args.checkpoint) if args.checkpoint else None,
        device,
    )

    report_path = output_dir / "reports.json"
    reports: List[Dict[str, object]] = []

    # Resume support: if reports.json exists, load completed reports.
    if report_path.exists():
        try:
            reports = json.loads(report_path.read_text(encoding="utf-8"))
            print(
                f"Loaded existing report with {len(reports)} completed cases.",
                flush=True,
            )
        except json.JSONDecodeError:
            print(
                "Warning: existing reports.json is corrupted. Starting a new report.",
                flush=True,
            )
            reports = []

    completed_cases = {
        str(report.get("volume_name"))
        for report in reports
        if isinstance(report, dict) and report.get("volume_name")
    }

    for case in cases:
        if case.volume_name in completed_cases:
            print(f"Skipping already reported case: {case.volume_name}", flush=True)
            continue

        predictions = run_case(
            model,
            case,
            device,
            output_dir,
            args.diseases,
            existence_threshold=args.existence_threshold,
            min_mask_voxels=args.min_mask_voxels,
            slice_batch_size=args.slice_batch_size,
            postprocess_threshold=args.postprocess_threshold,
        )

        case_report = build_case_report(case, predictions)
        reports.append(case_report)

        # Save reports.json after every completed case.
        save_reports_atomic(report_path, reports)

        print(f"Updated report after case: {case.volume_name}", flush=True)

    print(f"Wrote final report: {report_path}", flush=True)


if __name__ == "__main__":
    main()


"""
.\.venv\Scripts\python.exe pipelines\biomedparse_rexgroundingct\pipeline_fix.py `
  --metadata-json data\Govindu\rexgrounding-ct\dataset.json `
  --volume-root data\data_volumes `
  --checkpoint models\biomed-parse\model_weights\biomedparse_v2.ckpt `
  --output-dir outputs\biomedparse_rexgroundingct_fixed_cuda `
  --device cuda `
  --slice-batch-size 4 `
  --postprocess-threshold 0.30 `
  --existence-threshold 0.30 `
  --limit 20
"""
