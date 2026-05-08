from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import hydra
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "biomed-parse"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from inference import merge_multiclass_masks, postprocess  # type: ignore
from utils import process_input, process_output  # type: ignore

from dataset_adapter import RexCase, iter_target_cases, load_rexgroundingct_cases
from prompts import default_prompt_bundles
from visualize import save_overlay_png


def window_ct(volume: np.ndarray, width: float = 1500.0, level: float = -160.0) -> np.ndarray:
    lower = level - width / 2.0
    upper = level + width / 2.0
    clipped = np.clip(volume.astype(np.float32), lower, upper)
    scaled = (clipped - lower) / max(width, 1e-6)
    return np.clip(scaled * 255.0, 0.0, 255.0).astype(np.uint8)


def load_volume(volume_path: str | Path) -> np.ndarray:
    image = nib.load(str(volume_path))
    volume = image.get_fdata(dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D CT volume, got shape {volume.shape} from {volume_path}")
    return np.transpose(volume, (2, 0, 1))


def load_model(checkpoint_path: str | None, device: torch.device):
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=str(MODEL_DIR / "configs" / "model"), job_name="rexgroundingct_pipeline", version_base=None)
    cfg = compose(config_name="biomedparse_3D")
    model = hydra.utils.instantiate(cfg, _convert_="object")

    if checkpoint_path is None:
        checkpoint_path = hf_hub_download(repo_id="microsoft/BiomedParse", filename="biomedparse_v2.ckpt")

    model.load_pretrained(checkpoint_path)
    return model.to(device).eval()


def build_case_report(case: RexCase, predictions: Dict[str, Dict[str, float]]) -> Dict[str, object]:
    return {
        "volume_name": case.volume_name,
        "matched_diseases_in_finding_text": case.matched_diseases,
        "findings": case.findings,
        "predictions": predictions,
        "protocol": case.protocol,
    }


def run_case(model, case: RexCase, device: torch.device, output_dir: Path, diseases: List[str]) -> Dict[str, Dict[str, float]]:
    volume = load_volume(case.volume_path)
    volume = window_ct(volume)

    disease_scores: Dict[str, Dict[str, float]] = {}
    raw_mask_file = output_dir / "masks" / f"{case.volume_name}.npz"
    raw_mask_file.parent.mkdir(parents=True, exist_ok=True)

    bundle_outputs: Dict[str, np.ndarray] = {}
    for bundle in default_prompt_bundles():
        if bundle.disease not in diseases:
            continue

        image, pad_width, padded_size, valid_axis = process_input(volume, 512)
        image = image.to(device).int()
        input_tensor = {"image": image.unsqueeze(0), "text": [bundle.text]}

        with torch.no_grad():
            output = model(input_tensor, mode="eval", slice_batch_size=4)

        mask_preds = output["predictions"]["pred_gmasks"]
        mask_preds = F.interpolate(mask_preds, size=(512, 512), mode="bicubic", align_corners=False, antialias=True)
        mask_preds = postprocess(mask_preds, output["predictions"]["object_existence"])

        if mask_preds.ndim == 4:
            disease_mask = mask_preds.max(dim=0).values
        else:
            disease_mask = mask_preds.squeeze(0)

        disease_mask = process_output(disease_mask, pad_width, padded_size, valid_axis)
        existence_score = float(output["predictions"]["object_existence"].sigmoid().max().item())
        mask_voxels = float((disease_mask > 0).sum())

        disease_scores[bundle.disease] = {
            "existence_score": existence_score,
            "mask_voxels": mask_voxels,
        }
        bundle_key = bundle.disease.lower().replace(" ", "_")
        bundle_outputs[bundle_key] = np.asarray(disease_mask.cpu())

        overlay_path = output_dir / "overlays" / bundle.disease / f"{Path(case.volume_name).stem}.png"
        save_overlay_png(volume, bundle_outputs[bundle_key], overlay_path)

    np.savez_compressed(raw_mask_file, **bundle_outputs)
    return disease_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BiomedParse RexGroundingCT pipeline")
    parser.add_argument("--metadata-json", type=Path, default=ROOT / "data" / "rexgrounding-ct" / "dataset.json")
    parser.add_argument("--volume-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "biomedparse_rexgroundingct")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--diseases", nargs="*", default=["Lung nodule", "Lung opacity", "Consolidation", "Atelectasis"])
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

    model = load_model(str(args.checkpoint) if args.checkpoint else None, device)

    reports: List[Dict[str, object]] = []
    for case in cases:
        predictions = run_case(model, case, device, output_dir, args.diseases)
        reports.append(build_case_report(case, predictions))

    report_path = output_dir / "reports.json"
    report_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()