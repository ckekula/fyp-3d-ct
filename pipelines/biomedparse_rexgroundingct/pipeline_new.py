from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import hydra
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "biomed-parse"
DEFAULT_CHECKPOINT = MODEL_DIR / "model_weights" / "biomedparse_v2.ckpt"

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from inference import postprocess, merge_multiclass_masks  # type: ignore
from utils import process_input, process_output  # type: ignore


TARGETS = {
    "Lung Nodule": [
        "lung nodule",
        "pulmonary nodule",
        "nodule",
        "nodular opacity",
        "nodular lesion",
    ],
    "Lung opacity": [
        "lung opacity",
        "pulmonary opacity",
        "opacity",
        "ground-glass opacity",
        "ground glass opacity",
        "ggo",
    ],
    "Consolidation": [
        "consolidation",
        "pulmonary consolidation",
        "lobar consolidation",
        "airspace consolidation",
    ],
    "Atelectasis": [
        "atelectasis",
        "linear atelectasis",
        "segmental atelectasis",
        "subsegmental atelectasis",
        "collapse",
    ],
}


def setup_cpu(cpu_threads: int) -> None:
    if cpu_threads <= 0:
        cpu_threads = max(1, (os.cpu_count() or 4) // 2)

    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)

    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(1)
    torch.backends.mkldnn.enabled = True
    torch.set_grad_enabled(False)


def get_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")

    if device_name in ["xpu", "intel"]:
        try:
            import intel_extension_for_pytorch  # noqa: F401

            if hasattr(torch, "xpu") and torch.xpu.is_available():
                return torch.device("xpu")
        except Exception:
            print("Intel XPU requested, but IPEX/XPU is not available. Falling back to CPU.")

    return torch.device("cpu")


def match_target(text: str) -> Optional[str]:
    lower = text.lower()
    for label, keywords in TARGETS.items():
        if any(keyword in lower for keyword in keywords):
            return label
    return None


def window_ct_lung(volume: np.ndarray, width: float = 1500, level: float = -160) -> np.ndarray:
    lower = level - width / 2
    upper = level + width / 2

    volume = volume.astype(np.float32)
    volume = np.clip(volume, lower, upper)
    volume = (volume - lower) / (upper - lower)
    volume = np.clip(volume * 255.0, 0, 255)

    return volume.astype(np.uint8)


def load_ct_dhw(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    vol = img.get_fdata(dtype=np.float32)

    if vol.ndim != 3:
        raise ValueError(f"Expected 3D CT volume, got shape {vol.shape}")

    # ReXGroundingCT volume: H, W, D
    # BiomedParse input: D, H, W
    return np.transpose(vol, (2, 0, 1))


def load_gt_mask_dhw(path: Path, finding_index: int) -> np.ndarray:
    img = nib.load(str(path))
    mask = img.get_fdata(dtype=np.float32)

    # ReXGroundingCT mask usually: F, H, W, D
    if mask.ndim == 4:
        mask = mask[finding_index]

    if mask.ndim != 3:
        raise ValueError(f"Expected mask shape [H,W,D] or [F,H,W,D], got {mask.shape}")

    mask = np.transpose(mask, (2, 0, 1))
    return (mask > 0).astype(np.uint8)


def find_nii_file(root: Path, name: str) -> Optional[Path]:
    candidates = [
        root / name,
        root / f"{name}.nii.gz",
        root / f"{name}.nii",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = list(root.rglob(name))
    if matches:
        return matches[0]

    matches = list(root.rglob(f"{name}.nii.gz"))
    if matches:
        return matches[0]

    matches = list(root.rglob(f"{name}.nii"))
    if matches:
        return matches[0]

    return None


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float | int]:
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)

    tp = np.logical_and(pred_bool, gt_bool).sum()
    fp = np.logical_and(pred_bool, ~gt_bool).sum()
    fn = np.logical_and(~pred_bool, gt_bool).sum()

    pred_sum = pred_bool.sum()
    gt_sum = gt_bool.sum()
    union = np.logical_or(pred_bool, gt_bool).sum()

    dice = (2.0 * tp) / (pred_sum + gt_sum + 1e-8)
    iou = tp / (union + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "pred_voxels": int(pred_sum),
        "gt_voxels": int(gt_sum),
    }


def load_model(checkpoint: Path, device: torch.device):
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    GlobalHydra.instance().clear()

    initialize_config_dir(
        config_dir=str(MODEL_DIR / "configs" / "model"),
        job_name="biomedparse_rexgroundingct_cpu",
        version_base=None,
    )

    cfg = compose(config_name="biomedparse_3D")
    model = hydra.utils.instantiate(cfg, _convert_="object")

    print(f"Loading checkpoint: {checkpoint}")
    model.load_pretrained(str(checkpoint))

    model = model.to(device)
    model.eval()

    if device.type == "xpu":
        try:
            import intel_extension_for_pytorch as ipex

            model = ipex.optimize(model)
            print("Intel IPEX optimization enabled.")
        except Exception as e:
            print(f"IPEX optimization skipped: {e}")

    return model


def predict_mask(
    model,
    volume_dhw: np.ndarray,
    prompt: str,
    device: torch.device,
    image_size: int,
    slice_batch_size: int,
    threshold: float,
) -> Tuple[np.ndarray, float]:
    imgs, pad_width, padded_size, valid_axis = process_input(volume_dhw, image_size)
    imgs = imgs.to(device=device, dtype=torch.int32)

    input_tensor = {
        "image": imgs.unsqueeze(0),
        "text": [prompt],
    }

    with torch.inference_mode():
        output = model(input_tensor, mode="eval", slice_batch_size=slice_batch_size)

    pred = output["predictions"]["pred_gmasks"]
    existence = output["predictions"]["object_existence"]

    pred = F.interpolate(
        pred,
        size=(image_size, image_size),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )

    pred = postprocess(pred, existence)

    try:
        pred = merge_multiclass_masks(pred, [1])
    except Exception:
        if isinstance(pred, torch.Tensor):
            pred = pred.squeeze()

    pred = process_output(pred, pad_width, padded_size, valid_axis)
    pred = np.asarray(pred)

    pred = np.squeeze(pred)

    if pred.ndim != 3:
        raise ValueError(f"Expected predicted mask to be 3D after squeeze, got {pred.shape}")

    existence_score = float(torch.sigmoid(existence).max().detach().cpu().item())
    pred_binary = (pred > threshold).astype(np.uint8)

    return pred_binary, existence_score


def read_metadata(metadata_json: Path) -> list:
    with metadata_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    entries = []

    for key in ["train", "val", "valid", "validation", "test"]:
        value = data.get(key)
        if isinstance(value, list):
            entries.extend(value)

    if not entries:
        raise ValueError("Could not find entries in metadata JSON.")

    return entries


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument("--volume-root", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)

    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "biomedparse_rexgroundingct_cpu")

    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "xpu", "intel"])
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--slice-batch-size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--save-masks", action="store_true")

    args = parser.parse_args()

    setup_cpu(args.cpu_threads)

    device = get_device(args.device)
    print(f"Using device: {device}")
    print(f"Image size: {args.image_size}")
    print(f"Slice batch size: {args.slice_batch_size}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = args.output_dir / "pred_masks"
    pred_dir.mkdir(parents=True, exist_ok=True)

    entries = read_metadata(args.metadata_json)
    model = load_model(args.checkpoint, device)

    rows = []
    evaluated_cases = 0

    for entry in entries:
        volume_name = entry.get("name") or entry.get("volume") or entry.get("image")
        if not volume_name:
            continue

        findings = entry.get("findings", {})
        if not findings:
            continue

        selected = []

        for finding_id, finding_text in findings.items():
            target = match_target(str(finding_text))
            if target:
                selected.append((int(finding_id), str(finding_text), target))

        if not selected:
            continue

        volume_path = find_nii_file(args.volume_root, volume_name)
        mask_path = find_nii_file(args.mask_root, volume_name)

        if volume_path is None:
            print(f"Volume not found: {volume_name}")
            continue

        if mask_path is None:
            print(f"Mask not found: {volume_name}")
            continue

        print(f"\nCase: {volume_name}")
        print(f"Selected findings: {len(selected)}")

        try:
            volume = load_ct_dhw(volume_path)
            volume = window_ct_lung(volume)
        except Exception as e:
            print(f"Failed to load volume {volume_name}: {e}")
            continue

        for finding_index, finding_text, target in selected:
            print(f"  Finding {finding_index} | {target}")
            print(f"  Prompt: {finding_text}")

            try:
                gt = load_gt_mask_dhw(mask_path, finding_index)

                pred, existence_score = predict_mask(
                    model=model,
                    volume_dhw=volume,
                    prompt=finding_text,
                    device=device,
                    image_size=args.image_size,
                    slice_batch_size=args.slice_batch_size,
                    threshold=args.threshold,
                )

                if pred.shape != gt.shape:
                    print(f"  Shape mismatch: pred={pred.shape}, gt={gt.shape}")
                    continue

                metrics = compute_metrics(pred, gt)

                prediction_file = ""

                if args.save_masks:
                    safe_target = target.lower().replace(" ", "_")
                    safe_name = Path(volume_name).name.replace(".nii.gz", "").replace(".nii", "")
                    out_path = pred_dir / f"{safe_name}_finding_{finding_index}_{safe_target}.npz"
                    np.savez_compressed(out_path, pred=pred, gt=gt)
                    prediction_file = str(out_path)

                row = {
                    "volume_name": volume_name,
                    "finding_index": finding_index,
                    "target": target,
                    "prompt": finding_text,
                    "existence_score": existence_score,
                    **metrics,
                    "prediction_file": prediction_file,
                }

                rows.append(row)

                print(
                    f"  Dice={metrics['dice']:.4f} | "
                    f"IoU={metrics['iou']:.4f} | "
                    f"Precision={metrics['precision']:.4f} | "
                    f"Recall={metrics['recall']:.4f} | "
                    f"Existence={existence_score:.4f}"
                )

            except Exception as e:
                print(f"  Failed finding {finding_index}: {e}")

        evaluated_cases += 1

        if args.limit > 0 and evaluated_cases >= args.limit:
            break

    csv_path = args.output_dir / "metrics.csv"

    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nSaved metrics: {csv_path}")
        print(f"Total evaluated findings: {len(rows)}")

        summary = {}
        for target in TARGETS:
            target_rows = [r for r in rows if r["target"] == target]
            if not target_rows:
                continue

            summary[target] = {
                "count": len(target_rows),
                "mean_dice": float(np.mean([r["dice"] for r in target_rows])),
                "mean_iou": float(np.mean([r["iou"] for r in target_rows])),
                "mean_precision": float(np.mean([r["precision"] for r in target_rows])),
                "mean_recall": float(np.mean([r["recall"] for r in target_rows])),
            }

        summary_path = args.output_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Saved summary: {summary_path}")

    else:
        print("\nNo findings evaluated. Check paths, metadata keys, and target keywords.")


if __name__ == "__main__":
    main()