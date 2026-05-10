from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer


ROOT = Path(__file__).resolve().parents[2]
CT_CLIP_ROOT = ROOT / "models" / "ct-clip"
CT_CLIP_PKG = CT_CLIP_ROOT / "CT_CLIP"
MASKGIT_PKG = CT_CLIP_ROOT / "transformer_maskgit"
DEFAULT_BIOMEDVLP_DIR = ROOT / "models" / "BiomedVLP-CXR-BERT-specialized"

for package_root in (CT_CLIP_PKG, MASKGIT_PKG):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))


ABNORMALITIES: Dict[str, List[str]] = {
    "Atelectasis": [
        "atelectasis",
        "linear atelectasis",
        "segmental atelectasis",
        "subsegmental atelectasis",
        "fibroatelectatic",
        "atelectatic",
    ],
    "Lung nodule": [
        "lung nodule",
        "pulmonary nodule",
        "subcentimeter nodule",
        "subcentimeter nodules",
        "nodule",
        "nodules",
        "nodular lesion",
    ],
    "Lung opacity": [
        "lung opacity",
        "pulmonary opacity",
        "opacity",
        "opacities",
        "ground-glass opacity",
        "ground glass opacity",
        "ground-glass opacities",
        "ground glass opacities",
        "ggo",
    ],
    "Consolidation": [
        "consolidation",
        "consolidative",
        "airspace consolidation",
        "lobar consolidation",
        "nodular consolidation",
    ],
}


@dataclass
class EvalCase:
    case_id: str
    volume_path: Path
    labels: Dict[str, int]
    source: str


def setup_cpu_threads(cpu_threads: int) -> None:
    if cpu_threads <= 0:
        cpu_threads = max(1, (os.cpu_count() or 4) // 2)

    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)


def get_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def softmax_positive(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=0)
    return float(probs[0].detach().cpu().item())


def find_volume(volume_root: Path, name: str) -> Path | None:
    candidates = [volume_root / name, volume_root / f"{name}.nii.gz", volume_root / f"{name}.nii"]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    direct = list(volume_root.rglob(name))
    if direct:
        return direct[0]

    for suffix in (".nii.gz", ".nii"):
        matches = list(volume_root.rglob(f"{name}{suffix}"))
        if matches:
            return matches[0]
    return None


def normalize_text_labels(texts: Iterable[str]) -> Dict[str, int]:
    label_map = {name: 0 for name in ABNORMALITIES}
    lowered = " ".join(texts).lower()
    for abnormality, aliases in ABNORMALITIES.items():
        if any(alias in lowered for alias in aliases):
            label_map[abnormality] = 1
    return label_map


def read_ct_rate_cases(volume_root: Path, labels_csv: Path, limit: int = 0) -> List[EvalCase]:
    cases: List[EvalCase] = []
    with labels_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            case_id = row["VolumeName"]
            volume_path = find_volume(volume_root, case_id)
            if volume_path is None:
                continue

            labels = {}
            for abnormality in ABNORMALITIES:
                if abnormality not in row:
                    labels[abnormality] = 0
                else:
                    labels[abnormality] = int(str(row[abnormality]).strip() == "1")

            cases.append(EvalCase(case_id=case_id, volume_path=volume_path, labels=labels, source="ct-rate"))
            if limit > 0 and len(cases) >= limit:
                break
    return cases


def read_rex_cases(volume_root: Path, metadata_json: Path, limit: int = 0) -> List[EvalCase]:
    payload = json.loads(metadata_json.read_text(encoding="utf-8"))
    entries: List[dict] = []
    for key in ("train", "val", "valid", "validation", "test"):
        value = payload.get(key)
        if isinstance(value, list):
            entries.extend(value)

    cases: List[EvalCase] = []
    for entry in entries:
        case_id = entry["name"]
        volume_path = find_volume(volume_root, case_id)
        if volume_path is None:
            continue

        findings = entry.get("findings", {})
        texts = [str(v) for _, v in sorted(findings.items(), key=lambda item: int(item[0]))]
        labels = normalize_text_labels(texts)

        cases.append(EvalCase(case_id=case_id, volume_path=volume_path, labels=labels, source="rexgrounding-ct"))
        if limit > 0 and len(cases) >= limit:
            break
    return cases


def load_and_preprocess_ct(volume_path: Path) -> torch.Tensor:
    image = nib.load(str(volume_path))
    volume = image.get_fdata(dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI volume, got shape {volume.shape} for {volume_path}")

    slope, intercept = image.header.get_slope_inter()
    slope = 1.0 if slope is None else float(slope)
    intercept = 0.0 if intercept is None else float(intercept)

    spacings = tuple(float(v) for v in image.header.get_zooms()[:3])
    # header spacing is H, W, D -> convert to D, H, W after transpose below
    current_spacing = (spacings[2], spacings[0], spacings[1])
    target_spacing = (1.5, 0.75, 0.75)

    volume = slope * volume + intercept
    volume = np.clip(volume, -1000.0, 1000.0)
    volume = np.transpose(volume, (2, 0, 1))

    tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    original_shape = tensor.shape[2:]
    new_shape = []
    for axis, dim in enumerate(original_shape):
        scaled = dim * (current_spacing[axis] / target_spacing[axis])
        new_shape.append(max(1, int(round(scaled))))

    tensor = F.interpolate(tensor, size=tuple(new_shape), mode="trilinear", align_corners=False)
    tensor = tensor[0, 0]

    # convert back to H, W, D for crop/pad logic used in the reference repo
    tensor = tensor.permute(1, 2, 0)
    tensor = tensor / 1000.0

    target_h, target_w, target_d = 480, 480, 240
    h, w, d = tensor.shape

    h_start = max((h - target_h) // 2, 0)
    w_start = max((w - target_w) // 2, 0)
    d_start = max((d - target_d) // 2, 0)
    h_end = min(h_start + target_h, h)
    w_end = min(w_start + target_w, w)
    d_end = min(d_start + target_d, d)

    tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

    pad_h_before = (target_h - tensor.size(0)) // 2
    pad_h_after = target_h - tensor.size(0) - pad_h_before
    pad_w_before = (target_w - tensor.size(1)) // 2
    pad_w_after = target_w - tensor.size(1) - pad_w_before
    pad_d_before = (target_d - tensor.size(2)) // 2
    pad_d_after = target_d - tensor.size(2) - pad_d_before

    tensor = F.pad(
        tensor,
        (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after),
        value=-1.0,
    )

    tensor = tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
    return tensor


def build_zero_shot_model(device: torch.device, checkpoint: Path, text_encoder_dir: Path | None = None):
    from ct_clip import CTCLIP  # type: ignore
    from transformer_maskgit import CTViT  # type: ignore

    text_encoder_source = text_encoder_dir if text_encoder_dir is not None else DEFAULT_BIOMEDVLP_DIR
    if text_encoder_source is None or not Path(text_encoder_source).exists():
        text_encoder_source = "microsoft/BiomedVLP-CXR-BERT-specialized"

    tokenizer = BertTokenizer.from_pretrained(str(text_encoder_source), do_lower_case=True)
    text_encoder = BertModel.from_pretrained(str(text_encoder_source))
    text_encoder.resize_token_embeddings(len(tokenizer))

    image_encoder = CTViT(
        dim=512,
        codebook_size=8192,
        image_size=480,
        patch_size=20,
        temporal_patch_size=10,
        spatial_depth=4,
        temporal_depth=4,
        dim_head=32,
        heads=8,
    )

    model = CTCLIP(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        tokenizer_name_or_path=str(text_encoder_source),
        dim_image=294912,
        dim_text=768,
        dim_latent=512,
        extra_latent_projection=False,
        use_mlm=False,
        downsample_image_embeds=False,
        use_all_token_embeds=False,
    )

    model.load(str(checkpoint))
    model = model.to(device)
    model.eval()
    return model, tokenizer


def predict_case(
    model,
    tokenizer: BertTokenizer,
    volume_tensor: torch.Tensor,
    device: torch.device,
    abnormalities: Sequence[str],
) -> Dict[str, float]:
    volume_tensor = volume_tensor.to(device)
    outputs: Dict[str, float] = {}
    with torch.no_grad():
        for abnormality in abnormalities:
            texts = [f"{abnormality} is present.", f"{abnormality} is not present."]
            tokens = tokenizer(
                texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512,
            ).to(device)
            logits = model(tokens, volume_tensor, device=device)
            outputs[abnormality] = softmax_positive(logits)
    return outputs


def binary_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return {
        "count": len(y_true),
        "positives": int(sum(y_true)),
        "predicted_positives": int(sum(y_pred)),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


def rankdata(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def roc_auc_score_manual(y_true: Sequence[int], y_score: Sequence[float]) -> float | None:
    positives = sum(y_true)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return None

    ranks = rankdata(y_score)
    pos_rank_sum = sum(rank for rank, truth in zip(ranks, y_true) if truth == 1)
    auc = (pos_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def write_predictions_csv(rows: List[dict], output_path: Path, abnormalities: Sequence[str]) -> None:
    fieldnames = ["case_id", "source", "volume_path"]
    for abnormality in abnormalities:
        safe = abnormality.lower().replace(" ", "_")
        fieldnames.extend([f"{safe}_gt", f"{safe}_prob"])

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a pretrained zero-shot CT-CLIP model on CT abnormalities.")
    parser.add_argument("--dataset", choices=["ct-rate", "rexgrounding-ct"], required=True)
    parser.add_argument("--volume-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "ctclip_abnormality_eval")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ct-rate-labels-csv", type=Path, default=ROOT / "data" / "ct-rate" / "valid_labels.csv")
    parser.add_argument("--rex-metadata-json", type=Path, default=ROOT / "data" / "Govindu" / "rexgrounding-ct" / "dataset.json")
    parser.add_argument("--text-encoder-dir", type=Path, default=DEFAULT_BIOMEDVLP_DIR)
    parser.add_argument("--abnormalities", nargs="*", default=list(ABNORMALITIES.keys()))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_cpu_threads(args.cpu_threads)
    device = get_device(args.device)
    abnormalities = [name for name in args.abnormalities if name in ABNORMALITIES]
    if not abnormalities:
        raise ValueError("No supported abnormalities selected.")

    if args.dataset == "ct-rate":
        cases = read_ct_rate_cases(args.volume_root, args.ct_rate_labels_csv, limit=args.limit)
    else:
        cases = read_rex_cases(args.volume_root, args.rex_metadata_json, limit=args.limit)

    print(f"Resolved cases: {len(cases)}")
    if cases:
        print(f"First case: {cases[0].case_id}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        manifest = {
            "dataset": args.dataset,
            "resolved_cases": len(cases),
            "abnormalities": abnormalities,
            "sample_case_ids": [case.case_id for case in cases[:10]],
        }
        manifest_path = args.output_dir / "dry_run_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Wrote dry-run manifest: {manifest_path}")
        return

    if args.checkpoint is None:
        raise ValueError("--checkpoint is required unless --dry-run is used.")

    model, tokenizer = build_zero_shot_model(device, args.checkpoint, args.text_encoder_dir)

    rows: List[dict] = []
    case_predictions: Dict[str, Dict[str, List[float]]] = {abn: {"gt": [], "prob": []} for abn in abnormalities}

    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.case_id}", flush=True)
        volume_tensor = load_and_preprocess_ct(case.volume_path)
        probs = predict_case(model, tokenizer, volume_tensor, device, abnormalities)

        row = {
            "case_id": case.case_id,
            "source": case.source,
            "volume_path": str(case.volume_path),
        }
        for abnormality in abnormalities:
            safe = abnormality.lower().replace(" ", "_")
            gt = int(case.labels.get(abnormality, 0))
            prob = float(probs[abnormality])
            row[f"{safe}_gt"] = gt
            row[f"{safe}_prob"] = prob
            case_predictions[abnormality]["gt"].append(gt)
            case_predictions[abnormality]["prob"].append(prob)
        rows.append(row)

    predictions_path = args.output_dir / "predictions.csv"
    write_predictions_csv(rows, predictions_path, abnormalities)

    summary = {
        "dataset": args.dataset,
        "num_cases": len(rows),
        "threshold": args.threshold,
        "abnormalities": {},
    }

    for abnormality in abnormalities:
        gt = case_predictions[abnormality]["gt"]
        prob = case_predictions[abnormality]["prob"]
        pred = [1 if value >= args.threshold else 0 for value in prob]
        metrics = binary_metrics(gt, pred)
        auc = roc_auc_score_manual(gt, prob)
        metrics["roc_auc"] = auc
        summary["abnormalities"][abnormality] = metrics

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote predictions: {predictions_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
