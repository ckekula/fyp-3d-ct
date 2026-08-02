import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import nibabel as nib

from pathlib import Path
from scipy.ndimage import zoom

from transformers import BertTokenizer, BertModel

ROOT = Path(__file__).resolve().parents[1]
CT_CLIP_DIR = ROOT / "CT_CLIP"
TRANSFORMER_MASKGIT_DIR = ROOT / "transformer_maskgit"
sys.path.insert(0, str(CT_CLIP_DIR))
sys.path.insert(0, str(TRANSFORMER_MASKGIT_DIR))

from transformer_maskgit import CTViT
from ct_clip import CTCLIP

from data import resize_array

DATA_ROOT = Path("/home/chest_ct/code/data")
DATASET_JSON = DATA_ROOT / "rexgrounding-ct/dataset_2.json"
VOLUME_DIR = DATA_ROOT / "data_volumes/dataset/train_fixed"
SEG_DIR = DATA_ROOT / "segmentations/segmentations"
TRAIN_META = DATA_ROOT / "ct-rate/train_metadata.csv"
VALID_META = DATA_ROOT / "ct-rate/valid_metadata.csv"
CHECKPOINT = "/home/chest_ct/code/models/ct-clip/checkpointlipro/CT_LiPro_v2.pt"
OUT_DIR = ROOT / "rexground_predictions/pilot"
DEVICE = "cuda"

PATHOLOGIES = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia",
    "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule", "Lung opacity",
    "Pulmonary fibrotic sequela", "Pleural effusion", "Mosaic attenuation pattern",
    "Peribronchial thickening", "Consolidation", "Bronchiectasis",
    "Interlobular septal thickening",
]
LUNG_NODULE_IDX = PATHOLOGIES.index("Lung nodule")
LUNG_OPACITY_IDX = PATHOLOGIES.index("Lung opacity")

TARGET_SPACING = (1.5, 0.75, 0.75)  # z, x, y
TARGET_SHAPE = (480, 480, 240)  # h, w, d


# ============================================================
# Case selection
# ============================================================

def select_pilot_cases():
    with open(DATASET_JSON) as f:
        data = json.load(f)

    available = set(os.listdir(VOLUME_DIR))
    seg_available = set(os.listdir(SEG_DIR))
    meta_set = set(pd.read_csv(TRAIN_META)["VolumeName"]) | set(pd.read_csv(VALID_META)["VolumeName"])

    nodule_cases, ggo_cases = [], []
    for split_name, cases in data.items():
        for c in cases:
            name = c.get("name")
            if not name or name not in available or name not in seg_available or name not in meta_set:
                continue
            cats = set(c.get("categories", {}).values())
            px = sum(c.get("pixels", {}).values())
            entry = {"split": split_name, "name": name, "pixels": px, "case": c}
            if cats == {"2d"} and len(c.get("findings", {})) <= 2 and 300 <= px <= 3000:
                nodule_cases.append(entry)
            elif cats == {"2c"} and len(c.get("findings", {})) <= 1 and 20000 <= px <= 200000:
                ggo_cases.append(entry)

    nodule_cases.sort(key=lambda e: -e["pixels"])
    ggo_cases.sort(key=lambda e: -e["pixels"])
    return nodule_cases[:10], ggo_cases[:10]


# ============================================================
# Classifier (identical to ct_lipro_inference.py)
# ============================================================

class ImageLatentsClassifier(nn.Module):
    def __init__(self, trained_model, latent_dim, num_classes):
        super().__init__()
        self.trained_model = trained_model
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, text_tokens, image, device):
        _, image_latents, _ = self.trained_model(
            text_tokens, image, device=device, return_latents=True
        )
        image_latents = self.relu(image_latents)
        return self.classifier(image_latents)


def load_ctclip_classifier():
    tokenizer = BertTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized", do_lower_case=True)
    text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
    text_encoder.resize_token_embeddings(len(tokenizer))

    image_encoder = CTViT(
        dim=512, codebook_size=8192, image_size=480, patch_size=20,
        temporal_patch_size=10, spatial_depth=4, temporal_depth=4,
        dim_head=32, heads=8,
    )
    clip = CTCLIP(
        image_encoder=image_encoder, text_encoder=text_encoder,
        dim_image=294912, dim_text=768, dim_latent=512,
        extra_latent_projection=False, use_mlm=False,
        downsample_image_embeds=False, use_all_token_embeds=False,
    )
    model = ImageLatentsClassifier(clip, 512, len(PATHOLOGIES))

    state = torch.load(CHECKPOINT, map_location="cpu")
    if isinstance(state, dict):
        for k in ["state_dict", "model_state_dict", "weights"]:
            if k in state:
                state = state[k]
                break
    model.load_state_dict(state, strict=False)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


# ============================================================
# Official spacing-aware preprocessing (matches data.py)
# ============================================================

def preprocess_ct(path, meta_row):
    nii = nib.load(str(path))
    img_data = nii.get_fdata()

    xy_spacing = float(meta_row["XYSpacing"][1:][:-2].split(",")[0])
    z_spacing = float(meta_row["ZSpacing"])

    # NOTE: do NOT apply RescaleSlope/RescaleIntercept here. train_metadata.csv's
    # slope/intercept describe the *original* DICOM series, but files under
    # data_volumes/dataset/train_fixed/ already store calibrated HU directly
    # (get_fdata() already returns values like median~-800, 95th pct~+90 --
    # textbook lung/soft-tissue HU). Re-applying slope*x+intercept (typically
    # 1*x-1024) double-rescales and clips ~78% of every voxel to the -1000
    # floor, destroying virtually all lung/soft-tissue signal and leaving only
    # bone -- confirmed by inspecting raw vs rescaled percentiles directly, and
    # by a 2004-case AUROC collapse to chance (0.50) that recovers once this is
    # removed. This matches data.py/data_inference_nii.py's rescale step being
    # designed for raw DICOM-derived NIfTI, not this already-corrected folder.
    #
    # NOTE: clip BEFORE resampling, not after -- keeps out-of-FOV sentinel HU
    # values (commonly -2000..-3024 at the circular scan-field edge) from being
    # trilinear-blended into real tissue during resampling.
    hu_min, hu_max = -1000, 1000
    img_data = np.clip(img_data, hu_min, hu_max)
    img_data = img_data.transpose(2, 0, 1)  # Z, X, Y

    tensor = torch.tensor(img_data).unsqueeze(0).unsqueeze(0)
    current = (z_spacing, xy_spacing, xy_spacing)
    img_data = resize_array(tensor, current, TARGET_SPACING)
    img_data = img_data[0][0]
    img_data = np.transpose(img_data, (1, 2, 0))  # back to X, Y, Z

    img_data = (img_data / 1000.0).astype(np.float32)

    tensor = torch.tensor(img_data)
    h, w, d = tensor.shape
    dh, dw, dd = TARGET_SHAPE

    h_start = max((h - dh) // 2, 0)
    w_start = max((w - dw) // 2, 0)
    d_start = max((d - dd) // 2, 0)
    tensor = tensor[h_start:h_start + dh, w_start:w_start + dw, d_start:d_start + dd]

    pad_h = dh - tensor.size(0)
    pad_w = dw - tensor.size(1)
    pad_d = dd - tensor.size(2)
    tensor = F.pad(
        tensor,
        (pad_d // 2, pad_d - pad_d // 2, pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
        value=-1,
    )

    tensor = tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # (1, 1, 240, 480, 480)

    crop_info = {
        "h_start": h_start, "w_start": w_start, "d_start": d_start,
        "pad_h_before": pad_h // 2, "pad_w_before": pad_w // 2, "pad_d_before": pad_d // 2,
        "orig_shape": (h, w, d),
    }
    return tensor, crop_info


# ============================================================
# Grad-CAM (with the VQ straight-through fix)
# ============================================================

def run_gradcam(model, tokenizer, image_tensor, target_idx):
    device = next(model.parameters()).device
    image_tensor = image_tensor.clone().to(device)
    image_tensor.requires_grad_(True)

    text_tokens_empty = tokenizer(
        "", return_tensors="pt", padding="max_length", truncation=True, max_length=200
    ).to(device)

    vq = model.trained_model.visual_transformer.vq
    vq.training = True
    vq._codebook.training = False

    target_layer = model.trained_model.visual_transformer

    activations, gradients = {}, {}

    def _fwd_hook(module, inp, output):
        activations["value"] = output
        output.register_hook(lambda grad: gradients.update({"value": grad}))

    handle = target_layer.register_forward_hook(_fwd_hook)

    model.zero_grad()
    logits = model(text_tokens_empty, image_tensor, device)
    score = logits[:, target_idx]
    score.sum().backward()
    handle.remove()

    acts = activations["value"].permute(0, 4, 1, 2, 3)
    grads = gradients["value"].permute(0, 4, 1, 2, 3)
    weights = grads.mean(dim=(2, 3, 4), keepdim=True)
    cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=image_tensor.shape[2:], mode="trilinear", align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    probs = torch.sigmoid(logits.detach())[0].cpu().numpy()
    return cam, probs


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    nodule_cases, ggo_cases = select_pilot_cases()
    pilot = [(e, "nodule") for e in nodule_cases] + [(e, "ggo") for e in ggo_cases]
    print(f"Pilot set: {len(nodule_cases)} nodule cases, {len(ggo_cases)} GGO cases")

    train_meta = pd.read_csv(TRAIN_META).set_index("VolumeName")
    valid_meta = pd.read_csv(VALID_META).set_index("VolumeName")

    model, tokenizer = load_ctclip_classifier()
    device = next(model.parameters()).device

    results = []
    for entry, kind in pilot:
        name = entry["name"]
        target_idx = LUNG_NODULE_IDX if kind == "nodule" else LUNG_OPACITY_IDX

        meta_row = train_meta.loc[name] if name in train_meta.index else valid_meta.loc[name]
        volume_path = VOLUME_DIR / name

        try:
            image_tensor, crop_info = preprocess_ct(volume_path, meta_row)
            image_tensor = image_tensor.to(device)

            text_tokens_empty = tokenizer(
                "", return_tensors="pt", padding="max_length", truncation=True, max_length=200
            ).to(device)
            with torch.no_grad():
                logits = model(text_tokens_empty, image_tensor, device)
                probs = torch.sigmoid(logits)[0].cpu().numpy()

            findings_text = "; ".join(entry["case"].get("findings", {}).values())
            result = {
                "name": name, "split": entry["split"], "kind": kind,
                "pixels": entry["pixels"], "findings": findings_text,
                "prob_lung_nodule": float(probs[LUNG_NODULE_IDX]),
                "prob_lung_opacity": float(probs[LUNG_OPACITY_IDX]),
                "target_prob": float(probs[target_idx]),
                "predicted_positive": bool(probs[target_idx] >= 0.5),
            }
            results.append(result)
            print(f"[{kind:6s}] {name:30s} target_prob={result['target_prob']:.3f} "
                  f"nodule={result['prob_lung_nodule']:.3f} opacity={result['prob_lung_opacity']:.3f}")
        except Exception as e:
            print(f"FAILED {name}: {e}")
            results.append({"name": name, "split": entry["split"], "kind": kind, "error": str(e)})

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "predictions.csv", index=False)
    print(f"\nSaved predictions to {OUT_DIR / 'predictions.csv'}")

    # Select "best" cases per kind: highest-confidence correct predictions
    valid_df = df[df.get("error").isna()] if "error" in df.columns else df
    gradcam_targets = []
    for kind in ["nodule", "ggo"]:
        sub = valid_df[(valid_df["kind"] == kind) & (valid_df["predicted_positive"])]
        sub = sub.sort_values("target_prob", ascending=False)
        gradcam_targets.extend(sub.head(2).to_dict("records"))

    print(f"\nRunning Grad-CAM on {len(gradcam_targets)} best cases...")
    for rec in gradcam_targets:
        name = rec["name"]
        kind = rec["kind"]
        target_idx = LUNG_NODULE_IDX if kind == "nodule" else LUNG_OPACITY_IDX
        meta_row = train_meta.loc[name] if name in train_meta.index else valid_meta.loc[name]
        volume_path = VOLUME_DIR / name

        image_tensor, crop_info = preprocess_ct(volume_path, meta_row)
        cam, probs = run_gradcam(model, tokenizer, image_tensor, target_idx)

        # save CAM as npy (avoids re-deriving affine bookkeeping for this pilot)
        out_path = OUT_DIR / f"gradcam_{kind}_{name.replace('.nii.gz', '')}.npy"
        np.save(out_path, cam.astype(np.float32))

        # quick overlap check against the segmentation mask, in the same
        # cropped/padded/resampled space as the CAM
        seg_nii = nib.load(str(SEG_DIR / name))
        seg = seg_nii.get_fdata()
        if seg.ndim == 4:
            seg = seg[0]
        seg_mask = seg > 0

        # resample seg to the same spacing/shape pipeline as the image
        # (seg is X,Y,Z like the raw CT before transpose(2,0,1) -> match that transform)
        seg_zxy = np.transpose(seg_mask.astype(np.float32), (2, 0, 1))
        seg_t = torch.tensor(seg_zxy).unsqueeze(0).unsqueeze(0)
        h, w, d = crop_info["orig_shape"]
        xy_spacing = float(meta_row["XYSpacing"][1:][:-1].split(",")[0])
        z_spacing = float(meta_row["ZSpacing"])
        seg_resized = resize_array(seg_t, (z_spacing, xy_spacing, xy_spacing), TARGET_SPACING)[0][0]
        seg_resized = np.transpose(seg_resized, (1, 2, 0))  # X,Y,Z

        dh, dw, dd = TARGET_SHAPE
        hs, ws, ds = crop_info["h_start"], crop_info["w_start"], crop_info["d_start"]
        seg_cropped = seg_resized[hs:hs + dh, ws:ws + dw, ds:ds + dd]
        seg_padded = np.pad(
            seg_cropped,
            ((crop_info["pad_h_before"], dh - seg_cropped.shape[0] - crop_info["pad_h_before"]),
             (crop_info["pad_w_before"], dw - seg_cropped.shape[1] - crop_info["pad_w_before"]),
             (crop_info["pad_d_before"], dd - seg_cropped.shape[2] - crop_info["pad_d_before"])),
            mode="constant", constant_values=0,
        )
        # cam is in (Z,X,Y) order coming out of the model (matches tensor.permute(2,0,1) in preprocess)
        cam_xyz = np.transpose(cam, (1, 2, 0))
        gt_mask = seg_padded > 0.5

        best_dice, best_p = 0.0, 0
        for p in [70, 80, 90, 95]:
            thr = np.percentile(cam_xyz, p)
            cam_mask = cam_xyz >= thr
            inter = np.logical_and(cam_mask, gt_mask).sum()
            dice = 2 * inter / (cam_mask.sum() + gt_mask.sum() + 1e-8)
            if dice > best_dice:
                best_dice, best_p = dice, p

        max_point = np.unravel_index(np.argmax(cam_xyz), cam_xyz.shape)
        pointing_hit = bool(gt_mask[max_point]) if gt_mask.sum() > 0 else None

        print(f"  {kind:6s} {name:30s} prob={rec['target_prob']:.3f} "
              f"best_dice@{best_p}%={best_dice:.4f} pointing_hit={pointing_hit}")

        rec["gradcam_npy"] = str(out_path)
        rec["gradcam_best_dice"] = best_dice
        rec["gradcam_pointing_hit"] = pointing_hit

    gradcam_df = pd.DataFrame(gradcam_targets)
    gradcam_df.to_csv(OUT_DIR / "gradcam_summary.csv", index=False)
    print(f"\nSaved Grad-CAM summary to {OUT_DIR / 'gradcam_summary.csv'}")


if __name__ == "__main__":
    main()
