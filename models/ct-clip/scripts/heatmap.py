import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

# ============================================================
# Paths
# ============================================================

ROOT = Path("/home/chest_ct/code/models/ct-clip")
CT_CLIP_DIR = ROOT / "CT_CLIP"
TRANSFORMER_MASKGIT_DIR = ROOT / "transformer_maskgit"

sys.path.insert(0, str(CT_CLIP_DIR))
sys.path.insert(0, str(TRANSFORMER_MASKGIT_DIR))

from transformers import BertTokenizer, BertModel
from transformer_maskgit import CTViT
from ct_clip import CTCLIP

DEVICE = "cuda"
CHECKPOINT = str(ROOT / "checkpointlipro" / "CT_LiPro_v2.pt")

CT_PATH = "/home/chest_ct/code/data/data_volumes/dataset/train_fixed/train_1387_a_2.nii.gz"
GT_PATH = "/home/chest_ct/code/data/segmentations/segmentations/train_1387_a_2.nii.gz"

PATHOLOGIES = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification",
    "Hiatal hernia", "Lymphadenopathy", "Emphysema", "Atelectasis",
    "Lung nodule", "Lung opacity", "Pulmonary fibrotic sequela",
    "Pleural effusion", "Mosaic attenuation pattern",
    "Peribronchial thickening", "Consolidation", "Bronchiectasis",
    "Interlobular septal thickening"
]
TARGET_CLASS = "Lung nodule"
TARGET_IDX = PATHOLOGIES.index(TARGET_CLASS)


# ============================================================
# Model
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


def build_model():
    tokenizer = BertTokenizer.from_pretrained(
        "microsoft/BiomedVLP-CXR-BERT-specialized", do_lower_case=True
    )
    text_encoder = BertModel.from_pretrained(
        "microsoft/BiomedVLP-CXR-BERT-specialized"
    )
    text_encoder.resize_token_embeddings(len(tokenizer))

    image_encoder = CTViT(
        dim=512, codebook_size=8192, image_size=480, patch_size=20,
        temporal_patch_size=10, spatial_depth=4, temporal_depth=4,
        dim_head=32, heads=8
    )

    clip = CTCLIP(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        dim_image=294912,
        dim_text=768,
        dim_latent=512,
        extra_latent_projection=False,
        use_mlm=False,
        downsample_image_embeds=False,
        use_all_token_embeds=False
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
# Preprocessing (identical to your inference script)
# ============================================================

def preprocess_ct(path):
    nii = nib.load(str(path))
    img = nii.get_fdata()

    img = np.clip(img, -1000, 1000)
    img = img / 1000.0

    tensor = torch.tensor(img, dtype=torch.float32)
    tensor = tensor.permute(2, 0, 1)           # X,Y,Z -> Z,X,Y
    tensor = tensor.unsqueeze(0).unsqueeze(0)   # (1,1,Z,X,Y)

    tensor = F.interpolate(
        tensor, size=(240, 480, 480), mode="trilinear", align_corners=False
    )
    return tensor  # (1,1,240,480,480)


# ============================================================
# Load CT + GT
# ============================================================

ct_nii = nib.load(CT_PATH)
gt_nii = nib.load(GT_PATH)

ct = ct_nii.get_fdata()
gt = gt_nii.get_fdata()
if gt.ndim == 4:
    gt = gt[0]
gt_mask = gt > 0

print("CT shape :", ct.shape)
print("GT shape :", gt_mask.shape)
print("GT voxels:", gt_mask.sum())


# ============================================================
# Build model + hooks
# ============================================================

model, tokenizer = build_model()

activations = {}
gradients = {}

target_layer = model.trained_model.image_encoder.enc_temporal_transformer.norm_out

def forward_hook(module, inp, out):
    activations["value"] = out

def backward_hook(module, grad_in, grad_out):
    gradients["value"] = grad_out[0]

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)


# ============================================================
# Forward + backward pass (grad enabled — no torch.no_grad())
# ============================================================

image_tensor = preprocess_ct(CT_PATH).to(DEVICE)
image_tensor.requires_grad_(True)

text_tokens = tokenizer(
    "", return_tensors="pt", padding="max_length", truncation=True, max_length=200
).to(DEVICE)

logits = model(text_tokens, image_tensor, DEVICE)
print("\nLogits shape:", logits.shape)

act = activations["value"]
print("Activation shape:", act.shape)

model.zero_grad()
score = logits[0, TARGET_IDX]
score.backward()

grad = gradients["value"]
print("Gradient shape:", grad.shape)
print(f"Target: {TARGET_CLASS} (idx {TARGET_IDX}), logit = {score.item():.4f}")


# ============================================================
# Grad-CAM weighting (auto-detect token grid)
# ============================================================

B, N, C = act.shape
weights = grad.mean(dim=1)                     # (B, C) — GAP over tokens
cam_flat = (act * weights.unsqueeze(1)).sum(-1)  # (B, N)
cam_flat = F.relu(cam_flat)[0]                   # (N,)
cam_flat = cam_flat.detach().cpu().numpy()

SPATIAL = 24  # image_size=480 / patch_size=20

if N == SPATIAL * SPATIAL:
    # temporal dim already pooled away -> only a 2D (H,W) map
    print(f"\nToken count {N} = {SPATIAL}x{SPATIAL} (no depth dimension resolved).")
    cam_2d = cam_flat.reshape(SPATIAL, SPATIAL)
    cam = np.repeat(cam_2d[None, :, :], SPATIAL, axis=0)  # broadcast across depth
elif N == SPATIAL ** 3:
    print(f"\nToken count {N} = {SPATIAL}^3 (T,H,W all present).")
    cam = cam_flat.reshape(SPATIAL, SPATIAL, SPATIAL)  # order: (T,H,W) — verify visually below
else:
    raise ValueError(
        f"Unexpected token count N={N} (activation shape {act.shape}) — "
        f"doesn't match {SPATIAL}x{SPATIAL} or {SPATIAL}^3. Paste this shape back."
    )

print("CAM grid shape:", cam.shape)
print("Range:", cam.min(), cam.max(), "Std:", cam.std())


# ============================================================
# Resize CAM to full CT resolution
# ============================================================

cam_full = zoom(
    cam,
    (
        ct.shape[0] / cam.shape[0],
        ct.shape[1] / cam.shape[1],
        ct.shape[2] / cam.shape[2],
    ),
    order=1,
)
print("Full CAM shape:", cam_full.shape)


# ============================================================
# Threshold sweep (on low-res CAM vs. downsampled GT)
# ============================================================

gt_small = zoom(
    gt_mask.astype(float),
    (
        cam.shape[0] / gt_mask.shape[0],
        cam.shape[1] / gt_mask.shape[1],
        cam.shape[2] / gt_mask.shape[2],
    ),
    order=0,
) > 0.5

best_dice, best_threshold = 0, 0
print("\nThreshold evaluation")
for p in [70, 75, 80, 85, 90, 95]:
    threshold = np.percentile(cam, p)
    cam_mask = cam >= threshold
    intersection = np.logical_and(cam_mask, gt_small).sum()
    dice = 2 * intersection / (cam_mask.sum() + gt_small.sum() + 1e-8)
    print(f"{p}% Dice = {dice:.4f}")
    if dice > best_dice:
        best_dice, best_threshold = dice, threshold

print("\nBest threshold:", best_threshold, "| Best low-res Dice:", best_dice)


# ============================================================
# Full-resolution metrics
# ============================================================

cam_mask_full = cam_full >= best_threshold
tp = np.logical_and(cam_mask_full, gt_mask).sum()
fp = np.logical_and(cam_mask_full, ~gt_mask).sum()
fn = np.logical_and(~cam_mask_full, gt_mask).sum()
union = np.logical_or(cam_mask_full, gt_mask).sum()

dice = 2 * tp / (cam_mask_full.sum() + gt_mask.sum() + 1e-8)
iou = tp / (union + 1e-8)
precision = tp / (tp + fp + 1e-8)
recall = tp / (tp + fn + 1e-8)

print("\n==============================")
print("CT-CLIP Grad-CAM Evaluation")
print("==============================")
print(f"Dice      : {dice:.4f}")
print(f"IoU       : {iou:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")


# ============================================================
# Save + visualize
# ============================================================

nib.save(nib.Nifti1Image(cam_full.astype(np.float32), ct_nii.affine),
          "train_1387_a_2_ctclip_lung_nodule_gradcam.nii.gz")

slice_idx = np.argmax(gt_mask.sum(axis=(0, 1)))

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(ct[:, :, slice_idx], cmap="gray")
plt.title(f"CT Slice {slice_idx}")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(ct[:, :, slice_idx], cmap="gray")
plt.imshow(np.ma.masked_where(gt_mask[:, :, slice_idx] == 0, gt_mask[:, :, slice_idx]),
           cmap="Reds", alpha=0.6)
plt.title("Ground Truth")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(ct[:, :, slice_idx], cmap="gray")
plt.imshow(cam_full[:, :, slice_idx], cmap="jet", alpha=0.5)
plt.title(f"CT-CLIP Grad-CAM\nDice={dice:.4f}")
plt.axis("off")

plt.tight_layout()
plt.show()