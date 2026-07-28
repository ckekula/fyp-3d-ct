import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.ndimage import zoom

from transformers import BertTokenizer, BertModel


# ============================================================
# Paths — same as your inference script
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
CT_CLIP_DIR = ROOT / "CT_CLIP"
TRANSFORMER_MASKGIT_DIR = ROOT / "transformer_maskgit"

sys.path.insert(0, str(CT_CLIP_DIR))
sys.path.insert(0, str(TRANSFORMER_MASKGIT_DIR))

from transformer_maskgit import CTViT
from ct_clip import CTCLIP

# NOTE: your existing GradCAM3D (used for Merlin/i3_resnet) assumes a
# conv-style (b, c, d, h, w) activation. CTViT's return_encoded_tokens=True
# output is (b, t, h, w, d) -- channels LAST, and needs a reshape before that
# code's channel-weighting math is valid. Rather than patch the shared
# library, this script does the Grad-CAM math manually below.


ct_path = "/home/chest_ct/code/data/data_volumes/dataset/train_fixed/train_1387_a_2.nii.gz"
gt_path = "/home/chest_ct/code/data/segmentations/segmentations/train_1387_a_2.nii.gz"

CHECKPOINT = "/home/chest_ct/code/models/ct-clip/checkpointlipro/CT_LiPro_v2.pt"
DEVICE = "cuda"

PATHOLOGIES = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia",
    "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule", "Lung opacity",
    "Pulmonary fibrotic sequela", "Pleural effusion", "Mosaic attenuation pattern",
    "Peribronchial thickening", "Consolidation", "Bronchiectasis",
    "Interlobular septal thickening",
]
LUNG_NODULE_IDX = PATHOLOGIES.index("Lung nodule")  # 9

assert os.path.exists(ct_path), "CT not found"
assert os.path.exists(gt_path), "GT not found"


# ============================================================
# Load CT + Ground Truth
# ============================================================

ct_nii = nib.load(ct_path)
gt_nii = nib.load(gt_path)

ct = ct_nii.get_fdata()
gt = gt_nii.get_fdata()

if gt.ndim == 4:
    gt = gt[0]

gt_mask = gt > 0

print("CT shape :", ct.shape)
print("GT shape :", gt_mask.shape)
print("GT voxels:", gt_mask.sum())


# ============================================================
# Classifier — identical to your inference script
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
    model.eval()  # eval() is fine — just don't wrap forward calls in torch.no_grad()
    return model, tokenizer


model, tokenizer = load_ctclip_classifier()
device = next(model.parameters()).device
print("CT-CLIP device:", device)

# RESOLVED: CTViT's forward(..., return_encoded_tokens=True) already returns
# the last spatial feature map (right after VQ, before decode), shaped
# (b, t, h, w, d) -- so hooking the whole module is correct. The only thing
# needed is reshaping (b, t, h, w, d) -> (b, d, t, h, w) before the
# Grad-CAM channel-weighting math (which expects channels first).
target_layer = model.trained_model.visual_transformer


# ============================================================
# Image forward for Grad-CAM
# CT-CLIP's classifier target is a disease logit, not an image-text
# similarity score — so this returns classifier logits directly, and
# ClassifierTarget below just slices out the class of interest.
# ============================================================

text_tokens_empty = tokenizer(
    "", return_tensors="pt", padding="max_length", truncation=True, max_length=200
).to(device)


def image_forward(x):
    return model(text_tokens_empty, x, device)


# ============================================================
# Classifier target (replaces Merlin's TextSimilarityTarget)
# ============================================================

class ClassifierTarget:
    def __init__(self, class_idx):
        self.class_idx = class_idx

    def __call__(self, logits):
        return logits[:, self.class_idx]


# ============================================================
# Preprocess CT — identical to ReXGroundDataset.preprocess
# ============================================================

def preprocess_ct(path):
    nii = nib.load(str(path))
    img = nii.get_fdata()
    img = np.clip(img, -1000, 1000)
    img = img / 1000.0
    tensor = torch.tensor(img, dtype=torch.float32)
    tensor = tensor.permute(2, 0, 1)  # X,Y,Z -> Z,X,Y
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(240, 480, 480), mode="trilinear", align_corners=False)
    return tensor  # (1, 1, 240, 480, 480)


image_tensor = preprocess_ct(ct_path).to(device)
image_tensor.requires_grad_(True)  # gradients must flow — no torch.no_grad() anywhere above this

print("\nInput tensor:")
print(image_tensor.shape)


# ============================================================
# Verify forward pass (sanity check, like Merlin's embedding check)
# ============================================================

logits_test = image_forward(image_tensor)
probs_test = torch.sigmoid(logits_test)[0]
print("\nLung nodule probability:", probs_test[LUNG_NODULE_IDX].item())


# ============================================================
# Grad-CAM (manual — CTViT's token-shaped output needs a reshape
# that the shared conv-style GradCAM3D doesn't do)
# ============================================================

_activations = {}
_gradients = {}

def _fwd_hook(module, input, output):
    # output: (b, t, h, w, d) from return_encoded_tokens=True
    _activations["value"] = output

def _bwd_hook(module, grad_input, grad_output):
    # grad_output[0] matches the (b, t, h, w, d) shape of the forward output
    _gradients["value"] = grad_output[0]

h1 = target_layer.register_forward_hook(_fwd_hook)
h2 = target_layer.register_full_backward_hook(_bwd_hook)

model.zero_grad()
target = ClassifierTarget(LUNG_NODULE_IDX)
logits = image_forward(image_tensor)
score = target(logits)
score.sum().backward()

h1.remove()
h2.remove()

acts = _activations["value"]      # (b, t, h, w, d)
grads = _gradients["value"]       # (b, t, h, w, d)
print("Activation shape:", acts.shape)  # sanity check — should be (1, t, h, w, 512)

# channels-last -> channels-first: (b, t, h, w, d) -> (b, d, t, h, w)
acts = acts.permute(0, 4, 1, 2, 3)
grads = grads.permute(0, 4, 1, 2, 3)

# standard Grad-CAM: global-average-pool gradients over spatial dims -> channel weights
weights = grads.mean(dim=(2, 3, 4), keepdim=True)
cam = F.relu((weights * acts).sum(dim=1, keepdim=True))  # (b, 1, t, h, w)

# upsample from CTViT's small token grid to the full preprocessed volume size
cam = F.interpolate(cam, size=image_tensor.shape[2:], mode="trilinear", align_corners=False)
cam = cam.squeeze().detach().cpu().numpy()
cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

# FIX: preprocessing did tensor.permute(2,0,1), i.e. (X,Y,Z) -> (Z,X,Y),
# so the CAM comes out in (Z,X,Y) order. ct.shape and gt_mask are still
# (X,Y,Z). Transpose back BEFORE any resize/comparison against them,
# or every downstream Dice/IoU/pointing-game number is comparing
# mismatched axes and is meaningless.
cam = np.transpose(cam, (1, 2, 0))  # (Z,X,Y) -> (X,Y,Z)

print("\nCAM:")
print(cam.shape)
print("Range:", cam.min(), cam.max())


# ============================================================
# Resize GT to CAM resolution
# ============================================================

gt_small = zoom(
    gt_mask.astype(float),
    (cam.shape[0] / gt_mask.shape[0], cam.shape[1] / gt_mask.shape[1], cam.shape[2] / gt_mask.shape[2]),
    order=0
)
gt_small = gt_small > 0.5

print("\nCAM resolution comparison")
print("CAM:", cam.shape)
print("GT:", gt_small.shape)


# ============================================================
# Threshold sweep — identical to Merlin script
# ============================================================

print("\nThreshold evaluation")
print("--------------------")

best_dice = 0
best_threshold = 0

for p in [70, 75, 80, 85, 90, 95]:
    threshold = np.percentile(cam, p)
    cam_mask = cam >= threshold
    intersection = np.logical_and(cam_mask, gt_small).sum()
    dice = 2 * intersection / (cam_mask.sum() + gt_small.sum() + 1e-8)
    print(f"{p}% Dice = {dice:.4f}")
    if dice > best_dice:
        best_dice = dice
        best_threshold = threshold

print("\nBest CAM threshold:", best_threshold)
print("Best low-res Dice:", best_dice)


# ============================================================
# Pointing accuracy
# ============================================================

max_point = np.unravel_index(np.argmax(cam), cam.shape)
hit = gt_small[max_point]
print("\nPointing accuracy:", bool(hit))


# ============================================================
# Resize CAM to full CT resolution
# ============================================================

cam_full = zoom(
    cam,
    (ct.shape[0] / cam.shape[0], ct.shape[1] / cam.shape[1], ct.shape[2] / cam.shape[2]),
    order=1
)
print("\nFull CAM:", cam_full.shape)


# ============================================================
# Save CAM NIfTI
# ============================================================

cam_nii = nib.Nifti1Image(cam_full.astype(np.float32), ct_nii.affine)
nib.save(cam_nii, "train_1387_a_2_lung_nodule_ctclip_gradcam.nii.gz")
print("Saved CAM NIfTI")


# ============================================================
# Full resolution metrics
# ============================================================

cam_mask_full = cam_full >= best_threshold

intersection = np.logical_and(cam_mask_full, gt_mask).sum()
union = np.logical_or(cam_mask_full, gt_mask).sum()

dice = 2 * intersection / (cam_mask_full.sum() + gt_mask.sum() + 1e-8)
iou = intersection / (union + 1e-8)

tp = np.logical_and(cam_mask_full, gt_mask).sum()
fp = np.logical_and(cam_mask_full, ~gt_mask).sum()
fn = np.logical_and(~cam_mask_full, gt_mask).sum()

precision = tp / (tp + fp + 1e-8)
recall = tp / (tp + fn + 1e-8)

print("\n==============================")
print("Final CT-CLIP Grad-CAM Evaluation")
print("==============================")
print(f"Dice      : {dice:.4f}")
print(f"IoU       : {iou:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")


# ============================================================
# Visualization — identical layout to Merlin script
# ============================================================

slice_idx = np.argmax(gt_mask.sum(axis=(0, 1)))

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(ct[:, :, slice_idx], cmap="gray")
plt.title(f"CT Slice {slice_idx}")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(ct[:, :, slice_idx], cmap="gray")
plt.imshow(np.ma.masked_where(gt_mask[:, :, slice_idx] == 0, gt_mask[:, :, slice_idx]), cmap="Reds", alpha=0.6)
plt.title("Ground Truth")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(ct[:, :, slice_idx], cmap="gray")
plt.imshow(cam_full[:, :, slice_idx], cmap="jet", alpha=0.5)
plt.title(f"CT-CLIP Grad-CAM\nDice={dice:.4f}")
plt.axis("off")

plt.tight_layout()
plt.show()