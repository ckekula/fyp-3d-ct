"""
CT-CLIP  –  ClassFine (CT_LiPro) Inference  –  Single File
============================================================
Model   : CT_LiPro_v2.pt  (ClassFine – linear classification head)
Pathologies detected (subset of the 18 CT-RATE labels):
  • Lung Nodule    (label index 9)
  • Lung Opacity   (label index 10)
  • Consolidation  (label index 15)
  • Atelectasis    (label index 8)

Source repo : github.com/ibrahimethemhamamci/CT-CLIP
Paper       : arXiv:2403.17834

── Install (run once, from the cloned CT-CLIP repo root) ────────────────────
  git clone https://github.com/ibrahimethemhamamci/CT-CLIP.git
  cd CT-CLIP/transformer_maskgit && pip install -e . && cd ..
  cd CT_CLIP && pip install -e . && cd ..
  pip install torch nibabel numpy einops transformers
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
import nibabel as nib
from pathlib import Path
from types import SimpleNamespace

# ═══════════════════════════════════════════════════════════════════════════
#  USER SETTINGS  –  only edit this block
# ═══════════════════════════════════════════════════════════════════════════

MODEL_CHECKPOINT = r"C:\Users\yasiru\Desktop\fyp-3d-ct\models\ct-clip\checkpoints_ctrate_4pathologies\CT_LiPro_v2.pt"

# Folder that holds one or more .nii.gz reconstructions for this patient scan
CT_SCAN_FOLDER = r"C:\Users\yasiru\Desktop\fyp-3d-ct\data\ct-rate\data_Volumes\train_fixed\train_81"

# If there are multiple .nii.gz files in the folder, pick which one to use:
#   "first"  -> use the first file found  (usually reconstruction _1)
#   "all"    -> run inference on every file and show results for each
MULTI_RECON_MODE = "first"

DEVICE = "cpu"   # force CPU on this machine

# ═══════════════════════════════════════════════════════════════════════════


# ---------------------------------------------------------------------------
#  CT-RATE label index map  (18 abnormalities, zero-indexed)
#  Source: CT-RATE dataset / repo label ordering
# ---------------------------------------------------------------------------
ALL_LABELS = [
    "Medical material",                    # 0
    "Arterial wall calcification",         # 1
    "Cardiomegaly",                        # 2
    "Pericardial effusion",                # 3
    "Coronary artery wall calcification",  # 4
    "Hiatal hernia",                       # 5
    "Lymphadenopathy",                     # 6
    "Emphysema",                           # 7
    "Atelectasis",                         # 8  <- target
    "Lung nodule",                         # 9  <- target
    "Lung opacity",                        # 10 <- target
    "Pulmonary fibrotic sequela",          # 11
    "Pleural effusion",                    # 12
    "Mosaic attenuation pattern",          # 13
    "Peribronchial thickening",            # 14
    "Consolidation",                       # 15 <- target
    "Bronchiectasis",                      # 16
    "Interstitial lung disease",           # 17
]

# Pathologies we care about -> {display name: label index}
TARGET_PATHOLOGIES = {
    "Lung Nodule":   9,
    "Lung Opacity":  10,
    "Consolidation": 15,
    "Atelectasis":   8,
}

NUM_CLASSES = len(ALL_LABELS)  # 18


# ---------------------------------------------------------------------------
#  CT volume pre-processing
#  Target shape used by CT-ViT in the CT-RATE repo: (240, 480, 480)
#  HU clipped to [-1000, 1000], then normalised to [0, 1].
# ---------------------------------------------------------------------------
TARGET_SHAPE = (240, 480, 480)   # (D, H, W)
HU_MIN, HU_MAX = -1000, 1000


def load_and_preprocess_ct(nifti_path: str) -> torch.Tensor:
    """
    Load a NIfTI CT, resize to TARGET_SHAPE, and return
    a float32 tensor  (1, 1, D, H, W)  ready for the vision encoder.
    """
    img  = nib.load(str(nifti_path))
    data = img.get_fdata(dtype=np.float32)   # nibabel returns (H, W, D)

    # Reorder axes to (D, H, W)
    data = np.transpose(data, (2, 0, 1))

    # Clip HU and normalise to [0, 1]
    data = np.clip(data, HU_MIN, HU_MAX)
    data = (data - HU_MIN) / (HU_MAX - HU_MIN)

    tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    tensor = F.interpolate(
        tensor,
        size=TARGET_SHAPE,
        mode="trilinear",
        align_corners=False,
    )
    return tensor.float()


# ---------------------------------------------------------------------------
#  Model loading
#  CT_LiPro_v2.pt is the ClassFine checkpoint:
#    - CT-ViT vision encoder  (same architecture as base CT-CLIP)
#    - A linear head: latent_dim (512) -> 18 classes
# ---------------------------------------------------------------------------
def load_model(checkpoint_path: str, device: str):
    """
    Build the CT-CLIP ClassFine model and load weights.
    Returns the model in eval mode.
    """
    try:
        from ct_clip import CTCLIP
        from transformer_maskgit import CTViT
        from transformers import BertTokenizer, BertModel
    except ImportError as e:
        raise ImportError(
            "\n[ERROR] CT-CLIP packages not installed.\n"
            "Run from the cloned CT-CLIP repo root:\n"
            "  cd transformer_maskgit && pip install -e . && cd ..\n"
            "  cd CT_CLIP && pip install -e .\n"
            f"Original error: {e}"
        )

    # -- Vision encoder (CT-ViT) -------------------------------------------
    image_encoder = CTViT(
        dim                 = 512,
        codebook_size       = 8192,
        image_size          = 480,
        patch_size          = 20,
        temporal_patch_size = 10,
        spatial_depth       = 4,
        temporal_depth      = 4,
        dim_head            = 32,
        heads               = 8,
    )

    # -- Text encoder (needed to construct CTCLIP, not used at inference) ---
    tokenizer    = BertTokenizer.from_pretrained(
        "microsoft/BiomedVLP-CXR-BERT-specialized", do_lower_case=True
    )
    text_encoder = BertModel.from_pretrained(
        "microsoft/BiomedVLP-CXR-BERT-specialized"
    )

    # -- Full CT-CLIP wrapper -----------------------------------------------
    model = CTCLIP(
        image_encoder           = image_encoder,
        text_encoder            = text_encoder,
        dim_image               = 294912,
        dim_text                = 768,
        dim_latent              = 512,
        extra_latent_projection = False,
        use_mlm                 = False,
        downsample_image_embeds = False,
        use_all_token_embeds    = False,
    )

    # -- Classification head (ClassFine / LiPro) ---------------------------
    # The checkpoint stores a linear layer: 512 -> NUM_CLASSES
    model.classifier = torch.nn.Linear(512, NUM_CLASSES)

    # -- Load checkpoint ---------------------------------------------------
    print(f"  Loading: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    prefix = "trained_model."
    if isinstance(state, dict) and any(k.startswith(prefix) for k in state.keys()):
        remapped_state = {}
        for k, v in state.items():
            if k.startswith(prefix):
                remapped_state[k[len(prefix):]] = v
            elif k.startswith("classifier."):
                # Keep ClassFine head weights as-is so they are not dropped.
                remapped_state[k] = v
        state = remapped_state
        print("  [INFO] Stripped 'trained_model.' prefix and preserved classifier.* keys.")

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [WARN] Missing keys   ({len(missing)}): {missing[:3]} ...")
    if unexpected:
        print(f"  [WARN] Unexpected keys({len(unexpected)}): {unexpected[:3]} ...")

    model.to(device)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
#  Inference  -  ClassFine forward pass
#  1. CT-ViT  ->  image embedding  (1, 512)
#  2. Linear classifier head       (1, 18)  logits
#  3. Sigmoid                      (1, 18)  probabilities
#  4. Extract the 4 target indices
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(model, tokenizer, ct_tensor: torch.Tensor, device: str) -> dict:
    """Returns {pathology_display_name: probability_of_presence}."""
    ct_tensor = ct_tensor.to("cpu")  # Force CPU (device arg is ignored now)

    vit = find_image_encoder(model, tokenizer)
    img_embed = vit(ct_tensor)    # (1, 512)
    img_embed = F.normalize(img_embed, dim=-1)

    # Classification head -> (1, 18) -> sigmoid -> (18,)
    logits = model.classifier(img_embed)
    probs  = torch.sigmoid(logits)[0]

    return {
        display_name: probs[idx].item()
        for display_name, idx in TARGET_PATHOLOGIES.items()
    }


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def find_nifti_files(folder: Path) -> list:
    return sorted(list(folder.rglob("*.nii.gz")) + list(folder.rglob("*.nii")))


def find_image_encoder(model, tokenizer):
    def vit(ct_tensor: torch.Tensor) -> torch.Tensor:
        text_tokens = tokenizer(
            "",
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=200,
        )
        text_tokens = {key: value.to(ct_tensor.device) for key, value in text_tokens.items()}
        text_ns = SimpleNamespace(**text_tokens)
        _, image_latents, _ = model(
            text_ns,
            ct_tensor,
            ct_tensor.device,
            return_latents=True,
        )
        return image_latents

    return vit


def print_results(results: dict, volume_name: str, threshold: float = 0.5):
    print(f"\n  Volume : {volume_name}")
    print(f"  {'Pathology':<22} {'Probability':>11}  Finding")
    print(f"  {'-'*22}  {'-'*10}  {'-'*10}")
    for pathology, prob in results.items():
        flag = "POSITIVE" if prob >= threshold else "negative"
        print(f"  {pathology:<22} {prob:>10.4f}  {flag}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 62)
    print("  CT-CLIP  |  ClassFine Inference  (CT_LiPro_v2.pt)")
    print("=" * 62)

    # Validate checkpoint
    ckpt_path = Path(MODEL_CHECKPOINT)
    if not ckpt_path.exists():
        print(f"\n[ERROR] Checkpoint not found:\n  {ckpt_path}")
        sys.exit(1)

    # Validate scan folder
    scan_folder = Path(CT_SCAN_FOLDER)
    if not scan_folder.exists():
        print(f"\n[ERROR] Scan folder not found:\n  {scan_folder}")
        sys.exit(1)

    nifti_files = find_nifti_files(scan_folder)
    if not nifti_files:
        print(f"\n[ERROR] No .nii / .nii.gz files found in:\n  {scan_folder}")
        sys.exit(1)

    print(f"\n  Device       : {DEVICE}")
    print(f"  Checkpoint   : {ckpt_path.name}")
    print(f"  Scan folder  : {scan_folder.name}")
    print(f"  Reconstructions found ({len(nifti_files)}): "
          + ", ".join(f.name for f in nifti_files))

    # Load model
    print("\n[1/3] Loading model ...")
    model, tokenizer = load_model(str(ckpt_path), DEVICE)
    model = model.cpu()  # belt-and-suspenders: ensure everything is on CPU
    torch.set_default_device("cpu")  # prevent any stray CUDA allocations
    print("      Model ready.\n")

    # Select which files to process
    files_to_run = nifti_files if MULTI_RECON_MODE == "all" else [nifti_files[0]]

    # Run inference
    print("[2/3] Pre-processing and running inference ...")
    all_results = {}
    for nifti_path in files_to_run:
        print(f"\n  -> {nifti_path.name}")
        ct_tensor = load_and_preprocess_ct(str(nifti_path))
        print(f"     Tensor shape: {tuple(ct_tensor.shape)}")
        scores = predict(model, tokenizer, ct_tensor, DEVICE)
        all_results[nifti_path.name] = scores

    # Print summary
    THRESHOLD = 0.5
    print(f"\n[3/3] Results  (positive threshold = {THRESHOLD})")
    print("=" * 62)
    for vol_name, results in all_results.items():
        print_results(results, vol_name, threshold=THRESHOLD)

    print("\n" + "=" * 62)
    print("  Done.")
    print("=" * 62)


if __name__ == "__main__":
    main()