import os
import sys
import copy
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import nibabel as nib
import pandas as pd
import torch.nn.functional as F

from tqdm import tqdm


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            "ct_lipro_rexground_inference.log"
        )
    ]
)

logger = logging.getLogger(__name__)


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CT_CLIP_DIR = ROOT / "CT_CLIP"

TRANSFORMER_MASKGIT_DIR = ROOT / "transformer_maskgit"


sys.path.insert(0, str(CT_CLIP_DIR))
sys.path.insert(0, str(TRANSFORMER_MASKGIT_DIR))


logger.info(f"Added paths: {CT_CLIP_DIR}, {TRANSFORMER_MASKGIT_DIR}")


# ============================================================
# Imports
# ============================================================

from transformers import BertTokenizer, BertModel
from transformer_maskgit import CTViT
from ct_clip import CTCLIP

logger.info("Imported CT-CLIP modules")


# ============================================================
# Configuration
# ============================================================

DEVICE = "cuda"

CHECKPOINT = (
    "/home/chest_ct/code/models/"
    "ct-clip/checkpointlipro/"
    "CT_LiPro_v2.pt"
)

DATA_FOLDER = (
    "/home/chest_ct/code/data/"
    "data_volumes/dataset/"
    "train_fixed"
)

OUTPUT_DIR = (
    "/home/chest_ct/code/models/"
    "ct-clip/rexground_predictions"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_CLASSES = 18

PATHOLOGIES = [
    "Medical material",
    "Arterial wall calcification",
    "Cardiomegaly",
    "Pericardial effusion",
    "Coronary artery wall calcification",
    "Hiatal hernia",
    "Lymphadenopathy",
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening"
]

TARGET_FINDINGS = [
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Consolidation"
]

TARGET_INDEX = [8, 9, 10, 15]


# ============================================================
# ReXGround Dataset
# ============================================================

class ReXGroundDataset(Dataset):

    def __init__(self, folder):
        self.files = sorted(list(Path(folder).glob("*.nii.gz")))
        logger.info(f"Found {len(self.files)} CT scans")

    def __len__(self):
        return len(self.files)

    def preprocess(self, path):
        nii = nib.load(str(path))
        img = nii.get_fdata()

        logger.info(f"{path.name}: {img.shape}")

        # HU clipping
        img = np.clip(img, -1000, 1000)

        # normalize
        img = img / 1000.0

        tensor = torch.tensor(img, dtype=torch.float32)

        # X,Y,Z -> Z,X,Y
        tensor = tensor.permute(2, 0, 1)

        tensor = tensor.unsqueeze(0).unsqueeze(0)

        # CTCLIP input
        tensor = F.interpolate(
            tensor,
            size=(240, 480, 480),
            mode="trilinear",
            align_corners=False
        )

        tensor = tensor.squeeze(0)

        return tensor

    def __getitem__(self, idx):
        path = self.files[idx]
        volume = self.preprocess(path)
        name = path.name.replace(".nii.gz", "")
        return (volume, "", name)


# ============================================================
# Classifier
# ============================================================

class ImageLatentsClassifier(nn.Module):

    def __init__(self, trained_model, latent_dim, num_classes):
        super().__init__()
        self.trained_model = trained_model
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, text_tokens, image, device):
        _, image_latents, _ = self.trained_model(
            text_tokens,
            image,
            device=device,
            return_latents=True
        )
        image_latents = self.relu(image_latents)
        output = self.classifier(image_latents)
        return output


# ============================================================
# Sigmoid
# ============================================================

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


# ============================================================
# Load model
# ============================================================

def build_model():

    logger.info("Loading tokenizer")

    tokenizer = BertTokenizer.from_pretrained(
        "microsoft/BiomedVLP-CXR-BERT-specialized",
        do_lower_case=True
    )

    text_encoder = BertModel.from_pretrained(
        "microsoft/BiomedVLP-CXR-BERT-specialized"
    )

    text_encoder.resize_token_embeddings(len(tokenizer))

    logger.info("Creating CTViT")

    image_encoder = CTViT(
        dim=512,
        codebook_size=8192,
        image_size=480,
        patch_size=20,
        temporal_patch_size=10,
        spatial_depth=4,
        temporal_depth=4,
        dim_head=32,
        heads=8
    )

    logger.info("Creating CTCLIP")

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

    model = ImageLatentsClassifier(clip, 512, NUM_CLASSES)

    logger.info("Loading checkpoint")

    state = torch.load(CHECKPOINT, map_location="cpu")

    if isinstance(state, dict):
        for k in ["state_dict", "model_state_dict", "weights"]:
            if k in state:
                state = state[k]
                break

    result = model.load_state_dict(state, strict=False)

    logger.warning(f"Missing keys: {len(result.missing_keys)}")
    logger.warning(f"Unexpected keys: {len(result.unexpected_keys)}")

    model.to(DEVICE)
    model.eval()

    return (model, tokenizer)


# ============================================================
# Inference
# ============================================================

def inference():

    model, tokenizer = build_model()

    dataset = ReXGroundDataset(DATA_FOLDER)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )

    results = []

    with torch.no_grad():

        for volume, _, name in tqdm(loader):

            volume = volume.to(DEVICE)

            # ------------------------------------------------
            # FIX: keep the HuggingFace BatchEncoding object.
            # Do NOT convert to a plain dict — CTCLIP internally
            # accesses `text.input_ids` as an attribute, which a
            # plain dict does not support. BatchEncoding has its
            # own .to(device) that preserves attribute access.
            # ------------------------------------------------
            text_tokens = tokenizer(
                "",
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=200
            )

            text_tokens = text_tokens.to(DEVICE)

            logits = model(text_tokens, volume, DEVICE)

            probs = sigmoid(logits)[0].cpu().numpy()

            row = {"scan": name[0]}

            for i, p in enumerate(PATHOLOGIES):
                row[p] = float(probs[i])

            results.append(row)

    df = pd.DataFrame(results)

    save_path = os.path.join(OUTPUT_DIR, "rexground_predictions.csv")

    df.to_csv(save_path, index=False)

    logger.info(f"Saved predictions: {save_path}")

    print("\n==== Target findings ====")
    print(df[["scan"] + TARGET_FINDINGS])


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    logger.info("Starting ReXGround CT-CLIP inference")
    inference()
    logger.info("Finished")