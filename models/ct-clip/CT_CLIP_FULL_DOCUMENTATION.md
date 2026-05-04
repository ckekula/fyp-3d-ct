# CT-CLIP: Complete Technical Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [How It Works](#how-it-works)
4. [Dataset (CT-RATE)](#dataset-ct-rate)
5. [Model Variants](#model-variants)
6. [Creating the CT-CLIP Pipeline](#creating-the-ct-clip-pipeline) ⭐ NEW
7. [Training Pipeline](#training-pipeline)
8. [Inference Pipeline](#inference-pipeline)
9. [Loss Functions](#loss-functions)
10. [Code Walkthrough](#code-walkthrough)
11. [Performance Metrics](#performance-metrics)

---

## Overview

**CT-CLIP** is a pioneering 3D medical imaging model that learns from paired chest CT volumes and radiology text reports. It combines vision and language understanding for detecting abnormalities in CT scans through contrastive learning.

### Key Characteristics:

- **Modality**: 3D medical imaging (volumetric CT scans)
- **Training Data**: Radiology text reports paired with CT volumes
- **Architecture**: CLIP-style multimodal learning (Vision + Language)
- **Task**: Zero-shot abnormality detection across 18+ pathologies
- **Pretrained Models**: Available on HuggingFace

### Use Cases:

1. **Zero-shot pathology detection** - Detect abnormalities without task-specific training
2. **Fine-tuned classification** - Vocabulary or class-based fine-tuning for improved performance
3. **Semantic search** - Find similar CT scans based on radiology reports
4. **Multimodal representation learning** - Learn joint embeddings of images and text

---

## Architecture

### 1. **High-Level Components**

```
┌─────────────────────────────────────────────────────────────┐
│                    CT-CLIP Model                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐              ┌─────────────────────┐ │
│  │ 3D CT Volume    │              │ Radiology Report    │ │
│  │ (480×480×240)   │              │ (Text tokens)       │ │
│  └────────┬────────┘              └──────────┬──────────┘ │
│           │                                  │             │
│           ▼                                  ▼             │
│  ┌─────────────────┐              ┌─────────────────────┐ │
│  │ Image Encoder   │              │ Text Encoder        │ │
│  │ (CT-ViT)        │              │ (BiomedVLP-BERT)    │ │
│  └────────┬────────┘              └──────────┬──────────┘ │
│           │                                  │             │
│    dim: 294912                        dim: 768             │
│           │                                  │             │
│           ▼                                  ▼             │
│  ┌─────────────────┐              ┌─────────────────────┐ │
│  │ Image Projection│              │ Text Projection     │ │
│  │ Linear (→512)   │              │ Linear (→512)       │ │
│  └────────┬────────┘              └──────────┬──────────┘ │
│           │                                  │             │
│           └──────────────┬───────────────────┘             │
│                          ▼                                 │
│            ┌──────────────────────────┐                    │
│            │ Contrastive Loss         │                    │
│            │ (InfoNCE)                │                    │
│            └──────────────────────────┘                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2. **Image Encoder: CT-ViT (CT Vision Transformer)**

Located in: `transformer_maskgit` module

**Configuration:**

```python
CTViT(
    dim = 512,                      # Embedding dimens ion
    codebook_size = 8192,           # Codebook size for vector quantization
    image_size = 480,               # Input volume height/width
    patch_size = 20,                # Spatial patch size
    temporal_patch_size = 10,       # Temporal (depth) patch size
    spatial_depth = 4,              # Spatial transformer layers
    temporal_depth = 4,             # Temporal transformer layers
    dim_head = 32,                  # Attention head dimension
    heads = 8                       # Number of attention heads
)
```

**Process:**

1. **Input**: 3D CT volume (480×480×240 voxels)
2. **Patchification**: Split into spatial patches (20×20) and temporal patches (10 slices)
   - Output: ~576 patches (24×24×1 spatial, 24 temporal)
3. **Embedding**: Convert patches to embeddings
4. **Spatial Transformer**: Process spatial information within each temporal block
5. **Temporal Transformer**: Aggregate information across time
6. **Output**: Feature representation (294,912 dimensions)

### 3. **Text Encoder: BiomedVLP-CXR-BERT**

**Source**: `microsoft/BiomedVLP-CXR-BERT-specialized`

- Pretrained on biomedical and chest X-ray text
- Specialized for medical report understanding

**Process:**

1. **Tokenization**: Convert text to tokens using BertTokenizer
   - Max sequence length: 512 tokens
   - Medical domain vocabulary
2. **Token Embeddings**: Convert tokens to 768-dimensional embeddings
3. **BERT Layers**: Process through transformer layers
4. **Output**: [CLS] token representation (768 dimensions) representing the entire report

### 4. **Projection Layers**

Convert embeddings to shared latent space for contrastive learning:

```python
Text Projection:   768 → 512 dimensions (Linear layer)
Image Projection:  294,912 → 512 dimensions (Linear layer)
```

Both are then L2-normalized for cosine similarity computation.

---

## How It Works

### Core Learning Principle: Contrastive Learning

CT-CLIP learns by pushing similar image-text pairs close together and dissimilar pairs far apart in a shared embedding space.

### Mathematical Foundation

**Contrastive Loss (InfoNCE):**

$$
\log \frac{\exp(\text{sim}(z^v_i, z^t_i) / \tau)}{\sum_{j=1}^{B} \exp(\text{sim}(z^v_i, z^t_j) / \tau)} +
\log \frac{\exp(\text{sim}(z^t_i, z^v_i) / \tau)}{\sum_{j=1}^{B} \exp(\text{sim}(z^t_i, z^v_j) / \tau)}
\right]$$

Where:
- $z^v_i$ = image latent embedding for sample i
- $z^t_i$ = text latent embedding for sample i
- $\tau$ = temperature parameter (learnable)
- $\text{sim}(a, b) = a \cdot b$ (cosine similarity, since vectors are L2-normalized)

**Why it works:**
- Positive pair (matching image-text): High similarity → Low loss
- Negative pairs (mismatched): Low similarity → High loss
- Batch size > 1: Each sample in batch provides negatives for others

### Temperature Parameter

- Learnable scalar that scales similarity scores
- Higher temperature → softer probability distribution
- Lower temperature → sharper distribution (more confident)
- Initially: 1.0, learned during training

---

## Dataset: CT-RATE

### Overview

**CT-RATE** is a large-scale 3D medical imaging dataset with paired CT volumes and radiology reports.

**Composition:**
- **Total Volumes**: 25,692 unique CT scans
- **With Reconstructions**: 50,188 volumes (from different reconstruction parameters)
- **Unique Patients**: 21,304
- **CT Type**: Non-contrast chest CT
- **Split**:
  - Training: 20,000 patients
  - Validation: 1,304 patients
- **Metadata**:
  - Radiology reports (Findings + Impressions)
  - Multi-abnormality labels (18 pathologies)
  - Volume metadata (voxel spacing, rescale parameters)

### 18 Pathologies Detected

1. Medical material
2. Arterial wall calcification
3. Cardiomegaly
4. Pericardial effusion
5. Coronary artery wall calcification
6. Hiatal hernia
7. Lymphadenopathy
8. Emphysema
9. Atelectasis
10. Lung nodule
11. Lung opacity
12. Pulmonary fibrotic sequela
13. Pleural effusion
14. Mosaic attenuation pattern
15. Peribronchial thickening
16. Consolidation
17. Bronchiectasis
18. Interlobular septal thickening

### Data Organization

```
dataset/
├── train/
│   ├── patient_1/
│   │   ├── scan_a/
│   │   │   ├── reconstruction_1.nii.gz
│   │   │   ├── reconstruction_2.nii.gz
│   │   ├── scan_b/
│   ├── patient_2/
├── valid/
│   ├── ...
├── train_reports.csv      # Findings + Impressions
├── train_metadata.csv     # Voxel spacing, rescale params
├── train_labels.csv       # Multi-abnormality binary labels
```

### Data Preprocessing Pipeline

**File**: `data.py` in scripts directory

**Steps:**

1. **Load NIfTI Format**:
   - Convert NIfTI volume to numpy array
   - Extract metadata: voxel spacing, Hounsfield unit rescaling

2. **Normalize Hounsfield Units (HU)**:
   ```python
   img_data = slope * img_data + intercept
   img_data = np.clip(img_data, -1000, 1000)  # Clip to typical CT range
   img_data = img_data / 1000  # Normalize to [-1, 1]
   ```

3. **Resample to Standard Spacing**:
   - Input: Varies by patient (different scanner parameters)
   - Target: Standardized spacing
     - Axial (XY): 0.75 mm
     - Coronal (Z): 1.5 mm
   - Method: Trilinear interpolation

4. **Crop/Pad to Standard Size**:
   - Target shape: 480×480×240 voxels
   - If larger: Center crop
   - If smaller: Zero pad (value=-1)

5. **Reorder and Format**:
   - Output shape: (240, 480, 480) - (Depth, Height, Width)
   - Add channel dimension: (1, 240, 480, 480)
   - Tensor format: PyTorch float32

**Result**: Normalized, standardized 3D tensor ready for model input

### Text Data Preprocessing

1. **Load Reports**: Extract from CSV
   - Findings section
   - Impressions section
2. **Clean Text**:
   - Remove quotes and parentheses
   - Concatenate Findings + Impressions
   - Remove lines like "Not given."
3. **Tokenization**:
   - Use BiomedVLP-BERT tokenizer
   - Max length: 512 tokens
   - Padding and truncation applied

---

## Creating the CT-CLIP Pipeline

### Complete Step-by-Step Guide

This section provides detailed instructions for creating and setting up the complete CT-CLIP pipeline from scratch.

### Phase 1: Environment Setup

#### Step 1.1: Prerequisites

```bash
# System requirements
- Python 3.8+
- CUDA 11.8+ (for GPU support)
- 50GB+ disk space for models and data
- GPU: NVIDIA A100 (80GB for training) or RTX 4090 (24GB for inference)
```

#### Step 1.2: Clone and Install Dependencies

```bash
# Navigate to your workspace
cd /path/to/fyp-3d-ct/models/ct-clip

# Install transformer_maskgit first (dependency)
cd transformer_maskgit
pip install -e .
cd ..

# Install CT_CLIP
cd CT_CLIP
pip install -e .
cd ..

# Install additional dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate pandas nibabel tqdm scikit-learn
pip install einops tensorboard matplotlib seaborn
```

#### Step 1.3: Verify Installation

```python
# test_installation.py
import torch
from transformer_maskgit import CTViT
from transformers import BertModel, BertTokenizer
from ct_clip import CTCLIP

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Test imports
print("✓ All imports successful!")
```

Run with: `python test_installation.py`

---

### Phase 2: Data Preparation

#### Step 2.1: Download CT-RATE Dataset

```bash
# Option A: Download from HuggingFace (recommended)
from huggingface_hub import snapshot_download

# Download entire dataset
dataset_path = snapshot_download(
    repo_id="ibrahimhamamci/CT-RATE",
    repo_type="dataset",
    local_dir="./ct-rate-dataset"
)

# Option B: Use your own chest CT dataset
# Ensure your data is in NIfTI format (.nii.gz)
```

#### Step 2.2: Organize Dataset Structure

```
ct-rate-dataset/
├── train/
│   ├── patient_001/
│   │   ├── scan_a/
│   │   │   ├── vol_1.nii.gz
│   │   │   ├── vol_2.nii.gz
│   │   ├── scan_b/
│   ├── patient_002/
├── valid/
│   ├── ...
├── train_reports.csv
├── train_metadata.csv
├── train_labels.csv
├── valid_reports.csv
├── valid_metadata.csv
├── valid_labels.csv
```

**CSV File Formats:**

`reports.csv`:
```csv
VolumeName,Findings_EN,Impressions_EN
vol_001.nii.gz,"CT findings...",Impressions...
vol_002.nii.gz,"CT findings...",Impressions...
```

`metadata.csv`:
```csv
VolumeName,RescaleSlope,RescaleIntercept,XYSpacing,ZSpacing
vol_001.nii.gz,1.0,0,"(0.75, 0.75)",1.5
vol_002.nii.gz,1.0,0,"(0.75, 0.75)",1.5
```

`labels.csv`:
```csv
VolumeName,Medical_material,Cardiomegaly,Emphysema,...
vol_001.nii.gz,0,1,0,...
vol_002.nii.gz,0,0,1,...
```

#### Step 2.3: Verify Data Integrity

```python
# verify_data.py
import os
import pandas as pd
import nibabel as nib

data_dir = "./ct-rate-dataset"

# Check structure
for split in ["train", "valid"]:
    split_dir = os.path.join(data_dir, split)
    csv_files = ["reports.csv", "metadata.csv", "labels.csv"]

    for csv_file in csv_files:
        path = os.path.join(data_dir, f"{split}_{csv_file.split('_')[0]}.csv")
        df = pd.read_csv(path)
        print(f"{split}_{csv_file}: {len(df)} records")

    # Check NIfTI files
    nii_count = sum([len(f) for r, d, f in os.walk(split_dir) if 'nii' in str(f)])
    print(f"{split} NIfTI files: {nii_count}")
```

---

### Phase 3: Model Architecture Creation

#### Step 3.1: Create Image Encoder (CT-ViT)

```python
# create_image_encoder.py
import torch
from transformer_maskgit import CTViT

def create_image_encoder():
    """Create and configure CT Vision Transformer"""
    image_encoder = CTViT(
        dim=512,                      # Embedding dimension
        codebook_size=8192,           # Codebook for quantization
        image_size=480,               # Input volume size (H, W)
        patch_size=20,                # Spatial patch size
        temporal_patch_size=10,       # Temporal patch size (depth)
        spatial_depth=4,              # Spatial transformer layers
        temporal_depth=4,             # Temporal transformer layers
        dim_head=32,                  # Attention head dimension
        heads=8,                      # Number of heads
        attn_dropout=0.0,             # Attention dropout
        ff_dropout=0.0,               # Feedforward dropout
        norm='layer_norm'             # Normalization type
    )

    # Print model info
    total_params = sum(p.numel() for p in image_encoder.parameters())
    print(f"Image Encoder Parameters: {total_params:,}")

    return image_encoder

if __name__ == "__main__":
    encoder = create_image_encoder()

    # Test with random input
    test_input = torch.randn(1, 1, 240, 480, 480)  # (B, C, D, H, W)
    output = encoder(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
```

#### Step 3.2: Create Text Encoder (BiomedVLP-BERT)

```python
# create_text_encoder.py
import torch
from transformers import BertTokenizer, BertModel

def create_text_encoder():
    """Create and configure medical BERT encoder"""
    model_name = "microsoft/BiomedVLP-CXR-BERT-specialized"

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        model_name,
        do_lower_case=True
    )

    # Load model
    text_encoder = BertModel.from_pretrained(model_name)

    # Resize embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Print model info
    total_params = sum(p.numel() for p in text_encoder.parameters())
    print(f"Text Encoder Parameters: {total_params:,}")
    print(f"Vocabulary size: {len(tokenizer)}")

    return tokenizer, text_encoder

def test_text_encoder(tokenizer, text_encoder):
    """Test text encoder with sample reports"""
    sample_text = [
        "There is cardiomegaly and mild pulmonary edema.",
        "No acute cardiopulmonary abnormalities.",
        "Findings consistent with chronic obstructive pulmonary disease."
    ]

    # Tokenize
    tokens = tokenizer(
        sample_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )

    print(f"\nTokenized shape: {tokens['input_ids'].shape}")

    # Encode
    with torch.no_grad():
        output = text_encoder(**tokens)

    print(f"Embeddings shape: {output.last_hidden_state.shape}")
    print(f"CLS token shape: {output.pooler_output.shape}")

    return output

if __name__ == "__main__":
    tokenizer, encoder = create_text_encoder()
    test_text_encoder(tokenizer, encoder)
```

#### Step 3.3: Combine Encoders into CTCLIP

```python
# create_ctclip_model.py
import torch
import torch.nn as nn
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP

def create_ctclip_pipeline():
    """Complete CT-CLIP model creation"""

    print("=" * 50)
    print("Creating CT-CLIP Pipeline")
    print("=" * 50)

    # Step 1: Create image encoder
    print("\n[1/3] Creating Image Encoder (CT-ViT)...")
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
    print("✓ Image encoder created")

    # Step 2: Create text encoder
    print("\n[2/3] Creating Text Encoder (BiomedVLP-BERT)...")
    tokenizer = BertTokenizer.from_pretrained(
        'microsoft/BiomedVLP-CXR-BERT-specialized',
        do_lower_case=True
    )
    text_encoder = BertModel.from_pretrained(
        "microsoft/BiomedVLP-CXR-BERT-specialized"
    )
    text_encoder.resize_token_embeddings(len(tokenizer))
    print("✓ Text encoder created")

    # Step 3: Create CLIP model
    print("\n[3/3] Creating CTCLIP Model...")
    clip_model = CTCLIP(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        dim_image=294912,          # Output size of CT-ViT
        dim_text=768,              # Output size of BERT
        dim_latent=512,            # Shared latent space
        extra_latent_projection=False,
        use_mlm=False,             # Masked Language Modeling
        downsample_image_embeds=False,
        use_all_token_embeds=False
    )
    print("✓ CTCLIP model created")

    print("\n" + "=" * 50)
    print("Model Summary")
    print("=" * 50)

    total_params = sum(p.numel() for p in clip_model.parameters())
    trainable_params = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)

    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Model Size: {total_params * 4 / (1024**3):.2f} GB (fp32)")

    return clip_model, tokenizer

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, tokenizer = create_ctclip_pipeline()
    clip_model.to(device)

    # Test forward pass
    print("\n" + "=" * 50)
    print("Testing Forward Pass")
    print("=" * 50)

    # Create dummy inputs
    batch_size = 1
    dummy_image = torch.randn(batch_size, 1, 240, 480, 480).to(device)
    dummy_text = "Cardiomegaly and mild pulmonary edema"

    dummy_tokens = tokenizer(
        [dummy_text] * batch_size,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    ).to(device)

    print(f"Input image shape: {dummy_image.shape}")
    print(f"Input text: {dummy_text}")
    print(f"Input tokens shape: {dummy_tokens['input_ids'].shape}")

    with torch.no_grad():
        output = clip_model(
            dummy_tokens,
            dummy_image,
            device=device,
            return_latents=True
        )

    text_latent, image_latent = output
    print(f"\nOutput text latent shape: {text_latent.shape}")
    print(f"Output image latent shape: {image_latent.shape}")
    print("✓ Forward pass successful!")
```

---

### Phase 4: Data Loading Pipeline

#### Step 4.1: Create Dataset Class

```python
# create_dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
from scripts.data import CTReportDataset

def create_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 8,
    split: str = "train"
):
    """Create training/validation dataloaders"""

    print(f"Creating {split} DataLoader...")

    reports_file = f"{data_dir}/{split}_reports.csv"
    meta_file = f"{data_dir}/{split}_metadata.csv"
    labels_file = f"{data_dir}/{split}_labels.csv"
    data_folder = f"{data_dir}/{split}"

    # Create dataset
    dataset = CTReportDataset(
        data_folder=data_folder,
        reports_file=reports_file,
        meta_file=meta_file,
        min_slices=20,
        resize_dim=500,
        force_num_frames=True
    )

    print(f"Dataset size: {len(dataset)}")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=(split == "train"),
        pin_memory=True,
        drop_last=(split == "train")
    )

    print(f"Number of batches: {len(dataloader)}")

    return dataloader, dataset

def test_dataloader():
    """Test dataloader with sample data"""
    data_dir = "./ct-rate-dataset"

    train_loader, train_dataset = create_dataloaders(
        data_dir,
        batch_size=2,
        num_workers=4,
        split="train"
    )

    # Get one batch
    batch = next(iter(train_loader))
    volumes, texts = batch

    print(f"\nSample Batch:")
    print(f"Volumes shape: {volumes.shape}")
    print(f"Number of texts: {len(texts)}")
    print(f"Sample text: {texts[0][:100]}...")

    return train_loader

if __name__ == "__main__":
    test_dataloader()
```

---

### Phase 5: Training Setup

#### Step 5.1: Create Training Configuration

```python
# training_config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    """CT-CLIP Training Configuration"""

    # Data
    data_dir: str = "./ct-rate-dataset"
    batch_size: int = 8
    num_workers: int = 8

    # Model
    dim_text: int = 768
    dim_image: int = 294912
    dim_latent: int = 512

    # Training
    num_train_steps: int = 100000
    learning_rate: float = 1.25e-6
    weight_decay: float = 0.0
    max_grad_norm: float = 0.5
    warmup_steps: int = 10000

    # Optimization
    optimizer: str = "adam"
    lr_scheduler: str = "cosine"

    # Checkpointing
    save_every_steps: int = 1000
    eval_every_steps: int = 500
    checkpoint_dir: str = "./checkpoints"

    # Distributed training
    use_fsdp: bool = False
    use_accelerate: bool = True

    # Optional features
    use_mlm: bool = False
    use_visual_ssl: bool = False
    freeze_image_encoder: bool = False
    freeze_text_encoder: bool = False
```

#### Step 5.2: Create Training Script

```python
# train_ctclip.py
import torch
import torch.nn as nn
from pathlib import Path
from scripts.CTCLIPTrainer import CTClipTrainer
from create_ctclip_model import create_ctclip_pipeline
from training_config import TrainingConfig

def main():
    # Configuration
    config = TrainingConfig(
        data_dir="./ct-rate-dataset",
        batch_size=8,
        num_train_steps=100000,
        learning_rate=1.25e-6,
        warmup_steps=10000
    )

    print("=" * 60)
    print("CT-CLIP Training Pipeline")
    print("=" * 60)

    # Step 1: Create model
    print("\n[1/4] Creating Model...")
    clip_model, tokenizer = create_ctclip_pipeline()

    # Step 2: Create trainer
    print("\n[2/4] Creating Trainer...")
    trainer = CTClipTrainer(
        CTClip=clip_model,
        num_train_steps=config.num_train_steps,
        batch_size=config.batch_size,
        data_train=f"{config.data_dir}/train",
        data_valid=f"{config.data_dir}/valid",
        reports_file_train=f"{config.data_dir}/train_reports.csv",
        reports_file_valid=f"{config.data_dir}/valid_reports.csv",
        train_meta_file=f"{config.data_dir}/train_metadata.csv",
        valid_meta_file=f"{config.data_dir}/valid_metadata.csv",
        labels=f"{config.data_dir}/train_labels.csv",
        tokenizer=tokenizer,
        lr=config.learning_rate,
        wd=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        save_results_every=config.save_every_steps,
        save_model_every=config.save_every_steps,
        results_folder=config.checkpoint_dir,
        num_workers=config.num_workers
    )

    # Step 3: Training loop
    print("\n[3/4] Starting Training...")
    print(f"Total steps: {config.num_train_steps}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")

    try:
        for step in range(config.num_train_steps):
            trainer.train_step()

            if step % 100 == 0:
                print(f"Step {step}/{config.num_train_steps}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    # Step 4: Save final model
    print("\n[4/4] Saving Model...")
    final_path = Path(config.checkpoint_dir) / "final_model.pt"
    trainer.save(str(final_path))
    print(f"✓ Model saved to {final_path}")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

---

### Phase 6: Inference Pipeline

#### Step 6.1: Create Inference Script

```python
# inference_ctclip.py
import torch
from pathlib import Path
from scripts.zero_shot import CTClipInference
from create_ctclip_model import create_ctclip_pipeline

def run_inference(
    model_path: str,
    data_folder: str,
    results_folder: str = "./results"
):
    """Run zero-shot CT-CLIP inference"""

    print("=" * 60)
    print("CT-CLIP Inference Pipeline")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Create model
    print("\n[1/3] Creating Model...")
    clip_model, tokenizer = create_ctclip_pipeline()
    clip_model.to(device)

    # Step 2: Load pretrained weights
    print("\n[2/3] Loading Pretrained Weights...")
    clip_model.load(model_path)
    print(f"✓ Model loaded from {model_path}")

    # Step 3: Run inference
    print("\n[3/3] Running Inference...")
    inference = CTClipInference(
        CTClip=clip_model,
        data_folder=data_folder,
        reports_file=f"{data_folder}/reports.csv",
        meta_file=f"{data_folder}/metadata.csv",
        labels=f"{data_folder}/labels.csv",
        results_folder=results_folder
    )

    inference.infer()

    print("\n" + "=" * 60)
    print("Inference Complete!")
    print(f"Results saved to {results_folder}")
    print("=" * 60)

if __name__ == "__main__":
    # Configuration
    model_path = "./checkpoints/final_model.pt"
    data_folder = "./ct-rate-dataset/valid"
    results_folder = "./inference_results"

    run_inference(
        model_path=model_path,
        data_folder=data_folder,
        results_folder=results_folder
    )
```

---

### Phase 7: Quick Start Guide

#### Step 7.1: Complete Pipeline Script

```python
# pipeline_quickstart.py
"""
Complete CT-CLIP Pipeline - Ready to Run
Includes: Setup → Model Creation → Training → Inference
"""

import torch
from pathlib import Path
from create_ctclip_model import create_ctclip_pipeline
from create_dataset import create_dataloaders
from training_config import TrainingConfig

def complete_pipeline():
    """Run complete CT-CLIP pipeline"""

    print("\n" + "=" * 70)
    print(" " * 15 + "CT-CLIP COMPLETE PIPELINE")
    print("=" * 70)

    config = TrainingConfig(
        data_dir="./ct-rate-dataset",
        batch_size=8,
        num_train_steps=10000,  # Start small for testing
        learning_rate=1.25e-6,
        warmup_steps=1000
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n📍 Device: {device}")
    print(f"📍 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

    # ============================================================
    # PHASE 1: MODEL CREATION
    # ============================================================
    print("\n" + "─" * 70)
    print("PHASE 1: MODEL CREATION")
    print("─" * 70)

    clip_model, tokenizer = create_ctclip_pipeline()
    clip_model.to(device)

    # ============================================================
    # PHASE 2: DATA LOADING
    # ============================================================
    print("\n" + "─" * 70)
    print("PHASE 2: DATA LOADING")
    print("─" * 70)

    train_loader, train_dataset = create_dataloaders(
        config.data_dir,
        config.batch_size,
        config.num_workers,
        split="train"
    )

    valid_loader, valid_dataset = create_dataloaders(
        config.data_dir,
        config.batch_size,
        config.num_workers,
        split="valid"
    )

    # ============================================================
    # PHASE 3: SETUP TRAINING
    # ============================================================
    print("\n" + "─" * 70)
    print("PHASE 3: SETUP TRAINING")
    print("─" * 70)

    optimizer = torch.optim.Adam(
        clip_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.warmup_steps,
        T_mult=1,
        eta_max=config.learning_rate
    )

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    print(f"✓ Optimizer: Adam")
    print(f"✓ Learning Rate: {config.learning_rate}")
    print(f"✓ Scheduler: Cosine Annealing")
    print(f"✓ Checkpoint Dir: {checkpoint_dir}")

    # ============================================================
    # PHASE 4: TRAINING LOOP
    # ============================================================
    print("\n" + "─" * 70)
    print("PHASE 4: TRAINING LOOP")
    print("─" * 70)

    clip_model.train()

    for epoch in range(1):  # Single epoch for quick start
        for batch_idx, (volumes, texts) in enumerate(train_loader):
            if batch_idx >= 10:  # Quick test with 10 batches
                break

            volumes = volumes.to(device)

            tokens = tokenizer(
                list(texts),
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512
            ).to(device)

            optimizer.zero_grad()

            loss = clip_model(
                tokens,
                volumes,
                device=device,
                return_loss=True
            )

            loss.backward()

            if config.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    clip_model.parameters(),
                    config.max_grad_norm
                )

            optimizer.step()
            scheduler.step()

            if batch_idx % 5 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx:04d} | Loss: {loss.item():.4f}")

    # ============================================================
    # PHASE 5: SAVE MODEL
    # ============================================================
    print("\n" + "─" * 70)
    print("PHASE 5: SAVE MODEL")
    print("─" * 70)

    model_path = checkpoint_dir / "ctclip_quickstart.pt"
    torch.save(clip_model.state_dict(), model_path)
    print(f"✓ Model saved to {model_path}")

    print("\n" + "=" * 70)
    print(" " * 20 + "✓ PIPELINE COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    complete_pipeline()
```

#### Step 7.2: Run Quick Start

```bash
# Run complete pipeline
python pipeline_quickstart.py

# Expected output:
# [1/3] Creating Image Encoder...
# [2/3] Creating Text Encoder...
# [3/3] Creating CTCLIP Model...
# Total Parameters: X,XXX,XXX
# Training... (outputs loss values)
# ✓ Model saved to ./checkpoints/ctclip_quickstart.pt
```

---

## Model Variants

### Variant 1: CT-CLIP (Zero-Shot)

**Purpose**: Direct transfer learning without fine-tuning
**Inference Speed**: ~1.5 seconds per CT volume (18 pathologies)
**Process**:
```
For each pathology:
  1. Generate text prompts:
     - "There is {pathology}." (positive)
     - "There is no {pathology}." (negative)
  2. Encode both prompts with text encoder
  3. Encode CT volume with image encoder
  4. Compute similarity scores
  5. Apply softmax to get probabilities
```

### Variant 2: CT-VocabFine (Vocabulary Fine-Tuning)

**Purpose**: Improve performance with task-specific vocabulary
**Training Data**: Limited manual annotations
**Inference Speed**: ~1.5 seconds per CT volume
**Parameters**: Only fine-tune last projection layers

**Process**:
```
1. Train on CT-RATE with limited labels
2. Learn domain-specific text representations
3. Update only text projection layer
4. Freeze image encoder
```

### Variant 3: CT-LiPro / ClassFine (Classification Fine-Tuning)

**Purpose**: Direct classification for better accuracy
**Inference Speed**: ~0.5 seconds per CT volume (faster!)
**Parameters**: Add classification head on top of image features

**Process**:
```
1. Train classification head on image features
2. Direct prediction without text encoding
3. Faster inference (no text encoding needed)
4. Better accuracy on specific pathologies
```

---

## Training Pipeline

### Overall Training Flow

```
┌─────────────────────────────────────┐
│ Load CT-RATE Dataset                │
├─────────────────────────────────────┤
│ Training Loop:                      │
│  For each epoch:                    │
│   For each batch:                   │
│    1. Load CT volume                │
│    2. Load matching report          │
│    3. Encode image → 294,912 dims   │
│    4. Encode text → 768 dims        │
│    5. Project to 512-dim space      │
│    6. Compute contrastive loss      │
│    7. Backward pass                 │
│    8. Update weights                │
│    9. Log metrics                   │
│   Validation every N batches        │
└─────────────────────────────────────┘
```

### Key Hyperparameters

**Model Configuration**:
```python
dim_text = 768              # Text embedding dimension
dim_image = 294912          # Image embedding dimension
dim_latent = 512            # Shared latent space dimension
temperature = 1.0           # Learnable temperature (initial)
```

**Training Configuration**:
```python
batch_size = 8              # Global batch size (across GPUs)
learning_rate = 1.25e-6     # Very small learning rate
weight_decay = 0.0          # L2 regularization
max_grad_norm = 0.5         # Gradient clipping
num_train_steps = ~1M       # Total steps
```

**Learning Rate Scheduling**:
```
CosineAnnealingWarmUpRestarts:
  - Warmup phase: 10,000 steps
  - Linear increase to learning rate
  - Cosine decay after warmup
  - Support for periodic restarts
```

### Hardware Requirements

**Training**:
- GPU: NVIDIA A100 (80GB VRAM)
- Batch size: 8
- Why large GPU:
  - CT volumes are large (500M+ parameters for encoder)
  - Batch size must be large for contrastive learning
  - Large negative pairs pool

**Inference**:
- GPU: Any GPU with >8GB VRAM
- Can reduce model size by adjusting patch sizes

### Training Code Overview

**File**: `CTCLIPTrainer.py`

```python
class CTClipTrainer(nn.Module):
    def __init__(self, CTClip, num_train_steps, batch_size, ...):
        # Initialize model, optimizer, accelerator
        # Set up data loaders
        # Configure learning rate scheduler

    def train_step(self):
        # 1. Load batch of CT volumes and reports
        video, text = next(self.dl_iter)

        # 2. Tokenize text
        text_tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )

        # 3. Forward pass
        loss = self.CTClip(
            text_tokens,
            video,
            return_loss=True,
            device=device
        )

        # 4. Backward pass
        self.accelerator.backward(loss)

        # 5. Gradient clipping and optimization
        if self.max_grad_norm:
            self.accelerator.clip_grad_norm_(
                self.CTClip.parameters(),
                self.max_grad_norm
            )

        self.optim.step()
        self.optim.zero_grad()

        # 6. Log and save
        self.print(f"Step {steps}: loss: {loss.item()}")
        if not (steps % self.save_results_every):
            self.save(checkpoint_path)
```

### Distributed Training

Uses `accelerate` library with FSDP (Fully Sharded Data Parallelism):

```bash
# Without FSDP
accelerate launch run_train.py

# With FSDP (better memory efficiency)
accelerate launch --use_fsdp run_train.py
```

**Benefits**:
- Shards model parameters across GPUs
- Reduces per-GPU memory usage
- Enables training on larger models
- Automatic synchronization during forward/backward

---

## Inference Pipeline

### Zero-Shot Inference

**File**: `zero_shot.py`

```python
class CTClipInference:
    def __init__(self, clip_model, data_folder, reports_file, ...):
        # Load pretrained model
        # Prepare data loaders
        # Setup results folder

    def infer(self):
        pathologies = [
            'Medical material',
            'Arterial wall calcification',
            ... (18 total)
        ]

        for ct_volume in dataset:
            for pathology in pathologies:
                # Generate text prompts
                positive_text = f"There is {pathology}."
                negative_text = f"There is no {pathology}."

                # Encode text
                pos_embedding = text_encoder(positive_text)
                neg_embedding = text_encoder(negative_text)

                # Encode image
                img_embedding = image_encoder(ct_volume)

                # Compute similarities
                pos_sim = cosine_similarity(img_embedding, pos_embedding)
                neg_sim = cosine_similarity(img_embedding, neg_embedding)

                # Get probability
                logits = torch.cat([pos_sim, neg_sim])
                prob = softmax(logits)[0]  # Probability of positive

                predictions[pathology] = prob

            # Save predictions
            save_results(predictions, ct_volume_id)
```

### Inference Speed Comparison

| Model | Pathologies | Speed (sec) | Accuracy |
|-------|-------------|------------|----------|
| Zero-shot CT-CLIP | 18 | 1.5 | Baseline |
| CT-VocabFine | 18 | 1.5 | +X% |
| CT-ClassFine | 18 | 0.5 | +Y% |

---

## Loss Functions

### 1. Main Loss: Contrastive Learning (InfoNCE)

```python
# Compute normalized embeddings
text_latent = F.normalize(text_projection, dim=-1)
image_latent = F.normalize(image_projection, dim=-1)

# Compute similarity matrix (B x B)
# - Diagonal: positive pairs
# - Off-diagonal: negative pairs
similarity = image_latent @ text_latent.T / temperature

# Labels: [0, 1, 2, ..., B-1]
labels = torch.arange(B)

# Cross-entropy loss (both directions)
loss_img2txt = F.cross_entropy(similarity, labels)
loss_txt2img = F.cross_entropy(similarity.T, labels)

total_loss = (loss_img2txt + loss_txt2img) / 2
```

### 2. Optional: Masked Language Modeling (MLM) Loss

**Enabled with**: `use_mlm=True`

```python
text_ssl_loss = self.mlm(
    text.input_ids,
    attention_mask=text.attention_mask
)

total_loss += 0.05 * text_ssl_loss  # Weight = 0.05
```

**Purpose**: Additional objective for text encoder training
- Mask 15% of tokens during training
- Predict masked tokens
- Forces text encoder to learn better representations

### 3. Optional: Visual Self-Supervised Learning (SSL)

**Enabled with**: `use_visual_ssl=True`

**Options**:
- **SimSiam**: Siamese network with stop-gradient
- **SimCLR**: Contrastive with temperature

```python
if use_visual_ssl:
    image_ssl_loss = self.visual_ssl(image)
    total_loss += 0.05 * image_ssl_loss  # Weight = 0.05
```

**Purpose**: Improve image encoder without text supervision

### 4. Optional: Multiview Loss

**Purpose**: Handle data augmentation

```python
if is_multiview:
    # Handle augmented versions of same image/text
    multiview_loss = compute_multiview_alignment(...)
    total_loss += multiview_weight * multiview_loss
```

**Combined Loss**:
```
total_loss = L_contrastive + 0.05*L_mlm + 0.05*L_ssl + 0.1*L_multiview
```

---

## Code Walkthrough

### Main Components

#### 1. Model Initialization (`run_zero_shot.py`)

```python
# Step 1: Load text encoder
tokenizer = BertTokenizer.from_pretrained(
    'microsoft/BiomedVLP-CXR-BERT-specialized',
    do_lower_case=True
)
text_encoder = BertModel.from_pretrained(
    "microsoft/BiomedVLP-CXR-BERT-specialized"
)
text_encoder.resize_token_embeddings(len(tokenizer))

# Step 2: Create image encoder
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

# Step 3: Create CLIP model
clip = CTCLIP(
    image_encoder=image_encoder,
    text_encoder=text_encoder,
    dim_image=294912,
    dim_text=768,
    dim_latent=512,
    use_mlm=False,
    downsample_image_embeds=False,
    use_all_token_embeds=False
)

# Step 4: Load pretrained weights
clip.load("path_to_pretrained_model")

# Step 5: Create inference object
inference = CTClipInference(
    clip,
    data_folder='path_to_validation_folder',
    reports_file="path_to_reports_csv",
    meta_file="path_to_metadata_csv",
    labels="path_to_labels_csv",
    results_folder="inference_results/",
)

# Step 6: Run inference
inference.infer()
```

#### 2. Forward Pass (`ct_clip.py`)

```python
def forward(
    self,
    text,              # Tokenized text with attention masks
    image,             # 3D CT volume
    device,
    return_loss=False,
    return_encodings=False,
    freeze_image_encoder=False,
    freeze_text_encoder=False
):
    # 1. Encode text
    text_embeddings = self.text_transformer(
        text.input_ids,
        attention_mask=text.attention_mask
    )
    enc_text = text_embeddings[0]  # [CLS] token: (B, 768)

    # 2. Encode image
    enc_image = self.visual_transformer(image)  # (B, 294912)

    # 3. Optional: Compute SSL losses
    text_ssl_loss = self.mlm(text.input_ids) if self.use_mlm else 0
    image_ssl_loss = self.visual_ssl(image) if self.use_visual_ssl else 0

    # 4. Project to latent space
    text_latent = self.to_text_latent(enc_text)  # (B, 512)
    image_latent = self.to_visual_latent(enc_image)  # (B, 512)

    # 5. Normalize
    text_latent = F.normalize(text_latent, dim=-1)
    image_latent = F.normalize(image_latent, dim=-1)

    # 6. Compute similarity matrix
    sim = einsum('b i, b j -> b i j', image_latent, text_latent)
    sim = sim / self.temperature

    # 7. Contrastive loss (both directions)
    loss_img2txt = F.cross_entropy(sim, torch.arange(B))
    loss_txt2img = F.cross_entropy(sim.T, torch.arange(B))
    loss = (loss_img2txt + loss_txt2img) / 2

    # 8. Total loss with SSL
    total_loss = loss + text_ssl_loss + image_ssl_loss

    if return_loss:
        return total_loss

    if return_encodings:
        return enc_text, enc_image

    if return_latents:
        return text_latent, image_latent
```

#### 3. Data Loading (`data.py`)

```python
class CTReportDataset(Dataset):
    def __init__(self, data_folder, reports_file, meta_file, ...):
        # Load report text mapping
        self.accession_to_text = self.load_accession_text(reports_file)

        # Prepare file paths and metadata
        self.samples = self.prepare_samples()
        self.df = pd.read_csv(meta_file)

        # Create preprocessing function
        self.nii_to_tensor = partial(
            self.nii_img_to_tensor,
            df=self.df
        )

    def __getitem__(self, index):
        nii_file, input_text = self.samples[index]

        # Load and preprocess 3D volume
        video_tensor = self.nii_to_tensor(nii_file)  # (1, 240, 480, 480)

        # Clean text
        input_text = str(input_text).replace('"', '').replace("'", '')

        return video_tensor, input_text

# Usage in training
from torch.utils.data import DataLoader

train_dataset = CTReportDataset(
    data_folder='data/train',
    reports_file='data/train_reports.csv',
    meta_file='data/train_metadata.csv'
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    num_workers=8,
    shuffle=True
)

for batch in train_loader:
    volumes, texts = batch  # volumes: (B, 1, 240, 480, 480), texts: list[str]
```

---

## Performance Metrics

### Evaluation Metrics

1. **Per-Pathology Metrics**:
   - Accuracy: % correct classifications
   - AUC-ROC: Area under receiver operating characteristic curve
   - Sensitivity: True positive rate
   - Specificity: True negative rate

2. **Macro Metrics** (average across pathologies):
   - Macro-accuracy
   - Macro-F1 score
   - Macro-AUC

### Zero-Shot Performance

| Model | Metric | Value |
|-------|--------|-------|
| CT-CLIP (Base) | Macro-AUC | ~0.70 |
| CT-CLIP (Base) | Macro-Accuracy | ~0.65 |
| CT-CLIP (Base) | Inference Speed | 1.5 sec |

### Fine-Tuned Performance

| Model | Fine-tune Data | Macro-AUC | Improvement |
|-------|-----------------|-----------|-------------|
| CT-VocabFine | Limited labels | ~0.75 | +7.1% |
| CT-ClassFine | Limited labels | ~0.78 | +11.4% |

### Trade-offs

| Aspect | Zero-Shot | VocabFine | ClassFine |
|--------|-----------|-----------|-----------|
| Accuracy | Lower | Medium | Higher |
| Speed | Medium | Medium | Fastest |
| Flexibility | Highest | Medium | Lower |
| Fine-tune Data | None | Limited | Limited |

---

## Advanced Features

### 1. Multiview Learning

Support for augmented views of same image/text:

```python
# Multiple augmentations of same CT volume
aug_image_1 = random_rotation(image)
aug_image_2 = random_crop(image)

loss = model(
    text,
    image,
    aug_image=[aug_image_1, aug_image_2],
    return_loss=True
)

# Loss includes multiview alignment
```

### 2. Extra Latent Projection (CLOOB)

Separate projections for image-to-text vs text-to-image:

```python
clip = CTCLIP(
    ...,
    extra_latent_projection=True  # Enabled
)

# Now has:
# - to_text_latent / to_text_latent_extra
# - to_visual_latent / to_visual_latent_extra
```

**Benefit**: Can learn asymmetric relationships

### 3. All Token Embeddings (FILIP)

Instead of just [CLS] token, use all tokens:

```python
clip = CTCLIP(
    ...,
    use_all_token_embeds=True  # Enabled
)

# Fine-grained matching at token level
```

**Benefit**: Better alignment for localized pathologies

### 4. Image Embedding Downsampling

Reduce spatial dimensions for faster processing:

```python
clip = CTCLIP(
    ...,
    downsample_image_embeds=True  # Enabled
)

# Applies 3D convolution with stride 2
# Output size: 147,456 (half of original)
```

**Trade-off**: Faster but potentially less accurate

---

## Running the Code

### Setup

```bash
# Install dependencies
cd models/ct-clip/transformer_maskgit
pip install -e .
cd ..

cd CT_CLIP
pip install -e .
cd ..
```

### Training

```bash
# Edit paths in run_train.py
python run_train.py

# Or with distributed training
accelerate launch --use_fsdp run_train.py
```

### Inference (Zero-Shot)

```bash
# Edit paths in run_zero_shot.py
python run_zero_shot.py
```

### Fine-tuning

```bash
# Vocabulary-based fine-tuning
python ct_vocabfine_train.py \
    --lr 1e-5 \
    --epochs 10 \
    --pretrained path_to_pretrained_model \
    --data-folder path_to_train_data \
    --reports-file path_to_reports.csv

# Classification-based fine-tuning
python ct_lipro_train.py \
    --lr 1e-5 \
    --epochs 10 \
    --pretrained path_to_pretrained_model \
    --data-folder path_to_train_data \
    --reports-file path_to_reports.csv
```

---

## Key Takeaways

1. **Architecture**: Vision Transformer (CT-ViT) + Text Transformer (BiomedVLP-BERT) + Contrastive Learning
2. **Training**: Learns to align 3D CT volumes with radiology reports in shared embedding space
3. **Flexibility**: Zero-shot, vocabulary fine-tuning, or classification fine-tuning options
4. **Dataset**: CT-RATE provides 50K+ CT volumes with paired reports and labels
5. **Performance**: Trade-off between accuracy and inference speed across model variants
6. **Hardware**: A100 GPU (80GB) for training; smaller GPUs for inference
7. **Applications**: Pathology detection, semantic search, multimodal understanding of chest CT

---

## References

- **Paper**: CT-CLIP: Revolutionizing Abnormality Detection through Chest CT Volumes and Radiology Reports (arXiv:2403.17834)
- **Dataset**: CT-RATE on HuggingFace
- **Models**: Pretrained weights on HuggingFace
- **Related Work**: CT-CHAT (visual-language chat for CT-CLIP)
- **License**: CC-BY-NC-SA (free for non-commercial research)
$$
