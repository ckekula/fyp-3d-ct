# BiomedParse + RexGroundingCT Pipeline

An interpretable chest disease detection and localization pipeline using BiomedParse v2 for text-guided 3D CT segmentation.

## Overview

This pipeline uses **BiomedParse v2** (Microsoft Research, Nature Methods) to detect and localize four chest diseases in 3D CT volumes:

- **Lung Nodule**: Small spherical or oval lesions in the lung parenchyma
- **Lung Opacity**: Increased density or whiteness on CT, including ground-glass and consolidation patterns
- **Consolidation**: Dense opacity replacing normal lung tissue, typically indicating infection or fluid
- **Atelectasis**: Collapse of lung tissue, causing opacity and loss of volume

## Key Features

### 1. **Text-Guided Interpretability**

Each disease is queried with human-readable prompts such as "lung nodule" or "pulmonary consolidation". The model outputs explain where it detected the disease via pixel-level segmentation masks.

### 2. **3D Processing**

The model processes full 3D chest CT volumes (not just slices), encoding neighboring slices as RGB context for each slice during inference.

### 3. **Multi-Disease Inference**

Runs each disease detection independently, allowing per-disease confidence scores and mask outputs.

### 4. **Visual Explanations**

Generates overlay images showing the detected disease location on representative slices (25%, 50%, 75% depth).

## Architecture

```
Raw 3D CT Volume
    ↓
[Lung Window Normalization]  W:1500, L:-160
    ↓
[Image Encoder + Context Fusion]
    ↓
[Disease-Specific Text Prompt]
    ↓
[Transformer Decoder with BoltzFormer Attention]
    ↓
[Per-Slice NMS + Post-processing]
    ↓
Disease Masks + Existence Score
    ↓
[Overlay Visualization + JSON Report]
```

## Components

### `prompts.py`

Defines disease-to-prompt mapping with multiple prompt variants per disease for robustness.

```python
TARGET_DISEASES = {
    "Lung nodule": ["lung nodule", "pulmonary nodule", ...],
    "Lung opacity": ["lung opacity", "ground-glass opacity", ...],
    "Consolidation": ["consolidation", "pulmonary consolidation", ...],
    "Atelectasis": ["atelectasis", "linear atelectasis", ...],
}
```

### `dataset_adapter.py`

Parses RexGroundingCT metadata JSON and filters cases that mention target diseases in radiologist findings.

```python
cases = load_rexgroundingct_cases(metadata_json, volume_root)
target_cases = list(iter_target_cases(cases, diseases=["Lung nodule", ...]))
```

### `pipeline.py`

Main inference loop. For each case:

1. Loads 3D NIfTI volume
2. Applies lung window normalization
3. Runs BiomedParse with each disease prompt
4. Post-processes masks with NMS and object existence detection
5. Saves per-disease masks and overlay images
6. Writes JSON report

### `visualize.py`

Generates PNG overlays showing model predictions on CT slices.

## Usage

### Quick Start

```bash
python pipeline.py \
    --volume-root /path/to/RexGroundingCT/volumes \
    --output-dir ./outputs \
    --limit 5 \
    --device cuda
```

### Full Arguments

```
--metadata-json PATH
    Path to RexGroundingCT dataset.json (default: data/rexgrounding-ct/dataset.json)

--volume-root PATH (required)
    Path to folder containing .nii.gz chest CT volumes

--output-dir PATH
    Where to save masks, overlays, and reports (default: outputs/biomedparse_rexgroundingct)

--checkpoint PATH
    Path to biomedparse_v2.ckpt. If not provided, downloads from HuggingFace.

--device DEVICE
    cuda or cpu (default: auto-detect)

--limit N
    Process only first N cases (0 = all, default: 0)

--diseases LIST
    Space-separated disease names to detect
    (default: "Lung nodule" "Lung opacity" "Consolidation" "Atelectasis")
```

## Output Structure

```
outputs/biomedparse_rexgroundingct/
├── masks/
│   ├── train_1_a_1.nii.gz.npz
│   ├── train_1_a_2.nii.gz.npz
│   └── ...
├── overlays/
│   ├── Lung nodule/
│   │   ├── train_1_a_1.png
│   │   └── ...
│   ├── Lung opacity/
│   │   ├── train_1_a_1.png
│   │   └── ...
│   ├── Consolidation/
│   └── Atelectasis/
└── reports.json
```

### reports.json

```json
[
  {
    "volume_name": "train_1_a_1.nii.gz",
    "matched_diseases_in_finding_text": ["Consolidation", "Atelectasis"],
    "findings": ["Regression of consolidation areas in the middle lobe...", "Atelectatic changes in the posterobasal segment..."],
    "predictions": {
      "Lung nodule": {
        "existence_score": 0.12,
        "mask_voxels": 1245.0
      },
      "Lung opacity": {
        "existence_score": 0.87,
        "mask_voxels": 45382.0
      },
      "Consolidation": {
        "existence_score": 0.93,
        "mask_voxels": 78234.0
      },
      "Atelectasis": {
        "existence_score": 0.67,
        "mask_voxels": 32145.0
      }
    },
    "protocol": "protocol1"
  }
]
```

### Mask Files (NPZ)

Each `.npz` contains per-disease segmentation masks as NumPy arrays:

```python
import numpy as np

data = np.load("masks/train_1_a_1.nii.gz.npz")
print(data.files)  # ['lung_nodule', 'lung_opacity', 'consolidation', 'atelectasis']

lung_nodule_mask = data['lung_nodule']  # Shape: (D, H, W), dtype: float32
```

## Installation & Setup

### 1. Clone BiomedParse

```bash
cd models
git clone https://github.com/microsoft/BiomedParse.git
cd ..
```

### 2. Install Dependencies

```bash
pip install hydra-core torch nibabel matplotlib numpy huggingface_hub
```

### 3. Download RexGroundingCT Volumes

```bash
# Get metadata (usually already in data/rexgrounding-ct/dataset.json)
# Download volumes from HuggingFace
huggingface-cli download rajpurkarlab/ReXGroundingCT --repo-type dataset --local-dir data/rexgrounding-ct
```

### 4. Download BiomedParse Checkpoint

The pipeline automatically downloads the checkpoint on first run, or manually:

```bash
huggingface-cli download microsoft/BiomedParse biomedparse_v2.ckpt
```

## Interpretability & Validation

### How to Interpret Results

1. **Existence Score**: 0–1 confidence that the disease is present. Threshold typically at 0.5.
2. **Mask Voxels**: Number of voxels (3D pixels) the model marked as disease. Higher = larger detected region.
3. **Overlays**: Visual confirmation. If overlays show anatomically implausible regions, investigate the model's behavior on similar cases.

### Validation Workflow

1. Generate predictions on a test set
2. Compare mask overlays against radiologist annotations
3. Compute Dice score or IoU
4. Measure false positive rate (detections on healthy cases)

## Extending the Pipeline

### Add a New Disease

1. Edit `prompts.py`:

   ```python
   TARGET_DISEASES["Pleural effusion"] = [
       "pleural effusion",
       "fluid around the lungs",
       ...
   ]
   ```

2. Rerun the pipeline. It will automatically handle the new disease.

### Fine-tune on Your Data

See [models/biomed-parse/FINETUNING.md](../../models/biomed-parse/FINETUNING.md) for detailed instructions.

## References

- **BiomedParse v2**: https://aka.ms/biomedparse-paper
- **BoltzFormer**: https://openaccess.thecvf.com/content/CVPR2025/papers/Zhao_Boltzmann_Attention_Sampling_for_Image_Analysis_with_Small_Objects_CVPR_2025_paper.pdf
- **RexGroundingCT**: https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT
- **CT-RATE**: https://huggingface.co/datasets/ibrahimhamamci/CT-RATE

## Troubleshooting

### Out of Memory

- Reduce `--limit` to process fewer cases per run
- Use CPU (`--device cpu`) for lower memory overhead
- Reduce batch size in `pipeline.py` (line: `slice_batch_size=4`)

### Model Takes Too Long

- GPU required for practical inference times
- CPU inference is ~10x slower

### Checkpoint Download Fails

- Check internet connection
- Manually download from [HuggingFace](https://huggingface.co/microsoft/BiomedParse)
- Use `--checkpoint /local/path/to/biomedparse_v2.ckpt`

## License

This pipeline integrates Microsoft's BiomedParse (Apache 2.0) and HuggingFace's RexGroundingCT dataset. See individual repository licenses for details.
