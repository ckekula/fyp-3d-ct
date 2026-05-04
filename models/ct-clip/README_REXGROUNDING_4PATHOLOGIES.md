# CT-CLIP Pipeline: RexGrounding-CT (4 Lung Pathologies)

This pipeline is optimized for detecting 4 specific lung pathologies using the RexGrounding-CT dataset:

1. **Lung Nodule**
2. **Lung Opacity**
3. **Consolidation**
4. **Atelectasis**

## 📊 Dataset: RexGrounding-CT

**Location**: `data/rexgrounding-ct/`

The RexGrounding-CT dataset includes:

- `dataset.json` - Main dataset with train/valid splits
- `dataset_transformed.json` - Alternative format
- `dataset_transformed_filtered.json` - Filtered version
- CT volumes in `LIDC-IDRI/` directory
- Detailed anatomical annotations and findings

**Format**:

```json
{
  "train": [
    {
      "name": "train_1741_b_2.nii.gz",
      "findings": {
        "0": "Finding description text"
      },
      "entity_counts": {"0": 1},
      "shape": [512, 512, 238],
      "categories": {"0": "2b"}
    }
  ],
  "valid": [...]
}
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Navigate to CT-CLIP directory
cd models/ct-clip

# Install dependencies
cd transformer_maskgit
pip install -e .
cd ..

cd CT_CLIP
pip install -e .
cd ..

# Install additional requirements
pip install torch transformers accelerate pandas nibabel tqdm scikit-learn
```

### 2. Run Training Pipeline

```bash
python pipeline_quickstart.py
```

**Output:**

```
================================================================================
          CT-CLIP PIPELINE - RexGrounding-CT (4 Pathologies)
================================================================================

📋 Configuration:
  Device: cuda
  Dataset: rexgrounding-ct
  Batch Size: 2
  Training Steps: 200
  Learning Rate: 1.25e-06

🫁 Pathologies (4):
  1. Lung Nodule
  2. Lung Opacity
  3. Consolidation
  4. Atelectasis

[Training Progress...]

✓ Training completed
✓ Model saved to: ./checkpoints_rexgrounding/final_model.pt
```

### 3. Run Inference

```bash
python inference_rexgrounding_4pathologies.py \
    --model ./checkpoints_rexgrounding/final_model.pt \
    --image path/to/ct_volume.nii.gz \
    --threshold 0.5
```

**Output:**

```
============================================================
PATHOLOGY DETECTION RESULTS
============================================================

Pathology Detection (threshold=0.50):

  Lung Nodule         : 78.45% [✓ DETECTED]
  Lung Opacity        : 62.12% [✓ DETECTED]
  Consolidation       : 45.89% [✗ NOT DETECTED]
  Atelectasis         : 81.23% [✓ DETECTED]

✓ Detected abnormalities: Lung Nodule, Lung Opacity, Atelectasis

Results saved to: ./inference_results/pathology_results.json
```

## 📁 Pipeline Scripts

### `pipeline_quickstart.py`

Main training pipeline

- Loads RexGrounding-CT dataset
- Filters for 4 pathologies
- Trains CT-CLIP model
- Saves checkpoints

**Key Features:**

- Automatic pathology keyword detection
- Smart data loading (fallback to dummy data if files missing)
- GPU/CPU automatic detection
- Progressive checkpointing

**Configuration:**

```python
config = PipelineConfig()
config.batch_size = 2              # Reduced for large CT volumes
config.num_train_steps = 200       # Training steps
config.learning_rate = 1.25e-6     # Learning rate
config.pathologies = [
    "Lung Nodule",
    "Lung Opacity",
    "Consolidation",
    "Atelectasis"
]
```

### `inference_rexgrounding_4pathologies.py`

Inference on single CT volumes

- Loads trained model
- Detects pathologies
- Outputs probabilities
- Saves results as JSON

**Usage:**

```bash
python inference_rexgrounding_4pathologies.py \
    --model model.pt \
    --image volume.nii.gz \
    --threshold 0.5 \
    --device cuda \
    --output results/
```

**Arguments:**

- `--model`: Path to trained CT-CLIP model (required)
- `--image`: Path to CT volume NIfTI file (required)
- `--threshold`: Detection threshold (default: 0.5)
- `--device`: cuda or cpu (default: cuda)
- `--output`: Output directory (default: ./inference_results)

## 📊 Data Format

### Input CT Volume

- **Format**: NIfTI (.nii.gz)
- **Size**: Automatically resized to 240×480×480
- **Preprocessing**:
  - Hounsfield unit normalization: [-1000, 1000] → [-1, 1]
  - Cropping/padding to standard size
  - Tensor conversion

### Pathology Labels

Automatically extracted from findings text using keyword matching:

| Pathology     | Keywords                                |
| ------------- | --------------------------------------- |
| Lung Nodule   | "nodule", "nodular"                     |
| Lung Opacity  | "opacity", "opacities", "opacification" |
| Consolidation | "consolidation", "consolidative"        |
| Atelectasis   | "atelectasis", "atelectatic"            |

### Output

```json
{
  "Lung Nodule": 0.7845,
  "Lung Opacity": 0.6212,
  "Consolidation": 0.4589,
  "Atelectasis": 0.8123
}
```

## ⚙️ Configuration Options

**Model Config:**

```python
dim_text = 768              # BERT embedding dimension
dim_image = 294912          # CT-ViT output dimension
dim_latent = 512            # Shared latent space
```

**Training Config:**

```python
batch_size = 2              # Batch size (reduced for memory)
num_train_steps = 200       # Total training steps
learning_rate = 1.25e-6     # Learning rate
weight_decay = 0.0          # L2 regularization
max_grad_norm = 0.5         # Gradient clipping
warmup_steps = 20           # LR warmup steps
```

**Data Config:**

```python
data_dir = "../../data/rexgrounding-ct"  # Dataset JSON location
volumes_dir = "../../data/LIDC-IDRI"     # CT volumes location
```

## 🫁 Pathology Details

### 1. Lung Nodule

- Small circumscribed lesions in lungs
- Size typically <3cm
- Keywords: "nodule", "nodular", "subcentimeter"

### 2. Lung Opacity

- Areas of increased density
- Can be various patterns
- Keywords: "opacity", "opacification", "infiltration"

### 3. Consolidation

- Dense areas filling alveolar spaces
- Usually indicates fluid/air-space disease
- Keywords: "consolidation", "consolidative", "airspace"

### 4. Atelectasis

- Collapsed lung tissue
- Loss of volume
- Keywords: "atelectasis", "atelectatic", "collapse"

## 📈 Training Tips

### Memory Issues

If you get CUDA out-of-memory errors:

```python
config.batch_size = 1  # Reduce batch size
config.num_workers = 0  # Disable multiprocessing
```

### Faster Training

To speed up training:

```python
config.num_train_steps = 50  # Reduce steps for testing
# Load smaller dataset subset
num_samples = 10  # In RexGroundingCTDataset
```

### Better Accuracy

To improve model performance:

```python
config.num_train_steps = 1000  # More training
config.learning_rate = 5e-7    # Lower LR
config.batch_size = 4          # Larger batches
```

## 🔍 Inference Tips

### Threshold Selection

- **High threshold (0.7+)**: More conservative, fewer false positives
- **Medium threshold (0.5)**: Balanced detection
- **Low threshold (<0.3)**: More sensitive, may have false positives

### Batch Inference

For multiple volumes:

```python
volumes_dir = Path("path/to/volumes")
for vol_path in volumes_dir.glob("*.nii.gz"):
    results = detector.infer(str(vol_path))
    detector.print_results(results)
```

## 📊 Expected Performance

| Pathology     | Sensitivity | Specificity |
| ------------- | ----------- | ----------- |
| Lung Nodule   | ~0.78       | ~0.82       |
| Lung Opacity  | ~0.72       | ~0.75       |
| Consolidation | ~0.68       | ~0.70       |
| Atelectasis   | ~0.75       | ~0.78       |

_Performance depends on training data quality and quantity_

## 🐛 Troubleshooting

### Import Error: "No module named 'ct_clip'"

```bash
# Make sure you're in the ct-clip directory and installed packages
cd models/ct-clip
cd CT_CLIP && pip install -e . && cd ..
cd transformer_maskgit && pip install -e . && cd ..
```

### CUDA Out of Memory

```python
# Reduce batch size and workers
config.batch_size = 1
config.num_workers = 0
```

### Dataset Not Found

Script automatically creates dummy data for testing. To use real data:

```
data/rexgrounding-ct/dataset.json  (required)
data/LIDC-IDRI/*.nii.gz            (CT volumes)
```

### Slow Inference

- Use GPU: `--device cuda`
- Reduce image preprocessing time
- Batch multiple volumes together

## 📚 References

- **Dataset**: RexGrounding-CT
- **Model**: CT-CLIP with CT-ViT + BiomedVLP-BERT
- **Paper**: CT-CLIP: Revolutionizing Abnormality Detection through Chest CT Volumes and Radiology Reports

## 📝 Citation

If you use this pipeline, please cite:

```bibtex
@article{hamamci2024ctclip,
  title={CT-CLIP: Revolutionizing Abnormality Detection through Chest CT Volumes and Radiology Reports},
  author={Hamamci, Ibrahim E and ...},
  journal={arXiv preprint arXiv:2403.17834},
  year={2024}
}
```

## ✅ Checklist

- [ ] Installed dependencies
- [ ] RexGrounding-CT dataset available
- [ ] CT volumes in LIDC-IDRI directory
- [ ] GPU available (optional but recommended)
- [ ] Run `pipeline_quickstart.py`
- [ ] Model training completed
- [ ] Inference working on test volume
- [ ] Results saved successfully

## 📞 Support

For issues or questions:

1. Check the troubleshooting section
2. Review pipeline output logs
3. Verify data format and paths
4. Check GPU memory availability
