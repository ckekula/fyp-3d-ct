## BiomedParse + RexGroundingCT Pipeline - Setup Complete ✅

### Environment Status

- **Python Environment**: `venv_pipeline` (isolated venv)
- **All Dependencies**: ✅ Installed
  - PyTorch 2.11
  - HuggingFace Hub 1.13
  - nibabel, matplotlib, scikit-image
  - Hydra, OmegaConf
  - numpy, scipy

### Pipeline Components - All Validated ✅

- **Prompts Module**: ✅ Loads 4 disease types (Lung nodule, Lung opacity, Consolidation, Atelectasis)
- **Dataset Adapter**: ✅ Parses RexGroundingCT metadata (2,992 train cases loaded)
- **Visualization**: ✅ Overlay generation ready
- **Main Pipeline**: ✅ Structure complete, ready for inference

### What's Needed to Run Full Pipeline

#### 1. BiomedParse Checkpoint (`biomedparse_v2.ckpt`)

**Status**: Requires HuggingFace authentication

**Solution**: Get access at https://huggingface.co/microsoft/BiomedParse
Then authenticate:

```bash
cd d:\My\Projects\fyp-3d-ct
venv_pipeline\Scripts\python.exe -m huggingface_hub login
# Paste your HuggingFace token
```

Or set the token as environment variable:

```powershell
$env:HF_TOKEN = "your_token_here"
```

Then download:

```bash
venv_pipeline\Scripts\python.exe download_checkpoint_and_samples.py
```

#### 2. Sample CT Volumes

**Current**: No volumes downloaded yet
**Location**: `data/rexgrounding-ct/volumes/`

**To download samples**:

```bash
cd d:\My\Projects\fyp-3d-ct
venv_pipeline\Scripts\python.exe download_checkpoint_and_samples.py
```

This will download the first 3 training cases (~600MB each).

---

### Run the Full Pipeline (Once Checkpoint & Volumes Available)

```bash
cd d:\My\Projects\fyp-3d-ct
venv_pipeline\Scripts\python.exe pipelines/biomedparse_rexgroundingct/pipeline.py \
  --volume-root data/rexgrounding-ct/volumes \
  --output-dir pipelines/biomedparse_rexgroundingct/outputs \
  --limit 3 \
  --device cuda
```

**Output**:

- `outputs/masks/` - Per-disease segmentation masks (NPZ format)
- `outputs/overlays/` - Visualization overlays (PNG slices)
- `outputs/reports/` - JSON reports with disease predictions

---

### Project Structure

```
pipelines/biomedparse_rexgroundingct/
├── prompts.py                  # Disease-to-prompt mapping
├── dataset_adapter.py          # RexGroundingCT parser
├── visualize.py               # Overlay generation
├── pipeline.py                # Main inference CLI
├── download_checkpoint_and_samples.py  # Download script
├── test_pipeline.py           # Validation tests
├── quick_test.py              # Quick check script
├── README.md                  # Full documentation
└── outputs/                   # Will contain results
    ├── masks/
    ├── overlays/
    └── reports/
```

---

### Quick Test

To verify everything is working without the checkpoint:

```bash
cd d:\My\Projects\fyp-3d-ct
venv_pipeline\Scripts\python.exe pipelines/biomedparse_rexgroundingct/test_pipeline.py
```

✅ **Result**: All imports pass, dataset loads, prompts ready

---

### Next Steps

1. **Get HuggingFace Access**: Request access to microsoft/BiomedParse
2. **Authenticate**: Set HF_TOKEN or run `huggingface_hub login`
3. **Download**: Run `download_checkpoint_and_samples.py`
4. **Execute Pipeline**: Run `pipeline.py` with volume-root pointing to downloaded data
5. **Review Results**: Check outputs for masks, overlays, and reports

---

### Interpretability Features

The pipeline provides interpretability through:

- **Per-disease segmentation masks** - Shows exactly which regions are classified as each disease
- **Overlay visualizations** - Visual confirmation on sampled CT slices
- **Confidence scores** - Model confidence for each disease prediction
- **Structured reports** - JSON with findings for each case

This allows you to validate BiomedParse's ability to localize chest diseases in CT images.
