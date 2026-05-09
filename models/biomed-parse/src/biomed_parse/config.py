from __future__ import annotations

from pathlib import Path
from .prompts import DEFAULT_DISEASES


ROOT = Path("/home/chest_ct/code")
MODEL_DIR = ROOT / "models" / "biomed-parse"

METADATA_JSON = ROOT / "data" / "rexgrounding-ct" / "dataset_4.json"
VOLUME_ROOT = ROOT / "data" / "data_volumes"
CHECKPOINT = ROOT / "models" / "biomed-parse" / "model_weights" / "biomedparse_v2.ckpt"
OUTPUT_DIR = ROOT / "outputs" / "biomedparse"

DEVICE = "cuda"
DISEASES = DEFAULT_DISEASES

EXISTENCE_THRESHOLD = 0.30
POSTPROCESS_THRESHOLD = 0.30
MIN_MASK_VOXELS = 1
SLICE_BATCH_SIZE = 4