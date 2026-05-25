from __future__ import annotations

from pathlib import Path

try:
    from .prompts import DEFAULT_DISEASES
except ImportError:
    from prompts import DEFAULT_DISEASES


ROOT = Path(__file__).resolve().parents[2]

METADATA_JSON = ROOT / "data" / "Govindu" / "rexgrounding-ct" / "dataset.json"
VOLUME_ROOT = ROOT / "data" / "data_volumes"
CHECKPOINT = ROOT / "models" / "biomed-parse" / "model_weights" / "biomedparse_v2.ckpt"
OUTPUT_DIR = ROOT / "outputs" / "biomedparse_rexgroundingct_fixed_cuda"

DEVICE = "cuda"
DISEASES = DEFAULT_DISEASES

EXISTENCE_THRESHOLD = 0.30
POSTPROCESS_THRESHOLD = 0.30
MIN_MASK_VOXELS = 1
SLICE_BATCH_SIZE = 1