"""
config.py
Central configuration for the LC-KSVD chest CT training pipeline.
Edit paths and hyperparameters here; everything else reads from this file.
"""

from pathlib import Path

# ─── Dataset paths ────────────────────────────────────────────────────────────

DATASET_ROOT = Path("/home/chest_ct/code/data")
VOLUMES_DIR = DATASET_ROOT / "data_volumes" / "dataset" / "train_fixed"
MASKS_DIR = DATASET_ROOT / "segmentations" / "segmentations"
METADATA_JSON = DATASET_ROOT / "rexgrounding-ct" / "dataset_2_last.json"

# ─── Output paths ─────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("outputs")
PATCHES_DIR = OUTPUT_DIR / "patches"       # saved patch matrices (.npz)
MODELS_DIR  = OUTPUT_DIR / "models"        # saved LC-KSVD models (.pkl)
RESULTS_DIR = OUTPUT_DIR / "results"       # metrics, contribution maps
INFERENCE_DIR = OUTPUT_DIR / "inference"   # per-volume segmentation masks (.nii.gz)
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_RESUME = True

# ─── Abnormality classes ──────────────────────────────────────────────────────

ABNORMALITY_CATEGORIES = {
    "normal": "Normal (no findings)",
    "2c": "Groundglass opacity",
    "2d": "Pulmonary nodules/masses",
}

CLASS_ORDER = ["normal", "2c", "2d"]
NORMAL_CLASS_IDX = 0  # CLASS_ORDER[0] == "normal"

# ─── Preprocessing ────────────────────────────────────────────────────────────

# HU window for lung parenchyma
HU_MIN = -900
HU_MAX =  -200
BACKGROUND_HU = -1000  # value to fill outside the lung mask (air)
UPPER_HU = 500
LOWER_HU = -1000

# Target isotropic voxel spacing in mm after resampling
TARGET_SPACING_MM = 1.5   # resamples all voxel spacing to 1.5×1.5×1.5

# ─── Patch extraction ─────────────────────────────────────────────────────────

PATCH_SIZE = 16            # cubic patch: 16×16×16 voxels → 12mm³ at 1.5mm spacing
N_FEATURES = PATCH_SIZE ** 3  # 4096 — dimensionality of each patch vector

# Training-time patch-grid strides (voxels).
NORMAL_PATCH_STRIDE = PATCH_SIZE
ABNORMAL_PATCH_STRIDE = 4

# Retain a normal patch only when the fraction of zero voxels is below this threshold.
ZERO_FRACTION_THRESHOLD = 0.5
LESION_FRACTION_THRESHOLD = 0.5

# ─── LC-KSVD2 hyperparameters ────────────────────────────────────────────────
RANDOM_SEED = 42

LCKSVD_CONFIG = {
    "n_components":    N_FEATURES*5,   # number of dictionary atoms K
    "n_nonzero_coefs": 10,    # sparsity T
    "alpha":           4.0,   # label-consistency weight (√α in the paper)
    "beta":            2.0,   # classifier weight (√β); LC-KSVD2 only
    "variant":         "lcksvd2",
    "n_iter":          10,    # main training iterations
    "n_iter_init":     2,    # K-SVD warm-start iterations
    "verbose":         True,
    "random_state":    RANDOM_SEED,
}

KSVD_CONFIG = {
    "n_components":    N_FEATURES * 10,
    "n_nonzero_coefs": 10,
    "n_iter":          10,
    "exact_svd":       False,
    "mu_thresh":       0.99,
    "mem_usage":       "normal",
    "verbose":         True,
    "random_state":    RANDOM_SEED,
}