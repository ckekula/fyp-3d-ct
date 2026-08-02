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
PATCHES_DIR = OUTPUT_DIR / "patches"
MODELS_DIR  = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"
INFERENCE_DIR = OUTPUT_DIR / "inference"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
SPARSE_CODE_DIR = OUTPUT_DIR / "sparse_codes"
CHECKPOINT_RESUME = True

# ─── Abnormality classes ──────────────────────────────────────────────────────

ABNORMALITY_CATEGORIES = {
    "2c": "Ground-Glass Opacity",
    "2d": "Nodules"
}

CLASS_ORDER = ["normal", "2c", "2d"]
NORMAL_CLASS_IDX = 0  # CLASS_ORDER[0] == "normal"

# ─── Preprocessing ────────────────────────────────────────────────────────────

# HU window for lung parenchyma
UPPER_HU = 1000
LOWER_HU = -1000

# Target isotropic voxel spacing in mm after resampling
TARGET_SPACING_MM = (0.75, 0.75, 1.5)   # (x, y, z) = (H, W, D)
TARGET_SHAPE = (480, 480, 240)          # (H, W, D) = (coronal, sagittal, axial)

# ─── Patch extraction ─────────────────────────────────────────────────────────

PATCH_SIZE = 12
N_FEATURES = PATCH_SIZE ** 3

# Training-time patch-grid strides (voxels).
ABNORMAL_PATCH_STRIDE = 4

# Retain a normal patch only when the fraction of zero voxels is below this threshold.
ZERO_FRACTION_THRESHOLD = 0.5
LESION_FRACTION_THRESHOLD = 0.5
LESION_THRESHOLDS = {
    "2c": 0.50,
    "2d": 0.50
}

SHUFFLE_PATCHES = True

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
    "n_components":    N_FEATURES * 7,
    "n_nonzero_coefs": 10,
    "n_iter":          10,
    "exact_svd":       False,
    "mu_thresh":       0.99,
    "mem_usage":       "normal",
    "verbose":         True,
    "random_state":    RANDOM_SEED,
}

FDDL_CONFIG = {
    "n_components": N_FEATURES*7,
    "n_iter": 10,
    "dict_max_iter": 1,
    "coding_max_iter": 100,
    "random_state": RANDOM_SEED,
    "verbose": True,
    "coding_chunk_size": 131072
}
