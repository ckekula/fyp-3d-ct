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
METADATA_JSON = DATASET_ROOT / "rexgrounding-ct" / "dataset_4.json"

# ─── Output paths ─────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("outputs")
PATCHES_DIR = OUTPUT_DIR / "patches"       # saved patch matrices (.npz)
MODELS_DIR  = OUTPUT_DIR / "models"        # saved LC-KSVD models (.pkl)
RESULTS_DIR = OUTPUT_DIR / "results"       # metrics, contribution maps

# ─── Abnormality classes ──────────────────────────────────────────────────────

ABNORMALITY_CATEGORIES = {
    "2a": "Linear (including subsegmental atelectasis, scarring, fibrosis)",
    "2b": "Atelectasis, consolidation",
    "2c": "Groundglass opacity",
    "2d": "Pulmonary nodules/masses",
}

# ─── Preprocessing ────────────────────────────────────────────────────────────

# HU window for lung parenchyma
HU_MIN = -1000
HU_MAX =  200

# Target isotropic voxel spacing in mm after resampling
TARGET_SPACING_MM = 1.5   # resamples 0.75×0.75×1.5 and 1×1×3 to 1.5×1.5×1.5

# ─── Patch extraction ─────────────────────────────────────────────────────────

PATCH_SIZE = 32            # cubic patch: 32×32×32 voxels → 48mm³ at 1.5mm spacing
N_FEATURES = PATCH_SIZE ** 3  # 32768 — dimensionality of each patch vector

# Minimum fraction of patch voxels that must overlap the lesion mask
# for a patch to be considered a positive sample
# (lowered from 0.10 to 0.05 to capture small lesions like opacities/consolidations)
MIN_OVERLAP_RATIO = 0.05

# Number of positive patches to sample per lesion-containing scan
N_POSITIVE_PATCHES_PER_SCAN = 30

# Ratio of negative patches to positive patches in the final matrix
NEG_TO_POS_RATIO = 1.0     # balanced by default; increase if you want more negatives

# Random seed for reproducible patch sampling
RANDOM_SEED = 42

# ─── LC-KSVD2 hyperparameters ────────────────────────────────────────────────

LCKSVD_CONFIG = {
    "n_components":    128,   # number of dictionary atoms K; ablate [64, 128, 256]
    "n_nonzero_coefs": 10,    # sparsity T (~8% of K=128); ablate [5, 10, 20]
    "alpha":           4.0,   # label-consistency weight (√α in the paper)
    "beta":            2.0,   # classifier weight (√β); LC-KSVD2 only
    "variant":         "lcksvd2",
    "n_iter":          50,    # main training iterations
    "n_iter_init":     20,    # K-SVD warm-start iterations
    "verbose":         True,
    "random_state":    RANDOM_SEED,
}

# ─── Localization / contribution map ─────────────────────────────────────────

# Threshold strategy for binarizing the contribution map into a mask.
# "otsu"  → compute Otsu threshold from validation set contribution maps
# "fixed" → use CONTRIB_FIXED_THRESHOLD below
CONTRIB_THRESHOLD_MODE = "otsu"
CONTRIB_FIXED_THRESHOLD = 0.3

# Only atoms whose classifier weight |W[class, atom]| exceeds this percentile
# (computed per class over the full W matrix) are included in back-projection.
# Prevents reconstruction-only atoms from polluting the localization map.
DISCRIMINATIVE_ATOM_PERCENTILE = 75