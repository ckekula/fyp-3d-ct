"""
config.py
Central configuration for the LC-KSVD chest CT training pipeline.
Edit paths and hyperparameters here; everything else reads from this file.
"""

from pathlib import Path

# ─── Dataset paths ────────────────────────────────────────────────────────────

# Root of the ReXGroundingCT dataset
DATASET_ROOT = Path("/home/chest_ct/code/data")

# CT volumes: nested under data_volumes/dataset/train/<study>/<series>/
VOLUMES_DIR = DATASET_ROOT / "data_volumes" / "dataset" / "train_fixed"

# Segmentation masks: flat folder, same filename as volume
MASKS_DIR = DATASET_ROOT / "segmentations" / "segmentations"

# JSON metadata file mapping F-dimension index → finding label per scan
# Expected structure:  { "train_1_a_1": { "0": "Lung nodule", "1": "Lung opacity", ... }, ... }
METADATA_JSON = DATASET_ROOT / "rexgrounding-ct" / "dataset_transformed_filtered.json"

# CSV with per-volume one-hot labels
# Expected columns: filename, lung_nodule, lung_opacity, consolidation, atelectasis
LABELS_CSV = DATASET_ROOT / "ct-rate" / "train_labels.csv"

# ─── Output paths ─────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("outputs")
PATCHES_DIR = OUTPUT_DIR / "patches"       # saved patch matrices (.npz)
MODELS_DIR  = OUTPUT_DIR / "models"        # saved LC-KSVD models (.pkl)
RESULTS_DIR = OUTPUT_DIR / "results"       # metrics, contribution maps

# ─── Abnormality classes ──────────────────────────────────────────────────────

# Canonical names — order matters; used as dictionary keys and file suffixes
ABNORMALITIES = [
    "Lung nodule",
    "Lung opacity",
    "Consolidation",
    "Atelectasis",
]

# Matching label strings as they appear in the JSON metadata finding descriptions
# Add lowercase variants that appear in your dataset's JSON
ABNORMALITY_ALIASES = {
    "Lung nodule":    ["lung nodule", "pulmonary nodule", "nodule", "nodules"],
    "Lung opacity":   ["lung opacity", "opacity", "ground-glass opacity", "ground glass opacity", "groundglass opacity"],
    "Consolidation":  ["consolidation", "consolidations"],
    "Atelectasis":    ["atelectasis", "atelectatic"],
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
MIN_OVERLAP_RATIO = 0.10

# Number of positive patches to sample per lesion-containing scan
N_POSITIVE_PATCHES_PER_SCAN = 30

# Ratio of negative patches to positive patches in the final matrix
NEG_TO_POS_RATIO = 1.0     # balanced by default; increase if you want more negatives

# Random seed for reproducible patch sampling
RANDOM_SEED = 42

# ─── Train / val / test split ─────────────────────────────────────────────────

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15          # must sum to 1.0

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