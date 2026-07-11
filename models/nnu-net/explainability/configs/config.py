from pathlib import Path
import torch


# ======================================================
# ROOT DIRECTORY
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ======================================================
# nnU-Net storage
# ======================================================

NNUNET_ROOT = (
    PROJECT_ROOT /
    "models" /
    "nnu-net" /
    "storage"
)


# ======================================================
# DATASET
# ======================================================

DATASET_NAME = "Dataset100_GGO"



# ======================================================
# MODEL LOCATION
# ======================================================


MODEL_FOLDER = (
    NNUNET_ROOT /
    "nnUNet_results" /
    DATASET_NAME /
    "nnUNetTrainer__nnUNetPlans__3d_fullres"
)



CHECKPOINT_NAME = "checkpoint_best.pth"


# choose fold
FOLD = 4



# ======================================================
# DATA
# ======================================================


IMAGE_FOLDER = (
    NNUNET_ROOT /
    "nnUNet_raw" /
    DATASET_NAME /
    "imagesTs"
)


LABEL_FOLDER = (
    NNUNET_ROOT /
    "nnUNet_raw" /
    DATASET_NAME /
    "labelsTs"
)



# ======================================================
# OUTPUT
# ======================================================


OUTPUT_ROOT = (
    PROJECT_ROOT /
    "models" /
    "nnu-net" /
    "explainability" /
    "outputs"
)


PREDICTION_FOLDER = (
    OUTPUT_ROOT /
    "predictions"
)


HEATMAP_FOLDER = (
    OUTPUT_ROOT /
    "heatmaps"
)


OVERLAY_FOLDER = (
    OUTPUT_ROOT /
    "overlays"
)


FIGURE_FOLDER = (
    OUTPUT_ROOT /
    "figures"
)



for folder in [
    PREDICTION_FOLDER,
    HEATMAP_FOLDER,
    OVERLAY_FOLDER,
    FIGURE_FOLDER
]:

    folder.mkdir(
        parents=True,
        exist_ok=True
    )



# ======================================================
# DEVICE
# ======================================================


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)



# ======================================================
# SEGGRADCAM SETTINGS
# ======================================================


TARGET_CLASS = 1


USE_GAUSSIAN = True


USE_MIRRORING = False


TILE_STEP_SIZE = 0.5