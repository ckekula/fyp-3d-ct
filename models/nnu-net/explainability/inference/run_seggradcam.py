"""
run_seggradcam.py


Complete nnU-Net SegGradCAM pipeline.


Pipeline:

CT NIfTI

    |
    v

Load nnU-Net checkpoint

    |
    v

Preprocess CT

    |
    v

nnU-Net segmentation

    |
    v

Prediction mask

    |
    v

SlidingWindowSegGradCAM3D

    |
    v

3D CAM heatmap

    |
    v

Save NIfTI


"""


from pathlib import Path

import torch



# -------------------------------
# Model
# -------------------------------

from inference.load_model import (
    NNUNetModelLoader
)



# -------------------------------
# Preprocessing
# -------------------------------

from inference.preprocess import (
    NNUNetPreprocessor
)



# -------------------------------
# Segmentation
# -------------------------------

from inference.predict import (
    NNUNetPredictorWrapper
)



# -------------------------------
# GradCAM
# -------------------------------

from seggradcam.sliding_window_seggradcam3d import (
    SlidingWindowSegGradCAM3D
)



# -------------------------------
# IO
# -------------------------------

from utils.io import (
    save_nifti
)



from configs.config import (
    IMAGE_FOLDER,
    LABEL_FOLDER,
    HEATMAP_FOLDER,
    TARGET_CLASS
)





def run():


    print("\n")
    print("="*70)
    print(" nnU-Net 3D SegGradCAM Pipeline ")
    print("="*70)



    # ==================================================
    # 1. Load trained nnU-Net
    # ==================================================


    loader = NNUNetModelLoader()


    predictor = loader.load_model()



    # ==================================================
    # 2. Create preprocessing engine
    # ==================================================


    preprocessor = NNUNetPreprocessor(
        predictor
    )



    # ==================================================
    # 3. Create segmentation engine
    # ==================================================


    segmenter = NNUNetPredictorWrapper(
        predictor
    )



    # ==================================================
    # 4. Create GradCAM engine
    # ==================================================


    cam_engine = SlidingWindowSegGradCAM3D(

        predictor

    )



    # ==================================================
    # Select CT
    # ==================================================


    ct_files = sorted(
        Path(IMAGE_FOLDER)
        .glob("*.nii.gz")
    )



    if len(ct_files)==0:

        raise RuntimeError(
            "No CT files found"
        )



    ct_file = ct_files[0]



    print(
        "\nProcessing:",
        ct_file.name
    )



    # ==================================================
    # 5. Preprocess CT
    # ==================================================


    data, properties = (
        preprocessor.preprocess(
            ct_file
        )
    )



    print(
        "Input tensor:",
        data.shape
    )



    # ==================================================
    # 6. Segmentation
    # ==================================================

    print(
        "\nRunning segmentation..."
    )


    prediction_folder = (
        segmenter.predict(
            IMAGE_FOLDER
        )
    )



    print(
        "Prediction completed"
    )



    # ==================================================
    # 7. Load GT mask
    # ==================================================

    from utils.io import load_nifti


    patient_id = (
        ct_file.name
        .replace("_0000.nii.gz","")
    )



    mask_file = (

        Path(LABEL_FOLDER)

        /

        f"{patient_id}.nii.gz"

    )



    if not mask_file.exists():

        raise FileNotFoundError(
            f"GT mask not found: {mask_file}"
        )



    gt_mask, _ = load_nifti(
        mask_file
    )



    gt_mask = torch.from_numpy(
        gt_mask
    )



    # ==================================================
    # 8. Generate SegGradCAM
    # ==================================================


    print(
        "\nGenerating SegGradCAM..."
    )



    heatmap = cam_engine(

        data,

        gt_mask,

        TARGET_CLASS

    )



    print(
        "Heatmap generated"
    )



    # ==================================================
    # 9. Save CAM
    # ==================================================


    output_file = (

        Path(HEATMAP_FOLDER)

        /

        f"{patient_id}_cam.nii.gz"

    )


    save_nifti(

        heatmap,

        properties,

        output_file

    )


    print(
        "\nSaved:"
    )

    print(
        output_file
    )



    cam_engine.remove_hooks()



    print("\nDONE")






if __name__ == "__main__":

    run()