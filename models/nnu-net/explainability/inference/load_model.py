"""
load_model.py

Loads trained nnU-Net v2 model.

Responsibilities:
    - Create nnUNetPredictor
    - Load trained network
    - Load checkpoint_best.pth
    - Return initialized predictor

No inference.
No GradCAM.
"""


from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


from configs.config import (
    MODEL_FOLDER,
    CHECKPOINT_NAME,
    FOLD,
    DEVICE,
    TILE_STEP_SIZE,
    USE_GAUSSIAN,
    USE_MIRRORING
)



class NNUNetModelLoader:
    """
    Wrapper around nnUNetPredictor.
    """


    def __init__(self):

        self.predictor = None



    def load_model(self):
        """
        Initializes nnU-Net model.

        Returns
        -------
        nnUNetPredictor
        """


        print("="*60)
        print("Loading nnU-Net model")
        print("="*60)


        predictor = nnUNetPredictor(

            # same geometry as training inference
            tile_step_size=TILE_STEP_SIZE,


            # Gaussian importance weighting
            use_gaussian=USE_GAUSSIAN,


            # IMPORTANT:
            # Disabled for GradCAM
            # because gradients from flipped
            # copies are not meaningful
            use_mirroring=USE_MIRRORING,


            # Run network on GPU
            perform_everything_on_device=True,


            device=DEVICE,


            verbose=True,


            allow_tqdm=True

        )


        print("\nInitializing checkpoint...")


        predictor.initialize_from_trained_model_folder(

            model_training_output_dir=MODEL_FOLDER,


            use_folds=(FOLD,),


            checkpoint_name=CHECKPOINT_NAME

        )


        print("\nModel loaded successfully")


        print(
            "Network:",
            type(predictor.network)
        )


        self.predictor = predictor


        return predictor