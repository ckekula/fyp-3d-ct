"""
predict.py

Runs nnU-Net v2 inference.

Input:
    CT NIfTI files

Output:
    Segmentation prediction NIfTI

Uses:
    nnUNetPredictor.predict_from_files()

The prediction geometry is identical
to normal nnU-Net inference.
"""


from pathlib import Path


from configs.config import (
    PREDICTION_FOLDER
)





class NNUNetPredictorWrapper:
    """
    Wrapper around nnUNetPredictor.
    """



    def __init__(
            self,
            predictor
    ):

        self.predictor = predictor




    def predict(
            self,
            input_folder
    ):
        """
        Run nnU-Net segmentation.

        Parameters
        ----------
        input_folder:
            Folder containing CT volumes

        Returns
        -------
        prediction folder
        """



        print("="*60)

        print("Running nnU-Net inference")

        print("="*60)



        input_folder = str(
            input_folder
        )


        output_folder = str(
            PREDICTION_FOLDER
        )



        self.predictor.predict_from_files(

            input_folder,

            output_folder,


            # save softmax probabilities
            # required for some analyses
            save_probabilities=False

        )



        print(
            "\nPrediction saved:"
        )


        print(
            output_folder
        )



        return output_folder