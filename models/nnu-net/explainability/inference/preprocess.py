"""
preprocess.py

Uses nnU-Net v2 preprocessing pipeline.

Input:
    Raw CT NIfTI

Output:
    Preprocessed tensor ready for nnU-Net network

Shape:

    (1,1,D,H,W)

This is the same representation used by
predict_logits_from_preprocessed_data()
"""


from pathlib import Path
import torch
import numpy as np


from nnunetv2.preprocessing.preprocessors.default_preprocessor import (
    DefaultPreprocessor
)


from nnunetv2.utilities.plans_handling.plans_handler import (
    PlansManager
)


from nnunetv2.utilities.dataset_name_id_conversion import (
    maybe_convert_to_dataset_name
)


from batchgenerators.utilities.file_and_folder_operations import (
    load_json
)



from configs.config import (
    NNUNET_ROOT,
    DATASET_NAME,
    DEVICE
)





class NNUNetPreprocessor:
    """
    Wrapper around nnU-Net v2 preprocessing.
    """



    def __init__(
            self,
            predictor
    ):

        self.predictor = predictor


        self.configuration_manager = (
            predictor.configuration_manager
        )


        self.plans_manager = (
            predictor.plans_manager
        )



    def preprocess(
            self,
            image_file
    ):
        """
        Parameters
        ----------
        image_file:
            Path to CT nifti

        Returns
        -------
        data:
            torch tensor

        properties:
            nnU-Net metadata
        """


        image_file = str(
            image_file
        )



        print(
            "\nPreprocessing:",
            image_file
        )



        # ------------------------------------------------
        # nnU-Net uses its own data loader
        # ------------------------------------------------


        data, properties = (
            self.predictor
            .preprocessing_iterator
            if False
            else
            self._preprocess_single_file(
                image_file
            )
        )



        return data, properties





    def _preprocess_single_file(
            self,
            image_file
    ):

        """
        Internal nnU-Net preprocessing call.
        """


        dataset_json = (
            self.predictor.dataset_json
        )


        plans_manager = (
            self.predictor.plans_manager
        )


        configuration_manager = (
            self.predictor.configuration_manager
        )



        preprocessor = (
            DefaultPreprocessor()
        )



        data, _, properties = (
            preprocessor.run_case_npy(
                image_files=[image_file],
                seg_file=None,
                plans_manager=plans_manager,
                configuration_manager=configuration_manager,
                dataset_json=dataset_json
            )
        )



        # numpy -> torch

        data = torch.from_numpy(
            data
        ).float()



        # add batch dimension

        if data.ndim == 4:

            data = data.unsqueeze(0)



        data = data.to(
            DEVICE
        )



        return data, properties