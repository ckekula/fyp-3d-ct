import os
import glob
import nibabel as nib
import numpy as np

import torch
import torch.nn.functional as F

from torch.utils.data import Dataset


class ReXGroundDatasetInfer(Dataset):

    def __init__(self, data_folder):

        self.files = sorted(
            glob.glob(
                os.path.join(
                    data_folder,
                    "*.nii.gz"
                )
            )
        )

        if len(self.files) == 0:
            raise RuntimeError(
                f"No NIfTI files found in {data_folder}"
            )

        print(
            f"Found {len(self.files)} CT volumes"
        )


    def __len__(self):

        return len(self.files)



    def resize_array(
        self,
        array,
        current_spacing,
        target_spacing
    ):

        """
        Resize CT volume according to voxel spacing
        """

        original_shape = array.shape[2:]


        scaling_factors = [
            current_spacing[i] /
            target_spacing[i]
            for i in range(3)
        ]


        new_shape = [
            int(
                original_shape[i] *
                scaling_factors[i]
            )
            for i in range(3)
        ]


        resized = F.interpolate(
            array,
            size=new_shape,
            mode="trilinear",
            align_corners=False
        )


        return resized.cpu().numpy()



    def preprocess(self, path):

        nii = nib.load(path)


        img = nii.get_fdata()


        print(
            "Original CT shape:",
            img.shape
        )


        #
        # ReXGround NIfTI is already HU
        #

        img = np.clip(
            img,
            -1000,
            1000
        )



        #
        # spacing from NIfTI header
        #

        spacing = nii.header.get_zooms()


        current_spacing = (
            spacing[2],
            spacing[0],
            spacing[1]
        )


        target_spacing = (
            1.5,
            0.75,
            0.75
        )


        #
        # CT-LiPro expects Z,X,Y
        #

        img = img.transpose(
            2,
            0,
            1
        )


        tensor = torch.tensor(
            img
        ).float()



        tensor = (
            tensor
            .unsqueeze(0)
            .unsqueeze(0)
        )


        #
        # Resample
        #

        img = self.resize_array(
            tensor,
            current_spacing,
            target_spacing
        )


        img = img[0][0]


        img = np.transpose(
            img,
            (1,2,0)
        )


        #
        # Normalize
        #

        img = (
            img / 1000.0
        ).astype(
            np.float32
        )



        tensor = torch.tensor(
            img
        )



        #
        # CT-LiPro input size
        #

        target_shape = (
            480,
            480,
            240
        )


        h,w,d = tensor.shape

        th,tw,td = target_shape



        #
        # Center crop
        #

        h_start = max(
            (h-th)//2,
            0
        )

        w_start = max(
            (w-tw)//2,
            0
        )

        d_start = max(
            (d-td)//2,
            0
        )



        tensor = tensor[
            h_start:h_start+th,
            w_start:w_start+tw,
            d_start:d_start+td
        ]



        #
        # Padding
        #

        pad_h = th - tensor.shape[0]
        pad_w = tw - tensor.shape[1]
        pad_d = td - tensor.shape[2]


        tensor = F.pad(
            tensor,
            (
                0,
                pad_d,
                0,
                pad_w,
                0,
                pad_h
            ),
            value=-1
        )



        #
        # CTViT expects:
        # (C,D,H,W)
        #

        tensor = tensor.permute(
            2,
            0,
            1
        )


        tensor = tensor.unsqueeze(0)



        print(
            "Final CT tensor:",
            tensor.shape
        )


        return tensor



    def __getitem__(self,index):

        path = self.files[index]


        image = self.preprocess(
            path
        )


        name = os.path.basename(
            path
        ).replace(
            ".nii.gz",
            ""
        )


        #
        # Dummy labels
        # because inference has no GT requirement
        #

        labels = torch.zeros(
            18
        )


        return (
            image,
            "",
            labels,
            name
        )