"""
preprocess.py — mirrors load_and_preprocess_ct() from pipelineall.py exactly,
so Grad-CAM sees the same input distribution the model was actually run on.
"""

import numpy as np
import nibabel as nib
import torch

from explainability.configs.config import DEVICE, WINDOW_MIN, WINDOW_MAX


class CTPreprocessor:
    def __init__(self):
        self.affine = None

    def preprocess(self, nii_path):
        img = nib.load(str(nii_path))
        volume = img.get_fdata(dtype=np.float32)
        np.nan_to_num(volume, nan=0.0, copy=False)

        if volume.ndim == 4:
            volume = np.mean(volume, axis=0, dtype=np.float32)
        volume = np.squeeze(volume)
        assert volume.ndim == 3, f"Expected 3D volume, got {volume.shape}"

        np.clip(volume, WINDOW_MIN, WINDOW_MAX, out=volume)
        volume += abs(WINDOW_MIN)
        volume /= (WINDOW_MAX - WINDOW_MIN)

        tensor = torch.tensor(volume, dtype=torch.float32)
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)

        self.affine = img.affine.copy()
        orig_shape = volume.shape
        del img

        return tensor.to(DEVICE), orig_shape, self.affine


def preprocess_ct(image_path):
    proc = CTPreprocessor()
    tensor, orig_shape, affine = proc.preprocess(image_path)
    return tensor, orig_shape, affine