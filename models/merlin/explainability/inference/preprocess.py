"""
preprocess.py

Mirrors load_and_preprocess_ct() from merlin_ct_pipeline.py line for
line, including [::4, ::4, ::2] downsampling, so the tensor fed to
Grad-CAM is identical to what the model saw in production.
"""

import numpy as np
import nibabel as nib
import torch

from explainability.configs.config import DEVICE, WINDOW_MIN, WINDOW_MAX, DOWNSAMPLE_STRIDE


class CTPreprocessor:
    def __init__(self):
        self.affine = None
        self.full_res_shape = None

    def preprocess(self, nii_path):
        img = nib.load(str(nii_path))
        volume = img.get_fdata()

        if volume.ndim == 4:
            volume = volume[..., 0]

        volume = np.clip(volume, WINDOW_MIN, WINDOW_MAX)
        volume = (volume - WINDOW_MIN) / (WINDOW_MAX - WINDOW_MIN)
        volume = np.nan_to_num(volume, nan=0.0).astype(np.float32)

        self.affine = img.affine.copy()
        self.full_res_shape = volume.shape  # (H, W, D), pre-downsample
        del img

        sh, sw, sd = DOWNSAMPLE_STRIDE
        volume_ds = volume[::sh, ::sw, ::sd]

        tensor = torch.tensor(volume_ds, dtype=torch.float32)
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1,1,H,W,D)

        return tensor.to(DEVICE), volume_ds.shape, self.affine, self.full_res_shape


def preprocess_ct(image_path):
    proc = CTPreprocessor()
    return proc.preprocess(image_path)


if __name__ == "__main__":
    # STEP 2 sanity check -- see run order in chat.
    import sys
    tensor, ds_shape, affine, full_shape = preprocess_ct(sys.argv[1])
    print("Downsampled tensor shape:", tensor.shape)
    print("Full-res shape:", full_shape)
    print("Device:", tensor.device)
