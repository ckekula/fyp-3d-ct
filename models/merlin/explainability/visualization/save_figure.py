"""
save_figure.py — writes overlay PNGs and the raw CAM as .npy / .nii.gz.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from explainability.visualization.overlay import overlay_slice, best_slice_index


def save_cam_png(ct_volume, cam, out_path, axis=0):
    idx = best_slice_index(cam, axis=axis)
    ct_slice = np.take(ct_volume, idx, axis=axis)
    cam_slice = np.take(cam, idx, axis=axis)

    img = overlay_slice(ct_slice, cam_slice)
    plt.imsave(out_path, img)
    return out_path, idx


def save_cam_nifti(cam, affine, out_path):
    nib.save(nib.Nifti1Image(cam.astype(np.float32), affine), out_path)
    return out_path