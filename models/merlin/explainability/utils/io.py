import os
import numpy as np


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_cam_npy(cam, out_path):
    np.save(out_path, cam)
    return out_path
