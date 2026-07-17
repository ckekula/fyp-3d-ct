"""
overlay.py — overlays a normalized [0,1] 3D CAM on top of the CT volume slices.
"""

import numpy as np


def overlay_slice(ct_slice, cam_slice, alpha=0.45, cmap_thresh=0.15):
    """Returns an RGB uint8 image blending a grayscale CT slice with a heatmap."""
    import matplotlib.cm as cm

    ct_norm = (ct_slice - ct_slice.min()) / (np.ptp(ct_slice) + 1e-8)
    base = np.stack([ct_norm] * 3, axis=-1)

    heat = cm.get_cmap("jet")(cam_slice)[..., :3]
    mask = (cam_slice > cmap_thresh).astype(np.float32)[..., None]

    blended = base * (1 - alpha * mask) + heat * (alpha * mask)
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


def best_slice_index(cam, axis=0):
    """Pick the slice along `axis` with the strongest CAM response."""
    energy = cam.sum(axis=tuple(a for a in range(3) if a != axis))
    return int(np.argmax(energy))