"""
overlay.py

Creates CT + SegGradCAM overlay images.

Produces:

- CT slice
- CAM heatmap
- segmentation contour


Used for thesis figures.
"""


from pathlib import Path


import numpy as np

import matplotlib.pyplot as plt



from scipy.ndimage import binary_dilation





def normalize_ct(ct):

    """
    Window CT intensity for visualization.

    Lung window.
    """

    lower = -1000

    upper = 400


    ct = np.clip(
        ct,
        lower,
        upper
    )


    ct = (
        ct - lower
    ) / (
        upper - lower
    )


    return ct






def create_overlay(

        ct_slice,

        cam_slice,

        mask_slice=None,

        output_path=None,

        title="SegGradCAM"

):


    """
    Create overlay figure.


    Parameters
    ----------

    ct_slice:
        2D CT image


    cam_slice:
        2D heatmap


    mask_slice:
        optional GT mask


    """



    ct_slice = normalize_ct(
        ct_slice
    )



    plt.figure(
        figsize=(8,8)
    )



    # CT background

    plt.imshow(

        ct_slice,

        cmap="gray"

    )



    # CAM overlay

    plt.imshow(

        cam_slice,

        cmap="jet",

        alpha=0.45,

        vmin=0,

        vmax=1

    )



    # GT boundary

    if mask_slice is not None:


        boundary = binary_dilation(
            mask_slice
        ) - mask_slice



        plt.contour(

            boundary,

            colors="lime",

            linewidths=1

        )



    plt.title(
        title
    )


    plt.axis(
        "off"
    )



    if output_path:


        Path(
            output_path
        ).parent.mkdir(

            parents=True,

            exist_ok=True

        )


        plt.savefig(

            output_path,

            dpi=300,

            bbox_inches="tight"

        )



    plt.close()