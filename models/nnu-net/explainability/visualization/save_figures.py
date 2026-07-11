"""
save_figures.py

Generate 3D CT explanation figures:

Axial
Coronal
Sagittal

"""



from pathlib import Path


import matplotlib.pyplot as plt



from visualization.overlay import (
    create_overlay
)






def save_three_plane_figures(

        ct,

        cam,

        mask,

        output_folder,

        patient_id

):


    """
    Save three anatomical planes.

    """



    output_folder = Path(
        output_folder
    )



    output_folder.mkdir(

        parents=True,

        exist_ok=True

    )



    D,H,W = ct.shape



    # ---------------------------------
    # Axial
    # ---------------------------------

    axial = D//2



    create_overlay(

        ct[axial,:,:],

        cam[axial,:,:],

        mask[axial,:,:],

        output_folder /
        f"{patient_id}_axial.png",

        "Axial SegGradCAM"

    )




    # ---------------------------------
    # Coronal
    # ---------------------------------

    coronal = H//2



    create_overlay(

        ct[:,coronal,:],

        cam[:,coronal,:],

        mask[:,coronal,:],

        output_folder /
        f"{patient_id}_coronal.png",

        "Coronal SegGradCAM"

    )





    # ---------------------------------
    # Sagittal
    # ---------------------------------

    sagittal = W//2



    create_overlay(

        ct[:,:,sagittal],

        cam[:,:,sagittal],

        mask[:,:,sagittal],

        output_folder /
        f"{patient_id}_sagittal.png",

        "Sagittal SegGradCAM"

    )