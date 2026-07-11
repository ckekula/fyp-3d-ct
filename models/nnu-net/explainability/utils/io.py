"""
io.py

Medical image IO utilities.

Handles:

- Loading NIfTI (.nii.gz)
- Saving prediction masks
- Saving SegGradCAM heatmaps

Uses nibabel.

"""



from pathlib import Path

import numpy as np

import nibabel as nib





# ======================================================
# Load NIfTI
# ======================================================


def load_nifti(
        file_path
):
    """
    Load NIfTI volume.


    Parameters
    ----------
    file_path:
        Path to .nii.gz


    Returns
    -------

    volume:
        numpy array

    affine:
        transformation matrix

    """



    file_path = Path(
        file_path
    )



    if not file_path.exists():

        raise FileNotFoundError(
            file_path
        )



    nii = nib.load(
        str(file_path)
    )



    volume = nii.get_fdata()



    affine = nii.affine



    return volume, affine






# ======================================================
# Save NIfTI
# ======================================================


def save_nifti(
        volume,
        affine,
        output_path
):
    """
    Save numpy volume as NIfTI.


    Parameters
    ----------

    volume:
        numpy array


    affine:
        original CT affine


    output_path:
        destination .nii.gz


    """



    output_path = Path(
        output_path
    )



    output_path.parent.mkdir(

        parents=True,

        exist_ok=True

    )



    # convert tensor

    if hasattr(
        volume,
        "detach"
    ):

        volume = (
            volume
            .detach()
            .cpu()
            .numpy()
        )



    volume = np.asarray(
        volume
    )



    nii = nib.Nifti1Image(

        volume.astype(
            np.float32
        ),

        affine

    )



    nib.save(

        nii,

        str(output_path)

    )



    print(
        "Saved NIfTI:",
        output_path
    )