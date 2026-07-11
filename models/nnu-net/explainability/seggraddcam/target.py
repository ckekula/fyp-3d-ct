"""
target.py

Defines the segmentation target
used for SegGradCAM.

Instead of:

    class score

we use:

    lesion region score


Target:

sum(
    lesion logits
    inside target mask
)

"""


import torch




def segmentation_target(
        logits,
        mask,
        target_class
):
    """
    Calculate SegGradCAM target.


    Parameters
    ----------

    logits:
        Network output

        Shape:
            (1,C,D,H,W)


    mask:
        Ground truth lesion mask

        Shape:
            (D,H,W)


    target_class:
        lesion class index

        Example:
            1 = GGO



    Returns
    -------

    score:
        scalar tensor
    """



    # select lesion channel

    lesion_logits = (
        logits[:, target_class]
    )



    # remove batch dimension

    lesion_logits = (
        lesion_logits.squeeze(0)
    )



    # segmentation-aware score

    score = (
        lesion_logits *
        mask
    ).sum()



    return score