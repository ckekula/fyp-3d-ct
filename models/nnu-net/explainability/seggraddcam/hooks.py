"""
hooks.py

Forward and backward hooks for 3D SegGradCAM.

Forward hook:
    Saves feature maps from the target layer.

Backward hook:
    Saves gradients flowing through the target layer.

Used by:
    SlidingWindowSegGradCAM3D
"""


import torch



class GradCAMHooks:
    """
    Stores activations and gradients
    from a selected neural network layer.
    """



    def __init__(
            self,
            layer
    ):

        self.layer = layer

        self.activations = None

        self.gradients = None



        self.forward_handle = (
            layer.register_forward_hook(
                self.forward_hook
            )
        )



        self.backward_handle = (
            layer.register_full_backward_hook(
                self.backward_hook
            )
        )



    # --------------------------------------------------
    # Forward
    # --------------------------------------------------

    def forward_hook(
            self,
            module,
            input,
            output
    ):
        """
        Called during forward propagation.

        Saves feature maps.
        """

        self.activations = output.detach()



    # --------------------------------------------------
    # Backward
    # --------------------------------------------------

    def backward_hook(
            self,
            module,
            grad_input,
            grad_output
    ):
        """
        Called during backpropagation.

        Saves gradients.
        """

        self.gradients = (
            grad_output[0]
            .detach()
        )



    def clear(self):
        """
        Remove stored tensors.
        """

        self.activations = None

        self.gradients = None



    def remove(self):
        """
        Remove PyTorch hooks.
        """

        self.forward_handle.remove()

        self.backward_handle.remove()