"""
hooks.py — forward/backward hooks that capture the layer4 activations
and their gradients for Grad-CAM.
"""

class ActivationsAndGradients:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inp, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x):
        self.activations = None
        self.gradients = None
        return self.model(x)

    def release(self):
        for h in self.handles:
            h.remove()