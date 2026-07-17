"""
hooks.py -- forward/backward hooks that capture layer4 activations
and their gradients for Grad-CAM.
"""


class ActivationsAndGradients:
    def __init__(self, target_layer):
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

    def reset(self):
        self.activations = None
        self.gradients = None

    def release(self):
        for h in self.handles:
            h.remove()


if __name__ == "__main__":
    # STEP 4 sanity check -- see run order in chat.
    import sys
    import torch
    from explainability.inference.load_model import load_merlin
    from explainability.inference.preprocess import preprocess_ct
    from explainability.inference.predict import MerlinPredictor

    model, target_layer = load_merlin(image_embedding_only=True)
    predictor = MerlinPredictor(model, image_embedding_only=True)
    ag = ActivationsAndGradients(target_layer)

    tensor, ds_shape, affine, full_shape = preprocess_ct(sys.argv[1])
    output = predictor.forward(tensor)
    score = output.norm()
    score.backward()

    print("Activations shape:", ag.activations.shape)
    print("Gradients shape:", ag.gradients.shape)
    ag.release()
