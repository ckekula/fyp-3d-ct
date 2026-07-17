"""
predict.py — runs a forward pass with gradients kept intact (no torch.no_grad!),
since gradcam3d.py needs to backprop through this call.
"""

class MerlinPredictor:
    def __init__(self, model, image_embedding_only=True):
        self.model = model
        self.image_embedding_only = image_embedding_only

    def forward(self, image_tensor, text=None):
        image_tensor.requires_grad_(False)  # only layer4 activations need grad
        if self.image_embedding_only:
            output = self.model(image_tensor)
        else:
            assert text is not None, "text is required when image_embedding_only=False"
            output = self.model(image_tensor, [text])
        return output