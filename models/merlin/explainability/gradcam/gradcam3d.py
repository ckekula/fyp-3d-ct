"""
gradcam3d.py -- 3D Grad-CAM over the (H',W',D') volume produced by layer4.
"""

import torch
import torch.nn.functional as F

from explainability.gradcam.hooks import ActivationsAndGradients


class GradCAM3D:
    def __init__(self, model, target_layer, forward_fn):
        """
        forward_fn: callable(image_tensor) -> model output
                    (routes through predictor.forward, which knows
                     whether text needs to be passed).
        """
        self.model = model
        self.forward_fn = forward_fn
        self.ag = ActivationsAndGradients(target_layer)

    def __call__(self, input_tensor, target, output_size=None):
        self.model.zero_grad(set_to_none=True)
        self.ag.reset()

        output = self.forward_fn(input_tensor)   # gradients ENABLED
        score = target(output)
        score.backward(retain_graph=False)

        activations = self.ag.activations
        gradients = self.ag.gradients

        weights = gradients.mean(dim=(2, 3, 4), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        size = output_size or input_tensor.shape[2:]
        cam = F.interpolate(cam, size=size, mode="trilinear", align_corners=False)

        cam = cam.squeeze().detach().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam

    def release(self):
        self.ag.release()


if __name__ == "__main__":
    # STEP 5 sanity check -- see run order in chat.
    import sys
    from explainability.inference.load_model import load_merlin
    from explainability.inference.preprocess import preprocess_ct
    from explainability.inference.predict import MerlinPredictor
    from explainability.gradcam.target import ImageEmbeddingNormTarget

    model, target_layer = load_merlin(image_embedding_only=True)
    predictor = MerlinPredictor(model, image_embedding_only=True)
    tensor, ds_shape, affine, full_shape = preprocess_ct(sys.argv[1])

    cam_engine = GradCAM3D(model, target_layer, predictor.forward)
    cam = cam_engine(tensor, ImageEmbeddingNormTarget(), output_size=ds_shape)
    cam_engine.release()

    print("CAM shape:", cam.shape)
    print("CAM min/max:", cam.min(), cam.max())
