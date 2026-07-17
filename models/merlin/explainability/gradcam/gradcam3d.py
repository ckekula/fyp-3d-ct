"""
gradcam3d.py — 3D Grad-CAM over the (D,H,W) volume produced by layer4.
"""

import torch
import torch.nn.functional as F
import numpy as np

from explainability.gradcam.hooks import ActivationsAndGradients


class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.ag = ActivationsAndGradients(model, target_layer)

    def __call__(self, input_tensor, target, output_size=None):
        self.model.zero_grad(set_to_none=True)

        output = self.ag(input_tensor)          # gradients ENABLED — no torch.no_grad()
        score = target(output)
        score.backward(retain_graph=False)

        activations = self.ag.activations        # [B,C,D',H',W']
        gradients = self.ag.gradients             # [B,C,D',H',W']

        weights = gradients.mean(dim=(2, 3, 4), keepdim=True)   # GAP -> channel weights
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [B,1,D',H',W']
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