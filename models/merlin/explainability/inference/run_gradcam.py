"""
run_gradcam.py — end-to-end: CT -> Grad-CAM -> overlay PNG + NIfTI heatmap.
"""

import os
import numpy as np

from explainability.inference.load_model import load_merlin
from explainability.inference.preprocess import preprocess_ct
from explainability.inference.predict import MerlinPredictor
from explainability.gradcam.gradcam3d import GradCAM3D
from explainability.gradcam.target import ImageEmbeddingNormTarget, TextSimilarityTarget
from explainability.visualization.save_figure import save_cam_png, save_cam_nifti
from explainability.utils.io import ensure_dir, save_cam_npy


def run(nii_path, out_dir, text_query=None):
    ensure_dir(out_dir)

    # text_query given -> need full contrastive model (image_embedding_only=False)
    use_text = text_query is not None
    model, target_layer = load_merlin(image_embedding_only=not use_text)
    predictor = MerlinPredictor(model, image_embedding_only=not use_text)

    tensor, orig_shape, affine = preprocess_ct(nii_path)

    cam_engine = GradCAM3D(model, target_layer)

    target = (
        TextSimilarityTarget(model, text_query)
        if use_text
        else ImageEmbeddingNormTarget()
    )

    def fwd(x):
        return predictor.forward(x, text=text_query)

    cam_engine.ag.model = fwd  # route hook wrapper through predictor.forward
    cam = cam_engine(tensor, target, output_size=orig_shape)
    cam_engine.release()

    ct_volume = tensor.squeeze().detach().cpu().numpy()

    stem = os.path.splitext(os.path.splitext(os.path.basename(nii_path))[0])[0]
    save_cam_npy(cam, os.path.join(out_dir, f"{stem}_cam.npy"))
    save_cam_nifti(cam, affine, os.path.join(out_dir, f"{stem}_cam.nii.gz"))
    png_path, slice_idx = save_cam_png(ct_volume, cam, os.path.join(out_dir, f"{stem}_overlay.png"))

    print(f"Saved: {png_path} (slice {slice_idx})")
    return cam


if __name__ == "__main__":
    run(
        nii_path="sample.nii.gz",
        out_dir="gradcam_outputs",
        text_query="lung nodule",  # or None for generic saliency
    )