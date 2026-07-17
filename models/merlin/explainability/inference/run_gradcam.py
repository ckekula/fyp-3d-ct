"""
run_gradcam.py -- STEP 6/7 end-to-end orchestrator.
See run order in chat before running this file.
"""

import os
import argparse

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

from explainability.inference.load_model import load_merlin
from explainability.inference.preprocess import preprocess_ct
from explainability.inference.predict import MerlinPredictor
from explainability.gradcam.gradcam3d import GradCAM3D
from explainability.gradcam.target import ImageEmbeddingNormTarget, TextSimilarityTarget
from explainability.gradcam.metrics import evaluate_gradcam
from explainability.visualization.save_figure import save_cam_png, save_cam_nifti
from explainability.utils.io import ensure_dir, save_cam_npy


def run(nii_path, out_dir, text_query=None, gt_mask_path=None,
        deletion_steps=10, full_res=False):
    ensure_dir(out_dir)

    use_text = text_query is not None
    model, target_layer = load_merlin(image_embedding_only=not use_text)
    predictor = MerlinPredictor(model, image_embedding_only=not use_text)

    tensor, ds_shape, affine, full_res_shape = preprocess_ct(nii_path)

    cam_engine = GradCAM3D(model, target_layer, predictor.forward
                            if not use_text
                            else (lambda x: predictor.forward(x, text=text_query)))

    target = (
        TextSimilarityTarget(model, text_query)
        if use_text
        else ImageEmbeddingNormTarget()
    )

    cam = cam_engine(tensor, target, output_size=ds_shape)
    cam_engine.release()

    ct_volume = tensor.squeeze().detach().cpu().numpy()
    stem = os.path.splitext(os.path.splitext(os.path.basename(nii_path))[0])[0]

    save_cam_npy(cam, os.path.join(out_dir, f"{stem}_cam_downsampled.npy"))
    png_path, slice_idx = save_cam_png(ct_volume, cam, os.path.join(out_dir, f"{stem}_overlay.png"))
    print(f"Saved overlay: {png_path} (slice {slice_idx}, downsampled res {ds_shape})")

    if full_res:
        cam_t = torch.tensor(cam, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        cam_full = F.interpolate(cam_t, size=full_res_shape, mode="trilinear",
                                  align_corners=False).squeeze().numpy()
        save_cam_nifti(cam_full, affine, os.path.join(out_dir, f"{stem}_cam_fullres.nii.gz"))
        print(f"Saved full-res CAM ({full_res_shape}) with correct affine")

    gt_mask = None
    if gt_mask_path is not None:
        if gt_mask_path.endswith(".npz"):
            npz = np.load(gt_mask_path)
            gt_mask = npz[list(npz.keys())[0]]
        else:
            gt_mask = nib.load(gt_mask_path).get_fdata().astype(np.float32)
        if gt_mask.shape != cam.shape:
            gt_t = torch.tensor(gt_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            gt_mask = F.interpolate(gt_t, size=cam.shape, mode="trilinear",
                                     align_corners=False).squeeze().numpy()

    def forward_fn(x, text=None):
        return predictor.forward(x, text=text)

    metrics = evaluate_gradcam(
        predictor, tensor, cam, target, text=text_query,
        gt_mask=gt_mask, deletion_steps=deletion_steps,
    )

    print("\n=== Evaluation Metrics ===")
    for k, v in metrics.items():
        print(f"{k:28s}: {v}")

    return cam, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merlin Grad-CAM explainability")
    parser.add_argument("--nii", required=True)
    parser.add_argument("--out", default="./gradcam_outputs")
    parser.add_argument("--text", default=None)
    parser.add_argument("--gt-mask", default=None)
    parser.add_argument("--deletion-steps", type=int, default=10)
    parser.add_argument("--full-res", action="store_true")
    args = parser.parse_args()

    run(
        nii_path=args.nii,
        out_dir=args.out,
        text_query=args.text,
        gt_mask_path=args.gt_mask,
        deletion_steps=args.deletion_steps,
        full_res=args.full_res,
    )
