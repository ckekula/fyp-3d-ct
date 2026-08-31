"""
merlin_gradcam_explainability.py

Single-file Grad-CAM explainability pipeline for the Merlin 3D CT
vision-language model (https://github.com/StanfordMIMI/Merlin).

Built directly against your ACTUAL production script (merlin_ct_pipeline.py):
    - preprocessing (clip -1000..400, normalize (v+1000)/1400, then
      volume[::4, ::4, ::2] downsampling) is matched exactly, because
      Grad-CAM has to see precisely what the model saw.
    - tensor axis order is (1, 1, H, W, D), matching
      `torch.tensor(volume).unsqueeze(0).unsqueeze(0)` on a volume of
      shape (H, W, D) -- NOT (D, H, W). This matters for how the final
      CAM gets reshaped back onto the original scan.
    - model(ct_tensor) with ImageEmbedding=True returns the embedding
      tensor directly (not a tuple) -- handled by the isinstance check
      in ImageEmbeddingNormTarget below.

This REPLACES build_localization_mask() from merlin_ct_pipeline.py.
That function only ever highlights the top-5%-brightest voxels by raw
HU intensity -- the "embedding" it multiplies in is a single scalar
applied uniformly everywhere, so it carries no spatial information
about what the model actually attended to. Real Grad-CAM below uses
gradients through layer4, so the map is actually model-derived.

Sections:
    1. Config
    2. Model loading
    3. Preprocessing (exact match to merlin_ct_pipeline.py, incl. downsampling)
    4. Hooks (activations + gradients)
    5. Targets (what score Grad-CAM explains)
    6. Grad-CAM 3D core
    7. Predictor
    8. Visualization (overlay + saving)
    9. Evaluation metrics
    10. CLI entry point

Usage:
    python merlin_gradcam_explainability.py \
        --nii /path/to/scan.nii.gz \
        --out ./gradcam_outputs \
        --text "lung nodule" \
        --gt-mask /path/to/results_f/<case>/localization_mask.nii.gz \
        --full-res
"""

import os
import argparse

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

# merlin-vlm must be installed: pip install merlin-vlm
from merlin import Merlin


# =====================================================================
# 1. CONFIG
# =====================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HU windowing -- matches merlin_ct_pipeline.py exactly
WINDOW_MIN = -1000.0
WINDOW_MAX = 400.0

# Matches `volume[::4, ::4, ::2]` in load_and_preprocess_ct().
# This is a real accuracy/speed tradeoff: a 512x512xD scan becomes
# ~128x128x(D/2) before it ever reaches the model, so Grad-CAM's
# spatial precision is capped at that resolution no matter what.
# Set to (1, 1, 1) if you want to run Grad-CAM on full-resolution
# volumes instead (much slower, much more VRAM, sharper CAM).
DOWNSAMPLE_STRIDE = (4, 4, 2)

# Confirmed: model.model.encode_image.i3_resnet.layer4
TARGET_LAYER_PATH = "encode_image.i3_resnet.layer4"


# =====================================================================
# 2. MODEL LOADING
# =====================================================================

class MerlinLoader:
    """
    NOTE: Merlin() downloads/loads its pretrained weights internally
    (merlin-vlm package / HuggingFace). There is no local checkpoint
    file to load -- do not add torch.load() here.
    """

    def __init__(self, image_embedding_only=True):
        # image_embedding_only=True  -> forward(image) returns (embedding,)
        # image_embedding_only=False -> forward(image, text) returns
        #   (contrastive_img_emb, phenotype_logits, contrastive_text_emb)
        self.device = DEVICE
        self.image_embedding_only = image_embedding_only
        self.model = None

    def load(self):
        print("Loading Merlin...")
        model = Merlin(ImageEmbedding=self.image_embedding_only)
        model.to(self.device)
        model.eval()
        self.model = model
        print(f"Merlin loaded on {self.device}")
        return model

    def get_target_layer(self):
        layer = self.model.model  # -> model.model.encode_image...
        for m in TARGET_LAYER_PATH.split("."):
            layer = getattr(layer, m)
        return layer


def load_merlin(image_embedding_only=True):
    loader = MerlinLoader(image_embedding_only=image_embedding_only)
    model = loader.load()
    target_layer = loader.get_target_layer()
    return model, target_layer


# =====================================================================
# 3. PREPROCESSING -- exact match to merlin_ct_pipeline.py
# =====================================================================

class CTPreprocessor:
    """
    Mirrors load_and_preprocess_ct() from merlin_ct_pipeline.py line for
    line, including the [::4, ::4, ::2] downsampling, so the tensor fed
    to Grad-CAM is identical to what the model saw in production.

    Returns both the downsampled tensor (what the model/Grad-CAM run on)
    and the original full-resolution shape/affine, so the final CAM can
    optionally be upsampled back onto the true physical scan geometry
    instead of the affine-mismatched approach in the original pipeline
    (which saves the downsampled mask under the ORIGINAL affine).
    """

    def __init__(self):
        self.affine = None
        self.full_res_shape = None

    def preprocess(self, nii_path):
        img = nib.load(str(nii_path))
        volume = img.get_fdata()

        # Drop trailing singleton dim if present (H,W,D,1)
        if volume.ndim == 4:
            volume = volume[..., 0]

        volume = np.clip(volume, WINDOW_MIN, WINDOW_MAX)
        volume = (volume - WINDOW_MIN) / (WINDOW_MAX - WINDOW_MIN)
        volume = np.nan_to_num(volume, nan=0.0).astype(np.float32)

        self.affine = img.affine.copy()
        self.full_res_shape = volume.shape  # (H, W, D), pre-downsample
        del img

        # Exact match: volume[::4, ::4, ::2]
        sh, sw, sd = DOWNSAMPLE_STRIDE
        volume_ds = volume[::sh, ::sw, ::sd]

        # Shape -> (1, 1, H, W, D) -- NOT (1,1,D,H,W)
        tensor = torch.tensor(volume_ds, dtype=torch.float32)
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        return tensor.to(DEVICE), volume_ds.shape, self.affine, self.full_res_shape


def preprocess_ct(image_path):
    proc = CTPreprocessor()
    return proc.preprocess(image_path)


# =====================================================================
# 4. HOOKS
# =====================================================================

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


# =====================================================================
# 5. TARGETS -- what scalar score Grad-CAM explains
# =====================================================================

class ImageEmbeddingNormTarget:
    """Generic saliency: what drove the overall image embedding."""
    def __call__(self, output):
        emb = output[0] if isinstance(output, (tuple, list)) else output
        return emb.norm(dim=-1).sum()


class PhenotypeTarget:
    """Requires model built with image_embedding_only=False."""
    def __init__(self, class_idx):
        self.class_idx = class_idx

    def __call__(self, output):
        logits = output[1]
        return logits[:, self.class_idx].sum()


class TextSimilarityTarget:
    """
    Text-driven Grad-CAM: 'why does this region look like <text>?'
    Requires model built with image_embedding_only=False (needs encode_text).
    """
    def __init__(self, model, text):
        with torch.no_grad():
            text_emb = model.model.encode_text([text])
            self.text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    def __call__(self, output):
        image_emb = output[0]
        image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
        return (image_emb * self.text_emb).sum()


# =====================================================================
# 6. GRAD-CAM 3D CORE
# =====================================================================

class GradCAM3D:
    def __init__(self, model, target_layer, forward_fn):
        """
        forward_fn: callable(image_tensor) -> model output
                    (lets us route through predictor.forward, which
                     knows whether text needs to be passed).
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

        activations = self.ag.activations          # [B,C,D',H',W']
        gradients = self.ag.gradients               # [B,C,D',H',W']

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


# =====================================================================
# 7. PREDICTOR
# =====================================================================

class MerlinPredictor:
    """Runs forward WITHOUT torch.no_grad() -- Grad-CAM needs the graph."""

    def __init__(self, model, image_embedding_only=True):
        self.model = model
        self.image_embedding_only = image_embedding_only

    def forward(self, image_tensor, text=None):
        if self.image_embedding_only:
            return self.model(image_tensor)
        assert text is not None, "text is required when image_embedding_only=False"
        return self.model(image_tensor, [text])


# =====================================================================
# 8. VISUALIZATION
# =====================================================================

def overlay_slice(ct_slice, cam_slice, alpha=0.45, cmap_thresh=0.15):
    import matplotlib.cm as cm

    ct_norm = (ct_slice - ct_slice.min()) / (np.ptp(ct_slice) + 1e-8)
    base = np.stack([ct_norm] * 3, axis=-1)

    heat = cm.get_cmap("jet")(cam_slice)[..., :3]
    mask = (cam_slice > cmap_thresh).astype(np.float32)[..., None]

    blended = base * (1 - alpha * mask) + heat * (alpha * mask)
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


def best_slice_index(cam, axis=0):
    energy = cam.sum(axis=tuple(a for a in range(3) if a != axis))
    return int(np.argmax(energy))


def save_cam_png(ct_volume, cam, out_path, axis=0):
    import matplotlib.pyplot as plt

    idx = best_slice_index(cam, axis=axis)
    ct_slice = np.take(ct_volume, idx, axis=axis)
    cam_slice = np.take(cam, idx, axis=axis)
    img = overlay_slice(ct_slice, cam_slice)
    plt.imsave(out_path, img)
    return out_path, idx


def save_cam_nifti(cam, affine, out_path):
    nib.save(nib.Nifti1Image(cam.astype(np.float32), affine), out_path)
    return out_path


def save_cam_npy(cam, out_path):
    np.save(out_path, cam)
    return out_path


# =====================================================================
# 9. EVALUATION METRICS
# =====================================================================
# Three complementary families, standard in the Grad-CAM literature:
#
#   A) Faithfulness (does the CAM actually matter to the model?)
#      - Average Drop % / Increase in Confidence   (Chattopadhay et al., 2018)
#      - Deletion / Insertion AUC                    (Petsiuk et al., 2018 - RISE)
#
#   B) Localization (does the CAM agree with known findings?)
#      - Dice / IoU against a ground-truth / reference mask
#        (you already have these per-case in results_full/<case>/localization_masks.npz)
#      - Pointing Game hit rate
#
#   C) Sanity (is the CAM at least non-degenerate?)
#      - Coverage (fraction of voxels above threshold, should be << 1.0)
# =====================================================================

def average_drop_increase(predictor, image_tensor, cam, target, text=None):
    """
    Average Drop %:      how much the target score falls when the image
                          is masked by the CAM (lower is better -- means
                          the highlighted region really was important).
    Increase in Confidence: 1 if masking by the CAM *increases* the score
                          (can happen when the CAM removes distracting
                          background), 0 otherwise.
    """
    with torch.no_grad():
        orig_output = predictor.forward(image_tensor, text=text)
        orig_score = target(orig_output).item()

        cam_t = torch.tensor(cam, dtype=torch.float32, device=image_tensor.device)
        cam_t = cam_t.unsqueeze(0).unsqueeze(0)
        masked_input = image_tensor * cam_t

        masked_output = predictor.forward(masked_input, text=text)
        masked_score = target(masked_output).item()

    drop_percent = max(0.0, orig_score - masked_score) / (abs(orig_score) + 1e-8) * 100
    increase = 1.0 if masked_score > orig_score else 0.0

    return {
        "orig_score": orig_score,
        "masked_score": masked_score,
        "average_drop_percent": drop_percent,
        "increase_in_confidence": increase,
    }


def deletion_insertion_auc(predictor, image_tensor, cam, target, text=None, steps=10):
    """
    Deletion: start from the full image, progressively zero out voxels in
              order of CAM importance (highest first) -- score should drop
              fast (low AUC = good CAM).
    Insertion: start from a blank volume, progressively insert voxels in
              order of CAM importance -- score should rise fast
              (high AUC = good CAM).
    Downsampled to `steps` for tractability on large 3D volumes.
    """
    flat_cam = cam.flatten()
    order = np.argsort(-flat_cam)  # most important first
    n_voxels = flat_cam.size
    chunk = max(1, n_voxels // steps)

    orig_flat = image_tensor.squeeze().detach().cpu().numpy().flatten()

    def score_at(mask_flat):
        vol = (orig_flat * mask_flat).reshape(image_tensor.shape[2:])
        t = torch.tensor(vol, dtype=torch.float32, device=image_tensor.device)
        t = t.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            out = predictor.forward(t, text=text)
        return target(out).item()

    # Deletion curve
    del_mask = np.ones(n_voxels, dtype=np.float32)
    deletion_scores = [score_at(del_mask)]
    for i in range(steps):
        idx = order[i * chunk:(i + 1) * chunk]
        del_mask[idx] = 0.0
        deletion_scores.append(score_at(del_mask))

    # Insertion curve
    ins_mask = np.zeros(n_voxels, dtype=np.float32)
    insertion_scores = [score_at(ins_mask)]
    for i in range(steps):
        idx = order[i * chunk:(i + 1) * chunk]
        ins_mask[idx] = 1.0
        insertion_scores.append(score_at(ins_mask))

    deletion_auc = float(np.trapezoid(deletion_scores) / len(deletion_scores))
    insertion_auc = float(np.trapezoid(insertion_scores) / len(insertion_scores))

    return {
        "deletion_auc": deletion_auc,     # lower is better
        "insertion_auc": insertion_auc,   # higher is better
    }


def dice_iou_with_mask(cam, gt_mask, threshold=0.5):
    """
    Compare the binarized CAM against a reference/ground-truth soft mask
    (e.g. results_full/<case>/localization_masks.npz from your own pipeline).
    """
    cam_bin = (cam >= threshold).astype(np.uint8)
    gt_bin = (gt_mask >= threshold).astype(np.uint8)

    intersection = np.logical_and(cam_bin, gt_bin).sum()
    union = np.logical_or(cam_bin, gt_bin).sum()

    dice = (2 * intersection) / (cam_bin.sum() + gt_bin.sum() + 1e-8)
    iou = intersection / (union + 1e-8)

    return {"dice": float(dice), "iou": float(iou)}


def pointing_game(cam, gt_mask):
    """Does the single highest-activation voxel fall inside the reference mask?"""
    idx = np.unravel_index(np.argmax(cam), cam.shape)
    hit = bool(gt_mask[idx] > 0)
    return {"pointing_game_hit": hit}


def cam_coverage(cam, threshold=0.5):
    """Sanity check: fraction of the volume highlighted. Should be small."""
    return {"coverage_fraction": float((cam >= threshold).mean())}


def evaluate_gradcam(predictor, image_tensor, cam, target, text=None,
                      gt_mask=None, deletion_steps=10):
    """Runs the full evaluation suite and returns one flat dict of metrics."""
    metrics = {}
    metrics.update(average_drop_increase(predictor, image_tensor, cam, target, text=text))
    metrics.update(deletion_insertion_auc(predictor, image_tensor, cam, target,
                                           text=text, steps=deletion_steps))
    metrics.update(cam_coverage(cam))

    if gt_mask is not None:
        metrics.update(dice_iou_with_mask(cam, gt_mask))
        metrics.update(pointing_game(cam, gt_mask))

    return metrics


# =====================================================================
# 10. CLI ENTRY POINT
# =====================================================================

def run(nii_path, out_dir, text_query=None, gt_mask_path=None,
        deletion_steps=10, full_res=False):
    os.makedirs(out_dir, exist_ok=True)

    use_text = text_query is not None
    model, target_layer = load_merlin(image_embedding_only=not use_text)
    predictor = MerlinPredictor(model, image_embedding_only=not use_text)

    # tensor is the DOWNSAMPLED volume the model actually sees.
    # full_res_shape is the ORIGINAL scan resolution (e.g. 512x512xD).
    tensor, ds_shape, affine, full_res_shape = preprocess_ct(nii_path)

    def forward_fn(x):
        return predictor.forward(x, text=text_query)

    cam_engine = GradCAM3D(model, target_layer, forward_fn)

    target = (
        TextSimilarityTarget(model, text_query)
        if use_text
        else ImageEmbeddingNormTarget()
    )

    # CAM computed and upsampled to the DOWNSAMPLED tensor's own shape
    # (ds_shape) -- this is the resolution Grad-CAM can actually resolve.
    cam = cam_engine(tensor, target, output_size=ds_shape)
    cam_engine.release()

    ct_volume = tensor.squeeze().detach().cpu().numpy()  # downsampled, (H,W,D)
    stem = os.path.splitext(os.path.splitext(os.path.basename(nii_path))[0])[0]

    save_cam_npy(cam, os.path.join(out_dir, f"{stem}_cam_downsampled.npy"))
    png_path, slice_idx = save_cam_png(ct_volume, cam, os.path.join(out_dir, f"{stem}_overlay.png"))
    print(f"Saved overlay: {png_path} (slice {slice_idx}, downsampled res {ds_shape})")

    if full_res:
        # Upsample the CAM back to the TRUE original resolution and save
        # it with the ORIGINAL (correct) affine -- fixes the affine
        # mismatch present in build_localization_mask()'s output, which
        # saves a downsampled-resolution mask under the full-res affine.
        cam_t = torch.tensor(cam, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        cam_full = F.interpolate(cam_t, size=full_res_shape, mode="trilinear",
                                  align_corners=False).squeeze().numpy()
        save_cam_nifti(cam_full, affine, os.path.join(out_dir, f"{stem}_cam_fullres.nii.gz"))
        print(f"Saved full-res CAM ({full_res_shape}) with correct affine")

    gt_mask = None
    if gt_mask_path is not None:
        if gt_mask_path.endswith(".npz"):
            gt_mask = np.load(gt_mask_path)[list(np.load(gt_mask_path).keys())[0]]
        else:
            gt_mask = nib.load(gt_mask_path).get_fdata().astype(np.float32)
        if gt_mask.shape != cam.shape:
            gt_t = torch.tensor(gt_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            gt_mask = F.interpolate(gt_t, size=cam.shape, mode="trilinear",
                                     align_corners=False).squeeze().numpy()

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
    parser.add_argument("--nii", required=True, help="Path to input CT (.nii.gz)")
    parser.add_argument("--out", default="./gradcam_outputs", help="Output directory")
    parser.add_argument("--text", default=None,
                         help="Text query for text-driven Grad-CAM (omit for generic saliency)")
    parser.add_argument("--gt-mask", default=None,
                         help="Path to a reference mask for evaluation: either a "
                              ".nii.gz (e.g. results_f/<case>/localization_mask.nii.gz) "
                              "or a .npz (first array in the file is used)")
    parser.add_argument("--deletion-steps", type=int, default=10,
                         help="Number of steps for deletion/insertion AUC")
    parser.add_argument("--full-res", action="store_true",
                         help="Also upsample the CAM back to the original scan "
                              "resolution and save as NIfTI with the correct affine")
    args = parser.parse_args()

    run(
        nii_path=args.nii,
        out_dir=args.out,
        text_query=args.text,
        gt_mask_path=args.gt_mask,
        deletion_steps=args.deletion_steps,
        full_res=args.full_res,
    )