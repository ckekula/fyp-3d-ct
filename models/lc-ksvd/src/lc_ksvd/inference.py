"""
inference.py
Full inference pipeline for a trained LC-KSVD2 model on a single CT volume.

Stages:
  1. Load + preprocess the volume (resample, HU window, normalise)
  2. Extract overlapping patches on a regular grid
  3. Sparse-encode each patch using the learned dictionary (OMP)
  4. Classify each patch via W @ gamma → class scores
  5. Back-project patch scores into a 3D heatmap in voxel space
  6. Return the predicted class label and the localisation heatmap

Usage:
    python inference.py --volume train_1741_c_2 --model outputs/models/unified_lcksvd2.pkl
"""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import nibabel as nib
import numpy as np

from lc_ksvd.config import PATCH_SIZE
from lc_ksvd.data_loader import ScanLoader, MetadataRegistry

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ─── Patch grid extraction ────────────────────────────────────────────────────

def extract_patch_grid(
    volume: np.ndarray,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract overlapping patches on a regular stride grid across the volume.

    Returns:
        patches : float64 (n_features, n_patches)  — column-normalised
        centres : int32   (n_patches, 3)            — (x, y, z) voxel centres
    """
    p = PATCH_SIZE
    h = p // 2
    H, W, D = volume.shape

    xs = np.arange(h, H - h, stride)
    ys = np.arange(h, W - h, stride)
    zs = np.arange(h, D - h, stride)

    patches_list = []
    centres_list = []

    for x in xs:
        for y in ys:
            for z in zs:
                patch = volume[x-h:x-h+p, y-h:y-h+p, z-h:z-h+p]
                patches_list.append(patch.ravel().astype(np.float64))
                centres_list.append((x, y, z))

    if not patches_list:
        raise RuntimeError(
            f"No patches extracted — volume shape {volume.shape} may be "
            f"smaller than PATCH_SIZE={p}."
        )

    X = np.stack(patches_list, axis=1)          # (n_features, n_patches)
    centres = np.array(centres_list, dtype=np.int32)  # (n_patches, 3)

    # Column-normalise (same as training)
    norms = np.linalg.norm(X, axis=0)
    norms_safe = np.where(norms < 1e-10, 1.0, norms)
    X_norm = X / norms_safe[np.newaxis, :]

    logger.info(f"Extracted {X_norm.shape[1]} patches  (stride={stride}, grid={len(xs)}x{len(ys)}x{len(zs)})")
    return X_norm, centres


# ─── Back-projection ──────────────────────────────────────────────────────────

def backproject_scores(
    scores: np.ndarray,         # (n_classes, n_patches)
    centres: np.ndarray,        # (n_patches, 3)
    volume_shape: Tuple[int, int, int],
    class_idx: int,
) -> np.ndarray:
    """
    Accumulate the score for `class_idx` from every patch back into a 3D
    heatmap of the same spatial shape as the resampled volume.

    Each patch contributes its score to all voxels it covers. The final map
    is divided by the overlap count so overlapping patches are averaged rather
    than summed, giving a smooth localisation heatmap.

    Returns:
        heatmap : float32 (H, W, D) — score in [roughly 0..1] after avg
    """
    p = PATCH_SIZE
    h = p // 2
    H, W, D = volume_shape

    accumulator = np.zeros((H, W, D), dtype=np.float64)
    count       = np.zeros((H, W, D), dtype=np.float64)

    class_scores = scores[class_idx]   # (n_patches,)

    for i, (cx, cy, cz) in enumerate(centres):
        x0, x1 = cx - h, cx - h + p
        y0, y1 = cy - h, cy - h + p
        z0, z1 = cz - h, cz - h + p
        accumulator[x0:x1, y0:y1, z0:z1] += class_scores[i]
        count      [x0:x1, y0:y1, z0:z1] += 1.0

    # Avoid division by zero for voxels not covered by any patch
    mask = count > 0
    heatmap = np.zeros((H, W, D), dtype=np.float32)
    heatmap[mask] = (accumulator[mask] / count[mask]).astype(np.float32)

    return heatmap


# ─── Main inference ───────────────────────────────────────────────────────────

def run_inference(
    volume_name: str,
    model_path: Path,
    stride: Optional[int] = None,
    save_dir: Optional[Path] = None,
) -> Dict:
    """
    Full inference pipeline for one CT volume.

    Args:
        volume_name : scan ID (e.g. "train_1741_c_2")
        model_path  : path to the saved unified_lcksvd2.pkl
        stride      : patch stride in voxels (defaults to PATCH_SIZE // 2)
        save_dir    : if set, writes heatmap NIfTIs here

    Returns a dict with:
        "predicted_class"  : str  — CLASS_ORDER label with highest mean patch score
        "class_scores"     : dict — mean patch score per class
        "heatmaps"         : dict — { class_label: np.ndarray (H, W, D) }
        "volume_shape"     : tuple — spatial shape after resampling
    """
    stride = stride or (PATCH_SIZE // 2)

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info(f"Loading model from {model_path}")
    with open(model_path, "rb") as f:
        payload = pickle.load(f)

    model       = payload["model"]
    class_order = payload["class_order"]   # use saved order, not config constant

    # ── Load + preprocess volume ──────────────────────────────────────────────
    logger.info(f"Loading volume: {volume_name}")
    metadata = MetadataRegistry()
    loader   = ScanLoader(metadata)
    scan     = loader.load(volume_name)
    volume   = scan["volume"]              # float32 (H, W, D) in [0, 1]
    logger.info(f"Volume shape after resampling: {volume.shape}")

    # ── Extract patches ───────────────────────────────────────────────────────
    X_norm, centres = extract_patch_grid(volume, stride=stride)

    # ── Sparse coding ─────────────────────────────────────────────────────────
    logger.info("Sparse coding patches…")
    Gamma = model.transform(X_norm)        # (n_components, n_patches)
    logger.info(f"Gamma shape: {Gamma.shape}")

    # ── Classification ────────────────────────────────────────────────────────
    W      = model.W_                      # (n_classes, n_components)
    scores = W @ Gamma                     # (n_classes, n_patches)

    mean_scores = scores.mean(axis=1)      # (n_classes,) — volume-level score
    pred_idx    = int(np.argmax(mean_scores))
    predicted   = class_order[pred_idx]

    class_scores = {cls: float(mean_scores[i]) for i, cls in enumerate(class_order)}
    logger.info(f"Predicted class: {predicted}")
    for cls, score in class_scores.items():
        logger.info(f"  {cls}: {score:.4f}")

    # ── Back-projection ───────────────────────────────────────────────────────
    logger.info(f"Back-projecting scores for predicted class '{predicted}'…")
    heatmaps = {}
    for i, cls in enumerate(class_order):
        heatmaps[cls] = backproject_scores(
            scores       = scores,
            centres      = centres,
            volume_shape = volume.shape,
            class_idx    = i,
        )

    # ── Optionally save heatmaps as NIfTI ─────────────────────────────────────
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for cls, heatmap in heatmaps.items():
            out_path = save_dir / f"{volume_name}_{cls}_heatmap.nii.gz"
            nib.save(
                nib.Nifti1Image(heatmap, affine=np.eye(4)),
                str(out_path),
            )
            logger.info(f"Saved heatmap → {out_path}")

    return {
        "predicted_class": predicted,
        "class_scores":    class_scores,
        "heatmaps":        heatmaps,
        "volume_shape":    volume.shape,
    }


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LC-KSVD2 inference: classify a CT and localise the abnormality"
    )
    parser.add_argument("--volume",  required=True,  help="Volume name, e.g. train_1741_c_2")
    parser.add_argument("--model",   required=True,  help="Path to unified_lcksvd2.pkl")
    parser.add_argument("--stride",  type=int, default=None,
                        help="Patch stride in voxels (default: PATCH_SIZE // 2)")
    parser.add_argument("--save-dir", default=None,
                        help="Directory to write per-class heatmap NIfTIs")
    args = parser.parse_args()

    result = run_inference(
        volume_name = args.volume,
        model_path  = Path(args.model),
        stride      = args.stride,
        save_dir    = Path(args.save_dir) if args.save_dir else None,
    )

    print(f"\nPredicted class : {result['predicted_class']}")
    print("Mean patch scores:")
    for cls, score in result["class_scores"].items():
        print(f"  {cls}: {score:.4f}")


if __name__ == "__main__":
    main()