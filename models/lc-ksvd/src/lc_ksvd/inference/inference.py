"""
inference.py
End-to-end single-volume inference pipeline for the LC-KSVD chest CT project.

Given a single lung CT volume (NIfTI), this module runs the exact same
preprocessing/patch pipeline used at training time, sparse-codes the
resulting patches against the trained dictionary (default algorithm: frozen
K-SVD, i.e. IncrementalFrozenDictionary), classifies each patch with the
trained SVM, and reconstructs a per-voxel abnormality map that is overlaid
on the original volume for visualisation.

Pipeline (mirrors patch_extractor/* and inference/classify.py exactly):
  1. Load NIfTI volume (raw HU) + voxel spacing
  2. Resample to TARGET_SPACING_MM isotropic
  3. Lung segmentation + HU windowing -> [0, 1] volume (nifti_io.preprocess)
  4. Dense grid patch extraction (PATCH_SIZE^3 patches, configurable stride)
  5. Column-normalise patches, sparse-code against dictionary D via Batch-OMP
  6. Classify each patch's sparse code with the trained LinearSVC
  7. Reconstruct per-voxel label / confidence volumes from patch predictions
  8. Overlay abnormal regions on the original volume and visualise

Usage:
  python -m lc_ksvd.inference.inference --volume /path/to/scan.nii.gz
  python -m lc_ksvd.inference.inference --scan-id <id-in-VOLUMES_DIR>
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from lc_ksvd.config import (
    ABNORMAL_PATCH_STRIDE, CLASS_ORDER, INFERENCE_DIR, MODELS_DIR, N_FEATURES,
    NORMAL_CLASS_IDX, PATCH_SIZE, TARGET_SPACING_MM, ZERO_FRACTION_THRESHOLD,
)
from lc_ksvd.data_loader.nifti_io import preprocess, resample_volume, resolve_volume_path
from lc_ksvd.inference.classify import encode_patches, load_dictionary
from lc_ksvd.patch_extractor.patch_io import extract_patch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# Default algorithm is frozen K-SVD (IncrementalFrozenDictionary) — matches
# the payload produced by `main.py` / `train.py` with --algorithm frozen.
DICT_MODEL_PATH = MODELS_DIR / "unified_frozen.pkl"
SVM_MODEL_PATH = MODELS_DIR / "frozen_svm_model.pkl"

N_NONZERO_COEFS = 10

# Non-overlapping (stride=PATCH_SIZE) patches are 18mm blocks and look
# blocky/undersegmented in the overlay. Training's abnormal-patch sampling
# used ABNORMAL_PATCH_STRIDE (3x overlap in each axis) precisely to get finer
# lesion boundaries via majority voting over overlapping patches — use the
# same stride here by default for consistent, smoother localisation.
INFERENCE_STRIDE = ABNORMAL_PATCH_STRIDE

DEFAULT_SCAN_ID: Optional[str] = "train_10000_a_1"
DEFAULT_VOLUME_PATH: Optional[Path] = None
DEFAULT_OUTPUT_DIR = INFERENCE_DIR
DEFAULT_SHOW = False


# ─── 1-3. Volume loading & preprocessing ──────────────────────────────────────

def load_raw_volume(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI CT volume from an arbitrary path. Mirrors nifti_io.load_volume."""
    img = nib.load(str(path))
    vol = np.asarray(img.dataobj, dtype=np.float32)
    zooms = np.abs(np.array(img.header.get_zooms()[:3], dtype=np.float32))
    return vol, zooms


def load_and_preprocess_volume(
    volume_path: Optional[Path] = None,
    scan_id: Optional[str] = None,
) -> Dict:
    """
    Load + resample + lung-segment/window a single volume, exactly as
    ScanLoader.load() does at training time (minus the mask/label lookup,
    which don't apply to a fresh inference volume).

    Pass either `volume_path` (any NIfTI file) or `scan_id` (resolved via
    VOLUMES_DIR, same convention as training).
    """
    if (volume_path is None) == (scan_id is None):
        raise ValueError("Pass exactly one of volume_path or scan_id.")

    logger.info("[1/8] Resolving + loading volume...")
    path = resolve_volume_path(scan_id) if scan_id is not None else Path(volume_path)
    logger.info(f"  Loading volume: {path}")

    vol_hu, spacing = load_raw_volume(path)
    logger.info(f"  Raw volume shape={vol_hu.shape}, spacing={spacing}")

    logger.info("[2/8] Resampling to isotropic spacing...")
    vol_rs = resample_volume(vol_hu, spacing)
    logger.info(f"  Resampled to {vol_rs.shape} @ {TARGET_SPACING_MM}mm isotropic")

    logger.info("[3/8] Running lung segmentation + HU windowing...")
    vol = preprocess(vol_rs)  # lung segmentation + HU windowing -> [0, 1]
    logger.info(f"  Preprocessed volume ready: shape={vol.shape}, "
                f"lung-tissue voxels={(int((vol > 0).sum()))}")

    return {
        "name": scan_id or path.name.replace(".nii.gz", "").replace(".nii", ""),
        "volume_hu_resampled": vol_rs,  # kept for reference / alt. visualisation
        "volume": vol,                  # preprocessed [0,1] volume fed to patching
    }


# ─── 4. Dense patch extraction ────────────────────────────────────────────────

def extract_dense_patches(
    volume: np.ndarray,
    stride: int = INFERENCE_STRIDE,
    skip_background: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slide a PATCH_SIZE^3 window across the whole volume on a regular grid.

    Returns:
      X      : (N_FEATURES, n_patches) float64 patch matrix
      coords : (n_patches, 3) int64 array of (x0, y0, z0) patch origins

    Patches that are >50% zero (pure air/background, same rule as training's
    normal-patch filter) are skipped by default — they never reach the
    dictionary/classifier and are implicitly "normal" in the output map.
    """
    logger.info(f"[4/8] Extracting dense patches (stride={stride}, patch_size={PATCH_SIZE})...")
    H, W, D = volume.shape
    p = PATCH_SIZE
    patches, coords = [], []
    n_skipped_background = 0

    for x0 in range(0, H - p + 1, stride):
        for y0 in range(0, W - p + 1, stride):
            for z0 in range(0, D - p + 1, stride):
                patch = extract_patch(volume, x0, y0, z0)
                if patch is None:
                    continue
                if skip_background and (patch < 1e-6).mean() > ZERO_FRACTION_THRESHOLD:
                    n_skipped_background += 1
                    continue
                patches.append(patch.ravel())
                coords.append((x0, y0, z0))

    if not patches:
        logger.warning("  No patches survived the background filter — empty volume or all-air scan?")
        return np.empty((N_FEATURES, 0)), np.empty((0, 3), dtype=np.int64)

    X = np.stack(patches, axis=1).astype(np.float64)  # (N_FEATURES, n_patches)
    coords_arr = np.array(coords, dtype=np.int64)
    logger.info(f"  Extracted {X.shape[1]} patches (skipped {n_skipped_background} background-only "
                f"patches) from volume {volume.shape}")
    return X, coords_arr


# ─── 5-6. Sparse coding + classification ──────────────────────────────────────
#
# load_dictionary() and encode_patches() are imported from inference/classify.py
# (this project's canonical Batch-OMP encoding step) rather than redefined here
# — same dictionary payload format, same OMP config, no need for a second copy.

def classify_patches(
    Gamma: np.ndarray, svm_path: Path = SVM_MODEL_PATH
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classify each patch's sparse code with the trained LinearSVC.

    Returns (pred_labels, decision_scores):
      pred_labels     : (n_patches,) int class index per patch (into CLASS_ORDER)
      decision_scores : (n_patches, n_classes) signed distance to each class'
                        separating hyperplane, used as a confidence proxy.
    """
    logger.info("[6/8] Classifying patches with the trained SVM...")
    n_classes = len(CLASS_ORDER)
    if Gamma.shape[1] == 0:
        logger.warning("  No patches to classify (empty Gamma).")
        return np.empty((0,), dtype=np.int64), np.empty((0, n_classes))

    logger.info(f"  Loading SVM classifier from {svm_path}")
    svm_clf = joblib.load(svm_path)
    X = Gamma.T  # (n_patches, n_components)
    pred = svm_clf.predict(X)
    scores = svm_clf.decision_function(X)
    if scores.ndim == 1:  # binary case: decision_function returns a 1D margin
        scores = np.stack([-scores, scores], axis=1)

    counts = {CLASS_ORDER[i]: int((pred == i).sum()) for i in range(n_classes)}
    logger.info(f"  Classified {len(pred)} patches -> class counts: {counts}")
    return pred, scores


# ─── 7. Reconstruct per-voxel abnormality map ─────────────────────────────────

def build_abnormality_volume(
    volume: np.ndarray,
    coords: np.ndarray,
    pred_labels: np.ndarray,
    decision_scores: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Paint each patch's predicted class + confidence back into full-volume
    arrays, majority-voting where overlapping patches disagree.

    A classified patch is accepted as a whole cube even when up to
    ZERO_FRACTION_THRESHOLD of its voxels are background (same tolerance
    training used for normal patches) — boundary patches straddling the lung
    edge are common. Painting that whole cube would bleed the label past the
    lung boundary into background, so voxels with volume <= 0 (background,
    per nifti_io.preprocess) are excluded from the output regardless of what
    their covering patch(es) predicted.

    Returns:
      label_volume : (H,W,D) int, CLASS_ORDER index per voxel (NORMAL_CLASS_IDX
                     where no patch covered that voxel, or it's background)
      heat_volume  : (H,W,D) float, max abnormal decision score per voxel
                     (0 where never covered by an abnormal-predicted patch)
    """
    logger.info("[7/8] Reconstructing per-voxel abnormality map from patch predictions...")
    p = PATCH_SIZE
    n_classes = len(CLASS_ORDER)
    volume_shape = volume.shape
    votes = np.zeros((n_classes,) + volume_shape, dtype=np.int32)
    heat = np.zeros(volume_shape, dtype=np.float32)

    for (x0, y0, z0), label, scores in zip(coords, pred_labels, decision_scores):
        votes[label, x0:x0 + p, y0:y0 + p, z0:z0 + p] += 1
        if label != NORMAL_CLASS_IDX:
            score = float(scores[label])
            region = heat[x0:x0 + p, y0:y0 + p, z0:z0 + p]
            np.maximum(region, score, out=region)

    tissue = volume > 0
    covered = (votes.sum(axis=0) > 0) & tissue
    label_volume = np.full(volume_shape, NORMAL_CLASS_IDX, dtype=np.int64)
    label_volume[covered] = np.argmax(votes, axis=0)[covered]
    heat[~tissue] = 0

    abnormal_voxel_counts = {
        CLASS_ORDER[i]: int((label_volume == i).sum())
        for i in range(n_classes) if i != NORMAL_CLASS_IDX
    }
    logger.info(f"  Abnormal voxel counts (post tissue-masking): {abnormal_voxel_counts}")

    return {"label_volume": label_volume, "heat_volume": heat}


# ─── 8. Visualisation ──────────────────────────────────────────────────────────

def visualise_abnormalities(
    volume: np.ndarray,
    label_volume: np.ndarray,
    scan_name: str,
    output_dir: Path = INFERENCE_DIR,
    n_slices: int = 6,
    show: bool = False,
) -> Path:
    """
    Overlay abnormal regions (label_volume != NORMAL_CLASS_IDX) in colour on
    top of grayscale axial slices of `volume`, picking the slices with the
    most abnormal voxels. Saves a PNG grid and returns its path.
    """
    logger.info(f"[8/8] Building overlay visualisation for {scan_name}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    abnormal_mask = label_volume != NORMAL_CLASS_IDX

    voxels_per_slice = abnormal_mask.sum(axis=(0, 1))  # per z-slice
    if voxels_per_slice.sum() == 0:
        top_slices = np.linspace(0, volume.shape[2] - 1, n_slices).astype(int)
        logger.info(f"  [{scan_name}] No abnormal patches detected; showing evenly-spaced slices.")
    else:
        top_slices = np.argsort(voxels_per_slice)[::-1][:n_slices]
        top_slices = np.sort(top_slices)
        logger.info(f"  Selected slices with most abnormal voxels: {top_slices.tolist()}")

    class_colors = plt.get_cmap("tab10", len(CLASS_ORDER))

    n_cols = min(3, len(top_slices))
    n_rows = int(np.ceil(len(top_slices) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, z in zip(axes, top_slices):
        gray = volume[:, :, z].T
        ax.imshow(gray, cmap="gray", origin="lower")

        overlay = np.zeros((*gray.shape, 4))
        for cls_idx in range(len(CLASS_ORDER)):
            if cls_idx == NORMAL_CLASS_IDX:
                continue
            cls_mask = (label_volume[:, :, z] == cls_idx).T
            if not cls_mask.any():
                continue
            color = class_colors(cls_idx)
            overlay[cls_mask] = (*color[:3], 0.45)
        ax.imshow(overlay, origin="lower")
        ax.set_title(f"z={z}")
        ax.axis("off")

    for ax in axes[len(top_slices):]:
        ax.axis("off")

    handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=class_colors(i),
                   markersize=12, label=CLASS_ORDER[i])
        for i in range(len(CLASS_ORDER)) if i != NORMAL_CLASS_IDX
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles))
    fig.suptitle(f"Abnormality localisation — {scan_name}")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))

    out_path = output_dir / f"{scan_name}_abnormality_overlay.png"
    fig.savefig(out_path, dpi=150)
    logger.info(f"Saved overlay visualisation -> {out_path}")

    if show:
        plt.show()
    plt.close(fig)
    return out_path


def save_label_volume(
    label_volume: np.ndarray, scan_name: str, output_dir: Path = INFERENCE_DIR
) -> Path:
    """Save the per-voxel label volume as NIfTI for 3D viewers (ITK-SNAP, Slicer)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{scan_name}_abnormality_labels.nii.gz"
    # Voxels are TARGET_SPACING_MM isotropic post-resample (nifti_io.resample_volume);
    # an identity affine would misreport them as 1mm and misalign against the CT volume.
    affine = np.diag([TARGET_SPACING_MM, TARGET_SPACING_MM, TARGET_SPACING_MM, 1.0])
    nib.save(nib.Nifti1Image(label_volume.astype(np.int16), affine=affine), str(out_path))
    logger.info(f"Saved label volume -> {out_path}")
    return out_path


# ─── End-to-end pipeline ───────────────────────────────────────────────────────

def run_inference(
    volume_path: Optional[Path] = None,
    scan_id: Optional[str] = None,
    dict_model_path: Path = DICT_MODEL_PATH,
    svm_model_path: Path = SVM_MODEL_PATH,
    stride: int = INFERENCE_STRIDE,
    n_nonzero_coefs: int = N_NONZERO_COEFS,
    output_dir: Path = INFERENCE_DIR,
    show: bool = False,
) -> Dict:
    """
    Full pipeline: load -> preprocess -> patch -> sparse-code -> classify ->
    localise -> visualise, for a single volume. Returns a dict with the
    intermediate arrays and output file paths, for programmatic use.
    """
    t0 = time.time()
    logger.info(f"{'='*60}\nRunning inference (volume={volume_path}, scan_id={scan_id})\n{'='*60}")

    scan = load_and_preprocess_volume(volume_path=volume_path, scan_id=scan_id)
    volume, scan_name = scan["volume"], scan["name"]

    X, coords = extract_dense_patches(volume, stride=stride)

    logger.info(f"[5/8] Loading dictionary + sparse-coding patches (n_nonzero_coefs={n_nonzero_coefs})...")
    D = load_dictionary(dict_model_path)
    Gamma = encode_patches(X, D, n_nonzero_coefs=n_nonzero_coefs)

    pred_labels, decision_scores = classify_patches(Gamma, svm_path=svm_model_path)

    maps = build_abnormality_volume(volume, coords, pred_labels, decision_scores)

    overlay_path = visualise_abnormalities(
        volume, maps["label_volume"], scan_name, output_dir=output_dir, show=show,
    )
    label_path = save_label_volume(maps["label_volume"], scan_name, output_dir=output_dir)

    elapsed = time.time() - t0
    logger.info(f"Inference complete for {scan_name} in {elapsed:.1f}s -> "
                f"{overlay_path}, {label_path}")

    return {
        "scan_name":     scan_name,
        "volume":        volume,
        "coords":        coords,
        "pred_labels":   pred_labels,
        "label_volume":  maps["label_volume"],
        "heat_volume":   maps["heat_volume"],
        "overlay_png":   overlay_path,
        "label_nifti":   label_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end LC-KSVD inference + abnormality "
                    "localisation on a single lung CT volume."
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--volume", type=Path, default=DEFAULT_VOLUME_PATH,
                        help="Path to a NIfTI volume (.nii/.nii.gz).")
    group.add_argument("--scan-id", type=str, default=DEFAULT_SCAN_ID,
                        help="Scan ID resolved via VOLUMES_DIR.")
    parser.add_argument(
        "--stride", type=int, default=INFERENCE_STRIDE,
        help=f"Patch grid stride in voxels (default: {INFERENCE_STRIDE} = PATCH_SIZE, "
             "non-overlapping; use a smaller value for finer localisation).",
    )
    parser.add_argument("--n-nonzero-coefs", type=int, default=N_NONZERO_COEFS)
    parser.add_argument("--dict-model", type=Path, default=DICT_MODEL_PATH)
    parser.add_argument("--svm-model", type=Path, default=SVM_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--show", action="store_true", default=DEFAULT_SHOW,
                         help="Display the overlay interactively.")
    args = parser.parse_args()

    if args.volume is not None and args.scan_id is not None and args.scan_id != DEFAULT_SCAN_ID:
        parser.error("Pass only one of --volume / --scan-id.")
    if args.volume is not None:
        args.scan_id = None  # an explicit --volume overrides the default scan-id

    return args


def main() -> None:
    args = _parse_args()
    logger.info(f"CLI args: {vars(args)}")
    run_inference(
        volume_path=args.volume,
        scan_id=args.scan_id,
        dict_model_path=args.dict_model,
        svm_model_path=args.svm_model,
        stride=args.stride,
        n_nonzero_coefs=args.n_nonzero_coefs,
        output_dir=args.output_dir,
        show=args.show,
    )


if __name__ == "__main__":
    main()
