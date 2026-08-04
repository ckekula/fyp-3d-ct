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
  3. Lung segmentation + HU windowing -> [-1, 1] volume (nifti_io.preprocess),
     background/outside-lung voxels at -1.0
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
from scipy import ndimage

from lc_ksvd.config import (
    ABNORMAL_PATCH_STRIDE, CLASS_ORDER, INFERENCE_DIR, MODELS_DIR,
    N_FEATURES, NORMAL_CLASS_IDX, PATCH_SIZE, TARGET_SPACING_MM,
    ZERO_FRACTION_THRESHOLD,
)
from lc_ksvd.data_loader.nifti_io import preprocess, resample_volume, resolve_volume_path
from lc_ksvd.inference.classify import encode_patches_omp, load_dictionary
from lc_ksvd.patch_extractor.patch_io import extract_patch
from lc_ksvd.patch_extractor.patch_sampling_normal import is_background

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# Default algorithm is frozen K-SVD (IncrementalFrozenDictionary) — matches
# the payload produced by `main.py` / `train.py` with --algorithm frozen.
DICT_MODEL_PATH = MODELS_DIR / "unified_lcksvd2.pkl"
SVM_MODEL_PATH = MODELS_DIR / "lcksvd2_svm_model.pkl"

N_NONZERO_COEFS = 10

# Non-overlapping (stride=PATCH_SIZE) patches are 18mm blocks and look
# blocky/undersegmented in the overlay. Training's abnormal-patch sampling
# used ABNORMAL_PATCH_STRIDE (3x overlap in each axis) precisely to get finer
# lesion boundaries via majority voting over overlapping patches — use the
# same stride here by default for consistent, smoother localisation.
INFERENCE_STRIDE = ABNORMAL_PATCH_STRIDE

DEFAULT_SCAN_ID: Optional[str] = "valid_902_a_2"
DEFAULT_VOLUME_PATH: Optional[Path] = None
DEFAULT_OUTPUT_DIR = INFERENCE_DIR
DEFAULT_SHOW = False

# Raw [F, H, W, D] ground-truth mask (nifti_io.load_mask's output format,
# dumped straight to disk with an identity affine — MASKS_DIR / the metadata
# JSON aren't available in this environment for DEFAULT_SCAN_ID, so this
# stand-in file supplies the same array by hand). Auto-used only when running
# with the default scan (see _parse_args) — irrelevant to any other volume.
DEFAULT_GT_MASK_PATH: Optional[Path] = (
        Path(__file__).resolve().parent / "valid_342_a_2_finding3_upper.nii.gz"
)


# ─── 1-3. Volume loading & preprocessing ──────────────────────────────────────

def load_raw_volume(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a NIfTI CT volume from an arbitrary path. Mirrors nifti_io.load_volume."""
    img = nib.load(str(path))
    vol = np.asarray(img.dataobj, dtype=np.float32)
    zooms = np.abs(np.array(img.header.get_zooms()[:3], dtype=np.float32))
    return vol, zooms, img.affine


def output_affine_for(orig_affine: np.ndarray, spacing_mm: float = TARGET_SPACING_MM) -> np.ndarray:
    """
    Build the affine used for every exported (resampled-space) NIfTI file.

    resample_volume() rescales voxel spacing along each array axis but never
    reorders or flips axes, so the export affine must keep the same axis
    *directions* (signs) as the source scan's affine and only replace the
    per-axis spacing with the isotropic TARGET_SPACING_MM. Chest CT is
    commonly stored LPS (negative x/y direction cosines) rather than RAS;
    always writing a plain positive-diagonal affine silently assumes RAS and
    mirrors the exported volume left-right / front-back relative to the
    original scan -- this is the "orientation error" seen when the two are
    loaded side by side in a viewer.
    """
    direction = orig_affine[:3, :3]
    off_diag = direction - np.diag(np.diagonal(direction))
    if not np.allclose(off_diag, 0, atol=1e-3):
        logger.warning("  Source affine is not axis-aligned (has rotation/shear); "
                        "falling back to an identity-direction output affine -- "
                        "orientation of exported files may not match the source scan.")
        signs = np.ones(3)
    else:
        signs = np.sign(np.diagonal(direction))
        signs[signs == 0] = 1.0
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = np.diag(signs * spacing_mm)
    return affine


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

    vol_hu, spacing, orig_affine = load_raw_volume(path)
    logger.info(f"  Raw volume shape={vol_hu.shape}, spacing={spacing}, "
                f"orientation={nib.aff2axcodes(orig_affine)}")

    logger.info("[2/8] Resampling to isotropic spacing...")
    vol_rs = resample_volume(vol_hu, spacing)
    logger.info(f"  Resampled to {vol_rs.shape} @ {TARGET_SPACING_MM}mm isotropic")

    logger.info("[3/8] Running lung segmentation + HU windowing...")
    vol = preprocess(vol_rs)  # lung segmentation + HU windowing -> [-1, 1], background at -1.0
    logger.info(f"  Preprocessed volume ready: shape={vol.shape}, "
                f"lung-tissue voxels={int((~np.isclose(vol, -1.0, atol=1e-6)).sum())}")

    return {
        "name": scan_id or path.name.replace(".nii.gz", "").replace(".nii", ""),
        "volume_hu_resampled": vol_rs,  # kept for reference / alt. visualisation
        "volume": vol,                  # preprocessed [-1,1] volume fed to patching, background=-1.0
        "orig_affine": orig_affine,     # source orientation, carried to every export
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

    Patches that are >50% background are skipped by default — they never
    reach the dictionary/classifier and are implicitly "normal" in the
    output map. Background here uses the exact same test as training's
    normal-patch filter (patch_sampling_normal.is_background): voxels
    isclose to -1.0, not < 1e-6. nifti_io.preprocess() maps
    [LOWER_HU, UPPER_HU] = [-1000, 1000] HU to [-1, 1] (dividing by
    UPPER_HU, not the HU range), and sets everything outside the lung mask
    to LOWER_HU -> -1.0 -- it does NOT produce a [0, 1] volume. Real lung
    parenchyma sits well below 0 in this scale (aerated lung ~ -0.7 to
    -0.95), so a "< 1e-6" background test would misclassify almost every
    genuine lung patch as background and silently discard it.
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
                if skip_background and is_background(patch):
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
    lung boundary into background, so voxels isclose to -1.0 (background,
    per nifti_io.preprocess -- see extract_dense_patches's docstring for why
    it's -1.0 and not 0) are excluded from the output regardless of what
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

    tissue = ~np.isclose(volume, -1.0, atol=1e-6)
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


def compute_class_boundaries(
    label_volume: np.ndarray, thickness: int = 1
) -> Dict[int, np.ndarray]:
    """
    For each abnormal class, return a boolean mask of the voxels on the
    outer edge of that class's region (mask & ~erode(mask)) rather than the
    filled region -- an outline around each lesion instead of a solid blob,
    so the CT texture inside stays visible wherever the outline is painted.

    `thickness` controls how many voxels deep the outline is (erosion
    iterations); 1 gives a single-voxel-wide line.
    """
    boundaries: Dict[int, np.ndarray] = {}
    for cls_idx in range(len(CLASS_ORDER)):
        if cls_idx == NORMAL_CLASS_IDX:
            continue
        mask = label_volume == cls_idx
        if not mask.any():
            boundaries[cls_idx] = mask
            continue
        eroded = ndimage.binary_erosion(mask, iterations=thickness, border_value=0)
        boundaries[cls_idx] = mask & ~eroded
    return boundaries


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


def visualise_abnormality_boundaries(
    volume: np.ndarray,
    label_volume: np.ndarray,
    scan_name: str,
    output_dir: Path = INFERENCE_DIR,
    n_slices: int = 6,
    show: bool = False,
    thickness: int = 1,
    boundaries: Optional[Dict[int, np.ndarray]] = None,
) -> Path:
    """
    Same slice-picking as visualise_abnormalities(), but draws each class's
    boundary voxels (compute_class_boundaries()) as a solid outline instead
    of a semi-transparent filled region, so the CT texture inside the lesion
    stays visible. Saves a separate PNG grid and returns its path.
    """
    logger.info(f"Building boundary visualisation for {scan_name}...")
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

    if boundaries is None:
        boundaries = compute_class_boundaries(label_volume, thickness=thickness)

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
            boundary_mask = boundaries.get(cls_idx)
            if boundary_mask is None or not boundary_mask.any():
                continue
            cls_mask = boundary_mask[:, :, z].T
            if not cls_mask.any():
                continue
            color = class_colors(cls_idx)
            overlay[cls_mask] = (*color[:3], 1.0)  # solid outline, no blending
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
    fig.suptitle(f"Abnormality boundary outline — {scan_name}")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))

    out_path = output_dir / f"{scan_name}_abnormality_boundary.png"
    fig.savefig(out_path, dpi=150)
    logger.info(f"Saved boundary visualisation -> {out_path}")

    if show:
        plt.show()
    plt.close(fig)
    return out_path


# ─── Class colours for RGB overlay ────────────────────────────────────────────
# Each abnormal class gets a distinctive colour (R, G, B) in [0, 255].
# Normal tissue is rendered as pure greyscale (no tint).
_CLASS_COLOURS_RGB = {
    1: (230, 80, 50),    # 2c  — Groundglass opacity   → red-orange
    2: (50, 140, 230),   # 2d  — Pulmonary nodules     → blue-cyan
}
_OVERLAY_ALPHA = 0.50  # blend ratio: 0 = pure CT, 1 = pure colour


_NIFTI_RGB_DTYPE = np.dtype([("R", "u1"), ("G", "u1"), ("B", "u1")])


def save_ct_with_colored_overlay(
    volume_hu: np.ndarray,
    label_volume: np.ndarray,
    scan_name: str,
    output_dir: Path = INFERENCE_DIR,
    affine: Optional[np.ndarray] = None,
) -> Path:
    """
    Create and save a **true-color NIfTI volume** that shows the full,
    original CT anatomy with abnormal regions tinted in their class colour.

    The greyscale base is the resampled *raw HU* volume (not the lung-masked
    [0,1] volume fed to the classifier, which zeroes out everything outside
    the lungs) so that chest wall, mediastinum, etc. remain visible instead
    of being blacked out -- this is what makes it "the original CT" rather
    than the classifier's masked working volume. `label_volume` is already in
    the same resampled voxel grid, so the colour mask still lines up exactly.

    For every voxel:
      - Normal (class 0) or background → greyscale CT value
      - Abnormal (class 1 or 2)        → 50/50 blend of CT grey + class colour

    The array is (H, W, D, 3) uint8, but a plain uint8 array with a trailing
    size-3 axis is written by nibabel as a 4D *scalar* volume (3 separate
    frames/timepoints) -- NOT as colour -- regardless of any intent code set
    on the header (`set_intent("vector")` marks it as a statistical vector
    field, unrelated to display colour). Viewers therefore show it as
    grayscale/frames, never the coloured overlay. To get an actual NIfTI
    RGB24 volume (datatype code 128) that viewers like ITK-SNAP / 3D Slicer
    render in colour automatically, the data must be viewed through a
    structured (R, G, B) dtype before being wrapped in a Nifti1Image.
    """
    logger.info(f"Building RGB overlay volume for {scan_name}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Greyscale base: raw HU, windowed to [LOWER_HU, UPPER_HU] → [0, 255] uint8
    gray = np.clip(volume_hu, LOWER_HU, UPPER_HU)
    gray = ((gray - LOWER_HU) / (UPPER_HU - LOWER_HU) * 255.0).astype(np.uint8)

    # Start with 3-channel greyscale
    rgb = np.stack([gray, gray, gray], axis=-1)  # (H, W, D, 3)

    # Blend abnormal regions with their class colour
    for cls_idx, colour in _CLASS_COLOURS_RGB.items():
        mask = label_volume == cls_idx
        if not mask.any():
            continue
        n_voxels = int(mask.sum())
        for ch in range(3):
            blended = (
                gray[mask].astype(np.float32) * (1.0 - _OVERLAY_ALPHA)
                + colour[ch] * _OVERLAY_ALPHA
            )
            rgb[mask, ch] = np.clip(blended, 0, 255).astype(np.uint8)
        logger.info(f"  Coloured {n_voxels:,} voxels for class "
                    f"{CLASS_ORDER[cls_idx]} with RGB{colour}")

    out_path = output_dir / f"{scan_name}_ct_colored_overlay.nii.gz"
    if affine is None:
        affine = np.diag([TARGET_SPACING_MM, TARGET_SPACING_MM, TARGET_SPACING_MM, 1.0])

    rgb_struct = np.ascontiguousarray(rgb).view(_NIFTI_RGB_DTYPE).reshape(rgb.shape[:-1])
    img = nib.Nifti1Image(rgb_struct, affine=affine)
    nib.save(img, str(out_path))
    logger.info(f"Saved CT + coloured overlay volume (true RGB24) -> {out_path}")
    return out_path


def save_colored_segmentation_mask(
    label_volume: np.ndarray,
    scan_name: str,
    output_dir: Path = INFERENCE_DIR,
    affine: Optional[np.ndarray] = None,
) -> Path:
    """
    Save a standalone colour-coded segmentation mask as a true-colour NIfTI
    (RGB24) — normal/background voxels are black, each abnormal class is
    rendered in its full (unblended) class colour. This is the mask on its
    own, for viewers where the CT-blended overlay's grey base gets in the
    way of toggling the mask layer independently.
    """
    logger.info(f"Building colour-coded segmentation mask for {scan_name}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb = np.zeros((*label_volume.shape, 3), dtype=np.uint8)
    for cls_idx, colour in _CLASS_COLOURS_RGB.items():
        mask = label_volume == cls_idx
        if not mask.any():
            continue
        rgb[mask] = colour
        logger.info(f"  Coloured {int(mask.sum()):,} voxels for class "
                    f"{CLASS_ORDER[cls_idx]} with RGB{colour}")

    out_path = output_dir / f"{scan_name}_segmentation_mask.nii.gz"
    if affine is None:
        affine = np.diag([TARGET_SPACING_MM, TARGET_SPACING_MM, TARGET_SPACING_MM, 1.0])

    rgb_struct = np.ascontiguousarray(rgb).view(_NIFTI_RGB_DTYPE).reshape(rgb.shape[:-1])
    img = nib.Nifti1Image(rgb_struct, affine=affine)
    nib.save(img, str(out_path))
    logger.info(f"Saved colour-coded segmentation mask (true RGB24) -> {out_path}")
    return out_path


def save_ct_with_boundary_overlay(
    volume_hu: np.ndarray,
    label_volume: np.ndarray,
    scan_name: str,
    output_dir: Path = INFERENCE_DIR,
    affine: Optional[np.ndarray] = None,
    thickness: int = 1,
    boundaries: Optional[Dict[int, np.ndarray]] = None,
) -> Path:
    """
    Same true-colour CT base as save_ct_with_colored_overlay(), but paints
    only each class's boundary voxels (compute_class_boundaries()) in solid
    class colour instead of tinting the whole patch region -- an outline
    around each abnormal region rather than a filled, occluding blob, so
    the CT texture inside the lesion is left untouched.

    NOTE: this is an RGB24 volume, same caveats as save_ct_with_colored_overlay
    regarding Volume Rendering transfer functions (see save_boundary_label_map
    below for a scalar alternative that renders natively as a Segmentation).
    """
    logger.info(f"Building RGB boundary-outline volume for {scan_name}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Greyscale base: raw HU, windowed to [LOWER_HU, UPPER_HU] → [0, 255] uint8
    gray = np.clip(volume_hu, LOWER_HU, UPPER_HU)
    gray = ((gray - LOWER_HU) / (UPPER_HU - LOWER_HU) * 255.0).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)  # (H, W, D, 3)

    if boundaries is None:
        boundaries = compute_class_boundaries(label_volume, thickness=thickness)

    for cls_idx, colour in _CLASS_COLOURS_RGB.items():
        boundary = boundaries.get(cls_idx)
        if boundary is None or not boundary.any():
            continue
        rgb[boundary] = colour
        logger.info(f"  Outlined {int(boundary.sum()):,} boundary voxels for class "
                    f"{CLASS_ORDER[cls_idx]} with RGB{colour}")

    out_path = output_dir / f"{scan_name}_ct_boundary_overlay.nii.gz"
    if affine is None:
        affine = np.diag([TARGET_SPACING_MM, TARGET_SPACING_MM, TARGET_SPACING_MM, 1.0])

    rgb_struct = np.ascontiguousarray(rgb).view(_NIFTI_RGB_DTYPE).reshape(rgb.shape[:-1])
    img = nib.Nifti1Image(rgb_struct, affine=affine)
    nib.save(img, str(out_path))
    logger.info(f"Saved CT + boundary outline volume (true RGB24) -> {out_path}")
    return out_path


def save_boundary_label_map(
    label_volume: np.ndarray,
    scan_name: str,
    output_dir: Path = INFERENCE_DIR,
    affine: Optional[np.ndarray] = None,
    thickness: int = 1,
    boundaries: Optional[Dict[int, np.ndarray]] = None,
) -> Path:
    """
    Save class boundaries as a plain **scalar uint8 NIfTI label map**. Voxel
    value = class index (1, 2, ...) on the outline voxels, 0 elsewhere.

    A single-channel label map lets Slicer's Segmentations module import it
    natively (Segmentations -> Import), where a "Show 3D" button runs
    marching cubes to build a closed surface mesh straight from the label
    voxels -- correct for outline/boundary data, which is thin, sparse
    geometry rather than a dense volumetric signal. Per-segment colours can
    be set in the Segmentations module after import.

    `thickness` / `boundaries` behave exactly as in compute_class_boundaries();
    pass an already-computed `boundaries` dict (e.g. from run_inference) to
    avoid recomputing the erosion.
    """
    logger.info(f"Building scalar boundary label map for {scan_name}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    if boundaries is None:
        boundaries = compute_class_boundaries(label_volume, thickness=thickness)

    label_map = np.zeros(label_volume.shape, dtype=np.uint8)
    for cls_idx, boundary in boundaries.items():
        if boundary is None or not boundary.any():
            continue
        label_map[boundary] = cls_idx  # scalar label value, not colour
        logger.info(f"  Labelled {int(boundary.sum()):,} boundary voxels for class "
                    f"{CLASS_ORDER[cls_idx]} as label value {cls_idx}")

    out_path = output_dir / f"{scan_name}_boundary_labelmap.nii.gz"
    if affine is None:
        affine = np.diag([TARGET_SPACING_MM, TARGET_SPACING_MM, TARGET_SPACING_MM, 1.0])

    img = nib.Nifti1Image(label_map, affine=affine)
    # Explicit scalar int datatype so viewers treat it as a label map, not
    # a generic float/intensity volume.
    img.header.set_data_dtype(np.uint8)
    nib.save(img, str(out_path))
    logger.info(f"Saved scalar boundary label map -> {out_path}")
    return out_path


def save_resampled_ct_volume(
    volume_hu: np.ndarray,
    scan_name: str,
    output_dir: Path = INFERENCE_DIR,
    affine: Optional[np.ndarray] = None,
) -> Path:
    """
    Save the resampled CT volume as a plain scalar-HU NIfTI (real Hounsfield
    values, not the [0,255] display-window used for the RGB overlays above).
    This is the file to load for 3D Slicer's Volume Rendering module -- its
    CT presets are calibrated to real HU ranges, so they only work against
    genuine HU data, not the RGB24 exports.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{scan_name}_ct_resampled.nii.gz"
    if affine is None:
        affine = np.diag([TARGET_SPACING_MM, TARGET_SPACING_MM, TARGET_SPACING_MM, 1.0])
    nib.save(nib.Nifti1Image(volume_hu.astype(np.float32), affine=affine), str(out_path))
    logger.info(f"Saved resampled scalar HU CT volume -> {out_path}")
    return out_path


def build_ground_truth_label_volume(
    gt_mask_raw: np.ndarray, target_shape: Tuple[int, int, int]
) -> np.ndarray:
    """
    Collapse a raw [F, H, W, D] ground-truth mask (nifti_io.load_mask format —
    one channel per annotated finding, instance IDs 1/2/3.. within a channel)
    into a single-channel [H, W, D] label map on the resampled grid, for
    export as a plain scalar NIfTI alongside the predicted-abnormality maps.

    Instance IDs within a finding channel aren't needed for a Slicer overlay
    (only "which finding is here" is), so each channel is flattened to a
    boolean "present" mask first. Where two finding channels overlap at the
    same voxel (rare), the lower channel index wins — an arbitrary but
    deterministic tie-break, since we have no metadata JSON available here to
    rank categories by clinical relevance.

    nifti_io.resample_mask() does the actual resampling (nearest-neighbour,
    per channel, same as ScanLoader uses for training-time masks) so this
    mask ends up on the exact same voxel grid as the resampled CT volume.
    """
    gt_resampled = resample_mask(gt_mask_raw, target_shape=target_shape)  # (F,H,W,D)
    label_volume_gt = np.zeros(target_shape, dtype=np.uint8)
    for f in range(gt_resampled.shape[0]):
        unset = label_volume_gt == 0
        label_volume_gt[unset & (gt_resampled[f] > 0)] = f + 1
    return label_volume_gt


def save_ground_truth_mask(
    label_volume_gt: np.ndarray,
    scan_name: str,
    output_dir: Path = INFERENCE_DIR,
    affine: Optional[np.ndarray] = None,
) -> Path:
    """
    Save the collapsed ground-truth label map (see
    build_ground_truth_label_volume) as a plain scalar uint8 NIfTI — same
    label-map convention as save_boundary_label_map, so it imports cleanly
    into Slicer's Segmentations module (Segmentations -> Import) right next
    to the predicted boundary/segmentation outputs for a side-by-side check.
    Voxel value = finding-channel index + 1, 0 = no annotated finding.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{scan_name}_ground_truth_mask.nii.gz"
    if affine is None:
        affine = np.diag([TARGET_SPACING_MM, TARGET_SPACING_MM, TARGET_SPACING_MM, 1.0])

    img = nib.Nifti1Image(label_volume_gt, affine=affine)
    img.header.set_data_dtype(np.uint8)
    nib.save(img, str(out_path))
    logger.info(f"Saved ground-truth mask label map -> {out_path}")
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
    gt_mask_path: Optional[Path] = None,
) -> Dict:
    """
    Full pipeline: load -> preprocess -> patch -> sparse-code -> classify ->
    localise -> visualise, for a single volume. Returns a dict with the
    intermediate arrays and output file paths, for programmatic use.

    `gt_mask_path`, if given, points to a raw [F, H, W, D] ground-truth mask
    NIfTI (nifti_io.load_mask's format) on the *original* (pre-resample) CT
    grid; it's resampled and exported as an extra scalar label-map NIfTI
    (see save_ground_truth_mask) so it can be loaded into 3D Slicer right
    alongside the predicted outputs for a visual comparison.
    """
    t0 = time.time()
    logger.info(f"{'='*60}\nRunning inference (volume={volume_path}, scan_id={scan_id})\n{'='*60}")

    scan = load_and_preprocess_volume(volume_path=volume_path, scan_id=scan_id)
    volume, scan_name = scan["volume"], scan["name"]
    volume_hu_resampled = scan["volume_hu_resampled"]
    export_affine = output_affine_for(scan["orig_affine"])

    X, coords = extract_dense_patches(volume, stride=stride)

    logger.info(f"[5/8] Loading dictionary + sparse-coding patches (n_nonzero_coefs={n_nonzero_coefs})...")
    D = load_dictionary(dict_model_path)
    Gamma = encode_patches_omp(X, D, n_nonzero_coefs=n_nonzero_coefs)

    pred_labels, decision_scores = classify_patches(Gamma, svm_path=svm_model_path)

    maps = build_abnormality_volume(volume, coords, pred_labels, decision_scores)
    boundaries = compute_class_boundaries(maps["label_volume"])

    overlay_path = visualise_abnormalities(
        volume, maps["label_volume"], scan_name, output_dir=output_dir, show=show,
    )
    boundary_png_path = visualise_abnormality_boundaries(
        volume, maps["label_volume"], scan_name, output_dir=output_dir, show=show,
        boundaries=boundaries,
    )

    # ─── Step 9: NIfTI exports (boundary label map + resampled CT) ───────────
    # Scalar label-map version of the predicted boundaries -- imports as a
    # Slicer Segmentation with native "Show 3D" surface rendering.
    boundary_labelmap_path = save_boundary_label_map(
        maps["label_volume"], scan_name,
        output_dir=output_dir, affine=export_affine, boundaries=boundaries,
    )
    ct_resampled_path = save_resampled_ct_volume(
        volume_hu_resampled, scan_name, output_dir=output_dir, affine=export_affine,
    )

    # Ground-truth mask export -- DISABLED. Kept commented out rather than
    # deleted since build_ground_truth_label_volume/save_ground_truth_mask
    # still work correctly given a real [F,H,W,D] mask file; there just isn't
    # one available for the current default scan (DEFAULT_GT_MASK_PATH points
    # at the raw CT volume, not an actual finding mask, which crashes
    # resample_mask on a shape mismatch). Re-enable once a real mask is wired
    # up, or when gt_mask_path is guaranteed to point at a proper mask file.
    gt_mask_nifti_path = None
    # if gt_mask_path is not None:
    #     gt_mask_path = Path(gt_mask_path)
    #     if gt_mask_path.exists():
    #         logger.info(f"Loading ground-truth mask for export: {gt_mask_path}")
    #         gt_mask_raw = np.asarray(nib.load(str(gt_mask_path)).dataobj, dtype=np.uint8)
    #         label_volume_gt = build_ground_truth_label_volume(gt_mask_raw, target_shape=volume.shape)
    #         gt_mask_nifti_path = save_ground_truth_mask(
    #             label_volume_gt, scan_name, output_dir=output_dir, affine=export_affine,
    #         )
    #     else:
    #         logger.warning(f"  gt_mask_path does not exist, skipping: {gt_mask_path}")

    elapsed = time.time() - t0
    logger.info(f"Inference complete for {scan_name} in {elapsed:.1f}s -> "
                f"{overlay_path}, {boundary_png_path}, {boundary_labelmap_path}, "
                f"{ct_resampled_path}"
                + (f", {gt_mask_nifti_path}" if gt_mask_nifti_path else ""))

    return {
        "scan_name":               scan_name,
        "volume":                  volume,
        "coords":                  coords,
        "pred_labels":             pred_labels,
        "label_volume":            maps["label_volume"],
        "heat_volume":             maps["heat_volume"],
        "overlay_png":             overlay_path,
        "boundary_png":            boundary_png_path,
        "boundary_labelmap_nifti": boundary_labelmap_path,
        "ct_resampled_nifti":      ct_resampled_path,
        "ground_truth_mask_nifti": gt_mask_nifti_path,
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
    parser.add_argument(
        "--gt-mask", type=Path, default=None,
        help="Path to a raw [F,H,W,D] ground-truth mask NIfTI (nifti_io.load_mask "
             "format) to resample and export alongside the predicted outputs, for "
             "side-by-side comparison in a viewer like 3D Slicer. Defaults to "
             f"{DEFAULT_GT_MASK_PATH} when running against the default scan "
             "(no --volume/--scan-id override).",
    )
    args = parser.parse_args()

    if args.volume is not None and args.scan_id is not None and args.scan_id != DEFAULT_SCAN_ID:
        parser.error("Pass only one of --volume / --scan-id.")
    if args.volume is not None:
        args.scan_id = None  # an explicit --volume overrides the default scan-id

    if args.gt_mask is None and args.volume is None and args.scan_id == DEFAULT_SCAN_ID:
        args.gt_mask = DEFAULT_GT_MASK_PATH

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
        gt_mask_path=args.gt_mask,
    )


if __name__ == "__main__":
    main()