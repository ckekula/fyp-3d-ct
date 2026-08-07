"""
inference.py
End-to-end single-volume inference pipeline for the LC-KSVD chest CT project.

Given a single lung CT volume (NIfTI), this module runs the exact same
preprocessing pipeline used at training time (nifti_io.py / ScanLoader),
patches the resulting volume on a dense grid, sparse-codes each patch
against the trained dictionary (default: frozen K-SVD, i.e.
IncrementalFrozenDictionary), classifies each patch with the trained SVM,
and reconstructs a per-voxel abnormality map.

Pipeline (mirrors ScanLoader.load() / patch_extractor/* / inference/classify.py):
  1. Load NIfTI volume (raw HU) + voxel spacing
  2. Resample to TARGET_SPACING_MM isotropic                  (nifti_io.resample_volume)
  3. Center-crop/pad to TARGET_SHAPE, then lung segmentation
     + HU windowing -> [-1, 1]                                 (nifti_io.crop_or_pad / preprocess)
  4. Dense grid patch extraction (PATCH_SIZE^3 patches, configurable stride)
  5. Column-normalise patches, sparse-code against dictionary D via Batch-OMP
  6. Classify each patch's sparse code with the trained LinearSVC
  7. Reconstruct per-voxel label / confidence volumes from patch predictions
  8. Export outputs: resampled CT, boundary label map, ground-truth mask (if given)

NOTE on value range: nifti_io.preprocess() rescales HU by dividing by
UPPER_HU (not min-max normalisation), so the preprocessed volume fed to the
patcher/classifier lives in **[-1, 1]**, not [0, 1]. Voxels outside the lung
mask are set to LOWER_HU before that division, so background is *exactly*
-1.0 -- this is the same convention patch_sampling_normal.is_background()
uses to decide which patches are background-only, and it is what this module
uses throughout (extract_dense_patches, build_abnormality_volume).

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
import nibabel as nib
import numpy as np
from scipy import ndimage

from lc_ksvd.config import (
    ABNORMAL_PATCH_STRIDE, CLASS_ORDER, INFERENCE_DIR, LOWER_HU, MODELS_DIR,
    N_FEATURES, NORMAL_CLASS_IDX, PATCH_SIZE, TARGET_SHAPE, TARGET_SPACING_MM,
)
from lc_ksvd.data_loader.nifti_io import (
    crop_or_pad, crop_or_pad_mask, ensure_3d_volume, preprocess, resample_mask,
    resample_volume, resolve_volume_path,
)
from lc_ksvd.inference.classify import encode_patches_omp, load_dictionary
from lc_ksvd.patch_extractor.patch_io import extract_patch
from lc_ksvd.patch_extractor.patch_sampling_abnormal import _build_category_masks
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

DEFAULT_SCAN_ID: Optional[str] = None
DEFAULT_VOLUME_PATH: Optional[Path] = Path(__file__).resolve().parent / "valid_1067_a_2.nii.gz"
DEFAULT_OUTPUT_DIR = INFERENCE_DIR


# ─── 1-3. Volume loading & preprocessing ──────────────────────────────────────

def load_raw_volume(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a NIfTI CT volume from an arbitrary path. Mirrors nifti_io.load_volume."""
    img = nib.load(str(path))
    vol = ensure_3d_volume(np.asarray(img.dataobj, dtype=np.float32), context=str(path))
    zooms = np.abs(np.array(img.header.get_zooms()[:3], dtype=np.float32))
    return vol, zooms, img.affine


def output_affine_for(orig_affine: np.ndarray, spacing_mm: tuple = TARGET_SPACING_MM) -> np.ndarray:
    """
    Build the affine used for every exported (resampled-space) NIfTI file.

    resample_volume() rescales voxel spacing along each array axis but never
    reorders or flips axes, so the export affine must keep the same axis
    *directions* (signs) as the source scan's affine and replace the
    per-axis spacing with the TARGET_SPACING_MM while preserving the physical
    world origin coordinates (orig_affine[:3, 3]).
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
    affine = orig_affine.copy()
    affine[:3, :3] = np.diag(signs * np.array(spacing_mm))
    return affine


def load_and_preprocess_volume(
    volume_path: Optional[Path] = None,
    scan_id: Optional[str] = None,
) -> Dict:
    """
    Load + resample + center-crop/pad + lung-segment/window a single volume,
    exactly as ScanLoader.load() does at training time (minus the mask/
    finding-map lookup, which don't apply to a fresh inference volume).

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
    resampled_shape_pre_crop = vol_rs.shape
    logger.info(f"  Resampled to {vol_rs.shape} @ {TARGET_SPACING_MM}mm isotropic")

    logger.info("[3/8] Center-crop/pad to training matrix size, "
                "then lung segmentation + HU windowing...")
    vol_hu_cp = crop_or_pad(vol_rs, TARGET_SHAPE, pad_value=LOWER_HU)
    vol = preprocess(vol_hu_cp)  # lung segmentation + HU windowing -> [-1, 1]
    n_tissue = int((~np.isclose(vol, -1.0, atol=1e-6)).sum())
    logger.info(f"  Preprocessed volume ready: shape={vol.shape}, lung-tissue voxels={n_tissue}")

    return {
        "name": scan_id or path.name.replace(".nii.gz", "").replace(".nii", ""),
        "volume_hu_resampled": vol_hu_cp,  # raw HU, same grid as `volume` -- exported as ct_resampled
        "volume": vol,                     # preprocessed [-1,1] volume fed to patching
        "orig_affine": orig_affine,        # source orientation, carried to every export
        "resampled_shape_pre_crop": resampled_shape_pre_crop,  # needed to resample a GT mask correctly
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

    Patches are skipped when more than ZERO_FRACTION_THRESHOLD of their
    voxels are background (the lung-mask fill value, exactly -1.0 -- see
    nifti_io.preprocess) using patch_sampling_normal.is_background(), the
    same rule training's normal-patch grid sampling uses. Skipped patches
    never reach the dictionary/classifier and are implicitly "normal" in
    the output map.
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
# load_dictionary() and encode_patches_omp() are imported from
# inference/classify.py (this project's canonical Batch-OMP encoding step)
# rather than redefined here — same dictionary payload format, same OMP
# config, no need for a second copy.

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
    lung boundary into background, so voxels equal to the lung-mask
    background fill value (-1.0, per nifti_io.preprocess) are excluded from
    the output regardless of what their covering patch(es) predicted.

    Returns:
      label_volume : (H,W,D) int, CLASS_ORDER index per voxel (NORMAL_CLASS_IDX
                     where no patch covered that voxel, or it's background)
      heat_volume  : (H,W,D) float, max abnormal decision score per voxel
                     (0 where never covered by an abnormal-predicted patch)
      abnormal_voxel_counts : {class_name: voxel_count}, classes != normal
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

    return {
        "label_volume": label_volume,
        "heat_volume": heat,
        "abnormal_voxel_counts": abnormal_voxel_counts,
    }


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


# ─── 8. Export outputs ─────────────────────────────────────────────────────────

def save_boundary_label_map(
    label_volume: np.ndarray,
    scan_name: str,
    output_dir: Path = INFERENCE_DIR,
    affine: Optional[np.ndarray] = None,
    thickness: int = 1,
    boundaries: Optional[Dict[int, np.ndarray]] = None,
) -> Path:
    """
    Save class boundaries as a scalar uint8 NIfTI **label map**. Voxel
    value = class index (1, 2, ...) on the outline voxels, 0 elsewhere.

    A single-channel label map imports natively into 3D Slicer's
    Segmentations module (Segmentations -> Import) and exposes a "Show 3D"
    button that runs marching cubes straight from the label voxels -- the
    right fit for outline/boundary data, which is thin, sparse geometry
    rather than a dense volumetric signal.

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
    Save the resampled + cropped/padded CT volume as a plain scalar-HU
    NIfTI (real Hounsfield values). `volume_hu` must be on the same voxel
    grid as the label map (i.e. the `volume_hu_resampled` returned by
    load_and_preprocess_volume, already through crop_or_pad) so this file
    lines up voxel-for-voxel with save_boundary_label_map's output in a
    viewer. This is the file to load for 3D Slicer's Volume Rendering
    module -- its CT presets are calibrated to real HU ranges.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{scan_name}_ct_resampled.nii.gz"
    if affine is None:
        affine = np.diag([TARGET_SPACING_MM, TARGET_SPACING_MM, TARGET_SPACING_MM, 1.0])
    nib.save(nib.Nifti1Image(volume_hu.astype(np.float32), affine=affine), str(out_path))
    logger.info(f"Saved resampled scalar HU CT volume -> {out_path}")
    return out_path


def build_ground_truth_label_volume(
    gt_mask_raw: np.ndarray,
    resampled_shape_pre_crop: Tuple[int, int, int],
    target_shape: Tuple[int, int, int] = TARGET_SHAPE,
    finding_map: Optional[Dict[int, str]] = None,
) -> Tuple[np.ndarray, bool]:
    """
    Collapse a raw [F, H, W, D] ground-truth mask (nifti_io.load_mask's
    output format — one channel per annotated finding) into a single-channel
    [H, W, D] label map on the same grid as every other export.

    Mirrors ScanLoader.load()'s mask handling exactly: nearest-neighbour
    resample to the volume's *pre-crop* resampled shape
    (nifti_io.resample_mask), then center-crop/pad to target_shape
    (nifti_io.crop_or_pad_mask) — the same two-step grid alignment the CT
    volume itself goes through in load_and_preprocess_volume(). Skipping
    either step would leave the mask on a different grid than the CT/label
    exports and misalign it in a viewer.

    `finding_map` ({f_idx: category}, e.g. MetadataRegistry.get_finding_map())
    tells us which real abnormality category (e.g. "2c"/"2d") each channel
    is. When it's available, channels are collapsed per-category via
    patch_sampling_abnormal._build_category_masks() (the same helper
    training uses) and labelled with the *same* CLASS_ORDER index the
    predicted label map uses (build_abnormality_volume / CLASS_ORDER) --
    so ground truth and prediction share one label-value convention and
    line up (same colours/names) in a viewer.

    Without a finding_map (a standalone mask was supplied for a scan not
    present in the metadata), there's no way to know what category each
    channel represents, so this falls back to numbering channels in raw
    storage order (label = channel index + 1, lower index wins on overlap)
    with no category semantics -- purely "which finding is here", not "GGO
    vs nodule".

    Returns (label_volume_gt, class_order_aligned): the second value tells
    the caller whether label values follow CLASS_ORDER (True) or are just
    raw channel order (False), so callers (e.g. the webapp's Slicer segment
    renaming) only label things "GGO"/"Lung Nodule" when that's actually
    known to be correct.
    """
    mask_rs = resample_mask(gt_mask_raw, target_shape=resampled_shape_pre_crop)  # (F,H,W,D)
    mask_cp = crop_or_pad_mask(mask_rs, target_shape, pad_value=0)

    label_volume_gt = np.zeros(target_shape, dtype=np.uint8)

    if finding_map:
        category_masks = _build_category_masks(mask_cp, finding_map)
        class_to_idx = {cls: i for i, cls in enumerate(CLASS_ORDER)}
        for category in CLASS_ORDER:
            if category == "normal" or category not in category_masks:
                continue
            unset = label_volume_gt == 0
            label_volume_gt[unset & category_masks[category].astype(bool)] = class_to_idx[category]
        return label_volume_gt, True

    for f in range(mask_cp.shape[0]):
        unset = label_volume_gt == 0
        label_volume_gt[unset & (mask_cp[f] > 0)] = f + 1
    return label_volume_gt, False


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
    into Slicer's Segmentations module right next to the predicted output
    for a side-by-side check. Voxel value = finding-channel index + 1,
    0 = no annotated finding.
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
    gt_mask_path: Optional[Path] = None,
    gt_finding_map: Optional[Dict[int, str]] = None,
) -> Dict:
    """
    Full pipeline: load -> resample -> crop/pad + lung-mask -> dense-patch ->
    sparse-code -> classify -> reconstruct per-voxel map -> export, for a
    single volume.

    Exports exactly:
      - <scan>_ct_resampled.nii.gz       always
      - <scan>_boundary_labelmap.nii.gz  always
      - <scan>_ground_truth_mask.nii.gz  only if gt_mask_path is given

    `gt_mask_path`, if given, points to a raw [F, H, W, D] ground-truth mask
    NIfTI (nifti_io.load_mask's format) on the *original* (pre-resample) CT
    grid; it's resampled/cropped onto the same grid as the other exports.

    `gt_finding_map` ({f_idx: category}, e.g.
    MetadataRegistry.get_finding_map(scan_name)), if given, lets the
    ground-truth mask be labelled with the same CLASS_ORDER-based values the
    prediction uses (see build_ground_truth_label_volume) instead of raw
    per-scan channel order.

    Returns a dict with the intermediate arrays, per-class patch/voxel
    abnormality counts ("which patches are abnormal, and which category"),
    and the output file paths, for programmatic use / webapp display.
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

    patch_class_counts = {
        CLASS_ORDER[i]: int((pred_labels == i).sum()) for i in range(len(CLASS_ORDER))
    }
    logger.info(f"  Patch-level class counts: {patch_class_counts}")

    logger.info("[8/8] Exporting outputs...")
    boundary_labelmap_path = save_boundary_label_map(
        maps["label_volume"], scan_name,
        output_dir=output_dir, affine=export_affine, boundaries=boundaries,
    )
    ct_resampled_path = save_resampled_ct_volume(
        volume_hu_resampled, scan_name, output_dir=output_dir, affine=export_affine,
    )

    gt_mask_nifti_path = None
    gt_class_order_aligned = False
    if gt_mask_path is not None:
        gt_mask_path = Path(gt_mask_path)
        if gt_mask_path.exists():
            logger.info(f"  Loading ground-truth mask for export: {gt_mask_path}")
            gt_mask_raw = np.asarray(nib.load(str(gt_mask_path)).dataobj, dtype=np.uint8)
            label_volume_gt, gt_class_order_aligned = build_ground_truth_label_volume(
                gt_mask_raw, resampled_shape_pre_crop=scan["resampled_shape_pre_crop"],
                finding_map=gt_finding_map,
            )
            gt_mask_nifti_path = save_ground_truth_mask(
                label_volume_gt, scan_name, output_dir=output_dir, affine=export_affine,
            )
        else:
            logger.warning(f"  gt_mask_path does not exist, skipping: {gt_mask_path}")

    elapsed = time.time() - t0
    logger.info(f"Inference complete for {scan_name} in {elapsed:.1f}s -> "
                f"{ct_resampled_path}, {boundary_labelmap_path}"
                + (f", {gt_mask_nifti_path}" if gt_mask_nifti_path else ""))

    return {
        "scan_name":               scan_name,
        "volume":                  volume,
        "coords":                  coords,
        "pred_labels":             pred_labels,
        "decision_scores":         decision_scores,
        "label_volume":            maps["label_volume"],
        "heat_volume":             maps["heat_volume"],
        "patch_class_counts":      patch_class_counts,
        "abnormal_voxel_counts":   maps["abnormal_voxel_counts"],
        "ct_resampled_nifti":      ct_resampled_path,
        "boundary_labelmap_nifti": boundary_labelmap_path,
        "ground_truth_mask_nifti": gt_mask_nifti_path,
        "ground_truth_class_order_aligned": gt_class_order_aligned,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end LC-KSVD inference + abnormality "
                    "localisation on a single lung CT volume."
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--volume", type=Path, default=None,
                        help="Path to a NIfTI volume (.nii/.nii.gz).")
    group.add_argument("--scan-id", type=str, default=None,
                        help="Scan ID resolved via VOLUMES_DIR.")
    parser.add_argument(
        "--stride", type=int, default=INFERENCE_STRIDE,
        help=f"Patch grid stride in voxels (default: {INFERENCE_STRIDE}; use "
             f"{PATCH_SIZE} for a non-overlapping grid, or a smaller value "
             "for finer localisation).",
    )
    parser.add_argument("--n-nonzero-coefs", type=int, default=N_NONZERO_COEFS)
    parser.add_argument("--dict-model", type=Path, default=DICT_MODEL_PATH)
    parser.add_argument("--svm-model", type=Path, default=SVM_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--gt-mask", type=Path, default=None,
        help="Path to a raw [F,H,W,D] ground-truth mask NIfTI (nifti_io.load_mask "
             "format, original pre-resample grid) to resample and export "
             "alongside the predicted outputs, for side-by-side comparison in "
             "a viewer like 3D Slicer.",
    )
    args = parser.parse_args()

    if args.volume is not None and args.scan_id is not None:
        parser.error("Pass only one of --volume / --scan-id.")
    if args.volume is None and args.scan_id is None:
        args.volume = DEFAULT_VOLUME_PATH

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
        gt_mask_path=args.gt_mask,
    )


if __name__ == "__main__":
    main()
