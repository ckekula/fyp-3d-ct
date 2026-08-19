"""
inferencecopy.py
Single-volume LC-KSVD inference pipeline, extended with per-class
dictionary-restricted reconstruction and per-class segmentation mask export.

This builds on top of inference.py's pipeline (load -> resample -> crop/pad
+ lung-mask -> dense-patch -> sparse-code -> classify -> reconstruct) and
adds one more step, driven by the fact that D's columns are organised as
contiguous per-class blocks (see classify.load_dictionary_and_boundaries):

  1-6. Same as inference.py: preprocess volume, extract dense patches,
       sparse-code each patch against the full dictionary D (Gamma), and
       classify each patch's code with the trained SVM -> pred_labels.
  7.   For each patch the SVM classified as an abnormal class (e.g. "2d"):
       zero out every coefficient in its sparse code EXCEPT the rows
       belonging to that class's block of D (Gamma[start:end, patch]), and
       reconstruct the patch as D[:, start:end] @ Gamma[start:end, patch]
       -- "use only the 2d atoms to rebuild this 2d-classified patch".
  8.   Decide, per VOXEL (not per whole patch), whether that voxel is
       actually part of the abnormality (build_class_segmentation_masks,
       below) -- this is the part that keeps the whole patch from getting
       coloured in. See that function's docstring for why and how.
  9.   Export one NIfTI mask file per abnormal class, so "2c" and "2d"
       (etc.) each get their own standalone segmentation mask.

Usage:
  python -m lc_ksvd.inference.inferencecopy --volume /path/to/scan.nii.gz
  python -m lc_ksvd.inference.inferencecopy --scan-id <id-in-VOLUMES_DIR>
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

from lc_ksvd.config import CLASS_ORDER, NORMAL_CLASS_IDX, PATCH_SIZE, TARGET_SPACING_MM
from lc_ksvd.inference.classify import encode_patches_omp, load_dictionary_and_boundaries
from lc_ksvd.inference.inference import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VOLUME_PATH,
    DICT_MODEL_PATH,
    INFERENCE_STRIDE,
    N_NONZERO_COEFS,
    SVM_MODEL_PATH,
    build_abnormality_volume,
    build_ground_truth_label_volume,
    classify_patches,
    compute_class_boundaries,
    extract_dense_patches,
    load_and_preprocess_volume,
    output_affine_for,
    save_boundary_label_map,
    save_ground_truth_mask,
    save_resampled_ct_volume,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ─── Per-class dictionary-restricted reconstruction + voxel-level masking ────

def build_class_segmentation_masks(
    X: np.ndarray,
    coords: np.ndarray,
    pred_labels: np.ndarray,
    Gamma: np.ndarray,
    D: np.ndarray,
    class_boundaries: Dict[int, Tuple[int, int]],
    volume_shape: Tuple[int, int, int],
    class_order: List[str] = CLASS_ORDER,
    normal_class_idx: int = NORMAL_CLASS_IDX,
    background_value: float = -1.0,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    For every patch predicted as an abnormal class, reconstruct it from
    ONLY that class's block of D, and decide per voxel -- not per whole
    patch -- whether it belongs in that class's segmentation mask.

    Why per-patch reconstruction alone isn't a mask
    -------------------------------------------------
    D[:, start:end] @ Gamma[start:end, patch] rebuilds the *entire*
    PATCH_SIZE^3 cube, not just the diseased sub-region inside it: K-SVD
    atoms are dense (nonzero almost everywhere in the patch), and every
    training patch for a class like "2d" is mostly ordinary surrounding
    lung tissue with a lesion in one corner -- so the reconstruction fills
    the whole cube with plausible-looking tissue, lesion or not. Painting
    every voxel of a classified patch (or even every voxel above the
    background floor) colours the whole block, not the lesion.

    Also note encode_patches_omp() L2-normalises each patch column before
    OMP (metrics.normalise_columns), so Gamma is fit to explain the
    *unit-norm* patch, not the raw one -- D[:, start:end] @ Gamma[start:end]
    lives on that normalised scale. It has to be rescaled by the patch's
    original L2 norm (``patch_norm`` below) before it's comparable, voxel
    for voxel, to the real preprocessed CT intensity.

    The actual per-voxel decision
    -------------------------------
    For each abnormal-predicted patch, reconstruct it twice, both rescaled
    back to the real intensity scale:
      class_cube  = (D[:, c_start:c_end]  @ Gamma[c_start:c_end,  patch]) * patch_norm
      normal_cube = (D[:, n_start:n_end] @ Gamma[n_start:n_end, patch]) * patch_norm
    and compare each to the patch's actual (preprocessed) voxel value,
    ``orig_cube`` (straight from X, pre-normalisation):
      class_error  = |orig_cube - class_cube|
      normal_error = |orig_cube - normal_cube|
    A voxel is kept in this class's mask only if it is real lung tissue
    (not the -1.0 background fill, see nifti_io.preprocess) AND the
    class-restricted reconstruction fits the real signal there better than
    the normal-restricted reconstruction does (class_error < normal_error)
    -- i.e. "this specific voxel looks more like disease than like normal
    tissue", the sparse-representation-classification decision rule
    applied voxel-by-voxel instead of patch-by-patch. That is what carves
    the lesion's actual footprint out of the cube instead of colouring the
    whole patch.

    Returns
    -------
    dict[class_idx] -> {
        "mask": (H,W,D) bool, the voxel-level segmentation mask,
        "reconstruction": (H,W,D) float32, mean-pooled class-restricted
            reconstruction (real intensity scale) where covered, 0 elsewhere
            -- kept for QA/visualisation, not used to build "mask",
        "n_patches": int, number of patches predicted this class,
    }
    Only abnormal classes (`class_idx != normal_class_idx`) are included.
    """
    if normal_class_idx not in class_boundaries:
        raise KeyError(f"normal_class_idx={normal_class_idx} not in "
                        f"class_boundaries={list(class_boundaries)}")
    n_start, n_end = class_boundaries[normal_class_idx]

    p = PATCH_SIZE
    patch_norms = np.linalg.norm(X, axis=0)  # same per-column norm encode_patches_omp divided out
    out: Dict[int, Dict[str, np.ndarray]] = {}

    for class_idx, cls_name in enumerate(class_order):
        if class_idx == normal_class_idx:
            continue
        if class_idx not in class_boundaries:
            continue
        c_start, c_end = class_boundaries[class_idx]

        patch_idxs = np.nonzero(pred_labels == class_idx)[0]
        n_patches = int(patch_idxs.size)

        mask_vol = np.zeros(volume_shape, dtype=bool)
        recon_sum = np.zeros(volume_shape, dtype=np.float64)
        count = np.zeros(volume_shape, dtype=np.int32)

        for i in patch_idxs:
            x0, y0, z0 = coords[i]
            norm = patch_norms[i]

            orig_cube = X[:, i].reshape(p, p, p)
            class_cube = (D[:, c_start:c_end] @ Gamma[c_start:c_end, i]).reshape(p, p, p) * norm
            normal_cube = (D[:, n_start:n_end] @ Gamma[n_start:n_end, i]).reshape(p, p, p) * norm

            tissue = ~np.isclose(orig_cube, background_value, atol=1e-6)
            better_than_normal = np.abs(orig_cube - class_cube) < np.abs(orig_cube - normal_cube)
            voxel_mask = tissue & better_than_normal

            sl = np.s_[x0:x0 + p, y0:y0 + p, z0:z0 + p]
            mask_vol[sl] |= voxel_mask
            recon_sum[sl] += class_cube
            count[sl] += 1

        covered = count > 0
        mean_recon = np.zeros(volume_shape, dtype=np.float32)
        mean_recon[covered] = (recon_sum[covered] / count[covered]).astype(np.float32)

        logger.info(
            f"  Class '{cls_name}': {n_patches} patches classified, "
            f"{int(mask_vol.sum()):,} voxels kept as '{cls_name}' after per-voxel "
            f"reconstruction-residual comparison against normal tissue"
        )
        out[class_idx] = {
            "mask": mask_vol,
            "reconstruction": mean_recon,
            "n_patches": n_patches,
        }

    return out


def save_class_masks(
    class_masks: Dict[int, Dict[str, np.ndarray]],
    scan_name: str,
    output_dir: Path,
    affine: Optional[np.ndarray] = None,
    class_order: List[str] = CLASS_ORDER,
) -> Dict[str, Path]:
    """
    Save one standalone scalar uint8 NIfTI segmentation mask per abnormal
    class from `class_masks` (build_class_segmentation_masks' output).
    Mask voxel value = class index (CLASS_ORDER position, same convention
    save_boundary_label_map uses), 0 elsewhere. One file per class, e.g.
    "<scan>_2c_mask.nii.gz", "<scan>_2d_mask.nii.gz".

    Returns {class_name: output_path} for classes with a non-empty mask.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if affine is None:
        affine = np.diag([TARGET_SPACING_MM[0], TARGET_SPACING_MM[1], TARGET_SPACING_MM[2], 1.0])

    out_paths: Dict[str, Path] = {}
    for class_idx, cls_name in enumerate(class_order):
        if class_idx == NORMAL_CLASS_IDX or class_idx not in class_masks:
            continue

        voxel_mask = class_masks[class_idx]["mask"]
        if not voxel_mask.any():
            logger.info(f"  Class '{cls_name}': empty mask, skipping export.")
            continue

        mask = np.zeros(voxel_mask.shape, dtype=np.uint8)
        mask[voxel_mask] = class_idx

        out_path = output_dir / f"{scan_name}_{cls_name}_mask.nii.gz"
        img = nib.Nifti1Image(mask, affine=affine)
        img.header.set_data_dtype(np.uint8)
        nib.save(img, str(out_path))
        out_paths[cls_name] = out_path
        logger.info(f"  Saved '{cls_name}' segmentation mask ({int(voxel_mask.sum()):,} voxels) -> {out_path}")

    return out_paths


# ─── End-to-end pipeline ───────────────────────────────────────────────────────

def run_inference(
    volume_path: Optional[Path] = None,
    scan_id: Optional[str] = None,
    dict_model_path: Path = DICT_MODEL_PATH,
    svm_model_path: Path = SVM_MODEL_PATH,
    stride: int = INFERENCE_STRIDE,
    n_nonzero_coefs: int = N_NONZERO_COEFS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    gt_mask_path: Optional[Path] = None,
    gt_finding_map: Optional[Dict[int, str]] = None,
) -> Dict:
    """
    Full pipeline: load -> resample -> crop/pad + lung-mask -> dense-patch ->
    sparse-code -> classify -> reconstruct per-voxel map -> per-class
    dictionary-restricted, voxel-level segmentation -> export, for a single
    volume.

    Exports exactly:
      - <scan>_ct_resampled.nii.gz        always
      - <scan>_boundary_labelmap.nii.gz   always (all classes, outline only)
      - <scan>_<class>_mask.nii.gz        one per abnormal class with any
                                           mask voxels (e.g. _2c_mask,
                                           _2d_mask) -- lesion-shaped voxel
                                           mask from that class's
                                           dictionary-restricted
                                           reconstruction, not a solid
                                           patch block
      - <scan>_ground_truth_mask.nii.gz   only if gt_mask_path is given

    See module docstring / build_class_segmentation_masks for how the
    per-class masks are built.
    """
    t0 = time.time()
    logger.info(f"{'='*60}\nRunning inference (volume={volume_path}, scan_id={scan_id})\n{'='*60}")

    scan = load_and_preprocess_volume(volume_path=volume_path, scan_id=scan_id)
    volume, scan_name = scan["volume"], scan["name"]
    volume_hu_resampled = scan["volume_hu_resampled"]
    export_affine = output_affine_for(scan["orig_affine"])

    X, coords = extract_dense_patches(volume, stride=stride)

    logger.info(f"[5/9] Loading dictionary + sparse-coding patches (n_nonzero_coefs={n_nonzero_coefs})...")
    D, class_boundaries, class_order = load_dictionary_and_boundaries(dict_model_path)
    class_order = class_order or CLASS_ORDER
    Gamma = encode_patches_omp(X, D, n_nonzero_coefs=n_nonzero_coefs)

    pred_labels, decision_scores = classify_patches(Gamma, svm_path=svm_model_path)

    maps = build_abnormality_volume(volume, coords, pred_labels, decision_scores)
    boundaries = compute_class_boundaries(maps["label_volume"])

    patch_class_counts = {
        class_order[i]: int((pred_labels == i).sum()) for i in range(len(class_order))
    }
    logger.info(f"  Patch-level class counts: {patch_class_counts}")

    logger.info("[7/9] Reconstructing per-class voxel-level segmentation from "
                "class-restricted dictionary blocks...")
    class_masks = build_class_segmentation_masks(
        X, coords, pred_labels, Gamma, D, class_boundaries, volume.shape,
        class_order=class_order, normal_class_idx=NORMAL_CLASS_IDX,
    )
    mask_voxel_counts = {
        class_order[i]: int(class_masks[i]["mask"].sum()) for i in class_masks
    }
    logger.info(f"  Per-class segmentation mask voxel counts: {mask_voxel_counts}")

    logger.info("[8/9] Exporting combined outputs...")
    boundary_labelmap_path = save_boundary_label_map(
        maps["label_volume"], scan_name,
        output_dir=output_dir, affine=export_affine, boundaries=boundaries,
    )
    ct_resampled_path = save_resampled_ct_volume(
        volume_hu_resampled, scan_name, output_dir=output_dir, affine=export_affine,
    )

    logger.info("[9/9] Exporting per-class segmentation masks...")
    class_mask_paths = save_class_masks(
        class_masks, scan_name, output_dir=output_dir,
        affine=export_affine, class_order=class_order,
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
    logger.info(
        f"Inference complete for {scan_name} in {elapsed:.1f}s -> "
        f"{ct_resampled_path}, {boundary_labelmap_path}, class masks: {list(class_mask_paths)}"
        + (f", {gt_mask_nifti_path}" if gt_mask_nifti_path else "")
    )

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
        "class_masks":             class_masks,
        "mask_voxel_counts":       mask_voxel_counts,
        "ct_resampled_nifti":      ct_resampled_path,
        "boundary_labelmap_nifti": boundary_labelmap_path,
        "class_mask_niftis":       class_mask_paths,
        "ground_truth_mask_nifti": gt_mask_nifti_path,
        "ground_truth_class_order_aligned": gt_class_order_aligned,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end LC-KSVD inference + per-class "
                    "dictionary-restricted segmentation on a single lung CT volume."
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
