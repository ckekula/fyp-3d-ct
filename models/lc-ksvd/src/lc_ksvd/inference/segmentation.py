"""
segmentation.py
Per-abnormality-category segmentation from class-restricted dictionary
reconstruction, and evaluation against ground-truth masks.

Classification (inference.py) tells you *that* a patch is abnormal and
*which* category it belongs to. This module answers the follow-up question:
given a patch classified as e.g. "2d", reconstruct it using *only* the "2d"
block of the jointly-trained dictionary D, and use that per-category
reconstruction to build a segmentation mask per abnormality class — then
score it against ground truth (Dice / IoU / precision / recall).

Why you can't just do ``D_2d @ Gamma.T``
-----------------------------------------
D has shape (n_features, n_components); its columns are one contiguous
per-class block each (LCKSVD.class_boundaries_ /
IncrementalFrozenDictionary.class_boundaries_), e.g.
``{0: (0, 2880), 1: (2880, 5760), 2: (5760, 8640)}`` for normal/2c/2d.
Gamma — from ``encode_patches_omp(X, D, ...)`` — has shape
(n_components, n_patches): **row i of Gamma is the coefficient on D[:, i]**,
column j is patch j's full sparse code over the *whole* dictionary (OMP
picks atoms from any class's block when it minimises reconstruction error).

``D_2d`` alone is (n_features, n_atoms_2d). ``Gamma.T`` is
(n_patches, n_components) — its columns index the *full* n_components atom
space, not the n_atoms_2d atoms of the 2d block, and the shapes don't even
conform for ``D_2d @ Gamma.T``. The only correct restriction is to slice the
same atom-index range out of Gamma's *rows* (not transpose it) and match it
to D_2d's columns:

    start, end = class_boundaries[class_idx]
    D_class     = D[:, start:end]          # (n_features, n_atoms_class)
    Gamma_class = Gamma[start:end, :]       # (n_atoms_class, n_patches)
    X_hat_class = D_class @ Gamma_class     # (n_features, n_patches)

This is ``reconstruct_from_class_dictionary`` below.

Predicted mask vs. ``label_volume``
------------------------------------
``inference.build_abnormality_volume``'s ``label_volume`` resolves overlaps
between *all* classes (including "normal") via a single global argmax vote
per voxel — it answers "what's the single best label for this voxel".
Ground truth here (``build_ground_truth_class_masks``) is built the same way
training built it (``patch_sampling_abnormal._build_category_masks``): one
independent boolean mask per category, OR'd from whichever finding channels
share that category, with no cross-category exclusivity enforced. To compare
like with like, the predicted segmentation mask used for scoring here is
also independent per category — "did >=1 patch classified as this category
cover this voxel" (``coverage``, from ``build_class_reconstruction_volumes``)
— not the cross-class argmax in ``label_volume``.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from lc_ksvd.config import CLASS_ORDER, NORMAL_CLASS_IDX, PATCH_SIZE, RESULTS_DIR
from lc_ksvd.data_loader.metadata_registry import MetadataRegistry
from lc_ksvd.data_loader.scan_loader import ScanLoader
from lc_ksvd.inference.classify import encode_patches_omp, load_dictionary_and_boundaries
from lc_ksvd.inference.inference import (
    DICT_MODEL_PATH,
    INFERENCE_STRIDE,
    N_NONZERO_COEFS,
    SVM_MODEL_PATH,
    build_abnormality_volume,
    classify_patches,
    extract_dense_patches,
)
from lc_ksvd.patch_extractor.patch_sampling_abnormal import _build_category_masks

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ─── Class-restricted reconstruction ──────────────────────────────────────────

def reconstruct_from_class_dictionary(
    D: np.ndarray,
    Gamma: np.ndarray,
    class_boundaries: Dict[int, Tuple[int, int]],
    class_idx: int,
    patch_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Reconstruct signals using ONLY the dictionary atoms belonging to one
    class's block of a jointly-trained D. See module docstring for why this
    is a row-slice of Gamma matched to a column-slice of D, not ``Gamma.T``.

    Parameters
    ----------
    D : np.ndarray, shape (n_features, n_components)
    Gamma : np.ndarray, shape (n_components, n_patches)
        Sparse codes from encoding against this same D (same atom order).
    class_boundaries : dict[int, tuple[int, int]]
        e.g. ``model.class_boundaries_``.
    class_idx : int
        Which class's block to restrict to (index into CLASS_ORDER).
    patch_mask : np.ndarray or None, shape (n_patches,) bool
        If given, only reconstruct these columns of Gamma (e.g. the patches
        the SVM classified as `class_idx`). If None, reconstructs every
        patch using only this class's atoms (mostly for inspection/QA).

    Returns
    -------
    X_hat_class : np.ndarray, shape (n_features, n_selected_patches)
    """
    if class_idx not in class_boundaries:
        raise KeyError(f"class_idx={class_idx} not in class_boundaries={list(class_boundaries)}")
    start, end = class_boundaries[class_idx]
    D_class = D[:, start:end]
    Gamma_class = Gamma[start:end, :] if patch_mask is None else Gamma[start:end, patch_mask]
    return D_class @ Gamma_class


def build_class_reconstruction_volumes(
    volume_shape: Tuple[int, int, int],
    coords: np.ndarray,
    pred_labels: np.ndarray,
    Gamma: np.ndarray,
    D: np.ndarray,
    class_boundaries: Dict[int, Tuple[int, int]],
    class_order: List[str] = CLASS_ORDER,
    normal_class_idx: int = NORMAL_CLASS_IDX,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    For every abnormal class, take only the patches the SVM classified as
    that class, reconstruct each one using only that class's dictionary
    block (``reconstruct_from_class_dictionary``), and scatter the
    reconstructed cubes back into a full-volume array — mean-pooled where
    overlapping patches cover the same voxel (patches were extracted on an
    overlapping grid; averaging their independent reconstructions is the
    standard patch-based-reconstruction convention, as in K-SVD image
    denoising).

    Returns
    -------
    dict[class_idx] -> {
        "reconstruction": (H,W,D) float32, mean-pooled reconstructed
            intensity from this class's dictionary only (0 where uncovered),
        "coverage": (H,W,D) bool, True where >=1 patch predicted this class
            covers that voxel — the predicted segmentation mask for this
            class,
        "n_patches": int, number of patches predicted this class,
    }
    Only abnormal classes (`class_idx != normal_class_idx`) are included.
    """
    p = PATCH_SIZE
    out: Dict[int, Dict[str, np.ndarray]] = {}

    for class_idx, cls_name in enumerate(class_order):
        if class_idx == normal_class_idx:
            continue

        patch_mask = pred_labels == class_idx
        n_patches = int(patch_mask.sum())

        recon_sum = np.zeros(volume_shape, dtype=np.float64)
        count = np.zeros(volume_shape, dtype=np.int32)

        if n_patches > 0:
            X_hat = reconstruct_from_class_dictionary(D, Gamma, class_boundaries, class_idx, patch_mask)
            class_coords = coords[patch_mask]
            for i, (x0, y0, z0) in enumerate(class_coords):
                cube = X_hat[:, i].reshape(p, p, p)
                recon_sum[x0:x0 + p, y0:y0 + p, z0:z0 + p] += cube
                count[x0:x0 + p, y0:y0 + p, z0:z0 + p] += 1

        covered = count > 0
        mean_recon = np.zeros(volume_shape, dtype=np.float32)
        mean_recon[covered] = (recon_sum[covered] / count[covered]).astype(np.float32)

        logger.info(
            f"  Class '{cls_name}': {n_patches} patches classified, "
            f"{int(covered.sum())} voxels covered by their class-{cls_name} reconstruction"
        )
        out[class_idx] = {
            "reconstruction": mean_recon,
            "coverage": covered,
            "n_patches": n_patches,
        }

    return out


# ─── Ground truth ──────────────────────────────────────────────────────────────

def build_ground_truth_class_masks(
    mask_4d: np.ndarray,
    finding_map: Dict[int, str],
    class_order: List[str] = CLASS_ORDER,
) -> Dict[int, np.ndarray]:
    """
    Collapse a raw [F,H,W,D] mask + finding_map into one boolean mask per
    abnormal CLASS_ORDER index, using the exact same category-collapsing
    logic training used to build per-class patch labels
    (patch_extractor.patch_sampling_abnormal._build_category_masks) — so the
    ground truth here matches what the model was trained to predict.
    """
    category_masks = _build_category_masks(mask_4d, finding_map)  # {category_str: (H,W,D) uint8}
    class_to_idx = {cls: i for i, cls in enumerate(class_order)}

    out: Dict[int, np.ndarray] = {}
    for category, m in category_masks.items():
        class_idx = class_to_idx.get(category)
        if class_idx is None:
            continue
        out[class_idx] = m.astype(bool)
    return out


# ─── Metrics ────────────────────────────────────────────────────────────────────

def segmentation_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    """
    Voxel-level binary segmentation metrics for one class on one scan.

    Edge cases (no positives in prediction and/or ground truth) are defined
    explicitly rather than left to divide-by-zero:
      - gt empty & pred empty  -> dice/iou/precision/recall = 1.0 (nothing to find, found nothing)
      - gt empty & pred non-empty -> dice/iou/precision = 0.0, recall = NaN (no positives to recall)
      - gt non-empty & pred empty -> dice/iou/recall = 0.0, precision = NaN (no predictions to be precise about)
    """
    pred = np.asarray(pred_mask, dtype=bool)
    gt = np.asarray(gt_mask, dtype=bool)

    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())

    pred_sum = tp + fp
    gt_sum = tp + fn

    if gt_sum == 0 and pred_sum == 0:
        dice = iou = precision = recall = 1.0
    elif gt_sum == 0:
        dice = iou = precision = 0.0
        recall = float("nan")
    elif pred_sum == 0:
        dice = iou = recall = 0.0
        precision = float("nan")
    else:
        dice = 2 * tp / (pred_sum + gt_sum)
        iou = tp / (tp + fp + fn)
        precision = tp / pred_sum
        recall = tp / gt_sum

    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "gt_voxels": gt_sum, "pred_voxels": pred_sum,
    }


# ─── End-to-end per-scan evaluation ────────────────────────────────────────────

def evaluate_scan_segmentation(
    scan_id: str,
    split: str = "test",
    dict_model_path: Path = DICT_MODEL_PATH,
    svm_model_path: Path = SVM_MODEL_PATH,
    stride: int = INFERENCE_STRIDE,
    n_nonzero_coefs: int = N_NONZERO_COEFS,
    metadata: Optional[MetadataRegistry] = None,
) -> Dict:
    """
    Full pipeline for one scan that has a ground-truth mask: load (via
    ScanLoader, the exact same preprocessing + TARGET_SHAPE canvas training
    used, so the predicted and ground-truth arrays are voxel-aligned without
    any extra resampling) -> dense patch grid -> sparse-code against D ->
    classify -> class-restricted reconstruction -> per-class segmentation
    mask -> score against ground truth.

    Raises ValueError if the scan has no mask/finding_map in this split
    (nothing to evaluate against).
    """
    if metadata is None:
        metadata = MetadataRegistry(split=split)
    loader = ScanLoader(metadata)

    scan = loader.load(scan_id)
    volume = scan["volume"]
    mask_4d = scan["mask"]
    finding_map = scan["finding_map"]
    if mask_4d is None or not finding_map:
        raise ValueError(
            f"{scan_id!r} has no ground-truth mask/finding_map in split={split!r}; "
            "cannot evaluate segmentation."
        )

    X, coords = extract_dense_patches(volume, stride=stride)

    D, class_boundaries, class_order = load_dictionary_and_boundaries(dict_model_path)
    class_order = class_order or CLASS_ORDER

    Gamma = encode_patches_omp(X, D, n_nonzero_coefs=n_nonzero_coefs)
    pred_labels, decision_scores = classify_patches(Gamma, svm_path=svm_model_path)

    maps = build_abnormality_volume(volume, coords, pred_labels, decision_scores)
    recon = build_class_reconstruction_volumes(
        volume.shape, coords, pred_labels, Gamma, D, class_boundaries,
        class_order=class_order, normal_class_idx=NORMAL_CLASS_IDX,
    )
    gt_masks = build_ground_truth_class_masks(mask_4d, finding_map, class_order=class_order)

    per_class_metrics: Dict[str, Dict[str, float]] = {}
    pred_any = np.zeros(volume.shape, dtype=bool)
    gt_any = np.zeros(volume.shape, dtype=bool)

    for class_idx, cls_name in enumerate(class_order):
        if class_idx == NORMAL_CLASS_IDX:
            continue
        pred_mask = recon[class_idx]["coverage"]
        gt_mask = gt_masks.get(class_idx, np.zeros(volume.shape, dtype=bool))
        pred_any |= pred_mask
        gt_any |= gt_mask

        m = segmentation_metrics(pred_mask, gt_mask)
        m["n_patches_predicted"] = recon[class_idx]["n_patches"]
        per_class_metrics[cls_name] = m

    overall = segmentation_metrics(pred_any, gt_any)

    logger.info(
        f"[{scan_id}] segmentation — "
        + ", ".join(f"{c}: dice={m['dice']:.3f}" for c, m in per_class_metrics.items())
        + f", overall: dice={overall['dice']:.3f}"
    )

    return {
        "scan_id": scan_id,
        "split": split,
        "per_class": per_class_metrics,
        "overall": overall,
        "volume": volume,
        "coords": coords,
        "pred_labels": pred_labels,
        "label_volume": maps["label_volume"],
        "reconstruction": recon,
        "gt_masks": gt_masks,
    }


def evaluate_split_segmentation(
    split: str = "test",
    scan_ids: Optional[List[str]] = None,
    dict_model_path: Path = DICT_MODEL_PATH,
    svm_model_path: Path = SVM_MODEL_PATH,
    stride: int = INFERENCE_STRIDE,
    n_nonzero_coefs: int = N_NONZERO_COEFS,
    results_dir: Path = RESULTS_DIR,
) -> Path:
    """
    Run evaluate_scan_segmentation over every scan in `split` that has a
    ground-truth mask (or a caller-supplied subset), and save per-scan +
    summary (mean over scans, per class) CSVs.
    """
    metadata = MetadataRegistry(split=split)
    if scan_ids is None:
        scan_ids = [v for v in metadata.get_all_volume_names() if metadata.get_finding_map(v)]
        logger.info(f"Found {len(scan_ids)} scans with ground-truth findings in split={split!r}.")

    rows = []
    for scan_id in tqdm(scan_ids, desc="segmentation eval"):
        try:
            res = evaluate_scan_segmentation(
                scan_id, split=split,
                dict_model_path=dict_model_path, svm_model_path=svm_model_path,
                stride=stride, n_nonzero_coefs=n_nonzero_coefs, metadata=metadata,
            )
        except Exception as exc:
            logger.warning(f"Skipping {scan_id}: {exc}")
            continue

        for cls_name, m in res["per_class"].items():
            rows.append({"scan_id": scan_id, "class": cls_name, **m})
        rows.append({"scan_id": scan_id, "class": "overall", **res["overall"]})

    df = pd.DataFrame(rows)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = results_dir / f"segmentation_{split}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "segmentation_per_scan.csv", index=False)

    metric_cols = ["dice", "iou", "precision", "recall", "specificity"]
    summary = df.groupby("class")[metric_cols].mean(numeric_only=True)
    summary["n_scans"] = df.groupby("class").size()
    summary.to_csv(out_dir / "segmentation_summary.csv")

    logger.info(f"Saved segmentation evaluation ({len(df)} rows, {df['scan_id'].nunique()} scans) -> {out_dir}")
    logger.info(f"Summary:\n{summary}")
    return out_dir


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Evaluate per-category segmentation (class-restricted "
                    "dictionary reconstruction) against ground truth."
    )
    parser.add_argument("--scan-id", type=str, default=None,
                        help="Evaluate a single scan. Omit to evaluate the whole split.")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--stride", type=int, default=INFERENCE_STRIDE)
    parser.add_argument("--n-nonzero-coefs", type=int, default=N_NONZERO_COEFS)
    parser.add_argument("--dict-model", type=Path, default=DICT_MODEL_PATH)
    parser.add_argument("--svm-model", type=Path, default=SVM_MODEL_PATH)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.scan_id:
        res = evaluate_scan_segmentation(
            args.scan_id, split=args.split,
            dict_model_path=args.dict_model, svm_model_path=args.svm_model,
            stride=args.stride, n_nonzero_coefs=args.n_nonzero_coefs,
        )
        for cls_name, m in res["per_class"].items():
            print(f"{cls_name}: dice={m['dice']:.4f} iou={m['iou']:.4f} "
                  f"precision={m['precision']:.4f} recall={m['recall']:.4f} "
                  f"n_patches={m['n_patches_predicted']}")
        print(f"overall: dice={res['overall']['dice']:.4f} iou={res['overall']['iou']:.4f}")
    else:
        evaluate_split_segmentation(
            split=args.split,
            dict_model_path=args.dict_model, svm_model_path=args.svm_model,
            stride=args.stride, n_nonzero_coefs=args.n_nonzero_coefs,
        )


if __name__ == "__main__":
    main()
