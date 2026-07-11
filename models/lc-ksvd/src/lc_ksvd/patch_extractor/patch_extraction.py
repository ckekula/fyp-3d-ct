"""
patch_extractor.py
Assembles the (n_features, n_patches) matrix X, the (n_patches,) integer label
vector H, and the (n_patches,) scan-ID string array scan_ids required for
LC-KSVD2 training and scan-level evaluation, by combining the Phase 1 (normal)
and Phase 2 (abnormal) sampling passes. Also exposes the public
extract_unified / load_unified_patch_matrix entry points.
"""

import logging
import os
from typing import List, Tuple
import numpy as np

from lc_ksvd.config import CLASS_ORDER, N_FEATURES, PATCHES_DIR

from lc_ksvd.data_loader.scan_loader import (MetadataRegistry, ScanLoader)
from lc_ksvd.data_loader.nifti_io import resolve_volume_path
from lc_ksvd.data_loader.metadata_registry import LabelRegistry

from lc_ksvd.patch_extractor.patch_io import _PatchStreamWriter
from lc_ksvd.patch_extractor.patch_sampling_normal import collect_normal_patches
from lc_ksvd.patch_extractor.patch_sampling_abnormal import collect_abnormal_patches

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def build_unified_patch_matrix(
    normal_ids: List[str],
    positive_ids: List[str],
    loader: ScanLoader,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    PATCHES_DIR.mkdir(parents=True, exist_ok=True)
    scratch_path = PATCHES_DIR / f"_patch_scratch_{os.getpid()}.bin"
    writer = _PatchStreamWriter(scratch_path)

    try:
        normal_labels, normal_scan_ids = collect_normal_patches(normal_ids, loader, writer)
        abnormal_labels, abnormal_scan_ids = collect_abnormal_patches(positive_ids, loader, writer)

        if not abnormal_labels:
            raise RuntimeError(
                "No abnormal patches collected. "
                "Check MASKS_DIR, METADATA_JSON, and foreground mask contents."
            )

        all_labels = normal_labels + abnormal_labels
        all_scan_ids = normal_scan_ids + abnormal_scan_ids
        n_patches = len(all_labels)
        writer.close()

        patch_mm = np.memmap(scratch_path, dtype=np.float32, mode="r",
                              shape=(n_patches, N_FEATURES))
        X = np.empty((N_FEATURES, n_patches), dtype=np.float64)
        X[:] = patch_mm.T
        del patch_mm
    finally:
        writer.close()
        scratch_path.unlink(missing_ok=True)

    H = np.array(all_labels, dtype=np.int64)
    scan_ids = np.array(all_scan_ids, dtype=object)

    logger.info(f"Final matrix: X={X.shape}, H={H.shape}, scan_ids={scan_ids.shape}")
    return X, H, scan_ids


def _filter_existing(ids: List[str]) -> List[str]:
    valid, missing = [], []
    for vid in ids:
        try:
            resolve_volume_path(vid)
            valid.append(vid)
        except Exception:
            missing.append(vid)
    if missing:
        logger.debug(
            f"_filter_existing: {len(missing)} missing volumes "
            f"(sample ≤5): {missing[:5]}"
        )
    return valid


def extract_unified(split: str = "train") -> None:
    """
    Run both phases of patch extraction and save a single compressed .npz:
        patches/unified_{split}.npz
    Stores X, H, and scan_ids.
    """
    PATCHES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PATCHES_DIR / f"unified_{split}.npz"

    if out_path.exists():
        logger.info(f"Already exists: {out_path} — skipping.")
        return

    metadata = MetadataRegistry(split=split)
    labels   = LabelRegistry(metadata, split=split)
    loader   = ScanLoader(metadata)

    abnormality_keys = [k for k in CLASS_ORDER if k != "normal"]

    raw_by_cat = {ab: labels.get_positive_volume_names(ab) for ab in abnormality_keys}
    for ab, lst in raw_by_cat.items():
        logger.info(f"  volumes containing category '{ab}': {len(lst)}")

    positive_ids: List[str] = list({
        vid
        for ab in abnormality_keys
        for vid in raw_by_cat.get(ab, [])
    })
    logger.info(f"Raw positive IDs (deduped): {len(positive_ids)}")
    positive_ids = _filter_existing(positive_ids)
    logger.info(f"After path resolution: abnormal={len(positive_ids)}")

    raw_normals = labels.get_normal_volume_names()
    logger.info(f"Raw normal IDs from metadata: {len(raw_normals)}")
    if not raw_normals:
        logger.warning(
            "No normal volumes found in metadata — "
            "dataset may contain only abnormal scans."
        )
    normal_ids = _filter_existing(raw_normals)

    logger.info(
        f"Split={split!r} | normal={len(normal_ids)} | abnormal={len(positive_ids)}"
    )

    X, H, scan_ids = build_unified_patch_matrix(normal_ids, positive_ids, loader)

    np.savez_compressed(out_path, X=X, H=H, scan_ids=scan_ids)
    logger.info(f"Saved → {out_path}  (X: {X.shape}, H: {H.shape})")


def load_unified_patch_matrix(
    split: str = "train",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and return X (n_features, n_patches), H (n_patches,), scan_ids (n_patches,).
    """
    path = PATCHES_DIR / f"unified_{split}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Patch matrix not found: {path}. Run extract_unified() first."
        )
    data = np.load(path, allow_pickle=True)
    return data["X"], data["H"], data["scan_ids"]


if __name__ == "__main__":
    extract_unified(split="train")