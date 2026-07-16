"""
patch_extractor.py
Assembles the (n_features, n_patches) matrix X, the (n_patches,) integer label
vector H, and the (n_patches,) scan-ID string array scan_ids required for
LC-KSVD2 training and scan-level evaluation, by combining the Phase 1 (normal)
and Phase 2 (abnormal) sampling passes. Output is split into one .npz file
per class (per CLASS_ORDER) rather than a single unified file; the public
load_unified_patch_matrix entry point concatenates them back together in
CLASS_ORDER, so downstream callers see the same (X, H, scan_ids) contract
as before.
"""

import logging
import os
from pathlib import Path
from typing import Callable, List, Tuple
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

# Signature shared by collect_normal_patches / collect_abnormal_patches.
PatchCollector = Callable[
    [List[str], ScanLoader, _PatchStreamWriter],
    Tuple[List[int], List[str], List[Tuple[int, int, int]]],
]

def _class_out_path(split: str, class_name: str) -> Path:
    return PATCHES_DIR / f"unified_{split}_{class_name}.npz"


def build_class_patch_matrix(
    ids: List[str],
    loader: ScanLoader,
    collector: PatchCollector,
    scratch_tag: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a single collector (normal or abnormal) over `ids` and return the
    resulting (X, H, scan_ids, coords) for that class alone.
    """
    PATCHES_DIR.mkdir(parents=True, exist_ok=True)
    scratch_path = PATCHES_DIR / f"_patch_scratch_{os.getpid()}_{scratch_tag}.bin"
    writer = _PatchStreamWriter(scratch_path)

    try:
        labels, scan_ids, coords = collector(ids, loader, writer)
        n_patches = len(labels)
        writer.close()

        X = np.empty((N_FEATURES, n_patches), dtype=np.float64)
        if n_patches:
            patch_mm = np.memmap(scratch_path, dtype=np.float32, mode="r",
                                  shape=(n_patches, N_FEATURES))
            X[:] = patch_mm.T
            del patch_mm
    finally:
        writer.close()
        scratch_path.unlink(missing_ok=True)

    H = np.array(labels, dtype=np.int64)
    scan_ids_arr = np.array(scan_ids, dtype=object)
    coords_arr = np.array(coords, dtype=np.int64).reshape(n_patches, 3)
    return X, H, scan_ids_arr, coords_arr


def _filter_existing(ids: List[str], class_name: str = "") -> List[str]:
    valid, missing = [], []
    for vid in ids:
        try:
            resolve_volume_path(vid)
            valid.append(vid)
        except Exception:
            missing.append(vid)
    if missing:
        tag = f"[{class_name}] " if class_name else ""
        logger.info(
            f"_filter_existing: {tag}{len(missing)}/{len(ids)} missing volumes: {missing}"
        )
    return valid


def extract_unified(split: str = "train") -> None:
    """
    Run patch extraction for every class in CLASS_ORDER and save one
    compressed .npz per class:
        patches/unified_{split}_{class_name}.npz
    Each file stores X, H, scan_ids for that class only.
    """
    PATCHES_DIR.mkdir(parents=True, exist_ok=True)

    metadata = MetadataRegistry(split=split)
    labels   = LabelRegistry(metadata, split=split)
    loader   = ScanLoader(metadata)

    abnormality_keys = [k for k in CLASS_ORDER if k != "normal"]

    class_ids = {}

    raw_normals = labels.get_normal_volume_names()
    logger.info(f"Raw normal IDs from metadata: {len(raw_normals)}")
    if not raw_normals:
        logger.warning(
            "No normal volumes found in metadata — "
            "dataset may contain only abnormal scans."
        )
    class_ids["normal"] = _filter_existing(raw_normals, class_name="normal")

    for ab in abnormality_keys:
        raw_ids = labels.get_positive_volume_names(ab)
        logger.info(f"  category '{ab}': {len(raw_ids)} volumes")
        class_ids[ab] = _filter_existing(raw_ids, class_name=ab)

    total_abnormal = sum(len(class_ids[ab]) for ab in abnormality_keys)
    if total_abnormal == 0:
        raise RuntimeError(
            "No abnormal patches collected. "
            "Check MASKS_DIR, METADATA_JSON, and foreground mask contents."
        )

    logger.info(
        f"Split={split!r} | normal={len(class_ids['normal'])} | "
        f"abnormal={total_abnormal}"
    )

    for class_name in CLASS_ORDER:
        out_path = _class_out_path(split, class_name)
        if out_path.exists():
            logger.info(f"Already exists: {out_path} — skipping.")
            continue

        ids = class_ids[class_name]
        collector = collect_normal_patches if class_name == "normal" else collect_abnormal_patches

        X, H, scan_ids, coords = build_class_patch_matrix(ids, loader, collector, scratch_tag=class_name)
        np.savez_compressed(out_path, X=X, H=H, scan_ids=scan_ids, coords=coords)
        logger.info(f"Saved → {out_path}  (X: {X.shape}, H: {H.shape})")


def load_unified_patch_matrix(
    split: str = "train",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load each per-class .npz (in CLASS_ORDER) and concatenate them into
        X: (n_features, n_patches), H: (n_patches,), scan_ids: (n_patches,),
        coords: (n_patches, 3)
    """
    X_parts, H_parts, scan_id_parts, coord_parts = [], [], [], []

    for class_name in CLASS_ORDER:
        path = _class_out_path(split, class_name)
        if not path.exists():
            raise FileNotFoundError(
                f"Patch matrix not found: {path}. Run extract_unified() first."
            )
        data = np.load(path, allow_pickle=True)
        X_parts.append(data["X"])
        H_parts.append(data["H"])
        scan_id_parts.append(data["scan_ids"])
        coord_parts.append(data["coords"])

    X = np.concatenate(X_parts, axis=1)
    H = np.concatenate(H_parts, axis=0)
    scan_ids = np.concatenate(scan_id_parts, axis=0)
    coords = np.concatenate(coord_parts, axis=0)

    logger.info(f"Loaded matrix: X={X.shape}, H={H.shape}, scan_ids={scan_ids.shape}, coords={coords.shape}")
    return X, H, scan_ids, coords

if __name__ == "__main__":
    extract_unified(split="train")