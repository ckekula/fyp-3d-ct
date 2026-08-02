"""
patch_extractor.py
Assembles the (n_features, n_patches) matrix X, the (n_patches,) integer label
vector H, and the (n_patches,) scan-ID string array scan_ids required for
LC-KSVD2 training and scan-level evaluation, by combining the Phase 1 (normal)
and Phase 2 (abnormal) sampling passes. Output is split into one .npz file
per class (per CLASS_ORDER); the public load_unified_patch_matrix entry point
concatenates them back together in CLASS_ORDER, so downstream callers see the
same (X, H, scan_ids) contract as before.
"""

import logging
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np

from lc_ksvd.config import CLASS_ORDER, N_FEATURES, PATCHES_DIR
from lc_ksvd.data_loader.metadata_registry import LabelRegistry, MetadataRegistry
from lc_ksvd.data_loader.nifti_io import resolve_volume_path
from lc_ksvd.data_loader.scan_loader import ScanLoader
from lc_ksvd.patch_extractor.patch_io import _PatchStreamWriter
from lc_ksvd.patch_extractor.patch_sampling_abnormal import collect_abnormal_patches
from lc_ksvd.patch_extractor.patch_sampling_normal import collect_normal_patches

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# Signature shared by collect_normal_patches / collect_abnormal_patches.
PatchCollector = Callable[
    [list[str], ScanLoader, _PatchStreamWriter],
    tuple[list[int], list[str], list[tuple[int, int, int]]],
]

def _class_out_path(split: str, class_name: str) -> Path:
    return PATCHES_DIR / f"unified_{split}_{class_name}.npz"


def build_patch_matrix(
    ids: list[str],
    loader: ScanLoader,
    collector: PatchCollector,
    scratch_tag: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a single collector (normal or abnormal) over `ids` and return the
    resulting (X, H, scan_ids, coords).
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

def _filter_existing(ids: list[str], class_name: str = "") -> list[str]:
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

def _split_and_save_by_class(
    split: str,
    class_name: str,
    class_idx: int,
    X: np.ndarray,
    H: np.ndarray,
    scan_ids: np.ndarray,
    coords: np.ndarray,
) -> None:
    mask = H == class_idx
    out_path = _class_out_path(split, class_name)
    np.savez_compressed(
        out_path,
        X=X[:, mask],
        H=H[mask],
        scan_ids=scan_ids[mask],
        coords=coords[mask],
    )
    logger.info(f"Saved → {out_path}  (X: {X[:, mask].shape}, H: {H[mask].shape})")


def extract_unified(split: str = "train") -> None:
    """
    Run patch extraction once per phase (normal grid-sampling, abnormal
    bbox-sampling over the union of abnormal scans) and save one compressed
    .npz per class in CLASS_ORDER:
        patches/unified_{split}_{class_name}.npz
    """
    PATCHES_DIR.mkdir(parents=True, exist_ok=True)

    metadata = MetadataRegistry(split=split)
    labels   = LabelRegistry(metadata)
    loader   = ScanLoader(metadata)

    abnormality_keys = [k for k in CLASS_ORDER if k != "normal"]
    class_to_idx = {cls: i for i, cls in enumerate(CLASS_ORDER)}

    class_ids = {}

    raw_normals = labels.get_normal_volume_names()
    logger.info(f"Raw normal IDs from metadata: {len(raw_normals)}")
    if not raw_normals:
        logger.warning(
            "No normal volumes found in metadata — "
            "dataset may contain only abnormal scans."
        )
    class_ids["normal"] = _filter_existing(raw_normals, class_name="normal")

    seen = set()
    union_abnormal_ids: list[str] = []
    for ab in abnormality_keys:
        raw_ids = labels.get_positive_volume_names(ab)
        logger.info(f"  category '{ab}': {len(raw_ids)} volumes")
        filtered = _filter_existing(raw_ids, class_name=ab)
        class_ids[ab] = filtered
        for vid in filtered:
            if vid not in seen:
                seen.add(vid)
                union_abnormal_ids.append(vid)

    if not union_abnormal_ids:
        raise RuntimeError(
            "No abnormal patches collected. "
            "Check MASKS_DIR, METADATA_JSON, and foreground mask contents."
        )

    logger.info(
        f"Split={split!r} | normal scans={len(class_ids['normal'])} | "
        f"abnormal scans={len(union_abnormal_ids)}"
    )

    # --- Normal class ---
    normal_path = _class_out_path(split, "normal")
    if normal_path.exists():
        logger.info(f"Already exists: {normal_path} — skipping.")
    else:
        X, H, scan_ids, coords = build_patch_matrix(
            class_ids["normal"], loader, collect_normal_patches, scratch_tag="normal"
        )
        np.savez_compressed(normal_path, X=X, H=H, scan_ids=scan_ids, coords=coords)
        logger.info(f"Saved → {normal_path}  (X: {X.shape}, H: {H.shape})")

    # --- Abnormal classes: single pass, then split by label ---
    missing_abnormal = [c for c in abnormality_keys if not _class_out_path(split, c).exists()]
    if not missing_abnormal:
        logger.info("All abnormal class files already exist — skipping extraction.")
        return

    X, H, scan_ids, coords = build_patch_matrix(
        union_abnormal_ids, loader, collect_abnormal_patches, scratch_tag="abnormal"
    )

    for class_name in abnormality_keys:
        if _class_out_path(split, class_name).exists():
            logger.info(f"Already exists: {_class_out_path(split, class_name)} — skipping.")
            continue
        _split_and_save_by_class(split, class_name, class_to_idx[class_name], X, H, scan_ids, coords)


def load_unified_patch_matrix(
    split: str = "train",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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