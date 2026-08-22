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

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
from tqdm import tqdm

from lc_ksvd.config import CLASS_ORDER, N_FEATURES, PATCHES_DIR
from lc_ksvd.data_loader.metadata_registry import LabelRegistry, MetadataRegistry
from lc_ksvd.data_loader.nifti_io import resolve_volume_path
from lc_ksvd.data_loader.scan_loader import ScanLoader
from lc_ksvd.patch_extractor.patch_io import _PatchStreamWriter
from lc_ksvd.patch_extractor.patch_sampling_abnormal import collect_abnormal_patches
from lc_ksvd.patch_extractor.patch_sampling_normal import collect_normal_patches
from lc_ksvd.patch_extractor.scan_worker import ScanWorkerPool

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

OnScanDone = Callable[[str, list[int], list[tuple[int, int, int]]], None]
PatchCollector = Callable[
    [list[str], ScanLoader, _PatchStreamWriter, "OnScanDone | None"],
    tuple[list[int], list[str], list[tuple[int, int, int]]],
]


def _class_out_path(split: str, class_name: str) -> Path:
    return PATCHES_DIR / f"unified_{split}_{class_name}.npz"


def _scratch_path(scratch_tag: str) -> Path:
    return PATCHES_DIR / f"_patch_scratch_{scratch_tag}.bin"


def _checkpoint_path(scratch_tag: str) -> Path:
    return PATCHES_DIR / f"_patch_checkpoint_{scratch_tag}.jsonl"


class _Checkpoint:
    """
    Per-scan checkpoint, appended one line per completed scan
    (JSONL: {"scan_id": ..., "labels": [...], "coords": [[x,y,z],...]}).

    A truncated/corrupt final line (from a crash mid-write) is dropped on
    load; that scan is simply treated as not-yet-done and reprocessed —
    never silently accepted with mismatched label/coord counts.
    """

    def __init__(self, path: Path):
        self.path = path
        self.done_scan_ids: set[str] = set()
        self.labels: list[int] = []
        self.scan_ids: list[str] = []
        self.coords: list[list[int]] = []
        if path.exists():
            self._load()
        self._fh = open(path, "a", buffering=1)

    def _load(self) -> None:
        lines = self.path.read_text().splitlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                if i == len(lines) - 1:
                    logger.warning(f"{self.path}: dropping truncated final checkpoint line.")
                    break
                raise
            labels, coords = rec["labels"], rec["coords"]
            if len(labels) != len(coords):
                raise ValueError(
                    f"{self.path}: checkpoint line for scan_id={rec['scan_id']!r} has "
                    f"{len(labels)} labels but {len(coords)} coords — corrupt checkpoint."
                )
            self.done_scan_ids.add(rec["scan_id"])
            self.labels.extend(labels)
            self.scan_ids.extend([rec["scan_id"]] * len(labels))
            self.coords.extend(coords)

    @property
    def n_patches(self) -> int:
        return len(self.labels)

    def record_scan(
        self,
        scan_id: str,
        labels: list[int],
        coords: list[tuple[int, int, int]],
    ) -> None:
        self.done_scan_ids.add(scan_id)
        self.labels.extend(labels)
        self.scan_ids.extend([scan_id] * len(labels))
        coords_list = [list(c) for c in coords]
        self.coords.extend(coords_list)
        self._fh.write(json.dumps({"scan_id": scan_id, "labels": list(labels), "coords": coords_list}) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def finalize_pass(scratch_tag: str) -> None:
    """Delete the scratch + checkpoint files for a pass. Only call after
    every output file that depends on this pass is confirmed saved —
    never before, or a crash between deletion and save loses the pass's
    work irrecoverably."""
    _scratch_path(scratch_tag).unlink(missing_ok=True)
    _checkpoint_path(scratch_tag).unlink(missing_ok=True)
    logger.info(f"[{scratch_tag}] Pass finalized — scratch/checkpoint cleared.")


def build_patch_matrix(
    ids: list[str],
    loader: ScanLoader,
    collector: PatchCollector,
    scratch_tag: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    In-process pass (used for the normal/grid-sampling phase, which hasn't
    shown the crash pattern the abnormal phase has). Does NOT delete the
    scratch/checkpoint files — call finalize_pass(scratch_tag) once the
    output .npz is confirmed saved.
    """
    PATCHES_DIR.mkdir(parents=True, exist_ok=True)
    scratch_path = _scratch_path(scratch_tag)
    checkpoint = _Checkpoint(_checkpoint_path(scratch_tag))

    if checkpoint.done_scan_ids:
        logger.info(
            f"[{scratch_tag}] Resuming from checkpoint: "
            f"{len(checkpoint.done_scan_ids)} scans / {checkpoint.n_patches} patches already done."
        )
    remaining_ids = [i for i in ids if i not in checkpoint.done_scan_ids]

    writer = _PatchStreamWriter(scratch_path, resume_count=checkpoint.n_patches)

    try:
        collector(remaining_ids, loader, writer, on_scan_done=checkpoint.record_scan)

        labels, scan_ids, coords = checkpoint.labels, checkpoint.scan_ids, checkpoint.coords
        n_patches = len(labels)

        X = np.empty((N_FEATURES, n_patches), dtype=np.float64)
        if n_patches:
            patch_mm = np.memmap(scratch_path, dtype=np.float32, mode="r",
                                  shape=(n_patches, N_FEATURES))
            X[:] = patch_mm.T
            del patch_mm
    finally:
        writer.close()
        checkpoint.close()

    H = np.array(labels, dtype=np.int64)
    scan_ids_arr = np.array(scan_ids, dtype=object)
    coords_arr = np.array(coords, dtype=np.int64).reshape(n_patches, 3)
    return X, H, scan_ids_arr, coords_arr


def build_patch_matrix_isolated(
    ids: list[str],
    metadata: MetadataRegistry,
    scratch_tag: str,
    timeout_s: float = 60.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Worker-isolated counterpart to build_patch_matrix, used for the
    abnormal (bbox-sampling) pass. Each scan runs in a persistent
    subprocess (see scan_worker.ScanWorkerPool); a crash or hang on one
    scan costs at most one restart + one retry, never the whole pass.

    Does NOT delete the scratch/checkpoint files — call finalize_pass
    once every dependent output .npz is confirmed saved.
    """
    PATCHES_DIR.mkdir(parents=True, exist_ok=True)
    scratch_path = _scratch_path(scratch_tag)
    checkpoint = _Checkpoint(_checkpoint_path(scratch_tag))

    if checkpoint.done_scan_ids:
        logger.info(
            f"[{scratch_tag}] Resuming from checkpoint: "
            f"{len(checkpoint.done_scan_ids)} scans / {checkpoint.n_patches} patches already done."
        )
    remaining_ids = [i for i in ids if i not in checkpoint.done_scan_ids]

    writer = _PatchStreamWriter(scratch_path, resume_count=checkpoint.n_patches)

    try:
        with ScanWorkerPool(metadata, collect_abnormal_patches, timeout_s=timeout_s) as pool:
            logger.info(f"Phase 2 — bbox-sampling {len(remaining_ids)} abnormal scans (worker-isolated)…")
            for scan_id in tqdm(remaining_ids, desc="abnormal scans"):
                labels, coords, patches = pool.process(scan_id)
                if len(patches):
                    writer.write_batch(patches, coords)
                writer.flush_scan()
                checkpoint.record_scan(scan_id, labels, coords)

        labels, scan_ids, coords = checkpoint.labels, checkpoint.scan_ids, checkpoint.coords
        n_patches = len(labels)

        X = np.empty((N_FEATURES, n_patches), dtype=np.float64)
        if n_patches:
            patch_mm = np.memmap(scratch_path, dtype=np.float32, mode="r",
                                  shape=(n_patches, N_FEATURES))
            X[:] = patch_mm.T
            del patch_mm
    finally:
        writer.close()
        checkpoint.close()

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
        logger.info(f"_filter_existing: {tag}{len(missing)}/{len(ids)} missing volumes: {missing}")
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
        logger.warning("No normal volumes found in metadata — dataset may contain only abnormal scans.")
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
        finalize_pass("normal")

    # --- Abnormal classes: single worker-isolated pass, then split by label ---
    missing_abnormal = [c for c in abnormality_keys if not _class_out_path(split, c).exists()]
    if not missing_abnormal:
        logger.info("All abnormal class files already exist — skipping extraction.")
        return

    X, H, scan_ids, coords = build_patch_matrix_isolated(
        union_abnormal_ids, metadata, scratch_tag="abnormal"
    )

    for class_name in abnormality_keys:
        if _class_out_path(split, class_name).exists():
            logger.info(f"Already exists: {_class_out_path(split, class_name)} — skipping.")
            continue
        _split_and_save_by_class(split, class_name, class_to_idx[class_name], X, H, scan_ids, coords)

    if all(_class_out_path(split, c).exists() for c in abnormality_keys):
        finalize_pass("abnormal")

def load_unified_patch_matrix(
    split: str = "train",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and concatenate per-class patch matrices without holding
    duplicate copies in memory.

    Peak memory is ~= size(final X) + size(largest single class X),
    versus ~2x size(final X) (plus every intermediate) for the naive
    list-then-concatenate approach.
    """
    paths = []
    for class_name in CLASS_ORDER:
        path = _class_out_path(split, class_name)
        if not path.exists():
            raise FileNotFoundError(
                f"Patch matrix not found: {path}. Run extract_unified() first."
            )
        paths.append(path)

    # -- Pass 1: read shapes/dtypes only, don't retain array data --------
    n_features = None
    n_total = 0
    x_dtype = h_dtype = scan_id_dtype = coord_dtype = None
    coord_ndim1 = None
    per_class_n: list[int] = []

    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            X_shape = data["X"].shape
            if n_features is None:
                n_features = X_shape[0]
                x_dtype = data["X"].dtype
                h_dtype = data["H"].dtype
                scan_id_dtype = data["scan_ids"].dtype
                coord_dtype = data["coords"].dtype
                coord_ndim1 = data["coords"].shape[1]
            elif X_shape[0] != n_features:
                raise ValueError(
                    f"{path} has {X_shape[0]} features, expected {n_features}."
                )
            n_class = X_shape[1]
            per_class_n.append(n_class)
            n_total += n_class

    if x_dtype != np.float32:
        logger.warning(
            "Patch matrices are stored as %s, not float32 — this doubles "
            "(or worse) memory versus expected. Consider re-saving "
            "extract_unified() output as float32.",
            x_dtype,
        )

    # -- Pass 2: allocate final arrays once, fill in place ----------------
    X = np.empty((n_features, n_total), dtype=np.float32)
    H = np.empty((n_total,), dtype=h_dtype)
    scan_ids = np.empty((n_total,), dtype=scan_id_dtype)
    coords = np.empty((n_total, coord_ndim1), dtype=coord_dtype)

    offset = 0
    for path, n_class in zip(paths, per_class_n):
        with np.load(path, allow_pickle=True) as data:
            X[:, offset:offset + n_class] = data["X"]
            H[offset:offset + n_class] = data["H"]
            scan_ids[offset:offset + n_class] = data["scan_ids"]
            coords[offset:offset + n_class] = data["coords"]
        offset += n_class

    logger.info(
        "Loaded matrix: X=%s, H=%s, scan_ids=%s, coords=%s",
        X.shape, H.shape, scan_ids.shape, coords.shape,
    )
    return X, H, scan_ids, coords

if __name__ == "__main__":
    extract_unified(split="train")