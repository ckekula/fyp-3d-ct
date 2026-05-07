"""
localize.py
Atom back-projection: given a trained LC-KSVD2 model and a CT volume,
produces a 3D contribution map in voxel space — the inherent localization
mechanism of the LC-KSVD approach.

How it works:
  1. Extract overlapping patches from the volume with a sliding window.
  2. Compute the sparse code for each patch using the trained dictionary.
  3. Identify class-discriminative atoms (top percentile of |W[pos_class, :]|).
  4. Reconstruct each patch using only those discriminative atoms,
     weighted by the sparse code coefficients.
  5. Accumulate patch reconstructions back to the full volume grid
     (overlapping regions are averaged).
  6. The resulting 3D map is thresholded to produce a binary localization mask.

Usage:
  from localize import LocalizationEngine
  engine = LocalizationEngine.from_pickle("outputs/models/lung_nodule_lcksvd2.pkl")
  contrib_map = engine.contribution_map(volume_array)   # [H, W, D] float
  binary_mask = engine.localize(volume_array)           # [H, W, D] bool
"""

import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from skimage.filters import threshold_otsu

from .config import CONTRIB_THRESHOLD_MODE, DISCRIMINATIVE_ATOM_PERCENTILE, LCKSVD_CONFIG, PATCH_SIZE, TARGET_SPACING_MM, HU_MIN, HU_MAX
from .data_loader import window_and_normalise, resample_volume

try:
    from reppi import OMP
except ImportError:
    raise ImportError("reppi is not installed. Run: pip install reppi")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ─── Discriminative atom selection ───────────────────────────────────────────

def select_discriminative_atoms(W: np.ndarray, pos_class: int = 1) -> np.ndarray:
    """
    Return indices of atoms whose classifier weight for the positive class
    exceeds the DISCRIMINATIVE_ATOM_PERCENTILE threshold.

    W shape: (n_classes, n_components) — row 1 = positive class.
    """
    weights = np.abs(W[pos_class, :])             # (n_components,)
    threshold = np.percentile(weights, DISCRIMINATIVE_ATOM_PERCENTILE)
    atom_indices = np.where(weights >= threshold)[0]
    logger.debug(
        f"Selected {len(atom_indices)}/{W.shape[1]} discriminative atoms "
        f"(percentile={DISCRIMINATIVE_ATOM_PERCENTILE})"
    )
    return atom_indices


# ─── Sliding-window patch iterator ───────────────────────────────────────────

def sliding_window_patches(
    volume: np.ndarray,
    patch_size: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Yield (patch_flat, top_left_corner) for a 3D sliding window.

    Returns:
      patches   : (n_features, n_windows) float64
      corners   : (n_windows, 3) int — (x0, y0, z0) top-left corner of each patch
    """
    H, W, D = volume.shape
    p = patch_size

    xs = list(range(0, H - p + 1, stride))
    ys = list(range(0, W - p + 1, stride))
    zs = list(range(0, D - p + 1, stride))

    # Ensure the last patch reaches the boundary
    if xs[-1] + p < H:
        xs.append(H - p)
    if ys[-1] + p < W:
        ys.append(W - p)
    if zs[-1] + p < D:
        zs.append(D - p)

    patches = []
    corners = []

    for x0 in xs:
        for y0 in ys:
            for z0 in zs:
                patch = volume[x0:x0+p, y0:y0+p, z0:z0+p].ravel()
                patches.append(patch)
                corners.append((x0, y0, z0))

    patches_arr = np.array(patches, dtype=np.float64).T   # (n_features, n_windows)
    corners_arr = np.array(corners, dtype=np.int32)        # (n_windows, 3)
    return patches_arr, corners_arr


# ─── Contribution map computation ────────────────────────────────────────────

def compute_contribution_map(
    volume: np.ndarray,
    D: np.ndarray,
    W: np.ndarray,
    n_nonzero_coefs: int,
    discriminative_atoms: np.ndarray,
    stride: int,
) -> np.ndarray:
    """
    Core back-projection:
      For each patch, compute sparse code Gamma, restrict to discriminative
      atoms, reconstruct the patch contribution, and accumulate into a 3D map.

    Returns:
      contrib_map : float32 [H, W, D], values are mean reconstructed
                    intensity from discriminative atoms — high where the
                    class-relevant dictionary components are active.
    """
    p    = PATCH_SIZE
    H, W_vol, D_vol = volume.shape

    # Accumulation buffers
    accum  = np.zeros((H, W_vol, D_vol), dtype=np.float64)
    counts = np.zeros((H, W_vol, D_vol), dtype=np.float64)

    # Extract patches with sliding window
    patches, corners = sliding_window_patches(volume, p, stride)
    n_windows = patches.shape[1]
    logger.info(f"  Sliding window: {n_windows} patches (stride={stride})")

    # Column-normalise patches for OMP (same as training)
    col_norms  = np.linalg.norm(patches, axis=0)
    safe_norms = np.where(col_norms < 1e-10, 1.0, col_norms)
    patches_norm = patches / safe_norms[np.newaxis, :]

    # Sparse encode all patches at once
    omp = OMP(n_nonzero_coefs=n_nonzero_coefs)
    Gamma = omp.encode(patches_norm, D)             # (n_components, n_windows)

    # Restrict to discriminative atoms only
    Gamma_disc = np.zeros_like(Gamma)
    Gamma_disc[discriminative_atoms, :] = Gamma[discriminative_atoms, :]

    # Reconstruct each patch from discriminative atoms
    D_disc = D[:, discriminative_atoms]             # (n_features, n_disc_atoms)
    # Full reconstruction: D[:, disc] @ Gamma_disc[disc, :]
    reconstructions = D[:, discriminative_atoms] @ Gamma[discriminative_atoms, :]
    # Shape: (n_features, n_windows)

    # Accumulate back to volume space
    for j in range(n_windows):
        x0, y0, z0 = corners[j]
        contrib_patch = reconstructions[:, j].reshape(p, p, p)

        accum[x0:x0+p, y0:y0+p, z0:z0+p]  += np.abs(contrib_patch)
        counts[x0:x0+p, y0:y0+p, z0:z0+p] += 1.0

    # Average over overlapping windows
    safe_counts = np.where(counts == 0, 1.0, counts)
    contrib_map = (accum / safe_counts).astype(np.float32)

    return contrib_map


# ─── Thresholding ────────────────────────────────────────────────────────────

def binarize_contribution_map(
    contrib_map: np.ndarray,
    mode: str = "otsu",
    fixed_threshold: float = 0.3,
    otsu_reference: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Convert a continuous contribution map to a binary localization mask.

    mode="otsu"  → Otsu threshold on this map (or on otsu_reference if provided)
    mode="fixed" → threshold at fixed_threshold * max(contrib_map)
    """
    if mode == "otsu":
        ref = otsu_reference if otsu_reference is not None else contrib_map
        try:
            thresh = threshold_otsu(ref)
        except Exception:
            thresh = ref.mean()
    else:
        thresh = fixed_threshold * contrib_map.max()

    return (contrib_map >= thresh).astype(np.uint8)


# ─── LocalizationEngine ──────────────────────────────────────────────────────

class LocalizationEngine:
    """
    Wraps a trained LC-KSVD2 model and exposes:
      contribution_map(volume) → float32 [H, W, D]
      localize(volume)         → uint8   [H, W, D] binary mask
    """

    def __init__(self, model, abnormality: str, patch_size: int,
                 n_nonzero_coefs: int, target_spacing: float, hu_window: Tuple):
        self.model           = model
        self.abnormality     = abnormality
        self.patch_size      = patch_size
        self.n_nonzero_coefs = n_nonzero_coefs
        self.target_spacing  = target_spacing
        self.hu_window       = hu_window

        # Pre-select discriminative atoms once
        self.discriminative_atoms = select_discriminative_atoms(model.W_, pos_class=1)

        # Stride for sliding window: 50% overlap by default
        self.stride = max(1, patch_size // 2)

    @classmethod
    def from_pickle(cls, model_path: str) -> "LocalizationEngine":
        with open(model_path, "rb") as f:
            payload = pickle.load(f)

        model        = payload["model"]
        abnormality  = payload["abnormality"]
        patch_size   = payload.get("patch_size", PATCH_SIZE)
        cfg          = payload.get("lcksvd_config", LCKSVD_CONFIG)
        target_sp    = payload.get("target_spacing", TARGET_SPACING_MM)
        hu_window    = payload.get("hu_window", (HU_MIN, HU_MAX))

        return cls(
            model=model,
            abnormality=abnormality,
            patch_size=patch_size,
            n_nonzero_coefs=cfg["n_nonzero_coefs"],
            target_spacing=target_sp,
            hu_window=hu_window,
        )

    def preprocess(self, volume_hu: np.ndarray, current_spacing: np.ndarray) -> np.ndarray:
        """Resample and window a raw HU volume for inference."""
        vol = resample_volume(volume_hu, current_spacing)
        vol = window_and_normalise(vol)
        return vol

    def contribution_map(
        self,
        volume: np.ndarray,
        stride: Optional[int] = None,
    ) -> np.ndarray:
        """
        Produce a 3D contribution map for a preprocessed volume (already in [0,1]).

        volume : float32 [H, W, D] — must already be resampled and windowed.
        stride : sliding window stride in voxels. Default = patch_size // 2 (50% overlap).
        """
        s = stride or self.stride
        return compute_contribution_map(
            volume=volume,
            D=self.model.D_,
            W=self.model.W_,
            n_nonzero_coefs=self.n_nonzero_coefs,
            discriminative_atoms=self.discriminative_atoms,
            stride=s,
        )

    def localize(
        self,
        volume: np.ndarray,
        stride: Optional[int] = None,
        threshold_mode: Optional[str] = None,
    ) -> np.ndarray:
        """
        Produce a binary localization mask for a preprocessed volume.

        Returns uint8 [H, W, D] with 1 where the abnormality is predicted.
        """
        cmap  = self.contribution_map(volume, stride=stride)
        mode  = threshold_mode or CONTRIB_THRESHOLD_MODE
        mask  = binarize_contribution_map(cmap, mode=mode)
        return mask

    def classify(self, volume: np.ndarray) -> Tuple[int, float]:
        """
        Classify a preprocessed volume using the LC-KSVD2 classifier.

        Returns:
          prediction : 1 (abnormality present) or 0 (absent)
          score      : raw W[1, :] @ Gamma score (higher = more likely positive)
        """
        # Use a coarser stride for classification (speed)
        patches, _ = sliding_window_patches(volume, self.patch_size, stride=self.patch_size)

        col_norms = np.linalg.norm(patches, axis=0)
        safe_norms = np.where(col_norms < 1e-10, 1.0, col_norms)
        patches_norm = patches / safe_norms[np.newaxis, :]

        omp   = OMP(n_nonzero_coefs=self.n_nonzero_coefs)
        Gamma = omp.encode(patches_norm, self.model.D_)     # (K, n_windows)

        # Aggregate by taking the mean sparse code across patches
        gamma_mean = Gamma.mean(axis=1, keepdims=True)      # (K, 1)
        score = float(self.model.W_[1, :] @ gamma_mean[:, 0])
        prediction = int(score > 0)
        return prediction, score