"""
Sparse coding of patches against the K-SVD dictionary, and decomposition
of a patch's reconstruction into individual atom contributions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from reppi import OMP

from src.patches import reshape_patch


def normalize_columns(X: np.ndarray) -> np.ndarray:
    """Scale each column of X to unit L2 norm (columns near zero are left unchanged)."""
    norms = np.sqrt((X * X).sum(axis=0))
    norms = np.where(norms < 1e-10, 1.0, norms)
    return X / norms


def encode_patches_omp(X: np.ndarray, D: np.ndarray, n_nonzero_coefs: int) -> np.ndarray:
    """Sparse-code each column of X against dictionary D using batch OMP.

    Parameters
    ----------
    X : np.ndarray, shape (n_features, n_patches)
    D : np.ndarray, shape (n_features, n_components)
    n_nonzero_coefs : int

    Returns
    -------
    Gamma : np.ndarray, shape (n_components, n_patches)
    """
    X_norm = normalize_columns(X)
    omp = OMP(n_nonzero_coefs=n_nonzero_coefs, mode="batch", check_dict=True)
    return omp.encode(X_norm, D)


@dataclass(frozen=True)
class ReconstructionSimilarity:
    mae: float
    rmse: float
    correlation: float
    intensity_ratio: float


def top_k_atoms(gamma: np.ndarray, k: int = 3) -> np.ndarray:
    """Indices of the k active atoms with the largest |coefficient|, descending."""
    active = np.flatnonzero(gamma)
    order = np.argsort(np.abs(gamma[active]))[::-1]
    return active[order][:k]


def atom_contribution_patches(
    D: np.ndarray, gamma: np.ndarray, atoms: np.ndarray
) -> dict[int, np.ndarray]:
    """For each atom, the 3D patch obtained from D @ gamma with only that
    atom's coefficient kept (all others zeroed)."""
    contributions: dict[int, np.ndarray] = {}
    for atom in atoms:
        gamma_single = np.zeros_like(gamma)
        gamma_single[atom] = gamma[atom]
        contributions[int(atom)] = reshape_patch(D @ gamma_single)
    return contributions


def reconstruction_similarity(
    combined_patch: np.ndarray, original_patch: np.ndarray, gt_mask: np.ndarray
) -> ReconstructionSimilarity | None:
    """Compare a reconstruction to the original patch within the ground-truth region.

    Returns None if the patch contains no ground-truth voxels.
    """
    region = gt_mask > 0
    if not np.any(region):
        return None

    orig = original_patch[region]
    combined = combined_patch[region]

    mae = float(np.mean(np.abs(combined - orig)))
    rmse = float(np.sqrt(np.mean((combined - orig) ** 2)))
    correlation = float(np.corrcoef(combined, orig)[0, 1]) if region.sum() > 1 else float("nan")
    orig_sum = orig.sum()
    intensity_ratio = float(combined.sum() / orig_sum) if orig_sum != 0 else float("nan")

    return ReconstructionSimilarity(
        mae=mae, rmse=rmse, correlation=correlation, intensity_ratio=intensity_ratio
    )
