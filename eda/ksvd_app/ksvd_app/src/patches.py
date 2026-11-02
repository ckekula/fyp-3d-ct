"""
Reconstruct the patch grid of a scan into a common 3D volume, align the
ground-truth mask to it, and locate the patch closest to the nodule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import PATCH_SHAPE_XYZ


def patch_indices_for_scan(scan_ids: np.ndarray, scan_id: str) -> np.ndarray:
    """Indices (into X / coords) of all patches belonging to a scan."""
    return np.where(scan_ids == scan_id)[0]


def reshape_patch(flat_patch: np.ndarray) -> np.ndarray:
    """Reshape a flattened patch vector into (X, Y, Z) voxels."""
    return flat_patch.reshape(PATCH_SHAPE_XYZ)


@dataclass(frozen=True)
class ReconstructedVolume:
    """A scan's patches merged into one common 3D array via overlap-averaging."""

    volume: np.ndarray  # mean intensity per voxel, shape patch_space_shape
    min_coord: np.ndarray  # global (x, y, z) origin of this volume
    shape: tuple[int, int, int]


def reconstruct_patch_space(
    X: np.ndarray, coords: np.ndarray, indices: np.ndarray
) -> ReconstructedVolume:
    """Merge all patches of a scan into one volume, averaging overlaps."""
    patch_x, patch_y, patch_z = PATCH_SHAPE_XYZ
    scan_coords = coords[indices]

    min_coord = scan_coords.min(axis=0)
    max_coord = scan_coords.max(axis=0)
    shape = (
        int(max_coord[0] - min_coord[0] + patch_x),
        int(max_coord[1] - min_coord[1] + patch_y),
        int(max_coord[2] - min_coord[2] + patch_z),
    )

    volume = np.zeros(shape, dtype=np.float32)
    count = np.zeros(shape, dtype=np.int32)

    for idx in indices:
        patch = reshape_patch(X[:, idx])
        x0, y0, z0 = coords[idx] - min_coord
        x1, y1, z1 = x0 + patch_x, y0 + patch_y, z0 + patch_z
        volume[x0:x1, y0:y1, z0:z1] += patch
        count[x0:x1, y0:y1, z0:z1] += 1

    filled = count > 0
    volume[filled] /= count[filled]
    return ReconstructedVolume(volume=volume, min_coord=min_coord, shape=shape)


def crop_mask_to_patch_space(mask_final: np.ndarray, recon: ReconstructedVolume) -> np.ndarray:
    """Crop a full-volume ground-truth mask to the same region as a ReconstructedVolume."""
    x0, y0, z0 = recon.min_coord
    x1, y1, z1 = x0 + recon.shape[0], y0 + recon.shape[1], z0 + recon.shape[2]
    return mask_final[x0:x1, y0:y1, z0:z1]


def nodule_centroid_global(mask_crop: np.ndarray, recon: ReconstructedVolume) -> np.ndarray | None:
    """Centroid (global x, y, z) of ground-truth voxels within the patch space.

    Returns None if the mask has no positive voxels in this scan's patch region.
    """
    mx, my, mz = np.where(mask_crop > 0)
    if len(mx) == 0:
        return None
    local_centroid = np.array([mx.mean(), my.mean(), mz.mean()])
    return local_centroid + recon.min_coord


@dataclass(frozen=True)
class CenteredPatch:
    """The patch of a scan closest to the ground-truth nodule centroid."""

    index: int  # global index into X / coords
    position: int  # position within this scan's `indices` array
    coord: np.ndarray  # global (x, y, z) origin
    distance_to_centroid: float


def select_nodule_centered_patch(
    coords: np.ndarray, indices: np.ndarray, centroid_global: np.ndarray
) -> CenteredPatch:
    """Pick the patch (among a scan's patches) whose center is closest to the centroid."""
    half_extent = np.array(PATCH_SHAPE_XYZ) / 2.0
    patch_centers = coords[indices] + half_extent
    distances = np.linalg.norm(patch_centers - centroid_global, axis=1)

    position = int(np.argmin(distances))
    global_idx = int(indices[position])
    return CenteredPatch(
        index=global_idx,
        position=position,
        coord=coords[global_idx],
        distance_to_centroid=float(distances[position]),
    )


def ground_truth_voxels_in_patch(
    mask_final: np.ndarray, patch_coord: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Local (within-patch) coordinates of ground-truth voxels inside one patch."""
    x0, y0, z0 = patch_coord
    px, py, pz = PATCH_SHAPE_XYZ
    patch_mask = mask_final[x0 : x0 + px, y0 : y0 + py, z0 : z0 + pz]
    return np.where(patch_mask > 0)
