from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def _to_dhw(array: np.ndarray) -> np.ndarray:
    """
    Convert input to a simple 3D numpy array.

    The pipeline already passes (D, H, W), so this mainly validates shape.
    """
    array = np.asarray(array)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D array, got shape {array.shape}")
    return array


def select_mask_aware_slices(
    mask: np.ndarray,
    *,
    max_slices: int = 3,
) -> List[int]:
    """
    Select slices for visualization.

    If the model predicted foreground, choose slices around the largest predicted
    mask area. This is better for small nodules than fixed 25/50/75% slices.

    If the mask is empty, fall back to representative quartile/mid slices.
    """
    mask = _to_dhw(mask)
    depth = mask.shape[0]

    if depth <= 0:
        return []

    mask_area = (mask > 0).sum(axis=(1, 2))

    if int(mask_area.max()) > 0:
        best_slice = int(mask_area.argmax())

        candidates = [
            max(0, best_slice - 2),
            best_slice,
            min(depth - 1, best_slice + 2),
        ]

        # Remove duplicates while preserving order.
        unique: List[int] = []
        for idx in candidates:
            if idx not in unique:
                unique.append(idx)

        return unique[:max_slices]

    # Fallback when no foreground was predicted.
    fallback = [
        int(depth * 0.25),
        int(depth * 0.50),
        int(depth * 0.75),
    ]

    unique = []
    for idx in fallback:
        idx = max(0, min(depth - 1, idx))
        if idx not in unique:
            unique.append(idx)

    return unique[:max_slices]


def save_overlay_png(
    volume: np.ndarray,
    mask: np.ndarray,
    output_path: str | Path,
    *,
    slice_indices: Optional[Iterable[int]] = None,
    alpha: float = 0.40,
) -> None:
    """
    Save CT + predicted mask overlay.

    This visualizes the model-predicted abnormality location, not the ground-truth
    abnormality location. Use Dice/IoU against GT masks for quantitative validation.
    """
    volume = _to_dhw(volume)
    mask = _to_dhw(mask)

    if volume.shape != mask.shape:
        raise ValueError(
            f"Volume and mask must have the same shape. "
            f"Got volume={volume.shape}, mask={mask.shape}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if slice_indices is None:
        selected_slices = select_mask_aware_slices(mask, max_slices=3)
    else:
        selected_slices = [
            max(0, min(volume.shape[0] - 1, int(idx)))
            for idx in slice_indices
        ]

    if not selected_slices:
        raise ValueError("No slices selected for visualization.")

    fig, axes = plt.subplots(
        1,
        len(selected_slices),
        figsize=(5 * len(selected_slices), 5),
        squeeze=False,
    )

    axes_flat = axes.ravel()

    for ax, slice_idx in zip(axes_flat, selected_slices):
        ct_slice = volume[slice_idx]
        mask_slice = mask[slice_idx] > 0

        ax.imshow(ct_slice, cmap="gray")
        ax.imshow(np.ma.masked_where(~mask_slice, mask_slice), alpha=alpha)
        ax.set_title(f"Slice {slice_idx}")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
