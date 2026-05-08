from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_overlay_png(volume: np.ndarray, mask: np.ndarray, out_path: str | Path, slice_indices: list[int] | None = None) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if slice_indices is None:
        depth = volume.shape[0]
        slice_indices = [depth // 4, depth // 2, (3 * depth) // 4]

    fig, axes = plt.subplots(1, len(slice_indices), figsize=(5 * len(slice_indices), 5), constrained_layout=True)
    if len(slice_indices) == 1:
        axes = [axes]

    for axis, slice_index in zip(axes, slice_indices):
        axis.imshow(volume[slice_index], cmap="gray")
        axis.imshow(np.ma.masked_where(mask[slice_index] <= 0, mask[slice_index]), cmap="autumn", alpha=0.45)
        axis.set_title(f"Slice {slice_index}")
        axis.axis("off")

    fig.savefig(out_path, dpi=180)
    plt.close(fig)