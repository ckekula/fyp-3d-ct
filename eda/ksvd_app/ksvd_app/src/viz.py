"""
Plotly figure builders. Kept separate from app.py so the UI layer stays
thin and the plotting logic is independently testable.
"""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.patches import reshape_patch

_SCATTER3D_MARKER_SIZE = 4
_GT_MARKER_SIZE = 5


def patch_grid_figure(
    X: np.ndarray, coords: np.ndarray, indices: np.ndarray, scan_id: str
) -> go.Figure:
    """Grid of the middle Z-slice of every patch belonging to a scan."""
    n_patches = len(indices)
    n_cols = min(4, n_patches) or 1
    n_rows = math.ceil(n_patches / n_cols)

    subplot_titles = [
        f"idx {idx} · ({coords[idx][0]}, {coords[idx][1]}, {coords[idx][2]})"
        for idx in indices
    ]
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles)

    for i, idx in enumerate(indices):
        patch_3d = reshape_patch(X[:, idx])
        mid_slice = patch_3d[:, :, patch_3d.shape[2] // 2]
        row, col = divmod(i, n_cols)
        fig.add_trace(
            go.Heatmap(z=mid_slice.T, colorscale="gray", showscale=False),
            row=row + 1,
            col=col + 1,
        )

    fig.update_layout(
        height=260 * n_rows,
        title=f"{scan_id} — middle Z-slice of all {n_patches} patches",
        showlegend=False,
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False, scaleanchor="x")
    return fig


def single_slice_heatmap(slice_2d: np.ndarray, title: str) -> go.Figure:
    """A single 2D intensity slice."""
    fig = go.Figure(
        data=go.Heatmap(z=slice_2d.T, colorscale="gray", colorbar=dict(title="Intensity"))
    )
    fig.update_layout(
        title=title,
        xaxis_title="X",
        yaxis_title="Y",
        yaxis=dict(scaleanchor="x"),
        height=500,
    )
    return fig


def intensity_scatter_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    values: np.ndarray,
    title: str,
    *,
    colorscale: str = "RdBu",
    opacity: float = 0.35,
    symmetric_scale: bool = False,
    overlay: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    overlay_name: str = "Ground truth",
) -> go.Figure:
    """A 3D scatter of voxel intensities, with an optional highlighted overlay
    (e.g. ground-truth voxels) drawn on top in red.
    """
    marker: dict = dict(
        size=_SCATTER3D_MARKER_SIZE,
        color=values,
        colorscale=colorscale,
        opacity=opacity,
        colorbar=dict(title="Intensity"),
    )
    if symmetric_scale:
        peak = float(np.abs(values).max()) if values.size else 1.0
        marker["cmin"], marker["cmax"] = -peak, peak

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=x, y=y, z=z, mode="markers", marker=marker, name="Intensity", showlegend=False
        )
    )
    if overlay is not None and len(overlay[0]) > 0:
        ox, oy, oz = overlay
        fig.add_trace(
            go.Scatter3d(
                x=ox,
                y=oy,
                z=oz,
                mode="markers",
                marker=dict(size=_GT_MARKER_SIZE, color="red", opacity=0.9),
                name=overlay_name,
            )
        )

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        height=600,
        legend=dict(x=0.02, y=0.98),
    )
    return fig


def sparse_code_stem_figure(gamma: np.ndarray, title: str) -> go.Figure:
    """Stem plot of all sparse coefficients for one patch."""
    idx = np.arange(len(gamma))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.repeat(idx, 2),
            y=np.ravel(np.column_stack([np.zeros_like(gamma), gamma])),
            mode="lines",
            line=dict(color="lightgray", width=1),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=idx,
            y=gamma,
            mode="markers",
            marker=dict(size=5, color="steelblue"),
            showlegend=False,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Dictionary atom",
        yaxis_title="Sparse coefficient",
        height=420,
    )
    return fig


def atom_contribution_grid(
    atom_patches: dict[int, np.ndarray],
    gamma: np.ndarray,
    combined_patch: np.ndarray,
    origin: np.ndarray,
    title: str,
) -> go.Figure:
    """2x2 grid: each top atom's contribution plus the combined reconstruction."""
    x0, y0, z0 = origin
    px, py, pz = combined_patch.shape
    xi, yi, zi = np.indices((px, py, pz))
    xi, yi, zi = xi.ravel() + x0, yi.ravel() + y0, zi.ravel() + z0

    items = [
        (f"Atom {atom} (coef={gamma[atom]:.3f})", patch)
        for atom, patch in atom_patches.items()
    ]
    items.append((f"Combined (sum of top {len(atom_patches)} atoms)", combined_patch))

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}], [{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=[label for label, _ in items],
    )

    for i, (_, patch) in enumerate(items):
        row, col = divmod(i, 2)
        peak = float(np.abs(patch).max()) or 1.0
        fig.add_trace(
            go.Scatter3d(
                x=xi,
                y=yi,
                z=zi,
                mode="markers",
                marker=dict(
                    size=4,
                    color=patch.ravel(),
                    colorscale="RdBu",
                    cmin=-peak,
                    cmax=peak,
                    opacity=0.6,
                ),
                showlegend=False,
            ),
            row=row + 1,
            col=col + 1,
        )

    fig.update_layout(title=title, height=850)
    return fig
