"""
K-SVD Patch Explorer
=====================
Interactive Streamlit app for exploring K-SVD dictionary-learned patches
extracted from CT scans, their ground-truth abnormality overlays, and
their OMP sparse-code decomposition.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from src import config, data_io, patches, sparse_coding, viz

st.set_page_config(page_title="K-SVD Patch Explorer", layout="wide")


# --------------------------------------------------------------------------
# Cached per-scan computation
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _compute_ground_truth(scan_id: str, _recon: patches.ReconstructedVolume):
    """Load, resample, and align the ground-truth mask for a scan.

    Returns (mask_final, mask_crop, centroid_global) or None if no
    segmentation file is available for this scan.
    """
    from src import segmentation as seg

    if not data_io.has_segmentation(scan_id):
        return None
    raw_mask = data_io.load_raw_segmentation_mask(scan_id)
    mask_final = seg.resample_and_align_mask(raw_mask, scan_id)
    mask_crop = patches.crop_mask_to_patch_space(mask_final, _recon)
    centroid = patches.nodule_centroid_global(mask_crop, _recon)
    return mask_final, mask_crop, centroid


@st.cache_data(show_spinner="Running OMP sparse coding for this scan...")
def _encode_scan_patches(
    scan_id: str, _X_scan: np.ndarray, _D: np.ndarray, n_nonzero_coefs: int
) -> np.ndarray:
    return sparse_coding.encode_patches_omp(_X_scan, _D, n_nonzero_coefs)


def main() -> None:
    st.title("K-SVD Patch Explorer")
    st.caption(
        "Explore dictionary-learned CT patches: reconstruction, ground-truth "
        "alignment, and OMP sparse-code decomposition."
    )

    try:
        checkpoint = data_io.load_ksvd_checkpoint()
        dataset = data_io.load_patch_dataset()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info(f"Expected data files under: `{config.DATA_DIR}`")
        return

    scan_ids = data_io.available_scan_ids(dataset)
    if not scan_ids:
        st.warning("No scans found in the patch dataset.")
        return

    with st.sidebar:
        st.header("Scan selection")
        scan_id = st.selectbox("Scan ID", scan_ids)

        indices = patches.patch_indices_for_scan(dataset.scan_ids, scan_id)
        st.metric("Patches in this scan", len(indices))

        finding = data_io.get_finding_info(scan_id)
        if finding:
            st.subheader("Finding metadata")
            findings = finding.get("findings", {})
            for fid, desc in findings.items():
                st.write(f"**{fid}**: {desc}")
        elif data_io.load_dataset_metadata() is not None:
            st.caption("No matching entry in dataset_filtered.json.")

        if not data_io.has_segmentation(scan_id):
            st.warning(
                f"No segmentation file found at "
                f"`{config.segmentation_path(scan_id).name}`. "
                "Ground-truth views will be unavailable."
            )
        elif scan_id not in config.SCAN_ORIGINAL_SPACING_MM:
            st.warning(
                f"No original spacing registered for '{scan_id}' in "
                "`src/config.py` (SCAN_ORIGINAL_SPACING_MM). Ground-truth "
                "alignment will fail until it's added."
            )

    if len(indices) == 0:
        st.warning(f"No patches found for scan '{scan_id}'.")
        return

    recon = patches.reconstruct_patch_space(dataset.X, dataset.coords, indices)

    gt = None
    try:
        gt = _compute_ground_truth(scan_id, recon)
    except KeyError as exc:
        st.sidebar.error(str(exc))

    tab_names = ["Patch grid", "Ground truth", "Nodule-centered patch", "Sparse coding"]
    tab_grid, tab_gt, tab_centered, tab_sparse = st.tabs(tab_names)

    # ---------------------------------------------------------------- grid
    with tab_grid:
        st.plotly_chart(
            viz.patch_grid_figure(dataset.X, dataset.coords, indices, scan_id),
            use_container_width=True,
        )

    # ------------------------------------------------------------ ground truth
    with tab_gt:
        if gt is None:
            st.info("Ground truth unavailable for this scan (see sidebar warning).")
        else:
            mask_final, mask_crop, centroid = gt
            gt_voxel_count = int(np.count_nonzero(mask_crop))
            st.metric("Ground-truth voxels in patch region", gt_voxel_count)

            if gt_voxel_count == 0:
                st.info("No ground-truth voxels fall within this scan's patch region.")
            else:
                mx, my, mz = np.where(mask_crop > 0)
                st.plotly_chart(
                    viz.intensity_scatter_3d(
                        mx, my, mz, np.ones_like(mx), f"{scan_id} — ground-truth voxels",
                        colorscale="Reds", opacity=0.9, symmetric_scale=False,
                    ),
                    use_container_width=True,
                )

                xi, yi, zi = np.indices(recon.shape)
                st.plotly_chart(
                    viz.intensity_scatter_3d(
                        xi.ravel(), yi.ravel(), zi.ravel(), recon.volume.ravel(),
                        f"{scan_id} — reconstructed volume with ground-truth overlay",
                        overlay=(mx, my, mz),
                    ),
                    use_container_width=True,
                )

    # -------------------------------------------------------- nodule-centered
    with tab_centered:
        if gt is None or gt[2] is None:
            st.info("No ground-truth centroid available for this scan.")
        else:
            mask_final, _, centroid = gt
            centered = patches.select_nodule_centered_patch(dataset.coords, indices, centroid)
            st.write(
                f"Selected patch **#{centered.index}** "
                f"(position {centered.position} of {len(indices)}), "
                f"coord `{tuple(centered.coord)}`, "
                f"distance to centroid **{centered.distance_to_centroid:.2f}** voxels."
            )

            patch_3d = patches.reshape_patch(dataset.X[:, centered.index])
            gt_x, gt_y, gt_z = patches.ground_truth_voxels_in_patch(mask_final, centered.coord)
            x0, y0, z0 = centered.coord
            xi, yi, zi = np.indices(config.PATCH_SHAPE_XYZ)
            xi, yi, zi = xi.ravel() + x0, yi.ravel() + y0, zi.ravel() + z0

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    viz.intensity_scatter_3d(
                        xi, yi, zi, patch_3d.ravel(), "Patch intensity (no overlay)",
                        opacity=0.5,
                    ),
                    use_container_width=True,
                )
            with col2:
                st.plotly_chart(
                    viz.intensity_scatter_3d(
                        xi, yi, zi, patch_3d.ravel(), "Patch intensity (with ground truth)",
                        opacity=0.5,
                        overlay=(gt_x + x0, gt_y + y0, gt_z + z0),
                    ),
                    use_container_width=True,
                )

    # ------------------------------------------------------------- sparse coding
    with tab_sparse:
        if gt is None or gt[2] is None:
            st.info("Sparse-code analysis needs a ground-truth centroid to pick a patch.")
        else:
            mask_final, _, centroid = gt
            centered = patches.select_nodule_centered_patch(dataset.coords, indices, centroid)

            X_scan = dataset.X[:, indices]
            gamma_scan = _encode_scan_patches(
                scan_id, X_scan, checkpoint.dictionary, checkpoint.n_nonzero_coefs
            )
            gamma_centered = gamma_scan[:, centered.position]

            st.plotly_chart(
                viz.sparse_code_stem_figure(
                    gamma_centered,
                    f"Sparse code — {scan_id}, patch #{centered.index} "
                    f"({np.count_nonzero(gamma_centered)} active atoms)",
                ),
                use_container_width=True,
            )

            top_atoms = sparse_coding.top_k_atoms(gamma_centered, k=3)
            if len(top_atoms) == 0:
                st.info("No active atoms for this patch.")
            else:
                st.subheader("Top active atoms")
                st.table(
                    {
                        "Atom": top_atoms,
                        "Coefficient": [round(float(gamma_centered[a]), 6) for a in top_atoms],
                    }
                )

                atom_patches = sparse_coding.atom_contribution_patches(
                    checkpoint.dictionary, gamma_centered, top_atoms
                )
                combined_patch = sum(atom_patches.values())

                original_patch = patches.reshape_patch(dataset.X[:, centered.index])
                px, py, pz = config.PATCH_SHAPE_XYZ
                x0, y0, z0 = centered.coord
                patch_gt_mask = mask_final[x0 : x0 + px, y0 : y0 + py, z0 : z0 + pz]

                similarity = sparse_coding.reconstruction_similarity(
                    combined_patch, original_patch, patch_gt_mask
                )
                if similarity is not None:
                    st.subheader("Top-3 reconstruction vs. ground-truth region")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("MAE", f"{similarity.mae:.4f}")
                    c2.metric("RMSE", f"{similarity.rmse:.4f}")
                    c3.metric("Correlation", f"{similarity.correlation:.4f}")
                    c4.metric("Intensity ratio", f"{similarity.intensity_ratio:.2%}")

                st.plotly_chart(
                    viz.atom_contribution_grid(
                        atom_patches,
                        gamma_centered,
                        combined_patch,
                        centered.coord,
                        f"Top-3 OMP atoms + combined reconstruction — {scan_id}, "
                        f"patch #{centered.index}",
                    ),
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()
