# K-SVD Patch Explorer

Streamlit app for exploring K-SVD dictionary-learned CT patches: patch grids,
ground-truth abnormality overlays, and OMP sparse-code decomposition.

## Setup

```bash
pip install -r requirements.txt
```

## Data

Place the following files under `data/` (paths are fixed, see `src/config.py`):

```
data/
  ksvd_checkpoint.npz        # D, Gamma, n_components, n_nonzero_coefs, ...
  unified_train_2d.npz       # X, scan_ids, coords
  dataset_filtered.json      # optional — per-scan finding metadata
  segmentations/
    <scan_id>.nii.gz         # ground-truth mask, one file per scan
```

### Registering a scan's original voxel spacing

Ground-truth masks are resampled to the patch coordinate space using each
scan's original voxel spacing (mm), which isn't stored in the patch files
themselves. Register it in `src/config.py`:

```python
SCAN_ORIGINAL_SPACING_MM = {
    "train_7880_a_1": (0.6738281, 0.6738281, 1.5),
    "your_new_scan_id": (x_spacing, y_spacing, z_spacing),
}
```

A scan without a registered spacing (or without a matching `.nii.gz` file)
still works for the "Patch grid" tab — the ground-truth-dependent tabs are
disabled with a clear message instead of guessing a value.

## Run

```bash
streamlit run app.py
```

## Project structure

```
app.py                  Streamlit UI (thin orchestration layer)
src/config.py            Paths, patch geometry, resampling constants, spacing table
src/data_io.py           Cached loaders for checkpoints, patches, JSON metadata, masks
src/segmentation.py      Mask resampling / center-crop-or-pad alignment
src/patches.py           Patch-space reconstruction, centroid, nodule-centered patch selection
src/sparse_coding.py     OMP encoding (via `reppi`) and atom-contribution decomposition
src/viz.py               Plotly figure builders
```
