# LC-KSVD Training

## Step 1 — Configuration (`config.py`)

Every downstream module imports its constants from `config.py`. The key decisions baked in here are:

**HU windowing.** `HU_MIN = -1000`, `HU_MAX = 200` targets the lung parenchyma window. Values outside this range are clipped, then linearly rescaled to `[0, 1]` in `window_and_normalise()`.

**Resampling.** `TARGET_SPACING_MM = 1.5` collapses the variability of raw CT acquisitions (e.g. 0.75 × 0.75 × 1.5 mm or 1 × 1 × 3 mm) into a common isotropic grid before any patch is extracted, so every patch covers the same physical volume (~48 mm³).

**Patch geometry.** `PATCH_SIZE = 32` means each patch is a 32 × 32 × 32 voxel cube, giving `N_FEATURES = 32768` — the input dimensionality to the dictionary learning algorithm.

**Overlap threshold.** `MIN_OVERLAP_RATIO = 0.05` (5 %) — kept intentionally low to capture small lesions like ground-glass opacities that might not fill many voxels of a 32³ patch.

**Class taxonomy.** `CLASS_ORDER = ["normal", "2b", "2c", "2d"]` defines the four rows of the label matrix `H` (index 0 = normal, 1 = atelectasis/consolidation, 2 = ground-glass, 3 = nodules/masses).

---

## Step 2 — Metadata and label loading (`data_loader.py`)

`MetadataRegistry.__init__()` opens `dataset_3.json` and builds `self._volume_index`: a dict mapping each `volume_name` → `{f_idx: category_str}`. It strips file extensions via `_stem()` and casts F-index strings to integers.

`LabelRegistry._build_label_index()` iterates the same JSON and promotes finding-level categories to volume-level binary labels: a volume is positive for `"2c"` if *any* of its findings has `categories[f] == "2c"`. This is how `get_normal_volume_names()` is defined — volumes where every category flag is 0.

Design decision: normal volumes are defined by *absence of any recognised finding*, not by an explicit label in the CSV. This makes the normal class emergent from the metadata structure.

---

## Step 3 — Patch extraction (`patch_extractor.py`)

`extract_unified(split)` orchestrates both phases. It first checks whether `patches/unified_{split}.npz` already exists and short-circuits if so.

### Phase 1 — Normal patches

`collect_normal_patches()` iterates `normal_ids` and calls `sample_normal_patches()` on each loaded volume. Patches are sampled uniformly at random from the entire in-bounds voxel space — no mask constraint — because normal lung tissue is homogeneous. Each is labelled `CLASS_ORDER.index("normal") = 0`.

The valid centre range is computed by `_valid_centre_range()` as `(h, dim - h)` per axis, where `h = PATCH_SIZE // 2 = 16`, ensuring the patch never exceeds the volume boundary.

`_extract_patch()` slices `volume[x0:x1, y0:y1, z0:z1]` and returns `None` for out-of-bounds centres.

### Phase 2 — Abnormal patches

`collect_abnormal_patches()` follows a more elaborate path for each scan:

1. `ScanLoader.load()` calls `load_volume()` + `resample_volume()` + `window_and_normalise()`, then `load_mask()` + `resample_mask()`. Crucially, `resample_mask()` is passed `target_shape = vol_rs.shape` (the already-resampled volume shape) rather than relying on the mask's own header spacing — this avoids shape mismatches.

2. `_build_finding_masks()` collapses the 4D mask `[F, H, W, D]` into per-category binary masks by OR-ing together all finding slices that share the same category string. Findings whose category is not in `CLASS_ORDER` are silently skipped.

3. A `union_mask` is built by OR-ing all per-category masks — this defines the sampling region.

4. For each candidate centre drawn from `foreground_coords`, `_majority_class()` iterates every per-category mask and returns the one with the highest `_overlap_ratio()` (fraction of patch voxels that are foreground). If no category clears `MIN_OVERLAP_RATIO`, the patch is discarded.

Design decision: majority-class labelling instead of first-match correctly handles overlapping lesion regions — a patch that straddles a `2b` and `2d` region is labelled by whichever mask covers more of it.

Design decision: the per-scan budget `N_POSITIVE_PATCHES_PER_SCAN = 30` is flat for both normal and abnormal scans. An earlier design divided the budget by the number of scans × categories, which produced ~1 patch/scan on real datasets. The flat budget means class balance is governed by the dataset split, not the code.

### Matrix assembly

`build_unified_patch_matrix()` concatenates all patches and stacks them column-wise into `X` of shape `(N_FEATURES, n_patches)`, and fills `H` as a 1D int64 vector of class indices. This column-major convention matches what LC-KSVD2 expects.

---

## Step 4 — Loading and normalisation (`train.py`)

`load_unified_patch_matrix()` reads the `.npz` file and returns `X, H`.

`normalise_columns()` computes `norms = ||X[:, j]||₂` for each patch column and divides. Zero-norm patches (all-black, all-air, or outside the HU window after clipping) are identified by `zero_mask = norms < 1e-10` and their corresponding columns — and label entries — are dropped before training. This prevents division-by-zero and removes uninformative patches.

The same normalisation is applied to the validation set loaded as `X_val, H_val` via a second call to `load_unified_patch_matrix(split="val")`.

---

## Step 5 — Dictionary size adaptation (`train.py`)

Before constructing the model, the config is copied and guarded:

```python
max_atoms = max(8, X_norm.shape[1] // 2)
cfg["n_components"]    = min(cfg["n_components"], max_atoms)
cfg["n_nonzero_coefs"] = min(cfg["n_nonzero_coefs"], max(1, cfg["n_components"] // 2))
```

This prevents a crash when the number of patches is too small to support 128 dictionary atoms (e.g. on a tiny dev subset). It's a defensive measure for small datasets, not expected to trigger in production.

---

## Step 6 — One-hot conversion and model fit (`train.py`)

`label_binarize(H, classes=list(range(len(CLASS_ORDER))))` converts the 1D integer vector to a `(n_patches, n_classes)` binary matrix, which is then transposed to `H_onehot` of shape `(n_classes, n_patches)` — the format `LCKSVD.fit()` expects.

Design decision: the integer `H` is kept separate from `H_onehot`. Only the one-hot form is passed to `model.fit()`; all evaluation uses the integer form. This avoids a subtle bug where passing a one-hot matrix into `evaluate()` would make `y_true == c` comparisons fail (since `H` would be 2D).

`LCKSVD(**cfg)` is instantiated with `variant="lcksvd2"`, `alpha=4.0` (label-consistency weight), `beta=2.0` (classifier weight), `n_iter=50`, and `n_iter_init=20` warm-start K-SVD iterations. The model learns dictionary `D` and linear classifier `W` jointly.

---

## Step 7 — Evaluation (`train.py → evaluate()`)

`model.transform(X_norm)` produces sparse codes `Gamma` of shape `(n_components, n_patches)`. The classification scores are `scores = W @ Gamma` giving `(n_classes, n_patches)`.

Predicted labels are `np.argmax(scores, axis=0)`. Ground truth is the integer `H` vector. Per-class metrics are computed in a one-vs-rest fashion:

- `roc_auc_score(y_true_bin, score_c)` — uses the raw continuous score, not the argmax prediction, so it measures ranking quality not just accuracy.
- `average_precision_score(y_true_bin, score_c)` — area under the precision-recall curve, more informative than AUROC on imbalanced classes.
- `f1_score(y_true, y_pred, average="macro")` — harmonic mean of precision/recall across all four classes equally weighted.

`ValueError` is caught for classes that have no positive examples in a split (can occur in small val sets), recording `nan` rather than crashing.

---

## Step 8 — Saving (`train.py`)

`pickle.dump()` writes a payload dict to `outputs/models/unified_lcksvd2.pkl` containing not just `model` and `class_order` but full provenance: `train_metrics`, `val_metrics`, `lcksvd_config`, `patch_size`, `target_spacing`, `hu_window`, and `training_time_s`. This means the saved artifact is self-describing — inference or analysis code never needs to re-read `config.py` to know what preprocessing was applied.