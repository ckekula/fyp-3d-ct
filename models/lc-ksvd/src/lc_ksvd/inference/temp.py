import numpy as np
import matplotlib.pyplot as plt
from lc_ksvd.inference.inference import run_inference
from lc_ksvd.config import CLASS_ORDER, NORMAL_CLASS_IDX, PATCH_SIZE
from lc_ksvd.data_loader.metadata_registry import MetadataRegistry
from lc_ksvd.data_loader.scan_loader import ScanLoader

target_scan = "train_11150_b_2"
split = "train"

# --- Run the full pipeline: preprocess -> dense patch grid -> sparse-code -> classify -> per-voxel map ---
# stride=PATCH_SIZE -> non-overlapping grid, so any real grid/tissue misalignment is
# visible directly instead of being hidden under overlapping boxes at the default
# ABNORMAL_PATCH_STRIDE=4 (3x overlap per axis).
result = run_inference(scan_id=target_scan, stride=PATCH_SIZE)

volume = result["volume"]              # preprocessed [-1,1] volume, [H,W,D]
coords = result["coords"]              # (n_patches, 3) dense-grid patch origins
pred_labels = result["pred_labels"]    # (n_patches,) predicted class idx per patch
label_volume = result["label_volume"]  # (H,W,D) per-voxel predicted class idx

print("Predicted class counts:",
      {CLASS_ORDER[i]: int((pred_labels == i).sum()) for i in range(len(CLASS_ORDER))})

# --- Load ground truth mask on the same (resampled) grid, for comparison ---
metadata = MetadataRegistry(split=split)
scan_gt = ScanLoader(metadata).load(target_scan)
gt_mask = scan_gt["mask"]  # [F,H,W,D] or None
gt = gt_mask[0].astype(bool) if gt_mask is not None else None  # finding 0 ("2d" nodule) only

# --- Pick the slice with the most predicted-abnormal voxels ---
abnormal_mask = label_volume != NORMAL_CLASS_IDX
per_slice = abnormal_mask.sum(axis=(0, 1))
cz = int(np.argmax(per_slice)) if per_slice.sum() > 0 else volume.shape[2] // 2

fig, ax = plt.subplots(1, 1, figsize=(9, 9))
# volume is already normalised to [-1,1] over the whole scan (background/
# outside-lung == -1.0 exactly), but per-slice tissue values rarely exceed
# ~-0.7..-0.5 (aerated lung) -- fixing vmin/vmax (instead of letting imshow
# auto-scale to this slice's own min/max) stops real tissue from being
# crushed toward black.
ax.imshow(volume[:, :, cz].T, cmap="gray", origin="lower", vmin=-1.0, vmax=-0.5)

# Predicted abnormality overlay (per class, semi-transparent)
class_colors = plt.get_cmap("tab10", len(CLASS_ORDER))
pred_overlay = np.zeros((*volume[:, :, cz].T.shape, 4))
for cls_idx in range(len(CLASS_ORDER)):
    if cls_idx == NORMAL_CLASS_IDX:
        continue
    cls_mask = (label_volume[:, :, cz] == cls_idx).T
    if cls_mask.any():
        pred_overlay[cls_mask] = (*class_colors(cls_idx)[:3], 0.45)
ax.imshow(pred_overlay, origin="lower")

# Ground truth overlay (red outline, drawn on top so it's distinguishable from prediction fill)
if gt is not None:
    gt_slice = gt[:, :, cz].T
    if gt_slice.any():
        ax.contour(gt_slice, colors="red", linewidths=1.5, levels=[0.5])

# Patch grid boxes at this slice (green) — with stride=PATCH_SIZE, z0 == cz - (cz % PATCH_SIZE[2])
# is the only offset that can contain cz, so this is now a single non-overlapping mesh.
_px, _py, _pz = PATCH_SIZE
slice_boxes = [(x0, y0) for (x0, y0, z0) in coords if z0 <= cz < z0 + _pz]
print(f"n boxes drawn at z={cz}: {len(slice_boxes)}")

for (x0, y0) in slice_boxes:
    rect = plt.Rectangle((y0, x0), _py, _px,
                          linewidth=0.8, edgecolor="lime", facecolor="none", alpha=0.9)
    ax.add_patch(rect)

handles = [
    plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=class_colors(i),
               markersize=10, label=f"pred: {CLASS_ORDER[i]}")
    for i in range(len(CLASS_ORDER)) if i != NORMAL_CLASS_IDX
]
handles.append(plt.Line2D([0], [0], color="red", lw=1.5, label="ground truth"))
handles.append(plt.Line2D([0], [0], color="lime", lw=1, label="patch grid"))
ax.legend(handles=handles, loc="lower right", fontsize=8)

ax.set_title(f"{target_scan} — slice z={cz}\n(fill=prediction, red=GT, green=patch grid)")
ax.axis("off")
plt.tight_layout()
plt.show()
