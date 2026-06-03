import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class CTViewer:
    def __init__(self, ct_data, pred_mask=None, gt_mask=None, lung_mask=None):
        self.ct_data = ct_data
        self.pred_mask = pred_mask
        self.gt_mask = gt_mask
        self.lung_mask = lung_mask

        self.num_slices = ct_data.shape[2]

        if gt_mask is not None and gt_mask.sum() > 0:
            ggo_slices = np.where(gt_mask.sum(axis=(0, 1)) > 0)[0]
            self.slice_index = int(ggo_slices[len(ggo_slices) // 2])
        else:
            self.slice_index = self.num_slices // 2

        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 12))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.show_slice()

    def show_slice(self):
        for ax in self.axes.ravel():
            ax.clear()
            ax.axis("off")

        ct_slice = self.ct_data[:, :, self.slice_index]

        self.axes[0, 0].imshow(ct_slice.T, cmap="gray", origin="lower")
        self.axes[0, 0].set_title(f"CT Slice {self.slice_index}")

        self.axes[0, 1].imshow(ct_slice.T, cmap="gray", origin="lower")
        if self.lung_mask is not None:
            lung_slice = self.lung_mask[:, :, self.slice_index]
            self.axes[0, 1].imshow(
                np.ma.masked_where(lung_slice.T == 0, lung_slice.T),
                cmap="Blues",
                alpha=0.35,
                origin="lower"
            )
        self.axes[0, 1].set_title("Lung Mask")

        self.axes[1, 0].imshow(ct_slice.T, cmap="gray", origin="lower")
        if self.gt_mask is not None:
            gt_slice = self.gt_mask[:, :, self.slice_index]
            self.axes[1, 0].imshow(
                np.ma.masked_where(gt_slice.T == 0, gt_slice.T),
                cmap="Greens",
                alpha=0.45,
                origin="lower"
            )
        self.axes[1, 0].set_title("GT Mask - Green")

        self.axes[1, 1].imshow(ct_slice.T, cmap="gray", origin="lower")

        overlay = self.create_overlap_overlay()

        self.axes[1, 1].imshow(
            np.ma.masked_where(overlay.T == 0, overlay.T),
            cmap=self.get_overlay_cmap(),
            alpha=0.55,
            origin="lower",
            vmin=0,
            vmax=3
        )

        self.axes[1, 1].set_title("Overlay: Green=GT, Red=Pred, Yellow=Overlap")

        self.fig.suptitle(
            self.get_slice_title(),
            fontsize=13
        )

        self.fig.canvas.draw()

    def create_overlap_overlay(self):
        overlay = np.zeros(self.ct_data[:, :, self.slice_index].shape, dtype=np.uint8)

        if self.gt_mask is None and self.pred_mask is None:
            return overlay

        gt_slice = None
        pred_slice = None

        if self.gt_mask is not None:
            gt_slice = self.gt_mask[:, :, self.slice_index].astype(bool)

        if self.pred_mask is not None:
            pred_slice = self.pred_mask[:, :, self.slice_index].astype(bool)

        if gt_slice is not None:
            overlay[gt_slice] = 1

        if pred_slice is not None:
            overlay[pred_slice] = 2

        if gt_slice is not None and pred_slice is not None:
            overlap = gt_slice & pred_slice
            overlay[overlap] = 3

        return overlay

    def get_overlay_cmap(self):
        return ListedColormap([
            "black",   # 0 ignored by mask
            "green",   # 1 GT only
            "red",     # 2 prediction only
            "yellow"   # 3 overlap
        ])

    def get_slice_title(self):
        title = f"Slice {self.slice_index + 1}/{self.num_slices}"

        if self.gt_mask is not None:
            gt_voxels = int(self.gt_mask[:, :, self.slice_index].sum())
            title += f" | GT voxels: {gt_voxels}"

        if self.pred_mask is not None:
            pred_voxels = int(self.pred_mask[:, :, self.slice_index].sum())
            title += f" | Pred voxels: {pred_voxels}"

        if self.gt_mask is not None and self.pred_mask is not None:
            gt_slice = self.gt_mask[:, :, self.slice_index].astype(bool)
            pred_slice = self.pred_mask[:, :, self.slice_index].astype(bool)
            overlap = int((gt_slice & pred_slice).sum())
            title += f" | Overlap: {overlap}"

        return title

    def on_key(self, event):
        if event.key == "right":
            self.slice_index = min(self.slice_index + 1, self.num_slices - 1)

        elif event.key == "left":
            self.slice_index = max(self.slice_index - 1, 0)

        elif event.key == "up":
            self.slice_index = min(self.slice_index + 10, self.num_slices - 1)

        elif event.key == "down":
            self.slice_index = max(self.slice_index - 10, 0)

        elif event.key == "home":
            self.slice_index = 0

        elif event.key == "end":
            self.slice_index = self.num_slices - 1

        self.show_slice()