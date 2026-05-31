import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from pipeline import load_nifti_file, apply_hu_window

class CTViewer:
    def __init__(self, ct_data, pred_mask=None, gt_mask=None):
        """
        Interactive 3D CT viewer using Matplotlib.
        :param ct_data: 3D numpy array of the CT scan
        :param pred_mask: Optional 3D numpy array of the predicted segmentation mask
        :param gt_mask: Optional 3D numpy array of the ground truth mask
        """
        self.ct_data = np.asarray(ct_data)
        self.pred_mask = self._normalize_mask(pred_mask, self.ct_data.shape) if pred_mask is not None else None
        self.gt_mask = self._normalize_mask(gt_mask, self.ct_data.shape) if gt_mask is not None else None
        
        # Determine the number of slices
        self.num_slices = self.ct_data.shape[2]
        self.current_slice = self.num_slices // 2
        
        # Setup the figure and axes
        if self.pred_mask is not None and self.gt_mask is not None:
            self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 7))
            self.is_split = True
        else:
            self.fig, self.ax1 = plt.subplots(figsize=(8, 8))
            self.is_split = False
            
        plt.subplots_adjust(bottom=0.25)
        
        # Initial display for axis 1 (Prediction or CT only)
        self.im_ct1 = self.ax1.imshow(self.ct_data[:, :, self.current_slice], cmap='gray', vmin=0, vmax=1)
        if self.pred_mask is not None:
            p_slice = self.pred_mask[:, :, self.current_slice]
            p_overlay = np.ma.masked_where(p_slice == 0, p_slice)
            self.im_pred = self.ax1.imshow(p_overlay, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
        self.ax1.axis('off')
        
        # Initial display for axis 2 (Ground Truth)
        if self.is_split:
            self.im_ct2 = self.ax2.imshow(self.ct_data[:, :, self.current_slice], cmap='gray', vmin=0, vmax=1)
            gt_slice = self.gt_mask[:, :, self.current_slice]
            gt_overlay = np.ma.masked_where(gt_slice == 0, gt_slice)
            self.im_gt = self.ax2.imshow(gt_overlay, cmap='Greens', alpha=0.5, vmin=0, vmax=1)
            self.ax2.axis('off')
            
        self.update_titles()
        
        # Setup the slider
        axcolor = 'lightgoldenrodyellow'
        ax_slice = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor=axcolor)
        self.slider = Slider(ax_slice, 'Slice', 0, self.num_slices - 1, valinit=self.current_slice, valstep=1)
        
        # Connect the update function to the slider
        self.slider.on_changed(self.update)
        
    def _normalize_mask(self, mask, reference_shape):
        mask = np.asarray(mask)
        if mask.ndim == 4 and mask.shape[0] <= 8:
            mask = np.any(mask != 0, axis=0)
        elif mask.ndim == 4 and mask.shape[-1] == 1:
            mask = np.squeeze(mask, axis=-1)

        if mask.ndim != 3:
            raise ValueError(
                f"Mask must be 3D after normalization, got shape {mask.shape}. "
                "Provide a 3D binary mask or a 4D channel-first mask."
            )

        if tuple(mask.shape) != tuple(reference_shape):
            raise ValueError(
                f"Mask shape {mask.shape} does not match CT shape {reference_shape}."
            )

        return mask.astype(bool)

    def update_titles(self):
        if self.is_split:
            self.ax1.set_title(f'Predicted Mask - Slice {self.current_slice}/{self.num_slices-1}')
            self.ax2.set_title(f'Ground Truth - Slice {self.current_slice}/{self.num_slices-1}')
        else:
            self.ax1.set_title(f'Slice {self.current_slice}/{self.num_slices-1}')

    def update(self, val):
        """
        Updates the image when the slider changes.
        """
        self.current_slice = int(self.slider.val)
        
        # Update CT slice
        self.im_ct1.set_data(self.ct_data[:, :, self.current_slice])
        if self.pred_mask is not None:
            p_slice = self.pred_mask[:, :, self.current_slice]
            p_overlay = np.ma.masked_where(p_slice == 0, p_slice)
            self.im_pred.set_data(p_overlay)
            
        if self.is_split:
            self.im_ct2.set_data(self.ct_data[:, :, self.current_slice])
            gt_slice = self.gt_mask[:, :, self.current_slice]
            gt_overlay = np.ma.masked_where(gt_slice == 0, gt_slice)
            self.im_gt.set_data(gt_overlay)
            
        self.update_titles()
        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    print("Welcome to the Interactive CT Viewer!")
    print("Please run this script with a path to a NIfTI file.")
    print("Example usage:")
    print("ct_data = apply_hu_window(load_nifti_file('path/to/scan.nii.gz'))")
    print("viewer = CTViewer(ct_data)")
    print("plt.show()")
