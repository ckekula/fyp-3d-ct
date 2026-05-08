import numpy as np
from scipy.ndimage import zoom, gaussian_filter, median_filter
import pydicom
import os


def load_volume(patient_path: str):
    """
    Walk patient directory, load all DICOM slices, return
    sorted pixel array stack and the slice objects.
    """
    dcm_files = []
    for dirpath, _, filenames in os.walk(patient_path):
        for f in filenames:
            if f.endswith(".dcm"):
                dcm_files.append(os.path.join(dirpath, f))

    slices = [pydicom.dcmread(p) for p in dcm_files]

    # Filter to consistent 512x512 shape
    target_shape = (512, 512)
    slices = [s for s in slices if s.pixel_array.shape == target_shape]

    # Sort by InstanceNumber
    slices.sort(key=lambda x: int(x.InstanceNumber))

    return slices


def to_hounsfield(slices: list) -> np.ndarray:
    """
    Convert raw pixel values to Hounsfield Units (HU)
    using RescaleSlope and RescaleIntercept from DICOM headers.
    """
    volume = np.stack([s.pixel_array.astype(np.float32) for s in slices])

    for i, s in enumerate(slices):
        slope = float(getattr(s, "RescaleSlope", 1))
        intercept = float(getattr(s, "RescaleIntercept", 0))
        volume[i] = volume[i] * slope + intercept

    return volume


def apply_lung_window(volume: np.ndarray,
                      window_level: float = -600,
                      window_width: float = 1500) -> np.ndarray:
    """
    Clip HU volume to lung window and normalize to [0, 1].
    Default WL=-600, WW=1500 is standard lung window.
    Returns float32 array in [0.0, 1.0].
    """
    lower = window_level - window_width / 2   # -1350
    upper = window_level + window_width / 2   #  750

    windowed = np.clip(volume, lower, upper)
    windowed = (windowed - lower) / (upper - lower)
    return windowed.astype(np.float32)


def get_voxel_spacing(slices: list) -> np.ndarray:
    """
    Extract voxel spacing [slice_thickness, row_spacing, col_spacing] in mm.
    """
    pixel_spacing = slices[0].PixelSpacing          # [row, col] in mm
    slice_thickness = float(slices[0].SliceThickness)

    spacing = np.array([
        slice_thickness,
        float(pixel_spacing[0]),
        float(pixel_spacing[1])
    ])
    return spacing


def resample_volume(volume: np.ndarray,
                    spacing: np.ndarray,
                    target_spacing: list = [1.0, 1.0, 1.0]) -> tuple:
    """
    Resample volume to uniform voxel spacing (default 1x1x1 mm).
    Uses linear interpolation (order=1).
    Returns (resampled_volume, new_spacing).
    """
    target_spacing = np.array(target_spacing)
    zoom_factors = spacing / target_spacing

    resampled = zoom(volume, zoom_factors, order=1)
    return resampled, target_spacing


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """
    Z-score normalization across the full volume.
    Makes intensity values zero-mean and unit-variance.
    """
    mean = np.mean(volume)
    std = np.std(volume)
    if std == 0:
        return volume - mean
    return ((volume - mean) / std).astype(np.float32)


def apply_noise_reduction(volume: np.ndarray,
                           method: str = "gaussian",
                           sigma: float = 0.5) -> np.ndarray:
    """
    Apply noise reduction filter to reduce quantum noise while
    preserving GGO boundaries.

    Methods:
        "gaussian" — smooth noise, may slightly blur edges
        "median"   — better at preserving hard edges, removes salt/pepper
    """
    if method == "gaussian":
        return gaussian_filter(volume, sigma=sigma).astype(np.float32)
    elif method == "median":
        # scipy median_filter works slice-by-slice for speed
        filtered = np.zeros_like(volume)
        size = max(3, int(sigma * 2 + 1))
        for i in range(volume.shape[0]):
            filtered[i] = median_filter(volume[i], size=size)
        return filtered.astype(np.float32)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'gaussian' or 'median'.")


def full_pipeline(patient_path: str,
                  noise_method: str = "gaussian") -> dict:
    """
    Run the complete preprocessing pipeline for one patient.

    Returns a dict with all intermediate and final volumes:
        - raw_hu        : HU volume before any processing
        - spacing       : original voxel spacing [z, y, x] in mm
        - resampled_hu  : HU volume after resampling to 1x1x1 mm
        - windowed      : lung-windowed, normalized to [0,1]
        - normalized    : z-score normalized (for algorithms)
        - denoised      : after noise reduction filter
    """
    print("[1/5] Loading DICOM slices...")
    slices = load_volume(patient_path)
    print(f"      Loaded {len(slices)} slices")

    print("[2/5] Converting to Hounsfield Units...")
    raw_hu = to_hounsfield(slices)
    spacing = get_voxel_spacing(slices)
    print(f"      Spacing: {spacing} mm  |  HU range: {raw_hu.min():.0f} to {raw_hu.max():.0f}")

    print("[3/5] Resampling to 1x1x1 mm...")
    resampled_hu, new_spacing = resample_volume(raw_hu, spacing)
    print(f"      Shape before: {raw_hu.shape} → after: {resampled_hu.shape}")

    print("[4/5] Applying lung window...")
    windowed = apply_lung_window(resampled_hu)

    print("[5/5] Normalizing and denoising...")
    normalized = normalize_volume(windowed)
    denoised = apply_noise_reduction(normalized, method=noise_method)

    print("✅ Preprocessing complete.")

    return {
        "raw_hu":       raw_hu,
        "spacing":      spacing,
        "resampled_hu": resampled_hu,
        "windowed":     windowed,
        "normalized":   normalized,
        "denoised":     denoised,
    }