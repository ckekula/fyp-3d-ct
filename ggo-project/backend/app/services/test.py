import sys
sys.path.append(".")

from app.services.preprocessing import full_pipeline
import matplotlib.pyplot as plt

patient_path = "data/LIDC-IDRI/LIDC-IDRI-0001"

result = full_pipeline(patient_path, noise_method="gaussian")

# Show 3 stages side by side for one slice
mid = result["resampled_hu"].shape[0] // 2

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(result["windowed"][mid], cmap="gray")
axes[0].set_title("Lung Windowed")

axes[1].imshow(result["normalized"][mid], cmap="gray")
axes[1].set_title("Normalized")

axes[2].imshow(result["denoised"][mid], cmap="gray")
axes[2].set_title("Denoised (Gaussian)")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()

print(f"Final denoised shape: {result['denoised'].shape}")