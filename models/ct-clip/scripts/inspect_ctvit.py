import sys
from pathlib import Path

ROOT = Path("/home/chest_ct/code/models/ct-clip")
sys.path.insert(0, str(ROOT / "CT_CLIP"))
sys.path.insert(0, str(ROOT / "transformer_maskgit"))

from transformer_maskgit import CTViT

model = CTViT(
    dim=512,
    codebook_size=8192,
    image_size=480,
    patch_size=20,
    temporal_patch_size=10,
    spatial_depth=4,
    temporal_depth=4,
    dim_head=32,
    heads=8,
)

for name, module in model.named_modules():
    print(f"{name:70s} -> {module.__class__.__name__}")