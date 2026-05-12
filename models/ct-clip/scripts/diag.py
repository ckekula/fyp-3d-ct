# diag.py  –  paste and run this, share the output
import inspect
from ct_clip import CTCLIP
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
import torch

# Build a minimal model (no checkpoint needed)
ie = CTViT(dim=512, codebook_size=8192, image_size=480, patch_size=30,
           temporal_patch_size=15, spatial_depth=4, temporal_depth=4,
           dim_head=32, heads=8)
tok = BertTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized",
                                    do_lower_case=True)
te  = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
m   = CTCLIP(image_encoder=ie, text_encoder=te, dim_image=294912,
             dim_text=768, dim_latent=512, extra_latent_projection=False,
             use_mlm=False, downsample_image_embeds=False,
             use_all_token_embeds=False)

# 1. Print CTCLIP.forward() signature
print("=== CTCLIP.forward() signature ===")
print(inspect.signature(m.forward))
print()

# 2. Print the source of forward()
print("=== CTCLIP.forward() source ===")
print(inspect.getsource(m.forward))
print()

# 3. Print all top-level attributes
print("=== Top-level attributes ===")
for name, mod in m.named_children():
    print(f"  {name}: {type(mod).__name__}")