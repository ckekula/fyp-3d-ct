import torch
import merlin
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model = merlin.Merlin().eval().to(device)

print("Text features...")
text_labels = ["Lung Nodule", "Lung opacity", "Consolidation", "Atelectasis"]
with torch.no_grad():
    text_features = model.model.encode_text(text_labels)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
print(text_features.shape)

print("Image features...")
# Dummy image
image = torch.randn(1, 1, 16, 224, 224).to(device) # (B, C, D, H, W)

# We need to hook layer4 to get the dense feature map
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.model.encode_image.i3_resnet.layer4.register_forward_hook(get_activation('layer4'))

with torch.no_grad():
    image_features, ehr_features = model.model.encode_image(image)

layer4_out = activation['layer4'] # [1, 2048, D', H', W']
print("layer4_out shape:", layer4_out.shape)

dense_contrastive = model.model.encode_image.i3_resnet.contrastive_head(layer4_out) # [1, 512, D', H', W']
dense_contrastive = dense_contrastive / dense_contrastive.norm(dim=1, keepdim=True)
print("dense_contrastive shape:", dense_contrastive.shape)

# Cosine similarity for "Lung opacity" (idx 1)
similarity = torch.einsum('b c d h w, c -> b d h w', dense_contrastive, text_features[1])
print("similarity shape:", similarity.shape)
print("Min:", similarity.min().item(), "Max:", similarity.max().item())

