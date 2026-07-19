"""
load_model.py

NOTE: Merlin() downloads/loads its pretrained weights internally
(merlin-vlm package / HuggingFace). There is no local checkpoint
file to load -- do not add torch.load() here.
"""

from merlin import Merlin
from explainability.configs.config import DEVICE, TARGET_LAYER_PATH


class MerlinLoader:
    def __init__(self, image_embedding_only=True):
        # image_embedding_only=True  -> forward(image) returns embedding tensor directly
        # image_embedding_only=False -> forward(image, text) returns
        #   (contrastive_img_emb, phenotype_logits, contrastive_text_emb)
        self.device = DEVICE
        self.image_embedding_only = image_embedding_only
        self.model = None

    def load(self):
        print("Loading Merlin...")
        model = Merlin(ImageEmbedding=self.image_embedding_only)
        model.to(self.device)
        model.eval()
        self.model = model
        print(f"Merlin loaded on {self.device}")
        return model

    def get_target_layer(self):
        layer = self.model.model  # -> model.model.encode_image...
        for m in TARGET_LAYER_PATH.split("."):
            layer = getattr(layer, m)
        return layer


def load_merlin(image_embedding_only=True):
    loader = MerlinLoader(image_embedding_only=image_embedding_only)
    model = loader.load()
    target_layer = loader.get_target_layer()
    return model, target_layer


if __name__ == "__main__":
    # STEP 1 sanity check -- see run order in chat.
    model, target = load_merlin()
    print("Target layer:", target)
