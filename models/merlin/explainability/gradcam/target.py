"""
target.py — defines what scalar score Grad-CAM should backprop from.
Pick whichever matches how you constructed the model in load_model.py.
"""

import torch


class ImageEmbeddingNormTarget:
    """Generic saliency: what drove the overall image embedding."""
    def __call__(self, output):
        emb = output[0] if isinstance(output, (tuple, list)) else output
        return emb.norm(dim=-1).sum()


class PhenotypeTarget:
    """Requires model built with ImageEmbedding=False, PhenotypeCls not needed
    since default Merlin() already returns phenotype logits as output[1]."""
    def __init__(self, class_idx):
        self.class_idx = class_idx

    def __call__(self, output):
        logits = output[1]
        return logits[:, self.class_idx].sum()


class TextSimilarityTarget:
    """
    Text-driven Grad-CAM: 'why does this region look like <text>?'
    model must be built with ImageEmbedding=False (full contrastive model)
    since it needs encode_text.
    """
    def __init__(self, model, text):
        with torch.no_grad():
            text_emb = model.model.encode_text([text])
            self.text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    def __call__(self, output):
        image_emb = output[0]
        image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
        return (image_emb * self.text_emb).sum()