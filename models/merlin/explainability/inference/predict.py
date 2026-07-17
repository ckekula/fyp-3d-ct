"""
predict.py

Runs a forward pass WITHOUT torch.no_grad() -- gradcam3d.py needs to
backprop through this call. This is the one place people usually
accidentally break Grad-CAM by copy-pasting the no_grad() block from
merlin_ct_pipeline.py's process_one_volume().
"""


class MerlinPredictor:
    def __init__(self, model, image_embedding_only=True):
        self.model = model
        self.image_embedding_only = image_embedding_only

    def forward(self, image_tensor, text=None):
        if self.image_embedding_only:
            return self.model(image_tensor)
        assert text is not None, "text is required when image_embedding_only=False"
        return self.model(image_tensor, [text])


if __name__ == "__main__":
    # STEP 3 sanity check -- see run order in chat.
    import sys
    from explainability.inference.load_model import load_merlin
    from explainability.inference.preprocess import preprocess_ct

    model, _ = load_merlin(image_embedding_only=True)
    predictor = MerlinPredictor(model, image_embedding_only=True)

    tensor, ds_shape, affine, full_shape = preprocess_ct(sys.argv[1])
    output = predictor.forward(tensor)
    print("Output type:", type(output))
    print("Output shape:", output.shape)
