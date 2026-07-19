import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HU windowing -- matches merlin_ct_pipeline.py exactly
WINDOW_MIN = -1000.0
WINDOW_MAX = 400.0

# Matches `volume[::4, ::4, ::2]` in load_and_preprocess_ct().
# Set to (1, 1, 1) for full-resolution Grad-CAM (slower, sharper).
DOWNSAMPLE_STRIDE = (4, 4, 2)

# Confirmed: model.model.encode_image.i3_resnet.layer4
TARGET_LAYER_PATH = "encode_image.i3_resnet.layer4"
