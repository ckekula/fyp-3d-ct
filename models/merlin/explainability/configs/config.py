import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HU windowing — matches pipelineall.py exactly
WINDOW_MIN = -1000.0
WINDOW_MAX = 400.0

# Confirmed from pipelineall.py: model.model.encode_image.i3_resnet.layer4
TARGET_LAYER = "model.encode_image.i3_resnet.layer4"