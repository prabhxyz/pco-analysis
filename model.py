# model.py
import torch.nn as nn
import torchvision.models.segmentation as seg_models

# We have 1 background + 12 classes = 13 total
NUM_CLASSES = 13

class FastSegmentationModel(nn.Module):
    """
    A DeeplabV3 model with a MobileNetV3-Large backbone, pretrained on COCO.
    We replace the classifier head to output (NUM_CLASSES) channels.
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        # Create the model
        self.model = seg_models.deeplabv3_mobilenet_v3_large(pretrained=True)
        # Replace final classifier layer
        # The default classifier is a DeepLabHead with out_channels=21 (for COCO 21 classes).
        # We will override the final conv layer.
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        return self.model(x)  # returns dict with 'out' key
