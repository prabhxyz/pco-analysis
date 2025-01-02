# model.py
import torch
import torch.nn as nn
import torchvision.models.segmentation as seg_models

# We have background + 12 classes = 13 total
NUM_CLASSES = 13

class SimpleSegmentationModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        # Use FCN-ResNet50 from torchvision
        self.model = seg_models.fcn_resnet50(pretrained=True)
        # Replace final layer
        self.model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        return self.model(x)  # returns dict with "out" key
