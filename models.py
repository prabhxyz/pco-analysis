# models.py
import torch
import torch.nn as nn
import torchvision.models.segmentation as segm_models
from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

class LightweightSegModel(nn.Module):
    def __init__(self, num_classes=13, use_pretrained=True):
        super().__init__()
        if use_pretrained:
            weights = DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
            self.model = segm_models.deeplabv3_mobilenet_v3_large(weights=weights)
        else:
            self.model = segm_models.deeplabv3_mobilenet_v3_large(weights=None)

        self.model.classifier[4] = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        return self.model(x)

class PhaseRecognitionNet(nn.Module):
    def __init__(self, num_phases=12, use_pretrained=True):
        super().__init__()
        if use_pretrained:
            # e.g. MobileNet_V3_Large_Weights.IMAGENET1K_V1
            weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
            backbone = mobilenet_v3_large(weights=weights)
        else:
            backbone = mobilenet_v3_large(weights=None)

        # replace final layer
        # Usually backbone.classifier[3] => Linear(1280, 1000)
        backbone.classifier[3] = nn.Linear(1280, num_phases)
        self.base = backbone

    def forward(self, x):
        return self.base(x)
