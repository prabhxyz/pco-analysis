"""
segmentation_tracking.py

Contains:
  1. SimpleSegmentationModel - a DeepLabV3-based model with ResNet50 backbone.
  2. Train/inference functions for segmentation.
  3. Optical flow calculation using OpenCV Farneback method.
  4. Region-based tracking function that finds centroid of largest segmented region.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import cv2
import numpy as np
from tqdm import tqdm


class SimpleSegmentationModel(nn.Module):
    """
    A lightweight segmentation model based on torchvision's DeepLabV3 with ResNet50.
    In real usage, you might fine-tune YOLOv8 or a lighter model for real-time constraints.
    """

    def __init__(self, num_classes=2):
        super().__init__()
        # Load a pretrained DeepLabV3-ResNet50 model
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(
            weights=torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
        )
        # Modify the classifier to output num_classes
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the model.
        x: (B, 3, H, W)
        returns: (B, num_classes, H, W)
        """
        return self.model(x)['out']


def train_segmentation_model(seg_model, dataloader, device='cuda'):
    """
    Basic training loop for the segmentation model.
    Demonstrates an example of how to train; for real usage,
    you'd want multiple epochs, scheduling, and better evaluation.
    """
    seg_model.train()
    seg_model.to(device)
    optimizer = optim.Adam(seg_model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1):  # For demo, just 1 epoch
        print(f"[Segmentation Training] Epoch {epoch+1}")
        for batch in tqdm(dataloader):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            optimizer.zero_grad()
            outputs = seg_model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()

            # Just break after one batch for demonstration speed
            break


@torch.no_grad()
def run_segmentation_inference(seg_model, frames, device='cuda'):
    """
    Run segmentation inference on a batch of frames.
    frames: (B, 3, H, W)
    returns: seg_preds (B, H, W) - argmax segmentation output
    """
    seg_model.eval()
    seg_model.to(device)
    frames = frames.to(device)
    outputs = seg_model(frames)
    preds = torch.argmax(outputs, dim=1)  # (B, H, W)
    return preds


def compute_optical_flow(prev_frame, next_frame):
    """
    Compute optical flow using OpenCV's Farneback method.
    Args:
      prev_frame: (H, W, 3) or (H, W) - single frame
      next_frame: (H, W, 3) or (H, W) - single frame
    Returns:
      flow: (H, W, 2) float32
    """
    # Convert to grayscale for Farneback
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if prev_frame.ndim == 3 else prev_frame
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY) if next_frame.ndim == 3 else next_frame

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray,
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    return flow


def region_based_tracking(seg_preds):
    """
    Simple region-based tracking: for each segmentation mask, find
    the centroid of the largest region and store it.
    seg_preds: (B, H, W) integer mask
    Returns: list of (cx, cy) for each image in the batch.
    """
    tracked_info = []
    seg_preds_np = seg_preds.cpu().numpy().astype(np.uint8)

    for b in range(seg_preds_np.shape[0]):
        mask = seg_preds_np[b]
        # FindContours expects a single-channel image
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            tracked_info.append((0, 0))
            continue
        # largest by area
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M['m00'] == 0:
            tracked_info.append((0, 0))
        else:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            tracked_info.append((cx, cy))

    return tracked_info
