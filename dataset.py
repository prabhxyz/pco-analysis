# dataset.py
import os
import glob
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

# Chosen size for all frames
IMG_WIDTH = 512
IMG_HEIGHT = 512

# Classes: 0=Background, plus 12 relevant classes
CLASS_NAME_TO_ID = {
    "Iris": 1,
    "Pupil": 2,
    "Intraocular Lens": 3,
    "Slit/Incision Knife": 4,
    "Gauge": 5,
    "Spatula": 6,
    "Capsulorhexis Cystotome": 7,
    "Phacoemulsifier Tip": 8,
    "Irrigation-Aspiration": 9,
    "Lens Injector": 10,
    "Capsulorhexis Forceps": 11,
    "Katana Forceps": 12
}

def create_segmentation_mask(height, width, objects):
    """
    Returns an (height, width) numpy mask of uint8, with each pixel assigned
    to a class ID (0=Background). Polygons are drawn using cv2.fillPoly.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for obj in objects:
        class_title = obj["classTitle"]
        if class_title in CLASS_NAME_TO_ID:
            cls_id = CLASS_NAME_TO_ID[class_title]
        else:
            # Unrecognized class => treat as background
            continue

        pts = obj["points"]["exterior"]  # list of (x, y)
        polygon = np.array(pts, dtype=np.int32)
        cv2.fillPoly(mask, [polygon], color=cls_id)

    return mask

class CataractSegmentationDataset(Dataset):
    """
    Loads frames + polygon annotations from Supervisely JSON.
    Resizes each to 512x512, returns (image_tensor, mask_tensor).
    """
    def __init__(self, root_dir, transforms=None):
        super().__init__()
        self.root_dir = root_dir
        self.transforms = transforms
        self.samples = []

        ann_dir = os.path.join(root_dir, "Annotations", "Images-and-Supervisely-Annotations")
        case_folders = glob.glob(os.path.join(ann_dir, "case_*"))
        for cfold in case_folders:
            img_folder = os.path.join(cfold, "img")
            ann_folder = os.path.join(cfold, "ann")
            pngs = glob.glob(os.path.join(img_folder, "*.png"))
            for png_path in pngs:
                base_name = os.path.basename(png_path)  # e.g. "case_5000_01.png"
                json_path = os.path.join(ann_folder, base_name + ".json")
                if os.path.isfile(json_path):
                    self.samples.append((png_path, json_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, ann_path = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Load JSON
        with open(ann_path, "r") as f:
            data = json.load(f)

        height = data["size"]["height"]
        width = data["size"]["width"]
        objects = data["objects"]
        mask_np = create_segmentation_mask(height, width, objects)

        # Resize
        image = image.resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
        mask = Image.fromarray(mask_np).resize((IMG_WIDTH, IMG_HEIGHT), Image.NEAREST)

        if self.transforms:
            image = self.transforms(image)

        mask = np.array(mask, dtype=np.uint8)
        mask = torch.from_numpy(mask).long()  # shape [H, W]

        return image, mask
