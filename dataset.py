# dataset.py
import os
import glob
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

# Adjust if needed
IMG_WIDTH = 512
IMG_HEIGHT = 512

# Classes
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
    Creates a mask of shape (height, width), dtype=uint8,
    with integer class IDs for each polygon.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for obj in objects:
        class_title = obj["classTitle"]
        if class_title in CLASS_NAME_TO_ID:
            cls_id = CLASS_NAME_TO_ID[class_title]
        else:
            continue

        pts = obj["points"]["exterior"]  # [ [x,y], [x,y], ...]
        polygon = np.array(pts, dtype=np.int32)
        cv2.fillPoly(mask, [polygon], color=cls_id)

    return mask

class CataractSegmentationDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        super().__init__()
        self.root_dir = root_dir
        self.transforms = transforms
        self.samples = []

        ann_base = os.path.join(root_dir, "Annotations", "Images-and-Supervisely-Annotations")
        case_folders = glob.glob(os.path.join(ann_base, "case_*"))
        for case_folder in case_folders:
            img_folder = os.path.join(case_folder, "img")
            ann_folder = os.path.join(case_folder, "ann")
            png_paths = glob.glob(os.path.join(img_folder, "*.png"))
            for png_path in png_paths:
                base_name = os.path.basename(png_path)
                json_path = os.path.join(ann_folder, base_name + ".json")
                if os.path.isfile(json_path):
                    self.samples.append((png_path, json_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, ann_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        with open(ann_path, "r") as f:
            data = json.load(f)

        h = data["size"]["height"]
        w = data["size"]["width"]
        objects = data["objects"]
        mask_np = create_segmentation_mask(h, w, objects)

        # resize
        image = image.resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
        mask = Image.fromarray(mask_np).resize((IMG_WIDTH, IMG_HEIGHT), Image.NEAREST)

        if self.transforms:
            image = self.transforms(image)

        mask = np.array(mask, dtype=np.uint8)
        mask = torch.from_numpy(mask).long()

        return image, mask