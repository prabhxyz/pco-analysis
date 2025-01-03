import os
import glob
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A

INSTRUMENT_CLASSES = {
    "Background": 0,
    "Iris": 1,
    "Pupil": 2,
    "IntraocularLens": 3,
    "SlitKnife": 4,
    "Gauge": 5,
    "Spatula": 6,
    "CapsulorhexisCystotome": 7,
    "PhacoTip": 8,
    "IrrigationAspiration": 9,
    "LensInjector": 10,
    "CapsulorhexisForceps": 11,
    "KatanaForceps": 12
}
NUM_SEG_CLASSES = len(INSTRUMENT_CLASSES)

def create_seg_mask(height, width, objects):
    """
    Converts polygon annotations into an HxW mask of class IDs.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for obj in objects:
        class_name = obj["classTitle"]
        cid = INSTRUMENT_CLASSES.get(class_name, 0)
        pts = obj["points"]["exterior"]
        poly = np.array(pts, dtype=np.int32)
        cv2.fillPoly(mask, [poly], color=cid)
    return mask

class CataractSegDataset(Dataset):
    """
    Dataset for segmentation that loads PNG frames and polygon annotations.
    """
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        ann_dir = os.path.join(root_dir, "Annotations", "Images-and-Supervisely-Annotations")
        case_dirs = glob.glob(os.path.join(ann_dir, "case_*"))
        for cdir in case_dirs:
            img_folder = os.path.join(cdir, "img")
            ann_folder = os.path.join(cdir, "ann")
            pngs = glob.glob(os.path.join(img_folder, "*.png"))
            for p in pngs:
                filename = os.path.basename(p)
                jpath = os.path.join(ann_folder, filename + ".json")
                if os.path.isfile(jpath):
                    self.samples.append((p, jpath))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, ann_path = self.samples[idx]
        image_pil = Image.open(img_path).convert("RGB")
        image_np = np.array(image_pil, dtype=np.uint8)

        with open(ann_path, "r") as f:
            data = json.load(f)
        h = data["size"]["height"]
        w = data["size"]["width"]
        objs = data["objects"]
        mask_np = create_seg_mask(h, w, objs)

        if self.transform:
            transformed = self.transform(image=image_np, mask=mask_np)
            image_tensor = transformed["image"]
            mask_tensor = transformed["mask"].long()
        else:
            # Minimal fallback
            image_tensor = torch.from_numpy(image_np.transpose(2,0,1)).float() / 255.
            mask_tensor = torch.from_numpy(mask_np).long()

        return image_tensor, mask_tensor
