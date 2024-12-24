"""
cataract_dataset.py

Contains:
  1. Cataract1KDataset class to load and simulate the 'cataract-1k' dataset.
  2. get_cataract_dataloaders function to create train/val DataLoaders.
"""

import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class Cataract1KDataset(Dataset):
    """
    A placeholder for the actual 'cataract-1k' dataset loader.
    Assumes a structure like:
        root/
          images/
            frame_0.jpg
            frame_1.jpg
            ...
          masks/
            frame_0.png
            frame_1.png
          ...
    Also includes textual prompts or annotations in a real scenario.
    """

    def __init__(self, root_dir, transform=None, split='train'):
        """
        Args:
            root_dir (str): Path to the dataset root folder.
            transform (callable, optional): Transformation for images.
            split (str): 'train' or 'val' (placeholder usage).
        """
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.split = split

        # Mock a list of frames for demonstration
        self.samples = []
        for i in range(30):  # Adjust number as needed
            self.samples.append({
                'image_path': os.path.join(self.root_dir, f'images/frame_{i}.jpg'),
                'mask_path': os.path.join(self.root_dir, f'masks/frame_{i}.png'),
                'text_prompt': f"Surgical note describing frame {i}..."
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # In real usage, you would load the actual images like:
        #   image = cv2.imread(sample['image_path'])[..., ::-1]
        #   mask = cv2.imread(sample['mask_path'], 0)
        # For demonstration, use random data:
        image = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
        mask = (np.random.rand(224, 224) > 0.5).astype(np.uint8)

        text_prompt = sample['text_prompt']

        if self.transform:
            image = self.transform(image)
        else:
            # Convert to Torch tensor if no transform
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Convert mask to torch
        mask = torch.from_numpy(mask).long()

        return {
            'image': image,
            'mask': mask,
            'text_prompt': text_prompt
        }


def get_cataract_dataloaders(root_dir='./cataract-1k', batch_size=4):
    """
    Returns train and val data loaders for 'cataract-1k' dataset.
    Uses a basic transform for demonstration.
    """
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
    ])

    train_ds = Cataract1KDataset(root_dir, transform=transform, split='train')
    val_ds = Cataract1KDataset(root_dir, transform=transform, split='val')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader