import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from cataract_seg_dataset import CataractSegDataset, NUM_SEG_CLASSES
from cataract_phase_dataset import CataractPhaseDataset
from models import LightweightSegModel, PhaseRecognitionNet

def get_seg_transforms():
    return A.Compose([
        A.Resize(512,512),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])

def get_phase_transforms():
    return A.Compose([
        A.Resize(224,224),
        A.RandomBrightnessContrast(p=0.4),
        A.Rotate(limit=15, p=0.5),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])

def train_segmentation(model, train_loader, val_loader, device, epochs=10, lr=1e-4):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler(enabled=(device.type=="cuda"))
    model.train()
    for epoch in range(1, epochs+1):
        model.train()
        running_loss=0
        pbar = tqdm(train_loader, desc=f"Seg Epoch {epoch}/{epochs}")
        for imgs, masks in pbar:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=(device.type=="cuda")):
                out = model(imgs)["out"]
                loss = criterion(out, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = running_loss/len(train_loader)
        val_loss = validate_segmentation(model, val_loader, device, criterion)
        scheduler.step(val_loss)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

def validate_segmentation(model, val_loader, device, criterion):
    model.eval()
    losses=[]
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            out = model(imgs)["out"]
            loss = criterion(out, masks)
            losses.append(loss.item())
    return sum(losses)/len(losses)

def train_phase(model, train_loader, val_loader, device, epochs=10, lr=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=(device.type=="cuda"))

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0
        pbar = tqdm(train_loader, desc=f"Phase Epoch {epoch}/{epochs}")
        for frames, label in pbar:
            frames = frames.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                out = model(frames)
                loss = criterion(out, label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = running_loss/len(train_loader)
        val_loss = validate_phase(model, val_loader, device, criterion)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

def validate_phase(model, val_loader, device, criterion):
    model.eval()
    losses=[]
    with torch.no_grad():
        for frames,label in val_loader:
            frames,label = frames.to(device), label.to(device)
            out = model(frames)
            loss = criterion(out, label)
            losses.append(loss.item())
    return sum(losses)/len(losses)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seg_data_root", type=str, default="datasets/Cataract-1k/segmentation")
    parser.add_argument("--phase_data_root", type=str, default="datasets/Cataract-1k/phase")
    parser.add_argument("--seg_epochs", type=int, default=10)
    parser.add_argument("--phase_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    cudnn.benchmark = True

    # If we want to silence the "max number of worker" warning or avoid overhead, set e.g. num_workers=1
    num_workers=1

    # 1) Seg dataset
    seg_transform = get_seg_transforms()
    seg_dataset = CataractSegDataset(root_dir=args.seg_data_root, transform=seg_transform)
    val_size = int(0.2 * len(seg_dataset))
    train_size = len(seg_dataset) - val_size
    seg_train, seg_val = random_split(seg_dataset, [train_size, val_size])
    seg_train_loader = DataLoader(seg_train, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    seg_val_loader   = DataLoader(seg_val,   batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    # 2) Train segmentation
    seg_model = LightweightSegModel(num_classes=NUM_SEG_CLASSES, use_pretrained=True).to(device)
    print("Training Segmentation Model...")
    train_segmentation(seg_model, seg_train_loader, seg_val_loader, device, epochs=args.seg_epochs, lr=args.lr)
    torch.save(seg_model.state_dict(), "lightweight_seg.pth")
    print("Saved lightweight_seg.pth")

    # 3) Phase dataset
    phase_transform = get_phase_transforms()
    phase_dataset = CataractPhaseDataset(root_dir=args.phase_data_root, transform=phase_transform)
    val_size_p = int(0.2 * len(phase_dataset))
    train_size_p = len(phase_dataset) - val_size_p
    phase_train, phase_val = random_split(phase_dataset, [train_size_p, val_size_p])
    phase_train_loader = DataLoader(phase_train, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    phase_val_loader   = DataLoader(phase_val,   batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    # 4) Train phase recognition
    num_phases = len(phase_dataset.phase_label_map)  # from the dataset
    phase_model = PhaseRecognitionNet(num_phases=num_phases, use_pretrained=True).to(device)
    print("Training Phase Recognition Model...")
    train_phase(phase_model, phase_train_loader, phase_val_loader, device, epochs=args.phase_epochs, lr=args.lr)
    torch.save(phase_model.state_dict(), "phase_recognition.pth")
    print("Saved phase_recognition.pth")

if __name__ == "__main__":
    main()