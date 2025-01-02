# train.py
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import GPUtil  # so we can detect a T4 if available
from tqdm import tqdm

from dataset import CataractSegmentationDataset
from model import FastSegmentationModel, NUM_CLASSES

def detect_t4():
    """
    Returns True if an NVIDIA Tesla T4 is detected via GPUtil, else False.
    If no GPU is found, returns False.
    """
    gpus = GPUtil.getGPUs()
    for g in gpus:
        if "T4" in g.name:
            return True
    return False

def train_model(dataset_path, epochs=None, batch_size=8, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # If no epochs specified, pick automatically
    if epochs is None:
        if device.type == "cuda" and detect_t4():
            epochs = 10  # T4 is decent
            print("Detected T4 GPU: using 10 epochs.")
        elif device.type == "cuda":
            epochs = 8
            print("Using a non-T4 GPU: using 8 epochs.")
        else:
            epochs = 2
            print("CPU detected: using 2 epochs for demonstration.")

    # For max speed
    cudnn.benchmark = True
    # (Optional) cudnn.fastest might help sometimes
    # torch.backends.cudnn.deterministic = False

    # Transforms
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225))
    ])

    dataset = CataractSegmentationDataset(dataset_path, transforms=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = FastSegmentationModel(num_classes=NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    print(f"Training for {epochs} epochs, batch size={batch_size}, LR={lr}")
    model.train()
    for epoch in range(1, epochs+1):
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", total=len(dataloader))
        for images, masks in progress_bar:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                outputs = model(images)["out"]
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch}, Average Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), "cataract_seg_model.pth")
    print("Training completed, model saved: cataract_seg_model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="datasets/Cataract-1k/segmentation")
    parser.add_argument("--epochs", type=int, default=None, help="Leave as None to auto-pick based on device")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train_model(args.dataset_path, args.epochs, args.batch_size, args.lr)
