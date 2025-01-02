# train.py
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm  # <-- for progress bar

from dataset import CataractSegmentationDataset
from model import SimpleSegmentationModel, NUM_CLASSES

def train_model(dataset_path, epochs=5, batch_size=4, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on device:", device)

    # Speed up convolution operations
    cudnn.benchmark = True

    # Data transforms for images
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225))
    ])

    # Build dataset & dataloader
    dataset = CataractSegmentationDataset(dataset_path, transforms=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = SimpleSegmentationModel(num_classes=NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Automatic mixed precision
    scaler = torch.cuda.amp.GradScaler()

    model.train()
    for epoch in range(1, epochs+1):
        epoch_loss = 0.0
        # Wrap our dataloader in tqdm for a progress bar
        progress_bar = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch}/{epochs}")

        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                outputs = model(images)["out"]
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            current_loss = loss.item()

            # Update tqdm bar description or postfix
            progress_bar.set_postfix({"loss": f"{current_loss:.4f}"})

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch}/{epochs}] - Average Loss: {avg_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), "cataract_seg_model.pth")
    print("Training complete. Model saved to cataract_seg_model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="datasets/Cataract-1k/segmentation")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train_model(args.dataset_path, args.epochs, args.batch_size, args.lr)
