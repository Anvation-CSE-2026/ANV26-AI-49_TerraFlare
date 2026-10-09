"""
TerraFlare - Phase 3B CNN Verifier Training Pipeline
Trains a 3-class MobileNetV3-Small classifier (FIRE, SMOKE, NON_FIRE) for 2nd-stage false positive rejection.

Dataset Structure Expected:
dataset/
├── train/
│   ├── FIRE/
│   ├── SMOKE/
│   └── NON_FIRE/
└── val/
    ├── FIRE/
    ├── SMOKE/
    └── NON_FIRE/
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

CLASSES = ["NON_FIRE", "SMOKE", "FIRE"]

def build_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def train_cnn_verifier(data_dir, output_weights="weights/cnn_verifier.pt", epochs=15, batch_size=32, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] Starting CNN Verifier Training on device: {device}")

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if not os.path.exists(train_dir):
        print(f"[!] Error: Training directory '{train_dir}' not found.")
        print("Please structure your dataset into dataset/train/[FIRE, SMOKE, NON_FIRE] and dataset/val/[FIRE, SMOKE, NON_FIRE].")
        sys.exit(1)

    train_transform, val_transform = build_transforms()

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir if os.path.exists(val_dir) else train_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize MobileNetV3-Small with pretrained ImageNet weights
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, len(CLASSES))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    best_val_loss = float('inf')
    os.makedirs(os.path.dirname(output_weights), exist_ok=True)

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct.double() / total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)
                val_total += labels.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct.double() / val_total
        elapsed = time.time() - start_time

        scheduler.step(epoch_val_loss)

        print(f"Epoch {epoch+1:02d}/{epochs:02d} [{elapsed:.1f}s] - Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")

        # Save best checkpoint
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), output_weights)
            print(f"    --> Saved best model checkpoint to '{output_weights}'")

    print(f"[*] Training complete. Best weights saved at: '{output_weights}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TerraFlare 3-Class MobileNetV3 CNN Verifier")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Path to dataset root containing train/ and val/")
    parser.add_argument("--output", type=str, default="weights/cnn_verifier.pt", help="Path to save trained weights")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    train_cnn_verifier(args.data_dir, args.output, args.epochs, args.batch_size, args.lr)
