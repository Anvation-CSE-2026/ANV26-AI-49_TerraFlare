"""
TerraFlare - CNN MobileNetV3 Trainer Module
Handles PyTorch MobileNetV3-Small transfer learning, class weighting, data augmentation, epoch training loops, and experiment saving.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from PIL import Image
from datetime import datetime

from training.experiment_manager import ensure_experiments_dir
from dataset_validator import IMAGE_EXTENSIONS

BASE_CNN_EXPERIMENTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "CNN")

def get_next_cnn_experiment_id():
    """Generates next unique CNN experiment ID (e.g., EXP_001, EXP_002)."""
    os.makedirs(BASE_CNN_EXPERIMENTS_DIR, exist_ok=True)
    existing_dirs = [d for d in os.listdir(BASE_CNN_EXPERIMENTS_DIR) if os.path.isdir(os.path.join(BASE_CNN_EXPERIMENTS_DIR, d))]
    
    max_id = 0
    for folder in existing_dirs:
        if folder.startswith("EXP_"):
            try:
                num = int(folder.split("_")[1])
                if num > max_id:
                    max_id = num
            except (IndexError, ValueError):
                pass
                
    return f"EXP_{max_id + 1:03d}"

class ClassificationDataset(Dataset):
    """PyTorch Dataset for Classification directory layout."""
    def __init__(self, samples, transform=None):
        self.samples = samples  # List of (img_path, class_idx)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        try:
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
            if self.transform:
                img_rgb = self.transform(img_rgb)
            return img_rgb, class_idx
        except Exception as e:
            # Fallback for corrupt image during training: return black image tensor
            dummy = torch.zeros((3, 224, 224), dtype=torch.float32)
            return dummy, class_idx

def collect_classification_samples(dataset_root, class2idx):
    """
    Scans dataset_root for image samples grouped by split (train, val, test).
    Returns dict of split -> list of (img_path, class_idx).
    """
    splits_samples = {"train": [], "val": [], "test": []}
    
    # Check if dataset has train/val/test folders
    subdirs = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d)) and not d.startswith('.')]
    known_splits = [d for d in subdirs if d.lower() in ['train', 'val', 'valid', 'validation', 'test']]
    
    if known_splits:
        for sdir in known_splits:
            split_key = "train" if sdir.lower() == "train" else "val" if sdir.lower() in ["val", "valid", "validation"] else "test"
            split_path = os.path.join(dataset_root, sdir)
            for cname in os.listdir(split_path):
                if cname in class2idx:
                    cdir = os.path.join(split_path, cname)
                    if os.path.isdir(cdir):
                        for dp, _, files in os.walk(cdir):
                            for f in files:
                                if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                                    splits_samples[split_key].append((os.path.join(dp, f), class2idx[cname]))
    else:
        # Single root folder containing class folders
        for cname in os.listdir(dataset_root):
            if cname in class2idx:
                cdir = os.path.join(dataset_root, cname)
                if os.path.isdir(cdir):
                    for dp, _, files in os.walk(cdir):
                        for f in files:
                            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                                splits_samples["train"].append((os.path.join(dp, f), class2idx[cname]))

    return splits_samples

def train_cnn_mobilenetv3(
    dataset_metadata,
    experiment_name,
    epochs=20,
    batch_size=32,
    learning_rate=0.001,
    image_size=224,
    device_choice="Auto",
    use_class_weighting=True,
    progress_callback=None
):
    """
    Executes actual PyTorch transfer learning training of MobileNetV3-Small.
    Saves best_model.pth, last_model.pth, class_mapping.json, and training_history.json.
    """
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    exp_id = get_next_cnn_experiment_id()
    exp_dir = os.path.join(BASE_CNN_EXPERIMENTS_DIR, exp_id)
    os.makedirs(exp_dir, exist_ok=True)

    dataset_path = dataset_metadata.get("dataset_path") or dataset_metadata.get("dataset_dir")
    if not dataset_path or not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path '{dataset_path}' not found on disk. Please select or upload a valid dataset.")

    classes = dataset_metadata.get("classes", [])

    if not classes:
        raise ValueError("Selected dataset contains no detected class labels.")

    # Create class mappings
    classes = sorted(list(set(classes)))
    class2idx = {cname: i for i, cname in enumerate(classes)}
    idx2class = {i: cname for i, cname in enumerate(classes)}

    class_mapping = {
        "classes": classes,
        "class2idx": class2idx,
        "idx2class": {str(k): v for k, v in idx2class.items()}
    }

    # Save class_mapping.json
    with open(os.path.join(exp_dir, "class_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, indent=4)

    # Collect samples
    splits_samples = collect_classification_samples(dataset_path, class2idx)
    train_samples = splits_samples["train"]
    val_samples = splits_samples["val"] if splits_samples["val"] else train_samples[::5] # fallback 20% val if no val set
    test_samples = splits_samples["test"] if splits_samples["test"] else val_samples

    if not train_samples:
        raise ValueError("No training image samples found in dataset directory.")

    # Resolve device
    if device_choice.upper() == "AUTO":
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    elif device_choice.upper() == "CUDA":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_choice.upper() == "MPS":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # Define Data Preprocessing & Augmentation
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ClassificationDataset(train_samples, transform=train_transform)
    val_dataset = ClassificationDataset(val_samples, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Class Weighting Loss Calculation
    class_weights_tensor = None
    if use_class_weighting:
        class_counts = [0] * len(classes)
        for _, c_idx in train_samples:
            class_counts[c_idx] += 1
        total_samples = len(train_samples)
        weights = [total_samples / (len(classes) * max(1, count)) for count in class_counts]
        class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Build MobileNetV3-Small Model with Transfer Learning
    try:
        model = mobilenet_v3_small(pretrained=True)
    except Exception:
        try:
            model = mobilenet_v3_small(weights="DEFAULT")
        except Exception:
            model = mobilenet_v3_small()

    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, len(classes))
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    best_val_acc = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        if progress_callback:
            progress_callback(f"Training Epoch {epoch}/{epochs} on {device.type.upper()}...")

        # Train Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data).item()
            total_train += imgs.size(0)

        epoch_train_loss = running_loss / max(1, total_train)
        epoch_train_acc = correct_train / max(1, total_train)

        # Val Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        val_preds_all = []
        val_labels_all = []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data).item()
                total_val += imgs.size(0)
                val_preds_all.extend(preds.cpu().numpy())
                val_labels_all.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / max(1, total_val)
        epoch_val_acc = correct_val / max(1, total_val)

        history_entry = {
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 4),
            "train_acc": round(epoch_train_acc, 4),
            "val_loss": round(epoch_val_loss, 4),
            "val_acc": round(epoch_val_acc, 4)
        }
        history.append(history_entry)

        # Checkpoint Saving
        best_model_path = os.path.join(exp_dir, "best_model.pth")
        last_model_path = os.path.join(exp_dir, "last_model.pth")

        torch.save(model.state_dict(), last_model_path)
        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            best_val_loss = epoch_val_loss
            best_val_preds = list(val_preds_all)
            best_val_labels = list(val_labels_all)
            torch.save(model.state_dict(), best_model_path)

    # Save training_history.json
    with open(os.path.join(exp_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

    # Compute final metrics using sklearn
    from sklearn.metrics import precision_recall_fscore_support
    if 'best_val_preds' in locals() and best_val_preds:
        p, r, f1, _ = precision_recall_fscore_support(best_val_labels, best_val_preds, average='macro', zero_division=0)
    else:
        p, r, f1 = 0.0, 0.0, 0.0

    cnn_metrics = {
        "accuracy": round(best_val_acc, 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4)
    }

    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    config_dict = {
        "experiment_id": exp_id,
        "experiment_name": experiment_name,
        "architecture": "MobileNetV3-Small",
        "dataset_id": dataset_metadata.get("dataset_id"),
        "dataset_name": dataset_metadata.get("dataset_name"),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "image_size": image_size,
        "device": str(device),
        "use_class_weighting": use_class_weighting,
        "classes": classes,
        "num_classes": len(classes),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "training_start_time": start_time,
        "training_end_time": end_time,
        "status": "SUCCESS",
        "best_model_path": os.path.join(exp_dir, "best_model.pth"),
        "last_model_path": os.path.join(exp_dir, "last_model.pth"),
        "metrics": cnn_metrics,
        "history": history
    }

    from training.model_registry import register_trained_model

    reg_entry = register_trained_model(
        experiment_id=exp_id,
        experiment_name=experiment_name,
        dataset_id=dataset_metadata.get("dataset_id"),
        architecture="MobileNetV3-Small",
        best_weights_path=os.path.join(exp_dir, "best_model.pth"),
        metrics=cnn_metrics,
        classes=classes
    )
    config_dict["registered_model_id"] = reg_entry["model_id"]

    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4)

    return config_dict
