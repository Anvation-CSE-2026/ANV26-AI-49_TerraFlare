"""
TerraFlare - CNN Model Evaluator & Confusion Matrix Generator
Evaluates trained MobileNetV3-Small best_model.pth on test dataset and generates real confusion matrix, per-class metrics, false-positive analysis, and evaluation_report.json.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

from training.cnn_trainer import ClassificationDataset, collect_classification_samples
from training.model_registry import update_model_evaluation

def evaluate_cnn_experiment(exp_config, dataset_metadata):
    """
    Evaluates best_model.pth on the held-out test split of dataset_metadata.
    Calculates overall and per-class accuracy, precision, recall, F1, renders confusion matrix, and extracts false positives.
    """
    exp_dir = exp_config.get("experiment_dir")
    if not exp_dir or not os.path.exists(exp_dir):
        exp_dir = os.path.dirname(exp_config.get("best_model_path", ""))
        
    best_model_path = exp_config.get("best_model_path")
    dataset_path = dataset_metadata.get("dataset_path")

    # Load class mapping
    cmap_path = os.path.join(exp_dir, "class_mapping.json")
    if os.path.exists(cmap_path):
        with open(cmap_path, "r", encoding="utf-8") as f:
            cmap_data = json.load(f)
        classes = cmap_data["classes"]
        class2idx = cmap_data["class2idx"]
    else:
        classes = exp_config.get("classes", [])
        class2idx = {c: i for i, c in enumerate(classes)}

    idx2class = {i: c for c, i in class2idx.items()}

    # Collect test samples (fallback to val if test empty)
    splits_samples = collect_classification_samples(dataset_path, class2idx)
    test_samples = splits_samples["test"] if splits_samples["test"] else splits_samples["val"]
    eval_split_used = "TEST" if splits_samples["test"] else "VAL"
    if not test_samples:
        test_samples = splits_samples["train"]
        eval_split_used = "TRAIN_FALLBACK"

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    eval_transform = transforms.Compose([
        transforms.Resize((exp_config.get("image_size", 224), exp_config.get("image_size", 224))),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = ClassificationDataset(test_samples, transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Reconstruct MobileNetV3-Small model
    model = mobilenet_v3_small(weights=None)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, len(classes))

    try:
        state_dict = torch.load(best_model_path, map_location=device, weights_only=False)
    except Exception:
        state_dict = torch.load(best_model_path, map_location=device)
        
    if isinstance(state_dict, dict) and "model" in state_dict:
        sub = state_dict["model"]
        if hasattr(sub, "state_dict"):
            state_dict = sub.state_dict()
        elif isinstance(sub, dict):
            state_dict = sub

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    all_confidences = []
    false_positives = []

    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(test_loader):
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, 1)

            preds_np = preds.cpu().numpy()
            labels_np = labels.numpy()
            confs_np = confs.cpu().numpy()

            all_preds.extend(preds_np)
            all_targets.extend(labels_np)
            all_confidences.extend(confs_np)

            # Record false positive / misclassification instances
            batch_start = batch_idx * 32
            for i in range(len(preds_np)):
                pred_c = preds_np[i]
                target_c = labels_np[i]
                if pred_c != target_c:
                    sample_img_path, _ = test_samples[batch_start + i]
                    false_positives.append({
                        "image_path": sample_img_path,
                        "filename": os.path.basename(sample_img_path),
                        "actual_class": idx2class.get(target_c, f"Class {target_c}"),
                        "predicted_class": idx2class.get(pred_c, f"Class {pred_c}"),
                        "confidence": round(float(confs_np[i]), 4)
                    })

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = float(accuracy_score(all_targets, all_preds))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
    p_per_class, r_per_class, f1_per_class, _ = precision_recall_fscore_support(all_targets, all_preds, average=None, zero_division=0)

    per_class_metrics = {}
    for i, cname in enumerate(classes):
        per_class_metrics[cname] = {
            "precision": float(p_per_class[i]) if i < len(p_per_class) else 0.0,
            "recall": float(r_per_class[i]) if i < len(r_per_class) else 0.0,
            "f1": float(f1_per_class[i]) if i < len(f1_per_class) else 0.0
        }

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(len(classes))))

    # Render confusion matrix plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=classes, yticklabels=classes)
    plt.title(f"MobileNetV3-Small Confusion Matrix ({eval_split_used})")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Ground Truth Class")
    plt.tight_layout()
    cm_path = os.path.join(exp_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()

    conf_stats = {
        "mean_confidence": round(float(np.mean(all_confidences)), 4) if len(all_confidences) > 0 else 0.0,
        "min_confidence": round(float(np.min(all_confidences)), 4) if len(all_confidences) > 0 else 0.0,
        "max_confidence": round(float(np.max(all_confidences)), 4) if len(all_confidences) > 0 else 0.0
    }

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(float(p_macro), 4),
        "recall": round(float(r_macro), 4),
        "f1": round(float(f1_macro), 4),
        "test_samples": len(test_samples),
        "eval_split": eval_split_used,
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_path": cm_path,
        "false_positives": false_positives[:10],
        "confidence_stats": conf_stats,
        "classes": classes
    }

    # Save evaluation_report.json inside experiment directory
    report_dict = {
        "model_id": exp_config.get("registered_model_id", "CNN_MODEL"),
        "experiment_id": exp_config.get("experiment_id"),
        "experiment_name": exp_config.get("experiment_name"),
        "architecture": "MobileNetV3-Small",
        "dataset_id": dataset_metadata.get("dataset_id"),
        "dataset_name": dataset_metadata.get("dataset_name"),
        "eval_split": eval_split_used,
        "test_samples": len(test_samples),
        "metrics": metrics,
        "evaluation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    report_path = os.path.join(exp_dir, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=4)

    # Update Model Registry
    update_model_evaluation(
        model_id=exp_config.get("experiment_id"),
        metrics=metrics,
        test_samples_count=len(test_samples)
    )

    return metrics
