"""
TerraFlare - Dataset Analyzer & Visualization Module
Calculates dataset metrics, class imbalance metrics, quality summaries, and draws annotated ground-truth previews.
"""

import os
import cv2
import numpy as np
from PIL import Image
from dataset_validator import IMAGE_EXTENSIONS

# Predefined distinct colors for bounding box visualization (BGR format for OpenCV)
CLASS_COLORS = [
    (0, 0, 255),     # Red (Fire / Class 0)
    (0, 165, 255),   # Orange (Smoke / Class 1)
    (0, 255, 0),     # Green (Non-Fire / Class 2)
    (255, 255, 0),   # Cyan
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Yellow
    (128, 0, 128)    # Purple
]

def analyze_dataset_metrics(validation_report):
    """
    Computes class balance ratios, quality metrics, and imbalance warnings.
    """
    class_dist = validation_report.get("class_distribution", {})
    counts = [v for v in class_dist.values() if isinstance(v, (int, float))]
    
    imbalance_status = "Balanced"
    imbalance_ratio = 1.0
    imbalance_warning = None
    
    if len(counts) > 1:
        min_c = min(counts) if min(counts) > 0 else 1
        max_c = max(counts)
        imbalance_ratio = max_c / min_c
        
        if imbalance_ratio > 3.0:
            imbalance_status = "Severe Imbalance"
            imbalance_warning = f"⚠️ Dataset is severely imbalanced (max/min ratio: {imbalance_ratio:.1f}x). Consider balancing or data augmentation before training."
        elif imbalance_ratio > 1.5:
            imbalance_status = "Moderate Imbalance"
            imbalance_warning = f"⚠️ Dataset is moderately imbalanced (max/min ratio: {imbalance_ratio:.1f}x). Consider balancing/augmentation."
        else:
            imbalance_status = "Balanced"

    quality_summary = {
        "structure": "✅ Valid" if validation_report.get("valid_structure") else "❌ Invalid",
        "images": "✅ Valid" if validation_report.get("corrupted_images", 0) == 0 else f"⚠️ {validation_report.get('corrupted_images')} Corrupted",
        "labels": "✅ Valid" if validation_report.get("invalid_labels", 0) == 0 and validation_report.get("missing_labels", 0) == 0 else "⚠️ Issues Found",
        "corrupted_images": validation_report.get("corrupted_images", 0),
        "missing_labels": validation_report.get("missing_labels", 0),
        "invalid_labels": validation_report.get("invalid_labels", 0),
        "missing_files": validation_report.get("missing_files", 0),
        "class_balance_status": imbalance_status,
        "imbalance_ratio": round(imbalance_ratio, 2),
        "imbalance_warning": imbalance_warning
    }
    
    return quality_summary

def draw_yolo_ground_truth(image_path, label_path, class_names=None):
    """
    Loads image and draws ground-truth YOLO bounding boxes with class names.
    Returns RGB image array (numpy).
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        return None

    h_img, w_img, _ = img.shape
    
    if not label_path or not os.path.exists(label_path):
        return img
        
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
            
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                cid = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                
                # Convert normalized coords to pixel bounding box
                x1 = int((cx - w / 2) * w_img)
                y1 = int((cy - h / 2) * h_img)
                x2 = int((cx + w / 2) * w_img)
                y2 = int((cy + h / 2) * h_img)
                
                # Clip coordinates
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img - 1, x2), min(h_img - 1, y2)
                
                # Pick color
                color_rgb = CLASS_COLORS[cid % len(CLASS_COLORS)][::-1] # RGB
                
                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color_rgb, 2)
                
                # Label text
                label_text = class_names[cid] if class_names and 0 <= cid < len(class_names) else f"Class {cid}"
                
                # Draw text background pill
                (txt_w, txt_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, max(0, y1 - txt_h - 6)), (x1 + txt_w + 6, max(0, y1)), color_rgb, -1)
                cv2.putText(img, label_text, (x1 + 3, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            except Exception:
                continue
    except Exception:
        pass

    return img

def fetch_sample_previews(dataset_root, dataset_type="Object Detection", class_names=None, max_samples=6):
    """
    Fetches sample images with their ground-truth bounding boxes or labels.
    """
    samples = []
    
    if dataset_type == "Object Detection":
        image_label_pairs = []
        for dp, _, filenames in os.walk(dataset_root):
            for f in filenames:
                ext = os.path.splitext(f)[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    img_path = os.path.join(dp, f)
                    base_name = os.path.splitext(f)[0]
                    
                    # Find label file
                    lbl_path = None
                    # Check in labels directory matching image split
                    rel_dp = os.path.relpath(dp, dataset_root)
                    lbl_dp = rel_dp.replace("images", "labels")
                    lbl_candidate1 = os.path.join(dataset_root, lbl_dp, f"{base_name}.txt")
                    lbl_candidate2 = os.path.join(dp, f"{base_name}.txt")
                    
                    if os.path.exists(lbl_candidate1):
                        lbl_path = lbl_candidate1
                    elif os.path.exists(lbl_candidate2):
                        lbl_path = lbl_candidate2
                        
                    image_label_pairs.append((img_path, lbl_path, f))
                    if len(image_label_pairs) >= max_samples * 3:
                        break
            if len(image_label_pairs) >= max_samples * 3:
                break
                
        # Prefer samples that have label files
        valid_pairs = [p for p in image_label_pairs if p[1] is not None]
        if not valid_pairs:
            valid_pairs = image_label_pairs
            
        selected = valid_pairs[:max_samples]
        for img_path, lbl_path, fname in selected:
            annotated_img = draw_yolo_ground_truth(img_path, lbl_path, class_names)
            if annotated_img is not None:
                samples.append({
                    "filename": fname,
                    "image": annotated_img,
                    "label": "YOLO Annotations" if lbl_path else "No Label File"
                })
    else:
        # Classification
        class_samples = []
        for dp, _, filenames in os.walk(dataset_root):
            cname = os.path.basename(dp)
            for f in filenames:
                ext = os.path.splitext(f)[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    img_path = os.path.join(dp, f)
                    class_samples.append((img_path, cname, f))
                    if len(class_samples) >= max_samples * 3:
                        break
            if len(class_samples) >= max_samples * 3:
                break
                
        for img_path, cname, fname in class_samples[:max_samples]:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif len(img.shape) == 3 and img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                elif len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    continue
                samples.append({
                    "filename": fname,
                    "image": img,
                    "label": f"Class: {cname}"
                })
                
    return samples
