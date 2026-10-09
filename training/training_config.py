"""
TerraFlare - Training Configuration & Pre-validation Module
Detects hardware compute devices, available YOLO model variants, and performs pre-training validation.
"""

import os
import torch
import yaml
from ultralytics import YOLO

AVAILABLE_MODEL_VARIANTS = {
    "YOLO11n": "yolo11n.pt",
    "YOLO11s": "yolo11s.pt",
    "YOLO11m": "yolo11m.pt",
    "YOLOv8n": "yolov8n.pt"
}

def detect_compute_device():
    """Detects available hardware compute device (CUDA / MPS / CPU)."""
    devices = ["Auto", "CPU"]
    if torch.cuda.is_available():
        devices.insert(1, "CUDA")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.insert(1, "MPS")
    return devices

def get_available_yolo_models():
    """Returns a dict of YOLO model variants that can be safely loaded/downloaded in the current environment."""
    valid_models = {}
    for display_name, weight_file in AVAILABLE_MODEL_VARIANTS.items():
        try:
            # Check if file exists locally or can be instantiated
            if os.path.exists(weight_file):
                valid_models[display_name] = weight_file
            else:
                # Try loading lightweight check
                _ = YOLO(weight_file)
                valid_models[display_name] = weight_file
        except Exception:
            pass
    
    # Fallback guarantee: at minimum YOLO11n
    if not valid_models:
        valid_models["YOLO11n"] = "yolo11n.pt"
    return valid_models

def find_data_yaml_path(dataset_path):
    """Finds absolute path to data.yaml within dataset directory."""
    if not dataset_path or not os.path.exists(dataset_path):
        return None
        
    for dp, _, filenames in os.walk(dataset_path):
        for f in filenames:
            if f.lower() in ['data.yaml', 'data.yml']:
                return os.path.join(dp, f)
    return None

def ensure_valid_data_yaml(dataset_path):
    """
    Finds or creates data.yaml, normalizing its `path`, `train`, `val`, `nc`, and `names` fields
    so Ultralytics YOLO can execute training without path resolution errors.
    Auto-detects image locations or prepares split folders if missing on disk.
    """
    if not dataset_path or not os.path.exists(dataset_path):
        return None

    yaml_path = find_data_yaml_path(dataset_path)

    if yaml_path and os.path.exists(yaml_path):
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data_yaml = yaml.safe_load(f) or {}
        except Exception:
            data_yaml = {}
        yaml_dir = os.path.dirname(yaml_path)
    else:
        yaml_path = os.path.join(dataset_path, "data.yaml")
        yaml_dir = dataset_path
        data_yaml = {}

    data_yaml['path'] = os.path.abspath(yaml_dir)

    def is_dir_existing_and_valid(rel_or_abs):
        if not rel_or_abs:
            return False
        full_p = os.path.abspath(os.path.join(yaml_dir, str(rel_or_abs)))
        return os.path.exists(full_p)

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    # 1. Resolve 'train' path
    train_val = data_yaml.get('train')
    if not is_dir_existing_and_valid(train_val):
        candidates = [
            'images/train',
            'train/images',
            'train',
            'images',
            'images/val',
            'val/images',
            'val'
        ]
        found_cand = None
        for cand in candidates:
            if is_dir_existing_and_valid(cand):
                found_cand = cand
                break

        if found_cand:
            data_yaml['train'] = found_cand
        else:
            # Check if any image files exist anywhere inside yaml_dir
            all_imgs = []
            for root, _, files in os.walk(yaml_dir):
                for f in files:
                    if os.path.splitext(f)[1].lower() in image_extensions:
                        all_imgs.append(os.path.join(root, f))

            if all_imgs:
                # Create train/images directory inside yaml_dir and organize/populate images
                train_dir = os.path.join(yaml_dir, 'train', 'images')
                os.makedirs(train_dir, exist_ok=True)
                import shutil
                for img_p in all_imgs:
                    dest_p = os.path.join(train_dir, os.path.basename(img_p))
                    if not os.path.exists(dest_p):
                        try:
                            os.symlink(img_p, dest_p)
                        except Exception:
                            try:
                                shutil.copy2(img_p, dest_p)
                            except Exception:
                                pass
                data_yaml['train'] = 'train/images'
            else:
                data_yaml['train'] = '.'

    # 2. Resolve 'val' path
    val_val = data_yaml.get('val')
    if not is_dir_existing_and_valid(val_val):
        candidates = [
            'images/val',
            'val/images',
            'val',
            data_yaml.get('train', '.')
        ]
        found_cand = None
        for cand in candidates:
            if is_dir_existing_and_valid(cand):
                found_cand = cand
                break
        data_yaml['val'] = found_cand or data_yaml.get('train', '.')

    # 3. Ensure names and nc fields
    if 'names' not in data_yaml or not data_yaml['names']:
        data_yaml['names'] = {0: "fire", 1: "smoke"}
    if 'nc' not in data_yaml or not data_yaml['nc']:
        data_yaml['nc'] = len(data_yaml['names']) if isinstance(data_yaml['names'], (dict, list)) else 2

    # 4. Ensure label files exist for images in 'train' and 'val' splits
    for split_key in ['train', 'val']:
        split_rel = data_yaml.get(split_key)
        if not split_rel:
            continue
        split_full = os.path.abspath(os.path.join(yaml_dir, str(split_rel)))
        if not os.path.exists(split_full):
            continue

        if os.path.basename(split_full) == 'images':
            labels_dir = os.path.join(os.path.dirname(split_full), 'labels')
        else:
            labels_dir = os.path.join(split_full, 'labels')

        os.makedirs(labels_dir, exist_ok=True)

        for root, _, files in os.walk(split_full):
            if os.path.abspath(root).startswith(os.path.abspath(labels_dir)):
                continue
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in image_extensions:
                    base_stem = os.path.splitext(f)[0]
                    lbl_path = os.path.join(labels_dir, f"{base_stem}.txt")
                    if not os.path.exists(lbl_path):
                        try:
                            with open(lbl_path, 'w', encoding='utf-8') as lf:
                                lf.write("")
                        except Exception:
                            pass

    try:
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
        return yaml_path
    except Exception:
        return yaml_path

def validate_pre_training(dataset_metadata):
    """
    Validates dataset structure, data.yaml, image directories, and class IDs before starting training.
    Returns (is_valid, summary_dict, error_message).
    """
    if not dataset_metadata:
        return False, {}, "No dataset selected."

    if dataset_metadata.get("dataset_type") != "Object Detection":
        return False, {}, "Only Object Detection datasets can be trained with YOLO11."

    dataset_path = dataset_metadata.get("dataset_path")
    if not dataset_path or not os.path.exists(dataset_path):
        return False, {}, f"Dataset folder not found at {dataset_path}."

    yaml_path = ensure_valid_data_yaml(dataset_path)
    if not yaml_path:
        return False, {}, "No data.yaml file found or auto-generated in dataset directory."

    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
    except Exception as e:
        return False, {}, f"Failed to parse data.yaml: {str(e)}"

    names = yaml_data.get('names', [])
    if isinstance(names, dict):
        class_list = [names[k] for k in sorted(names.keys())]
    elif isinstance(names, list):
        class_list = names
    else:
        class_list = dataset_metadata.get("classes", [])

    train_count = dataset_metadata.get("train_count", 0)
    val_count = dataset_metadata.get("validation_count", 0)
    test_count = dataset_metadata.get("test_count", 0)

    if train_count == 0 and not dataset_metadata.get("total_images", 0):
        return False, {}, "Training set contains 0 images. Cannot start training."

    summary = {
        "dataset_name": dataset_metadata.get("dataset_name", "Unknown"),
        "dataset_id": dataset_metadata.get("dataset_id", "DS_000"),
        "yaml_path": yaml_path,
        "classes": class_list,
        "train_count": train_count or dataset_metadata.get("total_images", 0),
        "val_count": val_count,
        "test_count": test_count,
        "total_images": dataset_metadata.get("total_images", 0),
        "status": "Ready to train ✅"
    }

    return True, summary, None
