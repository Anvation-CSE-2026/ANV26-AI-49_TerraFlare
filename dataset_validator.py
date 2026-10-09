"""
TerraFlare - Dataset Validator Module
Handles safe ZIP extraction, dataset structural verification, label parsing, and integrity checks.
"""

import os
import zipfile
import yaml
import pandas as pd
from PIL import Image

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}

def is_safe_path(base_dir, target_path):
    """Prevents Path Traversal (Zip Slip) attacks."""
    abs_base = os.path.abspath(base_dir)
    abs_target = os.path.abspath(target_path)
    return os.path.commonpath([abs_base, abs_target]) == abs_base

def extract_zip_safely(zip_file_source, extract_to):
    """
    Extracts a zip file or file-like object to target directory with security checks.
    """
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_file_source, 'r') as zip_ref:
        for member in zip_ref.infolist():
            # Construct target path
            target_path = os.path.join(extract_to, member.filename)
            if not is_safe_path(extract_to, target_path):
                raise ValueError(f"Security Alert: Unsafe path in zip archive: {member.filename}")
        zip_ref.extractall(extract_to)
    
    # If the zip extracted into a single top-level wrapper directory, return that subdirectory if appropriate
    items = [i for i in os.listdir(extract_to) if not i.startswith('.') and not i.startswith('__MACOSX')]
    if len(items) == 1:
        single_item = os.path.join(extract_to, items[0])
        if os.path.isdir(single_item):
            # Check if dataset markers are inside this subfolder
            return single_item
    return extract_to

def find_data_yaml(root_dir):
    """Recursively search for data.yaml or data.yml in root_dir."""
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower() in ['data.yaml', 'data.yml']:
                return os.path.join(dirpath, f)
    return None

def validate_image_file(image_path):
    """Attempts to open and verify image file with PIL. Returns True if valid."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def validate_yolo_dataset(dataset_root, progress_callback=None):
    """
    Validates a YOLO format object detection dataset.
    Returns a comprehensive validation report dictionary.
    """
    report = {
        "dataset_type": "Object Detection",
        "valid_structure": True,
        "data_yaml_found": False,
        "total_images": 0,
        "train_count": 0,
        "validation_count": 0,
        "test_count": 0,
        "classes": [],
        "class_distribution": {},
        "corrupted_images": 0,
        "missing_labels": 0,
        "missing_files": 0,
        "invalid_labels": 0,
        "empty_labels": 0,
        "warnings": [],
        "errors": [],
        "status": "Valid",
        "splits_found": {}
    }

    from training.training_config import ensure_valid_data_yaml
    yaml_path = ensure_valid_data_yaml(dataset_root)
    class_names = []
    nc = 0
    yaml_dir = dataset_root

    if yaml_path and os.path.exists(yaml_path):
        report["data_yaml_found"] = True
        yaml_dir = os.path.dirname(yaml_path)
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data_yaml = yaml.safe_load(f) or {}
            
            names = data_yaml.get('names', [])
            if isinstance(names, dict):
                class_names = [names[k] for k in sorted(names.keys())]
            elif isinstance(names, list):
                class_names = names
            nc = data_yaml.get('nc', len(class_names))
            report["classes"] = class_names
        except Exception as e:
            report["warnings"].append(f"Failed to parse data.yaml: {str(e)}")

    # Search for image/label split directories
    # Potential bases: dataset_root or yaml_dir
    possible_bases = [dataset_root, yaml_dir]
    
    # Locate train, val, test directories
    splits = ["train", "val", "test"]
    split_paths = {}

    for split in splits:
        found_img_dir = None
        found_lbl_dir = None
        
        for base in possible_bases:
            # Common patterns: images/train, train/images, or train
            candidates_img = [
                os.path.join(base, "images", split),
                os.path.join(base, split, "images"),
                os.path.join(base, split)
            ]
            candidates_lbl = [
                os.path.join(base, "labels", split),
                os.path.join(base, split, "labels"),
                os.path.join(base, split)
            ]
            
            for c_img in candidates_img:
                if os.path.isdir(c_img):
                    # verify it contains images
                    files = [f for f in os.listdir(c_img) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]
                    if files or found_img_dir is None:
                        found_img_dir = c_img
                        break
            
            for c_lbl in candidates_lbl:
                if os.path.isdir(c_lbl) and c_lbl != found_img_dir:
                    found_lbl_dir = c_lbl
                    break
            
            if found_img_dir:
                break
                
        if found_img_dir:
            split_paths[split] = {
                "images": found_img_dir,
                "labels": found_lbl_dir
            }

    if not split_paths:
        # Fallback: scan root directory directly for images and labels
        img_files = []
        for dp, _, filenames in os.walk(dataset_root):
            for f in filenames:
                if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                    img_files.append(os.path.join(dp, f))
        if img_files:
            split_paths["train"] = {
                "images": dataset_root,
                "labels": dataset_root
            }
        else:
            report["valid_structure"] = False
            report["errors"].append("No valid image or split directories found in dataset.")
            report["status"] = "Invalid"
            return report

    # Initialize class distribution
    for cname in class_names:
        report["class_distribution"][cname] = 0

    total_scanned = 0
    all_image_paths = []
    
    # Collect all image paths first for progress reporting
    for split, paths in split_paths.items():
        img_dir = paths["images"]
        if not img_dir or not os.path.exists(img_dir):
            continue
            
        for dp, _, filenames in os.walk(img_dir):
            for f in filenames:
                ext = os.path.splitext(f)[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    all_image_paths.append((split, os.path.join(dp, f), paths["labels"]))

    total_images_count = len(all_image_paths)
    report["total_images"] = total_images_count

    for idx, (split, img_path, lbl_dir) in enumerate(all_image_paths):
        if progress_callback and total_images_count > 0:
            progress_callback((idx + 1) / total_images_count, f"Validating image {idx + 1}/{total_images_count}")
            
        # Update split counts
        if split == "train":
            report["train_count"] += 1
        elif split in ["val", "valid", "validation"]:
            report["validation_count"] += 1
        elif split == "test":
            report["test_count"] += 1

        # Check image validity
        if not validate_image_file(img_path):
            report["corrupted_images"] += 1
            continue

        # Look for label file
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        
        lbl_path = None
        if lbl_dir and os.path.exists(lbl_dir):
            candidate_lbl = os.path.join(lbl_dir, f"{base_name}.txt")
            if os.path.exists(candidate_lbl):
                lbl_path = candidate_lbl
        if not lbl_path:
            # Check same folder as image
            same_dir_lbl = os.path.join(os.path.dirname(img_path), f"{base_name}.txt")
            if os.path.exists(same_dir_lbl):
                lbl_path = same_dir_lbl

        if not lbl_path:
            report["missing_labels"] += 1
            continue

        # Read and validate label file
        try:
            with open(lbl_path, 'r', encoding='utf-8') as lf:
                lines = [l.strip() for l in lf.readlines() if l.strip()]
            
            if not lines:
                report["empty_labels"] += 1
                continue
                
            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    report["invalid_labels"] += 1
                    continue
                try:
                    cid = int(parts[0])
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    
                    if not (0 <= cx <= 1.05 and 0 <= cy <= 1.05 and 0 <= w <= 1.05 and 0 <= h <= 1.05):
                        report["invalid_labels"] += 1
                        continue
                        
                    if class_names and 0 <= cid < len(class_names):
                        cname = class_names[cid]
                        report["class_distribution"][cname] = report["class_distribution"].get(cname, 0) + 1
                    else:
                        cname = f"Class_{cid}"
                        report["class_distribution"][cname] = report["class_distribution"].get(cname, 0) + 1
                        if cname not in report["classes"]:
                            report["classes"].append(cname)
                except ValueError:
                    report["invalid_labels"] += 1
        except Exception:
            report["invalid_labels"] += 1

    # Check for orphan label files (labels with missing image)
    orphan_count = 0
    for split, paths in split_paths.items():
        lbl_dir = paths["labels"]
        if lbl_dir and os.path.exists(lbl_dir):
            for dp, _, files in os.walk(lbl_dir):
                for f in files:
                    if f.endswith(".txt"):
                        lbl_stem = os.path.splitext(f)[0]
                        # Check if any matching image exists
                        has_img = any(
                            os.path.exists(os.path.join(paths["images"], f"{lbl_stem}{ext}"))
                            for ext in IMAGE_EXTENSIONS
                        )
                        if not has_img:
                            orphan_count += 1
    report["missing_files"] = orphan_count

    # Determine overall status
    if report["corrupted_images"] > 0 or report["invalid_labels"] > 0 or not report["valid_structure"]:
        report["status"] = "Warning" if report["total_images"] > 0 else "Invalid"
    elif report["missing_labels"] > 0 or report["empty_labels"] > 0:
        report["status"] = "Warning"
    else:
        report["status"] = "Valid"

    return report

def validate_classification_dataset(dataset_root, csv_file_path=None, progress_callback=None):
    """
    Validates an image classification dataset (directory structure or CSV mapping).
    """
    report = {
        "dataset_type": "Classification",
        "valid_structure": True,
        "total_images": 0,
        "train_count": 0,
        "validation_count": 0,
        "test_count": 0,
        "classes": [],
        "class_distribution": {},
        "corrupted_images": 0,
        "missing_files": 0,
        "missing_labels": 0,
        "invalid_labels": 0,
        "warnings": [],
        "errors": [],
        "status": "Valid"
    }

    # Case A: CSV classification dataset
    if csv_file_path and os.path.exists(csv_file_path):
        try:
            df = pd.read_csv(csv_file_path)
            # Check if dataframe has valid image path and class columns
            path_col = None
            class_col = None
            
            for col in df.columns:
                col_lower = str(col).lower().strip()
                if col_lower in ['image_path', 'path', 'filepath', 'filename', 'image']:
                    path_col = col
                elif col_lower in ['class', 'label', 'target', 'category']:
                    class_col = col

            if not path_col or not class_col:
                if len(df.columns) >= 2:
                    path_col = df.columns[0]
                    class_col = df.columns[1]

            if not path_col or not class_col:
                report["valid_structure"] = False
                report["errors"].append("CSV does not contain valid image_path and class columns.")
                report["status"] = "Invalid"
                return report

            # Validate that the path column contains image extensions
            sample_paths = df[path_col].dropna().astype(str).tolist()[:20]
            valid_ext_count = sum(1 for p in sample_paths if os.path.splitext(p)[1].lower() in IMAGE_EXTENSIONS)
            
            if valid_ext_count == 0:
                report["valid_structure"] = False
                report["errors"].append("CSV file does not contain valid image paths (no image file extensions found).")
                report["status"] = "Invalid"
                return report

            total_rows = len(df)
            report["total_images"] = total_rows

            for idx, row in df.iterrows():
                if progress_callback and total_rows > 0:
                    progress_callback((idx + 1) / total_rows, f"Validating CSV row {idx + 1}/{total_rows}")
                    
                rel_img_path = str(row[path_col]).strip()
                cname = str(row[class_col]).strip()
                
                full_img_path = os.path.join(dataset_root, rel_img_path)
                if not os.path.exists(full_img_path):
                    # Check relative to CSV directory
                    full_img_path = os.path.join(os.path.dirname(csv_file_path), rel_img_path)

                if not os.path.exists(full_img_path):
                    report["missing_files"] += 1
                    continue

                if not validate_image_file(full_img_path):
                    report["corrupted_images"] += 1
                    continue

                if cname not in report["classes"]:
                    report["classes"].append(cname)
                report["class_distribution"][cname] = report["class_distribution"].get(cname, 0) + 1
                report["train_count"] += 1 # Default all CSV entries as train unless split specified

            return report
        except Exception as e:
            report["valid_structure"] = False
            report["errors"].append(f"Failed to parse CSV dataset: {str(e)}")
            report["status"] = "Invalid"
            return report

    # Case B: Directory structure (train/val/test or class folders)
    # Search for train, val, test subdirectories
    subdirs = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d)) and not d.startswith('.')]
    
    splits = [d for d in subdirs if d.lower() in ['train', 'val', 'valid', 'validation', 'test']]
    
    image_entries = [] # (split, class_name, img_path)

    if splits:
        for split in splits:
            split_dir = os.path.join(dataset_root, split)
            class_folders = [c for c in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, c))]
            for cname in class_folders:
                c_dir = os.path.join(split_dir, cname)
                for dp, _, files in os.walk(c_dir):
                    for f in files:
                        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                            image_entries.append((split.lower(), cname, os.path.join(dp, f)))
    else:
        # Class folders directly under dataset_root
        class_folders = [c for c in subdirs if c.lower() not in ['__macosx', 'metadata']]
        for cname in class_folders:
            c_dir = os.path.join(dataset_root, cname)
            for dp, _, files in os.walk(c_dir):
                for f in files:
                    if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                        image_entries.append(('train', cname, os.path.join(dp, f)))

    if not image_entries:
        report["valid_structure"] = False
        report["errors"].append("No classification class folders or image files found in dataset.")
        report["status"] = "Invalid"
        return report

    total_imgs = len(image_entries)
    report["total_images"] = total_imgs

    for idx, (split, cname, img_path) in enumerate(image_entries):
        if progress_callback and total_imgs > 0:
            progress_callback((idx + 1) / total_imgs, f"Validating image {idx + 1}/{total_imgs}")
            
        if split in ['train']:
            report["train_count"] += 1
        elif split in ['val', 'valid', 'validation']:
            report["validation_count"] += 1
        elif split in ['test']:
            report["test_count"] += 1
        else:
            report["train_count"] += 1

        if not validate_image_file(img_path):
            report["corrupted_images"] += 1
            continue

        if cname not in report["classes"]:
            report["classes"].append(cname)
        report["class_distribution"][cname] = report["class_distribution"].get(cname, 0) + 1

    if report["corrupted_images"] > 0:
        report["status"] = "Warning"
    else:
        report["status"] = "Valid"

    return report
