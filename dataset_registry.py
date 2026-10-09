"""
TerraFlare - Dataset Registry Module
Handles dataset directory structure, metadata storage, and dataset registry listing.
"""

import os
import json
from datetime import datetime

BASE_DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "datasets")

def ensure_datasets_dir():
    """Ensure the base dataset directory exists."""
    os.makedirs(BASE_DATASETS_DIR, exist_ok=True)
    return BASE_DATASETS_DIR

def get_next_dataset_id():
    """Generates the next unique dataset ID (e.g. DS_001, DS_002)."""
    ensure_datasets_dir()
    existing_dirs = [d for d in os.listdir(BASE_DATASETS_DIR) if os.path.isdir(os.path.join(BASE_DATASETS_DIR, d))]
    
    max_id = 0
    for folder in existing_dirs:
        if folder.startswith("DS_"):
            try:
                num = int(folder.split("_")[1])
                if num > max_id:
                    max_id = num
            except (IndexError, ValueError):
                pass
    
    next_num = max_id + 1
    return f"DS_{next_num:03d}"

def get_dataset_dir(dataset_id):
    """Returns absolute path to a specific dataset directory."""
    return os.path.join(ensure_datasets_dir(), dataset_id)

def save_dataset_metadata(dataset_id, metadata):
    """Saves metadata.json into the dataset directory."""
    dataset_dir = get_dataset_dir(dataset_id)
    os.makedirs(dataset_dir, exist_ok=True)
    
    metadata["dataset_id"] = dataset_id
    if "upload_timestamp" not in metadata or not metadata["upload_timestamp"]:
        metadata["upload_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    metadata_path = os.path.join(dataset_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
        
    return metadata_path

def get_dataset_metadata(dataset_id):
    """Reads metadata.json for a given dataset ID."""
    dataset_dir = get_dataset_dir(dataset_id)
    metadata_path = os.path.join(dataset_dir, "metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def list_registered_datasets():
    """Returns a list of all registered dataset metadata dicts sorted by timestamp descending."""
    ensure_datasets_dir()
    datasets = []
    
    if not os.path.exists(BASE_DATASETS_DIR):
        return datasets
        
    for item in sorted(os.listdir(BASE_DATASETS_DIR)):
        item_path = os.path.join(BASE_DATASETS_DIR, item)
        if os.path.isdir(item_path):
            meta = get_dataset_metadata(item)
            if meta:
                dpath = meta.get("dataset_path") or meta.get("dataset_dir") or item_path
                if os.path.exists(dpath):
                    datasets.append(meta)
                
    # Sort by upload timestamp descending if available
    datasets.sort(key=lambda x: x.get("upload_timestamp", ""), reverse=True)
    return datasets
