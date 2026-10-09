"""
TerraFlare - Experiment Manager Module
Manages experiment storage structure under experiments/YOLO/EXP_XXX, saves training metadata, and retrieves experiment logs.
"""

import os
import json
from datetime import datetime

BASE_EXPERIMENTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "YOLO")

def ensure_experiments_dir():
    """Ensures the base experiments directory exists."""
    os.makedirs(BASE_EXPERIMENTS_DIR, exist_ok=True)
    return BASE_EXPERIMENTS_DIR

def get_next_experiment_id():
    """Generates next unique experiment ID (e.g., EXP_001, EXP_002)."""
    ensure_experiments_dir()
    existing_dirs = [d for d in os.listdir(BASE_EXPERIMENTS_DIR) if os.path.isdir(os.path.join(BASE_EXPERIMENTS_DIR, d))]
    
    max_id = 0
    for folder in existing_dirs:
        if folder.startswith("EXP_"):
            try:
                num = int(folder.split("_")[1])
                if num > max_id:
                    max_id = num
            except (IndexError, ValueError):
                pass
                
    next_num = max_id + 1
    return f"EXP_{next_num:03d}"

def get_experiment_dir(exp_id):
    """Returns absolute path to specific experiment directory."""
    return os.path.join(ensure_experiments_dir(), exp_id)

def save_experiment_config(exp_id, config_dict):
    """Saves config.json into the experiment folder."""
    exp_dir = get_experiment_dir(exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    
    config_dict["experiment_id"] = exp_id
    if "saved_at" not in config_dict:
        config_dict["saved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4)
        
    return config_path

def get_experiment_metadata(exp_id):
    """Reads config.json for a given experiment ID."""
    exp_dir = get_experiment_dir(exp_id)
    config_path = os.path.join(exp_dir, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def list_all_experiments():
    """Lists all stored experiment metadata sorted by training start time descending."""
    ensure_experiments_dir()
    experiments = []
    
    if not os.path.exists(BASE_EXPERIMENTS_DIR):
        return experiments
        
    for item in sorted(os.listdir(BASE_EXPERIMENTS_DIR)):
        item_path = os.path.join(BASE_EXPERIMENTS_DIR, item)
        if os.path.isdir(item_path):
            meta = get_experiment_metadata(item)
            if meta:
                experiments.append(meta)
                
    experiments.sort(key=lambda x: x.get("training_start_time", ""), reverse=True)
    return experiments
