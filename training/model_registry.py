"""
TerraFlare - Trained Model Registry Module
Registers successfully trained models, stores model metadata, performance metrics, and deployment candidate flags.
"""

import os
import json
from datetime import datetime

REGISTRY_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "registered_models.json")

def ensure_registry_dir():
    """Ensures parent directory for registered_models.json exists."""
    os.makedirs(os.path.dirname(REGISTRY_FILE), exist_ok=True)

def load_registry():
    """Loads all registered models from JSON file."""
    ensure_registry_dir()
    if os.path.exists(REGISTRY_FILE):
        try:
            with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_registry(models_list):
    """Saves updated models list to JSON file."""
    ensure_registry_dir()
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(models_list, f, indent=4)

def get_next_model_id():
    """Generates next model ID (e.g. MODEL_001, MODEL_002)."""
    models = load_registry()
    max_id = 0
    for m in models:
        mid = m.get("model_id", "")
        if mid.startswith("MODEL_"):
            try:
                num = int(mid.split("_")[1])
                if num > max_id:
                    max_id = num
            except (IndexError, ValueError):
                pass
    return f"MODEL_{max_id + 1:03d}"

def register_trained_model(experiment_id, experiment_name, dataset_id, architecture, best_weights_path, metrics, classes=None):
    """
    Registers a successfully trained model into the model registry with status READY.
    """
    models = load_registry()
    model_id = get_next_model_id()
    
    is_yolo = "YOLO" in architecture.upper()
    
    entry = {
        "model_id": model_id,
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "dataset_id": dataset_id,
        "model_type": "YOLO Detection" if is_yolo else "CNN Classification",
        "architecture": architecture,
        "best_weights_path": best_weights_path,
        "classes": classes or [],
        "test_samples": metrics.get("test_samples", 0),
        "metrics": {
            "accuracy": f"{metrics.get('accuracy', 0.0)*100:.2f}%" if "accuracy" in metrics else "N/A",
            "precision": f"{metrics.get('precision', 0.0)*100:.2f}%" if "precision" in metrics else "N/A",
            "recall": f"{metrics.get('recall', 0.0)*100:.2f}%" if "recall" in metrics else "N/A",
            "f1": f"{metrics.get('f1', 0.0)*100:.2f}%" if "f1" in metrics else "N/A",
            "map50": f"{metrics.get('map50', 0.0)*100:.2f}%" if is_yolo else "N/A",
            "map50_95": f"{metrics.get('map50_95', 0.0)*100:.2f}%" if is_yolo else "N/A"
        },
        "raw_metrics": metrics,
        "status": "READY",
        "is_deployed": False,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Check if entry already exists for experiment_id
    existing_idx = next((i for i, m in enumerate(models) if m.get("experiment_id") == experiment_id), None)
    if existing_idx is not None:
        models[existing_idx] = entry
    else:
        models.append(entry)

    save_registry(models)
    return entry

def update_model_evaluation(model_id, metrics, test_samples_count=0):
    """
    Updates an existing model registry entry with fresh evaluation results.
    """
    models = load_registry()
    for m in models:
        if m.get("model_id") == model_id or m.get("experiment_id") == model_id:
            is_yolo = "YOLO" in m.get("architecture", "").upper()
            m["test_samples"] = test_samples_count
            m["raw_metrics"] = metrics
            m["metrics"] = {
                "accuracy": f"{metrics.get('accuracy', 0.0)*100:.2f}%" if "accuracy" in metrics else "N/A",
                "precision": f"{metrics.get('precision', 0.0)*100:.2f}%" if "precision" in metrics else "N/A",
                "recall": f"{metrics.get('recall', 0.0)*100:.2f}%" if "recall" in metrics else "N/A",
                "f1": f"{metrics.get('f1', 0.0)*100:.2f}%" if "f1" in metrics else "N/A",
                "map50": f"{metrics.get('map50', 0.0)*100:.2f}%" if is_yolo else "N/A",
                "map50_95": f"{metrics.get('map50_95', 0.0)*100:.2f}%" if is_yolo else "N/A"
            }
            m["last_evaluated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            save_registry(models)
            return m
    return None

def list_registered_models():
    """Returns all registered models sorted by created_at descending."""
    models = load_registry()
    models.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return models

def get_registered_model(model_id):
    """Returns metadata dict for specific model_id."""
    models = load_registry()
    for m in models:
        if m.get("model_id") == model_id or m.get("experiment_id") == model_id:
            return m
    return None
