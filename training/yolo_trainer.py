"""
TerraFlare - YOLO11 Trainer Module
Handles execution of actual Ultralytics YOLO11 training, saving experiment outputs, and metrics collection.
"""

import os
import shutil
from datetime import datetime
from ultralytics import YOLO

from training.experiment_manager import (
    get_next_experiment_id,
    get_experiment_dir,
    save_experiment_config
)
from training.training_config import ensure_valid_data_yaml
from training.evaluator import evaluate_yolo_model, extract_metrics_from_csv
from training.model_registry import register_trained_model

# Global lock state to prevent concurrent training runs
IS_TRAINING_RUNNING = False

def run_yolo_training(
    dataset_metadata,
    experiment_name,
    model_variant="yolo11n.pt",
    epochs=50,
    imgsz=640,
    batch_size=16,
    lr0=0.01,
    device_choice="Auto",
    progress_callback=None
):
    """
    Executes actual Ultralytics YOLO training on the selected dataset.
    Saves outputs in experiments/YOLO/{experiment_id}/ and registers trained model upon success.
    """
    global IS_TRAINING_RUNNING
    if IS_TRAINING_RUNNING:
        raise RuntimeError("Another training job is currently in progress. Please wait for it to complete.")

    IS_TRAINING_RUNNING = True
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    exp_id = get_next_experiment_id()
    exp_dir = get_experiment_dir(exp_id)
    os.makedirs(exp_dir, exist_ok=True)

    try:
        dataset_path = dataset_metadata.get("dataset_path")
        yaml_path = ensure_valid_data_yaml(dataset_path)

        if not yaml_path or not os.path.exists(yaml_path):
            raise FileNotFoundError(f"data.yaml could not be found or created for dataset at {dataset_path}")

        # Resolve device string
        if device_choice.upper() == "AUTO":
            import torch
            if torch.cuda.is_available():
                device_str = "0"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device_str = "mps"
            else:
                device_str = "cpu"
        elif device_choice.upper() == "CUDA":
            device_str = "0"
        elif device_choice.upper() == "MPS":
            device_str = "mps"
        else:
            device_str = "cpu"

        exp_config = {
            "experiment_id": exp_id,
            "experiment_name": experiment_name,
            "dataset_id": dataset_metadata.get("dataset_id"),
            "dataset_name": dataset_metadata.get("dataset_name"),
            "model_name": model_variant,
            "epochs": epochs,
            "image_size": imgsz,
            "batch_size": batch_size,
            "learning_rate": lr0,
            "device": device_str,
            "training_start_time": start_time,
            "training_end_time": None,
            "status": "IN_PROGRESS",
            "classes": dataset_metadata.get("classes", []),
            "experiment_dir": exp_dir,
            "best_model_path": None,
            "last_model_path": None,
            "results_csv_path": None,
            "metrics": {}
        }

        save_experiment_config(exp_id, exp_config)

        if progress_callback:
            progress_callback("Initializing YOLO11 model weights...")

        model = YOLO(model_variant)

        if progress_callback:
            progress_callback(f"Starting YOLO11 training ({epochs} epochs on {device_str.upper()})...")

        # Execute real training with automatic hardware fallback if MPS fails
        try:
            train_results = model.train(
                data=yaml_path,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch_size,
                lr0=lr0,
                device=device_str,
                project=exp_dir,
                name="ultralytics_run",
                exist_ok=True,
                verbose=True
            )
        except Exception as train_err:
            if device_str in ("mps", "0"):
                print(f"[YOLO Trainer Warning] Training failed on {device_str}: {train_err}. Retrying training on CPU...")
                if progress_callback:
                    progress_callback(f"Hardware error on {device_str.upper()}. Retrying training on CPU...")
                device_str = "cpu"
                exp_config["device"] = "cpu"
                train_results = model.train(
                    data=yaml_path,
                    epochs=epochs,
                    imgsz=imgsz,
                    batch=batch_size,
                    lr0=lr0,
                    device="cpu",
                    project=exp_dir,
                    name="ultralytics_run",
                    exist_ok=True,
                    verbose=True
                )
            else:
                raise train_err

        run_output_dir = os.path.join(exp_dir, "ultralytics_run")

        # Locate weights
        best_pt = os.path.join(run_output_dir, "weights", "best.pt")
        last_pt = os.path.join(run_output_dir, "weights", "last.pt")

        weights_dir = os.path.join(exp_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)

        final_best_pt = os.path.join(weights_dir, "best.pt")
        final_last_pt = os.path.join(weights_dir, "last.pt")

        if os.path.exists(best_pt):
            shutil.copy2(best_pt, final_best_pt)
        if os.path.exists(last_pt):
            shutil.copy2(last_pt, final_last_pt)

        # Copy plots and results.csv into exp_dir root
        results_csv = os.path.join(run_output_dir, "results.csv")
        final_results_csv = os.path.join(exp_dir, "results.csv")
        if os.path.exists(results_csv):
            shutil.copy2(results_csv, final_results_csv)

        # Copy plot files (confusion_matrix, results.png, etc.)
        for item in os.listdir(run_output_dir):
            item_path = os.path.join(run_output_dir, item)
            if os.path.isfile(item_path) and (item.endswith('.png') or item.endswith('.jpg')):
                shutil.copy2(item_path, os.path.join(exp_dir, item))

        if progress_callback:
            progress_callback("Evaluating trained model on validation set...")

        # Perform Evaluation on best checkpoint
        if os.path.exists(final_best_pt):
            try:
                metrics = evaluate_yolo_model(final_best_pt, yaml_path, device=device_str)
            except Exception:
                metrics = extract_metrics_from_csv(final_results_csv)
        else:
            metrics = extract_metrics_from_csv(final_results_csv)

        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        exp_config["status"] = "SUCCESS"
        exp_config["training_end_time"] = end_time
        exp_config["best_model_path"] = final_best_pt
        exp_config["last_model_path"] = final_last_pt
        exp_config["results_csv_path"] = final_results_csv
        exp_config["metrics"] = metrics

        save_experiment_config(exp_id, exp_config)

        # Register model in Model Registry
        registry_entry = register_trained_model(
            experiment_id=exp_id,
            experiment_name=experiment_name,
            dataset_id=dataset_metadata.get("dataset_id"),
            architecture=model_variant.replace('.pt', ''),
            best_weights_path=final_best_pt,
            metrics=metrics,
            classes=dataset_metadata.get("classes", [])
        )
        exp_config["registered_model_id"] = registry_entry["model_id"]

        return exp_config

    except Exception as e:
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if 'exp_config' in locals():
            exp_config["status"] = "FAILED"
            exp_config["training_end_time"] = end_time
            exp_config["error_message"] = str(e)
            save_experiment_config(exp_id, exp_config)
        raise e
    finally:
        IS_TRAINING_RUNNING = False
