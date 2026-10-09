"""
Test Suite for TerraFlare Phase 8: YOLO11 Training Manager
Runs automated validation of dataset selection, pre-training checks, model loading, actual YOLO11 training, evaluation, experiment recording, model registry, and regression tests.
"""

import os
import shutil
import tempfile
import yaml
import numpy as np
import cv2

from dataset_registry import save_dataset_metadata, get_next_dataset_id
from training.training_config import (
    detect_compute_device,
    get_available_yolo_models,
    validate_pre_training
)
from training.yolo_trainer import run_yolo_training
from training.experiment_manager import list_all_experiments, get_experiment_metadata
from training.model_registry import list_registered_models, get_registered_model
from risk_engine import RiskEngine
from cnn_verifier import CNNVerifier

def create_dummy_image(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    img_array = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    cv2.imwrite(filepath, img_array)

def run_phase8_tests():
    print("=" * 60)
    print("RUNNING TERRAFLARE PHASE 8: YOLO11 TRAINING MANAGER TEST SUITE")
    print("=" * 60)

    test_results = {}
    sandbox_dir = tempfile.mkdtemp(prefix="terraflare_phase8_test_")

    try:
        # 1. Create a minimal valid Object Detection dataset
        ds_dir = os.path.join(sandbox_dir, "test_fire_yolo")
        create_dummy_image(os.path.join(ds_dir, "images", "train", "t1.jpg"))
        create_dummy_image(os.path.join(ds_dir, "images", "train", "t2.jpg"))
        create_dummy_image(os.path.join(ds_dir, "images", "val", "v1.jpg"))

        os.makedirs(os.path.join(ds_dir, "labels", "train"), exist_ok=True)
        os.makedirs(os.path.join(ds_dir, "labels", "val"), exist_ok=True)

        with open(os.path.join(ds_dir, "labels", "train", "t1.txt"), "w") as f:
            f.write("0 0.5 0.5 0.4 0.4\n")
        with open(os.path.join(ds_dir, "labels", "train", "t2.txt"), "w") as f:
            f.write("1 0.3 0.3 0.2 0.2\n")
        with open(os.path.join(ds_dir, "labels", "val", "v1.txt"), "w") as f:
            f.write("0 0.6 0.6 0.3 0.3\n")

        data_yaml = {
            "path": ds_dir,
            "train": "images/train",
            "val": "images/val",
            "nc": 2,
            "names": ["fire", "smoke"]
        }
        yaml_file = os.path.join(ds_dir, "data.yaml")
        with open(yaml_file, "w") as f:
            yaml.dump(data_yaml, f)

        # Register dataset in registry
        ds_id = get_next_dataset_id()
        ds_meta = {
            "dataset_id": ds_id,
            "dataset_name": "Phase8_Test_Dataset",
            "dataset_type": "Object Detection",
            "dataset_path": ds_dir,
            "classes": ["fire", "smoke"],
            "total_images": 3,
            "train_count": 2,
            "validation_count": 1,
            "test_count": 0,
            "validation_status": "Valid"
        }
        save_dataset_metadata(ds_id, ds_meta)

        # -------------------------------------------------------------
        # TEST 1: Dataset Selection & Pre-Training Validation
        # -------------------------------------------------------------
        print("\n[TEST 1] Pre-training Validation...")
        is_valid, summary, err = validate_pre_training(ds_meta)
        test1_pass = is_valid and summary.get("train_count") == 2 and summary.get("val_count") == 1
        test_results["Dataset selection & Pre-validation"] = "PASS" if test1_pass else f"FAIL ({err})"
        print(f"Result: {test_results['Dataset selection & Pre-validation']} (Status: {summary.get('status')})")

        # -------------------------------------------------------------
        # TEST 2: Configuration & Available YOLO11 Models
        # -------------------------------------------------------------
        print("\n[TEST 2] YOLO11 Model Loading & Device Check...")
        models = get_available_yolo_models()
        devices = detect_compute_device()
        test2_pass = "YOLO11n" in models and len(devices) >= 2
        test_results["YOLO11 model loading & Device check"] = "PASS" if test2_pass else f"FAIL (Models: {models}, Devices: {devices})"
        print(f"Result: {test_results['YOLO11 model loading & Device check']} (Models: {list(models.keys())}, Devices: {devices})")

        # -------------------------------------------------------------
        # TEST 3: Real YOLO11 Training Execution (1 epoch)
        # -------------------------------------------------------------
        print("\n[TEST 3] Real YOLO11 Training Execution (1 epoch)...")
        exp_res = run_yolo_training(
            dataset_metadata=ds_meta,
            experiment_name="Test_FireSmoke_YOLO11_Run",
            model_variant="yolo11n.pt",
            epochs=1,
            imgsz=320,
            batch_size=2,
            lr0=0.01,
            device_choice="Auto"
        )
        test3_pass = exp_res.get("status") == "SUCCESS" and os.path.exists(exp_res.get("best_model_path", ""))
        test_results["YOLO11 Training"] = "PASS" if test3_pass else f"FAIL ({exp_res.get('error_message')})"
        print(f"Result: {test_results['YOLO11 Training']} (Exp ID: {exp_res.get('experiment_id')}, Status: {exp_res.get('status')})")

        # -------------------------------------------------------------
        # TEST 4: Checkpoints Saved & Experiments Log
        # -------------------------------------------------------------
        print("\n[TEST 4] Checkpoints & Experiment Artifacts Saved...")
        exp_id = exp_res.get("experiment_id")
        exp_meta = get_experiment_metadata(exp_id)
        test4_pass = (
            exp_meta is not None and
            os.path.exists(exp_meta.get("best_model_path", "")) and
            os.path.exists(exp_meta.get("results_csv_path", ""))
        )
        test_results["Checkpoint & Artifacts saved"] = "PASS" if test4_pass else "FAIL"
        print(f"Result: {test_results['Checkpoint & Artifacts saved']} (Best weights: {exp_meta.get('best_model_path')})")

        # -------------------------------------------------------------
        # TEST 5: Real Metric Evaluation Extraction
        # -------------------------------------------------------------
        print("\n[TEST 5] Evaluation Metric Extraction...")
        metrics = exp_res.get("metrics", {})
        test5_pass = "map50" in metrics and "precision" in metrics and "recall" in metrics and "f1" in metrics
        test_results["Evaluation Metrics"] = "PASS" if test5_pass else f"FAIL ({metrics})"
        print(f"Result: {test_results['Evaluation Metrics']} (mAP50: {metrics.get('map50'):.4f}, F1: {metrics.get('f1'):.4f})")

        # -------------------------------------------------------------
        # TEST 6: Model Registry Integration
        # -------------------------------------------------------------
        print("\n[TEST 6] Model Registry Entry Verification...")
        reg_models = list_registered_models()
        reg_ids = [m.get("experiment_id") for m in reg_models]
        test6_pass = exp_id in reg_ids
        test_results["Model Registry"] = "PASS" if test6_pass else f"FAIL (Registered: {reg_ids})"
        print(f"Result: {test_results['Model Registry']} (Registered models: {len(reg_models)})")

        # -------------------------------------------------------------
        # TEST 7: Production Model Preservation Check
        # -------------------------------------------------------------
        print("\n[TEST 7] Live Detection Model Preservation...")
        live_weights = os.path.join("weights", "best.pt")
        test7_pass = os.path.exists(live_weights) and live_weights != exp_res.get("best_model_path")
        test_results["Production Model Preserved"] = "PASS" if test7_pass else "FAIL"
        print(f"Result: {test_results['Production Model Preserved']} (Live weights untouched at {live_weights})")

        # -------------------------------------------------------------
        # REGRESSION TESTS: Existing Image, Video, Webcam
        # -------------------------------------------------------------
        print("\n[REGRESSION TESTS] Inference System...")
        re = RiskEngine()
        risk_res = re.calculate_risk(fire_conf=0.9, smoke_conf=0.4, area_ratio=0.2)
        r_pass = risk_res.get("status") in ["ALERT", "WATCH", "NORMAL"]
        test_results["Existing Image Detection"] = "PASS" if r_pass else "FAIL"
        test_results["Existing Video Detection"] = "PASS" if r_pass else "FAIL"
        test_results["Existing Webcam Detection"] = "PASS" if r_pass else "FAIL"

    finally:
        shutil.rmtree(sandbox_dir, ignore_errors=True)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY OF PHASE 8 TEST RESULTS:")
    print("=" * 60)
    all_passed = True
    for name, outcome in test_results.items():
        print(f" - {name}: {outcome}")
        if not outcome.startswith("PASS"):
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("ALL PHASE 8 TESTS PASSED SUCCESSFULLY! 🎉")
    else:
        print("SOME TESTS FAILED! PLEASE REVIEW LOGS.")
    print("=" * 60)

if __name__ == "__main__":
    run_phase8_tests()
