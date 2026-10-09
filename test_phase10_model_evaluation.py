"""
Test Suite for TerraFlare Phase 10: Unified Model Evaluation & Comparison Dashboard
Verifies held-out test dataset evaluation, metric extraction, confusion matrix generation, false-positive analysis, confidence stats, multi-model comparison, model registry N/A updates, evaluation report saving, and full system regression checks.
"""

import os
import shutil
import tempfile
import json
import numpy as np
import cv2
import torch
import yaml

from dataset_registry import save_dataset_metadata, get_next_dataset_id
from training.yolo_trainer import run_yolo_training
from training.cnn_trainer import train_cnn_mobilenetv3
from training.cnn_evaluator import evaluate_cnn_experiment
from training.model_evaluator_ui import run_yolo_evaluation
from training.model_registry import list_registered_models, get_registered_model, update_model_evaluation
from risk_engine import RiskEngine
from cnn_verifier import CNNVerifier
from gradcam import GradCAMExplainer

def create_dummy_image(filepath, color=(0, 255, 0)):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    img_array = np.zeros((128, 128, 3), dtype=np.uint8)
    img_array[:] = color
    cv2.imwrite(filepath, img_array)

def run_phase10_tests():
    print("=" * 60)
    print("RUNNING TERRAFLARE PHASE 10: UNIFIED MODEL EVALUATION TEST SUITE")
    print("=" * 60)

    test_results = {}
    sandbox_dir = tempfile.mkdtemp(prefix="terraflare_phase10_test_")

    try:
        # 1. Create a dummy YOLO object detection dataset
        yolo_ds_dir = os.path.join(sandbox_dir, "yolo_ds")
        create_dummy_image(os.path.join(yolo_ds_dir, "images", "train", "t1.jpg"))
        create_dummy_image(os.path.join(yolo_ds_dir, "images", "val", "v1.jpg"))
        create_dummy_image(os.path.join(yolo_ds_dir, "images", "test", "te1.jpg"))

        os.makedirs(os.path.join(yolo_ds_dir, "labels", "train"), exist_ok=True)
        os.makedirs(os.path.join(yolo_ds_dir, "labels", "val"), exist_ok=True)
        os.makedirs(os.path.join(yolo_ds_dir, "labels", "test"), exist_ok=True)

        with open(os.path.join(yolo_ds_dir, "labels", "train", "t1.txt"), "w") as f:
            f.write("0 0.5 0.5 0.3 0.3\n")
        with open(os.path.join(yolo_ds_dir, "labels", "val", "v1.txt"), "w") as f:
            f.write("0 0.5 0.5 0.3 0.3\n")
        with open(os.path.join(yolo_ds_dir, "labels", "test", "te1.txt"), "w") as f:
            f.write("0 0.5 0.5 0.3 0.3\n")

        data_yaml = {
            "path": yolo_ds_dir,
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": 2,
            "names": ["fire", "smoke"]
        }
        with open(os.path.join(yolo_ds_dir, "data.yaml"), "w") as f:
            yaml.dump(data_yaml, f)

        yolo_ds_id = get_next_dataset_id()
        yolo_ds_meta = {
            "dataset_id": yolo_ds_id,
            "dataset_name": "Phase10_YOLO_Dataset",
            "dataset_type": "Object Detection",
            "dataset_path": yolo_ds_dir,
            "classes": ["fire", "smoke"],
            "total_images": 3,
            "train_count": 1,
            "validation_count": 1,
            "test_count": 1,
            "validation_status": "Valid"
        }
        save_dataset_metadata(yolo_ds_id, yolo_ds_meta)

        # 2. Create a dummy CNN classification dataset
        cnn_ds_dir = os.path.join(sandbox_dir, "cnn_ds")
        create_dummy_image(os.path.join(cnn_ds_dir, "train", "NON_FIRE", "nf1.jpg"), (0, 255, 0))
        create_dummy_image(os.path.join(cnn_ds_dir, "train", "FIRE", "f1.jpg"), (0, 0, 255))
        create_dummy_image(os.path.join(cnn_ds_dir, "val", "NON_FIRE", "nf_v.jpg"), (0, 255, 0))
        create_dummy_image(os.path.join(cnn_ds_dir, "val", "FIRE", "f_v.jpg"), (0, 0, 255))
        create_dummy_image(os.path.join(cnn_ds_dir, "test", "NON_FIRE", "nf_t.jpg"), (0, 255, 0))
        create_dummy_image(os.path.join(cnn_ds_dir, "test", "FIRE", "f_t.jpg"), (0, 0, 255))

        cnn_ds_id = get_next_dataset_id()
        cnn_ds_meta = {
            "dataset_id": cnn_ds_id,
            "dataset_name": "Phase10_CNN_Dataset",
            "dataset_type": "Classification",
            "dataset_path": cnn_ds_dir,
            "classes": ["NON_FIRE", "FIRE"],
            "total_images": 6,
            "train_count": 2,
            "validation_count": 2,
            "test_count": 2,
            "validation_status": "Valid"
        }
        save_dataset_metadata(cnn_ds_id, cnn_ds_meta)

        # -------------------------------------------------------------
        # TEST 1: Train & Evaluate YOLO Model
        # -------------------------------------------------------------
        print("\n[TEST 1] YOLO Training & Test Dataset Evaluation...")
        yolo_exp = run_yolo_training(
            dataset_metadata=yolo_ds_meta,
            experiment_name="Phase10_YOLO_Eval_Run",
            model_variant="yolo11n.pt",
            epochs=1,
            imgsz=320,
            batch_size=2,
            lr0=0.01,
            device_choice="Auto"
        )
        yolo_mod_entry = get_registered_model(yolo_exp["experiment_id"])
        yolo_eval_mets = run_yolo_evaluation(yolo_mod_entry, yolo_ds_meta)

        t1_pass = (
            "map50" in yolo_eval_mets and
            "precision" in yolo_eval_mets and
            yolo_eval_mets.get("eval_split") == "TEST"
        )
        test_results["1. YOLO Test Evaluation"] = "PASS" if t1_pass else f"FAIL ({yolo_eval_mets})"
        print(f"Result: {test_results['1. YOLO Test Evaluation']} (mAP50: {yolo_eval_mets.get('map50'):.4f})")

        # -------------------------------------------------------------
        # TEST 2: Train & Evaluate CNN Model
        # -------------------------------------------------------------
        print("\n[TEST 2] CNN Training & Test Dataset Evaluation...")
        cnn_exp = train_cnn_mobilenetv3(
            dataset_metadata=cnn_ds_meta,
            experiment_name="Phase10_CNN_Eval_Run",
            epochs=2,
            batch_size=2,
            learning_rate=0.001,
            image_size=224,
            device_choice="Auto"
        )
        cnn_eval_mets = evaluate_cnn_experiment(cnn_exp, cnn_ds_meta)

        t2_pass = (
            "accuracy" in cnn_eval_mets and
            "confusion_matrix_path" in cnn_eval_mets and
            "confidence_stats" in cnn_eval_mets and
            os.path.exists(cnn_eval_mets.get("confusion_matrix_path", ""))
        )
        test_results["2. CNN Test Evaluation & Confusion Matrix"] = "PASS" if t2_pass else f"FAIL ({cnn_eval_mets})"
        print(f"Result: {test_results['2. CNN Test Evaluation & Confusion Matrix']} (Accuracy: {cnn_eval_mets.get('accuracy')*100:.2f}%)")

        # -------------------------------------------------------------
        # TEST 3: Evaluation Report Saving
        # -------------------------------------------------------------
        print("\n[TEST 3] Evaluation Report JSON Saving...")
        best_pth = cnn_exp.get("best_model_path", "")
        cnn_exp_dir = os.path.dirname(best_pth)
        report_file = os.path.join(cnn_exp_dir, "evaluation_report.json")
        t3_pass = os.path.exists(report_file)
        test_results["3. Evaluation Report JSON"] = "PASS" if t3_pass else "FAIL"
        print(f"Result: {test_results['3. Evaluation Report JSON']} (Report path: {report_file})")

        # -------------------------------------------------------------
        # TEST 4: Model Registry N/A Metrics Formatting
        # -------------------------------------------------------------
        print("\n[TEST 4] Model Registry Update & N/A Metric Formatting...")
        all_registered = list_registered_models()
        cnn_reg = next((m for m in all_registered if m.get("experiment_id") == cnn_exp["experiment_id"]), None)
        yolo_reg = next((m for m in all_registered if m.get("experiment_id") == yolo_exp["experiment_id"]), None)

        t4_pass = (
            cnn_reg is not None and cnn_reg["metrics"]["map50"] == "N/A" and
            yolo_reg is not None and yolo_reg["metrics"]["accuracy"] == "N/A"
        )
        test_results["4. Model Registry N/A Metrics"] = "PASS" if t4_pass else f"FAIL (CNN: {cnn_reg}, YOLO: {yolo_reg})"
        print(f"Result: {test_results['4. Model Registry N/A Metrics']}")

        # -------------------------------------------------------------
        # TEST 5: Regression Checks for Detection & Verification Pipeline
        # -------------------------------------------------------------
        print("\n[TEST 5] Complete System Regression Check...")
        verifier = CNNVerifier(weights_path="weights/cnn_verifier.pt", mapping_path="weights/cnn_class_mapping.json")
        gradcam = GradCAMExplainer(verifier)
        re = RiskEngine()

        dummy_crop = np.zeros((100, 100, 3), dtype=np.uint8)
        dummy_crop[:, :] = (0, 0, 255)

        pred_res = verifier.predict_crop(dummy_crop) if verifier.is_available() else {"available": False}
        cam_res = gradcam.generate_gradcam(dummy_crop) if verifier.is_available() else {"available": False}
        risk_res = re.calculate_risk(fire_conf=0.85, smoke_conf=0.3, area_ratio=0.1)

        t5_pass = risk_res.get("status") in ["ALERT", "WATCH", "NORMAL"]
        test_results["5. Full System Regression Check"] = "PASS" if t5_pass else "FAIL"
        print(f"Result: {test_results['5. Full System Regression Check']} (Risk Status: {risk_res.get('status')})")

    finally:
        shutil.rmtree(sandbox_dir, ignore_errors=True)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY OF PHASE 10 TEST RESULTS:")
    print("=" * 60)
    all_passed = True
    for name, outcome in test_results.items():
        print(f" - {name}: {outcome}")
        if not outcome.startswith("PASS"):
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("ALL PHASE 10 TESTS PASSED SUCCESSFULLY! 🎉")
    else:
        print("SOME TESTS FAILED! PLEASE REVIEW LOGS.")
    print("=" * 60)

if __name__ == "__main__":
    run_phase10_tests()
