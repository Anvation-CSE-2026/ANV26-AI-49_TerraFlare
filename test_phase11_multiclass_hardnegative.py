"""
Test Suite for TerraFlare Phase 11: Multi-Class Fire/Smoke + Hard-Negative Detection
Verifies multi-class dataset recognition (FIRE, SMOKE, CLOUD, FOG, NORMAL_FOREST), hard-negative risk suppression, Grad-CAM explanation, False-Positive Protection rendering, and regression checks.
"""

import os
import shutil
import tempfile
import json
import numpy as np
import cv2
import torch

from dataset_registry import save_dataset_metadata, get_next_dataset_id
from training.cnn_trainer import train_cnn_mobilenetv3
from training.cnn_evaluator import evaluate_cnn_experiment
from training.training_ui import deploy_cnn_model
from cnn_verifier import CNNVerifier
from gradcam import GradCAMExplainer
from risk_engine import RiskEngine

def create_dummy_image(filepath, color=(0, 255, 0)):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    img_array = np.zeros((128, 128, 3), dtype=np.uint8)
    img_array[:] = color
    cv2.imwrite(filepath, img_array)

def run_phase11_tests():
    print("=" * 60)
    print("RUNNING TERRAFLARE PHASE 11: MULTI-CLASS & HARD-NEGATIVE TEST SUITE")
    print("=" * 60)

    test_results = {}
    sandbox_dir = tempfile.mkdtemp(prefix="terraflare_phase11_test_")

    try:
        # 1. Create a multi-class dataset with hard negatives: FIRE, SMOKE, CLOUD, FOG, NORMAL_FOREST
        cls_dir = os.path.join(sandbox_dir, "multiclass_hardnegative_ds")
        classes = ["FIRE", "SMOKE", "CLOUD", "FOG", "NORMAL_FOREST"]
        
        for cname in classes:
            color = (0, 0, 255) if cname == "FIRE" else (128, 128, 128) if cname in ["SMOKE", "CLOUD", "FOG"] else (0, 255, 0)
            create_dummy_image(os.path.join(cls_dir, "train", cname, "t1.jpg"), color)
            create_dummy_image(os.path.join(cls_dir, "val", cname, "v1.jpg"), color)
            create_dummy_image(os.path.join(cls_dir, "test", cname, "te1.jpg"), color)

        ds_id = get_next_dataset_id()
        ds_meta = {
            "dataset_id": ds_id,
            "dataset_name": "Phase11_MultiClass_HardNegative_Dataset",
            "dataset_type": "Classification",
            "dataset_path": cls_dir,
            "classes": classes,
            "total_images": 15,
            "train_count": 5,
            "validation_count": 5,
            "test_count": 5,
            "validation_status": "Valid"
        }
        save_dataset_metadata(ds_id, ds_meta)

        # -------------------------------------------------------------
        # TEST 1: Multi-Class MobileNetV3 Training & Mapping Saving
        # -------------------------------------------------------------
        print("\n[TEST 1] Multi-Class MobileNetV3 Transfer Learning Training...")
        exp_config = train_cnn_mobilenetv3(
            dataset_metadata=ds_meta,
            experiment_name="Phase11_MultiClass_Run",
            epochs=1,
            batch_size=2,
            learning_rate=0.001,
            image_size=224,
            device_choice="Auto",
            use_class_weighting=True
        )

        best_pth = exp_config.get("best_model_path", "")
        exp_dir = os.path.dirname(best_pth)
        cmap_file = os.path.join(exp_dir, "class_mapping.json")

        t1_pass = (
            exp_config.get("status") == "SUCCESS" and
            os.path.exists(best_pth) and
            os.path.exists(cmap_file)
        )
        test_results["1. Multi-Class MobileNetV3 Training"] = "PASS" if t1_pass else f"FAIL ({exp_config})"
        print(f"Result: {test_results['1. Multi-Class MobileNetV3 Training']} (Saved mapping at {cmap_file})")

        # -------------------------------------------------------------
        # TEST 2: Deploy Multi-Class Model to CNNVerifier
        # -------------------------------------------------------------
        print("\n[TEST 2] Deploying Multi-Class Model to CNNVerifier...")
        model_entry = {
            "model_id": "CNN_MODEL_PHASE11",
            "best_weights_path": best_pth
        }
        deploy_cnn_model(model_entry)

        verifier = CNNVerifier(weights_path="weights/cnn_verifier.pt", mapping_path="weights/cnn_class_mapping.json")
        t2_pass = verifier.is_available() and set(verifier.classes) == set(classes)
        test_results["2. Deploy Multi-Class Model to CNNVerifier"] = "PASS" if t2_pass else f"FAIL (Classes: {verifier.classes})"
        print(f"Result: {test_results['2. Deploy Multi-Class Model to CNNVerifier']} (Verifier classes: {verifier.classes})")

        # -------------------------------------------------------------
        # TEST 3: Hard-Negative Prediction & Suppression in Risk Engine
        # -------------------------------------------------------------
        print("\n[TEST 3] Hard-Negative Suppression in Risk Engine...")
        re = RiskEngine()

        # Simulated crop prediction for CLOUD
        cloud_verifier_res = {
            "available": True,
            "status": "ACTIVE",
            "result": "CLOUD",
            "confidence": 0.94,
            "verified": False,
            "verifier_score": 0.0,
            "suppression_factor": 0.15,
            "message": "Rejected as False Positive (CLOUD: 94.0%)"
        }

        # Calculate risk with high YOLO smoke confidence (0.85) but suppressed by CLOUD verification
        risk_res_cloud = re.calculate_risk(fire_conf=0.0, smoke_conf=0.85, area_ratio=0.1, verifier_result=cloud_verifier_res)

        # Risk score should be strongly suppressed to NORMAL status (< 0.40)
        t3_pass = risk_res_cloud.get("status") == "NORMAL" and risk_res_cloud.get("risk_score") < 0.40
        test_results["3. Hard-Negative Suppression in Risk Engine"] = "PASS" if t3_pass else f"FAIL ({risk_res_cloud})"
        print(f"Result: {test_results['3. Hard-Negative Suppression in Risk Engine']} (Status: {risk_res_cloud.get('status')}, Risk Score: {risk_res_cloud.get('risk_score'):.2f})")

        # -------------------------------------------------------------
        # TEST 4: Grad-CAM Explanation for CLOUD / FOG
        # -------------------------------------------------------------
        print("\n[TEST 4] Grad-CAM Explanation for Hard-Negative Classes...")
        gradcam = GradCAMExplainer(verifier)
        dummy_crop = np.zeros((100, 100, 3), dtype=np.uint8)
        dummy_crop[:, :] = (128, 128, 128)

        cam_res = gradcam.generate_gradcam(dummy_crop, target_class_name="CLOUD")
        t4_pass = cam_res.get("available") == True and cam_res.get("target_class") == "CLOUD"
        test_results["4. Grad-CAM Hard-Negative Explanation"] = "PASS" if t4_pass else f"FAIL ({cam_res})"
        print(f"Result: {test_results['4. Grad-CAM Hard-Negative Explanation']} (Target class explained: {cam_res.get('target_class')})")

        # -------------------------------------------------------------
        # TEST 5: Full Detection Pipeline Regression Check
        # -------------------------------------------------------------
        print("\n[TEST 5] Full Detection System Regression Check...")
        fire_verifier_res = {
            "available": True,
            "status": "ACTIVE",
            "result": "FIRE",
            "confidence": 0.95,
            "verified": True,
            "verifier_score": 0.95,
            "suppression_factor": 1.0,
            "message": "Verified as FIRE (95.0%)"
        }
        risk_res_fire = re.calculate_risk(fire_conf=0.95, smoke_conf=0.40, area_ratio=0.15, verifier_result=fire_verifier_res)
        t5_pass = risk_res_fire.get("status") == "ALERT" and risk_res_fire.get("risk_score") >= 0.70
        test_results["5. Full Detection System Regression"] = "PASS" if t5_pass else f"FAIL ({risk_res_fire})"
        print(f"Result: {test_results['5. Full Detection System Regression']} (Status: {risk_res_fire.get('status')}, Risk Score: {risk_res_fire.get('risk_score'):.2f})")

    finally:
        shutil.rmtree(sandbox_dir, ignore_errors=True)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY OF PHASE 11 TEST RESULTS:")
    print("=" * 60)
    all_passed = True
    for name, outcome in test_results.items():
        print(f" - {name}: {outcome}")
        if not outcome.startswith("PASS"):
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("ALL PHASE 11 TESTS PASSED SUCCESSFULLY! 🎉")
    else:
        print("SOME TESTS FAILED! PLEASE REVIEW LOGS.")
    print("=" * 60)

if __name__ == "__main__":
    run_phase11_tests()
