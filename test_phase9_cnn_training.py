"""
Test Suite for TerraFlare Phase 9: CNN MobileNetV3 Training Manager
Runs automated validation of classification dataset selection, MobileNetV3-Small transfer learning, class-weighted loss, test evaluation, confusion matrix generation, model registry, deployment to CNNVerifier & Grad-CAM, and regression tests.
"""

import os
import shutil
import tempfile
import numpy as np
import cv2
import torch
from PIL import Image

from dataset_registry import save_dataset_metadata, get_next_dataset_id
from training.cnn_trainer import train_cnn_mobilenetv3
from training.cnn_evaluator import evaluate_cnn_experiment
from training.model_registry import list_registered_models, get_registered_model
from training.training_ui import deploy_cnn_model
from cnn_verifier import CNNVerifier
from gradcam import GradCAMExplainer
from risk_engine import RiskEngine

def create_dummy_image(filepath, color=(0, 255, 0)):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    img_array = np.zeros((128, 128, 3), dtype=np.uint8)
    img_array[:] = color
    cv2.imwrite(filepath, img_array)

def run_phase9_tests():
    print("=" * 60)
    print("RUNNING TERRAFLARE PHASE 9: CNN MOBILENETV3 TRAINING TEST SUITE")
    print("=" * 60)

    test_results = {}
    sandbox_dir = tempfile.mkdtemp(prefix="terraflare_phase9_test_")

    try:
        # 1. Create a dummy classification dataset (NON_FIRE, SMOKE, FIRE)
        cls_dir = os.path.join(sandbox_dir, "fire_smoke_cls")
        
        # Train images
        create_dummy_image(os.path.join(cls_dir, "train", "NON_FIRE", "nf1.jpg"), (0, 255, 0))
        create_dummy_image(os.path.join(cls_dir, "train", "NON_FIRE", "nf2.jpg"), (0, 250, 0))
        create_dummy_image(os.path.join(cls_dir, "train", "SMOKE", "s1.jpg"), (128, 128, 128))
        create_dummy_image(os.path.join(cls_dir, "train", "SMOKE", "s2.jpg"), (130, 130, 130))
        create_dummy_image(os.path.join(cls_dir, "train", "FIRE", "f1.jpg"), (0, 0, 255))
        create_dummy_image(os.path.join(cls_dir, "train", "FIRE", "f2.jpg"), (0, 10, 250))

        # Val images
        create_dummy_image(os.path.join(cls_dir, "val", "NON_FIRE", "nf_v.jpg"), (0, 255, 0))
        create_dummy_image(os.path.join(cls_dir, "val", "SMOKE", "s_v.jpg"), (128, 128, 128))
        create_dummy_image(os.path.join(cls_dir, "val", "FIRE", "f_v.jpg"), (0, 0, 255))

        # Test images
        create_dummy_image(os.path.join(cls_dir, "test", "NON_FIRE", "nf_t.jpg"), (0, 255, 0))
        create_dummy_image(os.path.join(cls_dir, "test", "SMOKE", "s_t.jpg"), (128, 128, 128))
        create_dummy_image(os.path.join(cls_dir, "test", "FIRE", "f_t.jpg"), (0, 0, 255))

        ds_id = get_next_dataset_id()
        ds_meta = {
            "dataset_id": ds_id,
            "dataset_name": "Phase9_FireSmoke_Classification",
            "dataset_type": "Classification",
            "dataset_path": cls_dir,
            "classes": ["NON_FIRE", "SMOKE", "FIRE"],
            "total_images": 12,
            "train_count": 6,
            "validation_count": 3,
            "test_count": 3,
            "validation_status": "Valid"
        }
        save_dataset_metadata(ds_id, ds_meta)

        # -------------------------------------------------------------
        # TEST 1: MobileNetV3 Transfer Learning Training Execution
        # -------------------------------------------------------------
        print("\n[TEST 1] MobileNetV3 Training Execution (2 epochs)...")
        exp_config = train_cnn_mobilenetv3(
            dataset_metadata=ds_meta,
            experiment_name="Phase9_Test_MobileNetV3",
            epochs=2,
            batch_size=2,
            learning_rate=0.001,
            image_size=224,
            device_choice="Auto",
            use_class_weighting=True
        )
        test1_pass = (
            exp_config.get("status") == "SUCCESS" and
            os.path.exists(exp_config.get("best_model_path", "")) and
            os.path.exists(exp_config.get("last_model_path", ""))
        )
        test_results["1. MobileNetV3 Training Execution"] = "PASS" if test1_pass else "FAIL"
        print(f"Result: {test_results['1. MobileNetV3 Training Execution']} (Exp ID: {exp_config.get('experiment_id')})")

        # -------------------------------------------------------------
        # TEST 2: Test Dataset Evaluation & Confusion Matrix Generation
        # -------------------------------------------------------------
        print("\n[TEST 2] Test Dataset Evaluation & Confusion Matrix...")
        metrics = evaluate_cnn_experiment(exp_config, ds_meta)
        test2_pass = (
            "accuracy" in metrics and
            "precision" in metrics and
            "recall" in metrics and
            "f1" in metrics and
            os.path.exists(metrics.get("confusion_matrix_path", ""))
        )
        test_results["2. Test Evaluation & Confusion Matrix"] = "PASS" if test2_pass else "FAIL"
        print(f"Result: {test_results['2. Test Evaluation & Confusion Matrix']} (Accuracy: {metrics.get('accuracy')*100:.2f}%)")

        # -------------------------------------------------------------
        # TEST 3: Model Registry Entry Verification
        # -------------------------------------------------------------
        print("\n[TEST 3] Model Registry Entry...")
        cnn_models = [m for m in list_registered_models() if m.get("experiment_id") == exp_config.get("experiment_id")]
        test3_pass = len(cnn_models) > 0
        test_results["3. Model Registry Entry"] = "PASS" if test3_pass else "FAIL"
        print(f"Result: {test_results['3. Model Registry Entry']} (Found registered entries: {len(cnn_models)})")

        # -------------------------------------------------------------
        # TEST 4: Deploy as CNN Verifier
        # -------------------------------------------------------------
        print("\n[TEST 4] Deploying Model as Active CNN Verifier...")
        if test3_pass:
            model_entry = cnn_models[0]
            deploy_cnn_model(model_entry)
            
            target_weights = "weights/cnn_verifier.pt"
            target_cmap = "weights/cnn_class_mapping.json"
            test4_pass = os.path.exists(target_weights) and os.path.exists(target_cmap)
        else:
            test4_pass = False
        test_results["4. Deploy as CNN Verifier"] = "PASS" if test4_pass else "FAIL"
        print(f"Result: {test_results['4. Deploy as CNN Verifier']}")

        # -------------------------------------------------------------
        # TEST 5: Integration with Deployed CNNVerifier & Grad-CAM
        # -------------------------------------------------------------
        print("\n[TEST 5] Deployed CNNVerifier & Grad-CAM Integration...")
        verifier = CNNVerifier(weights_path="weights/cnn_verifier.pt", mapping_path="weights/cnn_class_mapping.json")
        gradcam = GradCAMExplainer(verifier)

        test_crop = np.zeros((100, 100, 3), dtype=np.uint8)
        test_crop[:, :] = (0, 0, 255) # Red fire-like crop

        pred_res = verifier.predict_crop(test_crop)
        cam_res = gradcam.generate_gradcam(test_crop, target_class_name="FIRE")

        test5_pass = (
            verifier.is_available() and
            pred_res.get("available") == True and
            cam_res.get("available") == True
        )
        test_results["5. CNNVerifier & Grad-CAM Integration"] = "PASS" if test5_pass else f"FAIL (Pred: {pred_res}, CAM: {cam_res})"
        conf_val = pred_res.get('confidence')
        conf_str = f"{conf_val:.2f}" if conf_val is not None else "N/A"
        print(f"Result: {test_results['5. CNNVerifier & Grad-CAM Integration']} (Prediction: {pred_res.get('result')}, Conf: {conf_str})")

        # -------------------------------------------------------------
        # TEST 6: Regression Tests for Detection Pipeline
        # -------------------------------------------------------------
        print("\n[TEST 6] Inference System Regression Check...")
        re = RiskEngine()
        risk_res = re.calculate_risk(fire_conf=0.9, smoke_conf=0.4, area_ratio=0.2, verifier_result=pred_res)
        test6_pass = risk_res.get("status") in ["ALERT", "WATCH", "NORMAL"]
        test_results["6. Existing Detection System Regression"] = "PASS" if test6_pass else "FAIL"
        print(f"Result: {test_results['6. Existing Detection System Regression']} (Status: {risk_res.get('status')})")

    finally:
        shutil.rmtree(sandbox_dir, ignore_errors=True)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY OF PHASE 9 TEST RESULTS:")
    print("=" * 60)
    all_passed = True
    for name, outcome in test_results.items():
        print(f" - {name}: {outcome}")
        if not outcome.startswith("PASS"):
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("ALL PHASE 9 TESTS PASSED SUCCESSFULLY! 🎉")
    else:
        print("SOME TESTS FAILED! PLEASE REVIEW LOGS.")
    print("=" * 60)

if __name__ == "__main__":
    run_phase9_tests()
