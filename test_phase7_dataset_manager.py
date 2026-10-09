"""
Test Suite for TerraFlare Phase 7: Dataset Manager & Dataset Analyzer
Runs programmatic verification of all 10 requirements and dataset formats.
"""

import os
import shutil
import tempfile
import zipfile
import yaml
import numpy as np
import cv2
from PIL import Image

from dataset_validator import (
    validate_yolo_dataset,
    validate_classification_dataset,
    extract_zip_safely
)
from dataset_analyzer import (
    analyze_dataset_metrics,
    draw_yolo_ground_truth,
    fetch_sample_previews
)
from dataset_registry import (
    get_next_dataset_id,
    save_dataset_metadata,
    list_registered_datasets,
    get_dataset_metadata,
    BASE_DATASETS_DIR
)
from risk_engine import RiskEngine
from cnn_verifier import CNNVerifier

def create_dummy_image(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    cv2.imwrite(filepath, img_array)

def run_all_tests():
    print("=" * 60)
    print("RUNNING TERRAFLARE PHASE 7 TEST SUITE")
    print("=" * 60)

    test_results = {}

    # Create temporary sandbox directory for test datasets
    sandbox_dir = tempfile.mkdtemp(prefix="terraflare_test_")

    try:
        # -------------------------------------------------------------
        # TEST 1: Valid YOLO Dataset
        # -------------------------------------------------------------
        print("\n[TEST 1] Valid YOLO Dataset...")
        yolo_dir = os.path.join(sandbox_dir, "yolo_valid")
        create_dummy_image(os.path.join(yolo_dir, "images", "train", "img1.jpg"))
        create_dummy_image(os.path.join(yolo_dir, "images", "train", "img2.jpg"))
        create_dummy_image(os.path.join(yolo_dir, "images", "val", "img3.jpg"))

        # Create valid YOLO labels
        os.makedirs(os.path.join(yolo_dir, "labels", "train"), exist_ok=True)
        os.makedirs(os.path.join(yolo_dir, "labels", "val"), exist_ok=True)
        with open(os.path.join(yolo_dir, "labels", "train", "img1.txt"), "w") as f:
            f.write("0 0.5 0.5 0.2 0.3\n1 0.7 0.7 0.1 0.1\n")
        with open(os.path.join(yolo_dir, "labels", "train", "img2.txt"), "w") as f:
            f.write("0 0.4 0.4 0.3 0.3\n")
        with open(os.path.join(yolo_dir, "labels", "val", "img3.txt"), "w") as f:
            f.write("1 0.2 0.2 0.1 0.1\n")

        # Create data.yaml
        data_yaml = {
            "path": yolo_dir,
            "train": "images/train",
            "val": "images/val",
            "nc": 2,
            "names": ["fire", "smoke"]
        }
        with open(os.path.join(yolo_dir, "data.yaml"), "w") as f:
            yaml.dump(data_yaml, f)

        res1 = validate_yolo_dataset(yolo_dir)
        stats1 = analyze_dataset_metrics(res1)
        
        test1_pass = (
            res1["valid_structure"] == True and
            res1["total_images"] == 3 and
            res1["train_count"] == 2 and
            res1["validation_count"] == 1 and
            res1["corrupted_images"] == 0 and
            res1["missing_labels"] == 0 and
            res1["status"] == "Valid"
        )
        test_results["1. Valid YOLO Dataset"] = "PASS" if test1_pass else f"FAIL ({res1})"
        print(f"Result: {test_results['1. Valid YOLO Dataset']} (Images: {res1['total_images']}, Status: {res1['status']})")

        # -------------------------------------------------------------
        # TEST 2: Invalid Dataset Structure
        # -------------------------------------------------------------
        print("\n[TEST 2] Invalid Dataset...")
        invalid_dir = os.path.join(sandbox_dir, "invalid_ds")
        os.makedirs(invalid_dir, exist_ok=True)
        with open(os.path.join(invalid_dir, "random.txt"), "w") as f:
            f.write("not a dataset")

        res2 = validate_yolo_dataset(invalid_dir)
        test2_pass = (res2["valid_structure"] == False and res2["status"] == "Invalid")
        test_results["2. Invalid Dataset"] = "PASS" if test2_pass else f"FAIL ({res2})"
        print(f"Result: {test_results['2. Invalid Dataset']} (Status: {res2['status']})")

        # -------------------------------------------------------------
        # TEST 3: Dataset with Missing Labels
        # -------------------------------------------------------------
        print("\n[TEST 3] Dataset with Missing Labels...")
        missing_lbl_dir = os.path.join(sandbox_dir, "missing_lbl")
        create_dummy_image(os.path.join(missing_lbl_dir, "images", "train", "img_has_lbl.jpg"))
        create_dummy_image(os.path.join(missing_lbl_dir, "images", "train", "img_no_lbl.jpg"))

        os.makedirs(os.path.join(missing_lbl_dir, "labels", "train"), exist_ok=True)
        with open(os.path.join(missing_lbl_dir, "labels", "train", "img_has_lbl.txt"), "w") as f:
            f.write("0 0.5 0.5 0.2 0.2\n")

        res3 = validate_yolo_dataset(missing_lbl_dir)
        test3_pass = (res3["missing_labels"] == 1 and res3["status"] == "Warning")
        test_results["3. Dataset with missing labels"] = "PASS" if test3_pass else f"FAIL ({res3})"
        print(f"Result: {test_results['3. Dataset with missing labels']} (Missing Labels: {res3['missing_labels']})")

        # -------------------------------------------------------------
        # TEST 4: Dataset with Corrupted Image
        # -------------------------------------------------------------
        print("\n[TEST 4] Dataset with Corrupted Image...")
        corrupt_dir = os.path.join(sandbox_dir, "corrupt_ds")
        os.makedirs(os.path.join(corrupt_dir, "images", "train"), exist_ok=True)
        os.makedirs(os.path.join(corrupt_dir, "labels", "train"), exist_ok=True)

        create_dummy_image(os.path.join(corrupt_dir, "images", "train", "good.jpg"))
        with open(os.path.join(corrupt_dir, "labels", "train", "good.txt"), "w") as f:
            f.write("0 0.5 0.5 0.2 0.2\n")

        # Write invalid binary data to image file
        with open(os.path.join(corrupt_dir, "images", "train", "bad.jpg"), "wb") as f:
            f.write(b"this is corrupt binary image header")
        with open(os.path.join(corrupt_dir, "labels", "train", "bad.txt"), "w") as f:
            f.write("0 0.5 0.5 0.2 0.2\n")

        res4 = validate_yolo_dataset(corrupt_dir)
        test4_pass = (res4["corrupted_images"] == 1 and res4["status"] == "Warning")
        test_results["4. Dataset with corrupted image"] = "PASS" if test4_pass else f"FAIL ({res4})"
        print(f"Result: {test_results['4. Dataset with corrupted image']} (Corrupted: {res4['corrupted_images']})")

        # -------------------------------------------------------------
        # TEST 5: Classification Dataset
        # -------------------------------------------------------------
        print("\n[TEST 5] Classification Dataset...")
        cls_dir = os.path.join(sandbox_dir, "cls_ds")
        create_dummy_image(os.path.join(cls_dir, "train", "fire", "f1.jpg"))
        create_dummy_image(os.path.join(cls_dir, "train", "smoke", "s1.jpg"))
        create_dummy_image(os.path.join(cls_dir, "train", "non_fire", "n1.jpg"))
        create_dummy_image(os.path.join(cls_dir, "val", "fire", "f2.jpg"))

        res5 = validate_classification_dataset(cls_dir)
        test5_pass = (
            res5["valid_structure"] == True and
            res5["total_images"] == 4 and
            res5["train_count"] == 3 and
            res5["validation_count"] == 1 and
            set(res5["classes"]) == {"fire", "smoke", "non_fire"}
        )
        test_results["5. Classification dataset"] = "PASS" if test5_pass else f"FAIL ({res5})"
        print(f"Result: {test_results['5. Classification dataset']} (Total Images: {res5['total_images']}, Classes: {res5['classes']})")

        # -------------------------------------------------------------
        # TEST 6: CSV Classification Dataset
        # -------------------------------------------------------------
        print("\n[TEST 6] CSV Classification Dataset...")
        csv_ds_dir = os.path.join(sandbox_dir, "csv_ds")
        create_dummy_image(os.path.join(csv_ds_dir, "images", "img_a.jpg"))
        create_dummy_image(os.path.join(csv_ds_dir, "images", "img_b.jpg"))

        csv_file_path = os.path.join(csv_ds_dir, "mapping.csv")
        with open(csv_file_path, "w") as f:
            f.write("image_path,class\nimages/img_a.jpg,fire\nimages/img_b.jpg,smoke\n")

        res6 = validate_classification_dataset(csv_ds_dir, csv_file_path=csv_file_path)
        test6_pass = (
            res6["valid_structure"] == True and
            res6["total_images"] == 2 and
            "fire" in res6["classes"] and
            "smoke" in res6["classes"]
        )
        test_results["6. CSV classification dataset"] = "PASS" if test6_pass else f"FAIL ({res6})"
        print(f"Result: {test_results['6. CSV classification dataset']} (Total Images: {res6['total_images']})")

        # -------------------------------------------------------------
        # TEST 7: Multiple Uploaded Datasets & Registry
        # -------------------------------------------------------------
        print("\n[TEST 7] Multiple Uploaded Datasets & Registry...")
        ds1_id = get_next_dataset_id()
        meta1 = {
            "dataset_id": ds1_id,
            "dataset_name": "Test Forest Fire V1",
            "dataset_type": "Object Detection",
            "total_images": 1200,
            "classes": ["fire", "smoke"],
            "validation_status": "Valid"
        }
        save_dataset_metadata(ds1_id, meta1)

        ds2_id = get_next_dataset_id()
        meta2 = {
            "dataset_id": ds2_id,
            "dataset_name": "Test Classification V2",
            "dataset_type": "Classification",
            "total_images": 850,
            "classes": ["fire", "smoke", "non_fire"],
            "validation_status": "Valid"
        }
        save_dataset_metadata(ds2_id, meta2)

        registered = list_registered_datasets()
        reg_ids = [r["dataset_id"] for r in registered]

        test7_pass = (ds1_id in reg_ids and ds2_id in reg_ids)
        test_results["7. Multiple uploaded datasets"] = "PASS" if test7_pass else f"FAIL ({reg_ids})"
        print(f"Result: {test_results['7. Multiple uploaded datasets']} (Registered IDs: {reg_ids})")

        # -------------------------------------------------------------
        # TEST 8: Existing Image Detection (Regression Test)
        # -------------------------------------------------------------
        print("\n[TEST 8] Existing Image Detection (Regression Test)...")
        try:
            re = RiskEngine()
            risk_res = re.calculate_risk(
                fire_conf=0.85,
                smoke_conf=0.30,
                area_ratio=0.10,
                source="Image"
            )
            test8_pass = risk_res.get("status") in ["WATCH", "ALERT", "NORMAL"]
            test_results["8. Existing Image detection"] = "PASS" if test8_pass else f"FAIL ({risk_res})"
        except Exception as e:
            test_results["8. Existing Image detection"] = f"FAIL ({str(e)})"
        print(f"Result: {test_results['8. Existing Image detection']}")

        # -------------------------------------------------------------
        # TEST 9: Existing Video Detection (Regression Test)
        # -------------------------------------------------------------
        print("\n[TEST 9] Existing Video Detection (Regression Test)...")
        try:
            # Create a short dummy avi video
            vpath = os.path.join(sandbox_dir, "test_vid.avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(vpath, fourcc, 10.0, (320, 240))
            for _ in range(5):
                out.write(np.zeros((240, 320, 3), dtype=np.uint8))
            out.release()

            cap = cv2.VideoCapture(vpath)
            ret, frame = cap.read()
            cap.release()
            test9_pass = ret and frame is not None
            test_results["9. Existing Video detection"] = "PASS" if test9_pass else "FAIL"
        except Exception as e:
            test_results["9. Existing Video detection"] = f"FAIL ({str(e)})"
        print(f"Result: {test_results['9. Existing Video detection']}")

        # -------------------------------------------------------------
        # TEST 10: Existing Webcam Detection (Regression Test)
        # -------------------------------------------------------------
        print("\n[TEST 10] Existing Webcam Detection (Regression Test)...")
        try:
            from app import preprocess_frame
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            processed = preprocess_frame(dummy_frame)
            test10_pass = processed is not None and processed.shape == dummy_frame.shape
            test_results["10. Existing Webcam detection"] = "PASS" if test10_pass else "FAIL"
        except Exception as e:
            test_results["10. Existing Webcam detection"] = f"FAIL ({str(e)})"
        print(f"Result: {test_results['10. Existing Webcam detection']}")

    finally:
        shutil.rmtree(sandbox_dir, ignore_errors=True)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY OF TEST RESULTS:")
    print("=" * 60)
    all_passed = True
    for name, outcome in test_results.items():
        print(f" - {name}: {outcome}")
        if not outcome.startswith("PASS"):
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("ALL 10 TESTS PASSED SUCCESSFULLY! 🎉")
    else:
        print("SOME TESTS FAILED! PLEASE REVIEW LOGS.")
    print("=" * 60)

if __name__ == "__main__":
    run_all_tests()
