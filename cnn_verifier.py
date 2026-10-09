"""
TerraFlare - Phase 3 CNN False-Positive Verification Module
Lightweight PyTorch MobileNetV3 architecture for 3-class verification:
- FIRE
- SMOKE
- NON_FIRE (sunset, fog, dust, foliage, glare, etc.)
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

CLASSES = ["NON_FIRE", "SMOKE", "FIRE"]

class CNNVerifier:
    def __init__(self, weights_path="weights/cnn_verifier.pt", mapping_path="weights/cnn_class_mapping.json"):
        self.weights_path = weights_path
        self.mapping_path = mapping_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        if os.path.exists(self.mapping_path):
            try:
                import json
                with open(self.mapping_path, "r", encoding="utf-8") as f:
                    cmap_data = json.load(f)
                if isinstance(cmap_data, dict) and "classes" in cmap_data:
                    self.classes = cmap_data["classes"]
                elif isinstance(cmap_data, list):
                    self.classes = cmap_data
                else:
                    self.classes = CLASSES
            except Exception:
                self.classes = CLASSES
        else:
            self.classes = CLASSES

        self.model = None
        self.is_trained = False
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self._initialize_model()

    def _initialize_model(self):
        """
        Builds MobileNetV3-Small architecture for 3-class classification.
        Attempts to load fine-tuned weights if available on disk.
        """
        try:
            base_model = mobilenet_v3_small(weights=None)
            num_ftrs = base_model.classifier[3].in_features
            base_model.classifier[3] = nn.Linear(num_ftrs, len(self.classes))

            if os.path.exists(self.weights_path):
                try:
                    loaded = torch.load(self.weights_path, map_location=self.device, weights_only=False)
                except Exception:
                    loaded = torch.load(self.weights_path, map_location=self.device)
                
                if isinstance(loaded, dict) and "model" in loaded:
                    sub = loaded["model"]
                    if hasattr(sub, "state_dict"):
                        state_dict = sub.state_dict()
                    elif isinstance(sub, dict):
                        state_dict = sub
                    else:
                        state_dict = loaded
                elif hasattr(loaded, "state_dict"):
                    state_dict = loaded.state_dict()
                else:
                    state_dict = loaded

                base_model.load_state_dict(state_dict)
                base_model.to(self.device)
                base_model.eval()
                self.model = base_model
                self.is_trained = True
                print(f"[CNNVerifier] Loaded fine-tuned weights from {self.weights_path}")
            else:
                self.model = base_model
                self.is_trained = False
                print(f"[CNNVerifier] Fine-tuned weights not found at '{self.weights_path}'. Verifier set to NOT AVAILABLE.")
        except Exception as e:
            self.model = None
            self.is_trained = False
            print(f"[CNNVerifier] Error initializing CNN verifier: {e}")

    def is_available(self) -> bool:
        """Returns True only if trained weights are loaded and model is ready."""
        return self.is_trained and self.model is not None

    def crop_detection_regions(self, image_bgr, results):
        """
        Crops detected bounding box regions from the original BGR frame.
        """
        crops = []
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                h_img, w_img = image_bgr.shape[:2]

                for box in xyxy:
                    x1 = max(0, int(box[0]))
                    y1 = max(0, int(box[1]))
                    x2 = min(w_img, int(box[2]))
                    y2 = min(h_img, int(box[3]))

                    if (x2 - x1) > 5 and (y2 - y1) > 5:
                        crop = image_bgr[y1:y2, x1:x2]
                        crops.append(crop)
        return crops

    def predict_crop(self, crop_bgr):
        """
        Runs CNN inference on a cropped BGR image region.
        Returns prediction dict if available, else unavailable status.
        """
        if not self.is_available():
            return {
                "available": False,
                "status": "NOT AVAILABLE",
                "result": None,
                "confidence": None,
                "verified": None,
                "verifier_score": None,
                "message": "CNN verifier fine-tuned weights not available."
            }

        try:
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            tensor_img = self.transform(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(tensor_img)
                probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

            top_idx = int(np.argmax(probs))
            predicted_class = self.classes[top_idx]
            confidence = float(probs[top_idx])
            pred_upper = str(predicted_class).upper()

            # Verified if predicted FIRE or SMOKE; Rejected if CLOUD, FOG, NORMAL_FOREST, NON_FIRE
            verified = (pred_upper in ["FIRE", "SMOKE"])
            
            # Verifier score & suppression factor passed to Risk Engine
            if pred_upper == "FIRE":
                verifier_score = confidence
                suppression_factor = 1.0
            elif pred_upper == "SMOKE":
                verifier_score = confidence * 0.7
                suppression_factor = 1.0
            elif pred_upper in ["CLOUD", "FOG"]:
                verifier_score = 0.0  # False positive rejection
                suppression_factor = 0.15 # Strong non-fire suppression
            elif pred_upper in ["NORMAL_FOREST", "NON_FIRE"]:
                verifier_score = 0.0  # False positive rejection
                suppression_factor = 0.10 # Max non-fire suppression
            else:
                verifier_score = 0.0
                suppression_factor = 0.30

            return {
                "available": True,
                "status": "ACTIVE",
                "result": predicted_class,
                "confidence": confidence,
                "verified": verified,
                "verifier_score": verifier_score,
                "suppression_factor": suppression_factor,
                "all_probs": {cls_name: float(probs[i]) for i, cls_name in enumerate(self.classes)},
                "message": f"Verified as {predicted_class} ({confidence*100:.1f}%)" if verified else f"Rejected as False Positive ({predicted_class}: {confidence*100:.1f}%)"
            }
        except Exception as e:
            return {
                "available": False,
                "status": "ERROR",
                "result": None,
                "confidence": None,
                "verified": None,
                "verifier_score": None,
                "message": f"Inference error: {e}"
            }

    def verify_yolo_detections(self, image_bgr, results):
        """
        Extracts crops for all detections and evaluates them with the CNN verifier.
        Returns the overall verification summary.
        """
        if not self.is_available():
            return {
                "available": False,
                "status": "NOT AVAILABLE",
                "result": None,
                "confidence": None,
                "verified": None,
                "verifier_score": None,
                "message": "CNN Verifier: Not trained / Not available (20% weight reserved)"
            }

        crops = self.crop_detection_regions(image_bgr, results)
        if not crops:
            full_res = self.predict_crop(image_bgr)
            if full_res and full_res.get("available"):
                full_res["message"] += " (Full Image Verification)"
            return full_res

        # Run verification across crops
        crop_results = [self.predict_crop(c) for c in crops]
        
        # Take crop with highest verifier score
        best_crop_res = max(crop_results, key=lambda x: x.get("verifier_score", 0.0) or 0.0)
        return best_crop_res
