"""
TerraFlare - Phase 4 Grad-CAM Explainability Module
Generates visual Grad-CAM heatmap overlays for the MobileNetV3 2nd-stage verifier.
Explains model predictions (FIRE, SMOKE, NON_FIRE) by highlighting key spatial features.
"""

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

class GradCAMExplainer:
    def __init__(self, cnn_verifier):
        self.cnn_verifier = cnn_verifier
        self.transform = cnn_verifier.transform if cnn_verifier else None

    def _get_target_layer(self, model):
        """
        Selects the final convolutional feature layer of MobileNetV3-Small.
        Target layer: model.features[-1] (Conv2dNormActivation block).
        """
        if hasattr(model, 'features') and len(model.features) > 0:
            return model.features[-1]
        return None

    def generate_gradcam(self, crop_bgr, target_class_name=None, target_class_idx=None):
        """
        Generates Grad-CAM heatmap and overlay for a cropped BGR bounding box region.
        """
        if self.cnn_verifier is None or not self.cnn_verifier.is_available():
            return {
                "available": False,
                "crop_rgb": cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB) if crop_bgr is not None else None,
                "overlay_rgb": None,
                "heatmap_rgb": None,
                "target_class": None,
                "confidence": None,
                "message": "Grad-CAM unavailable: CNN verifier weights are not currently loaded."
            }

        if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 5 or crop_bgr.shape[1] < 5:
            return {
                "available": False,
                "crop_rgb": None,
                "overlay_rgb": None,
                "heatmap_rgb": None,
                "target_class": None,
                "confidence": None,
                "message": "Grad-CAM unavailable: Invalid crop dimensions."
            }

        try:
            model = self.cnn_verifier.model
            # Infer device directly from model parameters to guarantee matching types
            device = next(model.parameters()).device if model is not None else self.cnn_verifier.device
            classes = self.cnn_verifier.classes

            target_layer = self._get_target_layer(model)
            if target_layer is None:
                return {
                    "available": False,
                    "crop_rgb": cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB),
                    "overlay_rgb": None,
                    "heatmap_rgb": None,
                    "target_class": None,
                    "confidence": None,
                    "message": "Grad-CAM unavailable: Could not identify target conv layer."
                }

            # Register forward and backward hooks on target layer
            activation_map = []
            gradient_map = []

            def forward_hook(module, input, output):
                activation_map.append(output)

            def backward_hook(module, grad_in, grad_out):
                gradient_map.append(grad_out[0])

            handle_fwd = target_layer.register_forward_hook(forward_hook)
            handle_bwd = target_layer.register_full_backward_hook(backward_hook)

            # Preprocess crop image
            h_orig, w_orig = crop_bgr.shape[:2]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            tensor_img = self.transform(pil_img).unsqueeze(0).to(device)
            tensor_img.requires_grad = True

            # Forward pass
            model.eval()
            model.zero_grad()
            outputs = model(tensor_img)
            probs = torch.softmax(outputs, dim=1)[0].detach().cpu().numpy()

            # Determine target class
            if target_class_idx is None:
                if target_class_name is not None and target_class_name in classes:
                    target_class_idx = classes.index(target_class_name)
                else:
                    target_class_idx = int(np.argmax(probs))

            target_class_name = classes[target_class_idx]
            confidence = float(probs[target_class_idx])

            # Backward pass for target class logit
            score = outputs[0, target_class_idx]
            score.backward()

            # Remove hooks
            handle_fwd.remove()
            handle_bwd.remove()

            if not activation_map or not gradient_map:
                return {
                    "available": False,
                    "crop_rgb": crop_rgb,
                    "overlay_rgb": None,
                    "heatmap_rgb": None,
                    "target_class": target_class_name,
                    "confidence": confidence,
                    "message": "Grad-CAM unavailable: Failed to capture activation or gradients."
                }

            act = activation_map[0].detach().cpu().numpy()[0]  # (C, H, W)
            grad = gradient_map[0].detach().cpu().numpy()[0]   # (C, H, W)

            # Global average pooling of gradients for channel weights
            weights = np.mean(grad, axis=(1, 2))  # (C,)
            cam = np.zeros(act.shape[1:], dtype=np.float32)  # (H, W)

            for i, w in enumerate(weights):
                cam += w * act[i]

            # Apply ReLU and normalize
            cam = np.maximum(cam, 0)
            if np.max(cam) > 0:
                cam = cam / np.max(cam)

            # Resize heatmap to original crop size
            heatmap_resized = cv2.resize(cam, (w_orig, h_orig))
            heatmap_uint8 = np.uint8(255 * heatmap_resized)

            # Apply Jet colormap
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

            # Blend overlay
            overlay_bgr = cv2.addWeighted(crop_bgr, 0.6, heatmap_color, 0.4, 0)
            overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

            return {
                "available": True,
                "crop_rgb": crop_rgb,
                "heatmap_rgb": heatmap_color_rgb,
                "overlay_rgb": overlay_rgb,
                "target_class": target_class_name,
                "confidence": confidence,
                "message": f"Grad-CAM explanation for '{target_class_name}' ({confidence * 100:.1f}%)"
            }

        except Exception as e:
            return {
                "available": False,
                "crop_rgb": cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB) if crop_bgr is not None else None,
                "overlay_rgb": None,
                "heatmap_rgb": None,
                "target_class": None,
                "confidence": None,
                "message": f"Grad-CAM error: {e}"
            }
