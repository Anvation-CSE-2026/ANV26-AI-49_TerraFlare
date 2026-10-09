import numpy as np
import cv2
import time
from severity_engine import classify_severity, generate_simulated_alert
from cnn_verifier import CNNVerifier
from gradcam import GradCAMExplainer
from map_engine import render_geospatial_map

class RiskEngine:
    """
    TERRAFLARE Risk Assessment Engine (Phase 1, 2, 3, 4, 5 Compatible)
    Calculates unified risk score:
    - Fire Confidence Weight: 45%
    - Smoke Confidence Weight: 20%
    - Area Ratio Weight: 15%
    - CNN Verifier Weight: 20%
    """
    def __init__(self, w_fire=0.45, w_smoke=0.20, w_area=0.15, w_verifier=0.20, cnn_verifier=None):
        self.w_fire = w_fire
        self.w_smoke = w_smoke
        self.w_area = w_area
        self.w_verifier = w_verifier
        if cnn_verifier is not None:
            self.cnn_verifier = cnn_verifier
        else:
            self.cnn_verifier = CNNVerifier()
        self.verifier_available = self.cnn_verifier.is_available()

    def extract_metrics_from_results(self, results):
        if not results or len(results) == 0:
            return 0.0, 0.0, 0.0

        boxes = results[0].boxes
        if len(boxes) == 0:
            return 0.0, 0.0, 0.0

        names = results[0].names
        orig_shape = results[0].orig_shape
        frame_area = orig_shape[0] * orig_shape[1] if orig_shape else 1.0

        max_fire_conf = 0.0
        max_smoke_conf = 0.0
        total_bbox_area = 0.0

        for box in boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = names.get(cls_id, "").lower()

            xyxy = box.xyxy[0].cpu().numpy()
            w = xyxy[2] - xyxy[0]
            h = xyxy[3] - xyxy[1]
            box_area = w * h
            total_bbox_area += box_area

            if "fire" in cls_name:
                if conf > max_fire_conf:
                    max_fire_conf = conf
            elif "smoke" in cls_name:
                if conf > max_smoke_conf:
                    max_smoke_conf = conf

        area_ratio = min(total_bbox_area / frame_area, 1.0)
        return max_fire_conf, max_smoke_conf, area_ratio

    def calculate_risk(
        self,
        fire_conf: float,
        smoke_conf: float,
        area_ratio: float,
        verifier_result: dict = None,
        gradcam_result: dict = None,
        source: str = "Image",
        latitude: float = 12.9716,
        longitude: float = 77.5946,
        source_location: str = "Bengaluru Demo Location",
        accuracy_m: float = None,
        location_source_type: str = None,
        satellite_verification: dict = None
    ) -> dict:
        v_score = 0.0
        suppression = 1.0
        verifier_status_text = "Phase 1 Fallback (No CNN weights loaded)"

        if verifier_result and verifier_result.get("available"):
            v_score = verifier_result.get("verifier_score", 0.0)
            suppression = verifier_result.get("suppression_factor", 1.0)
            verifier_status_text = verifier_result.get("message", "CNN Active")
        elif self.verifier_available:
            verifier_status_text = "CNN Verifier Weights Available"

        base_risk = (
            (self.w_fire * fire_conf) +
            (self.w_smoke * smoke_conf) +
            (self.w_area * area_ratio)
        )
        risk_score = (base_risk + (self.w_verifier * v_score)) * suppression
        risk_score = max(0.0, min(1.0, risk_score))

        if risk_score >= 0.70:
            status = "ALERT"
            color_code = "#FF5722"
            badge_icon = "🚨"
        elif risk_score >= 0.40:
            status = "WATCH"
            color_code = "#FFB300"
            badge_icon = "🟡"
        else:
            status = "NORMAL"
            color_code = "#00E676"
            badge_icon = "🟢"

        sev_info = classify_severity(risk_score)
        severity = sev_info["severity"]

        simulated_alert = None
        detection_event = None

        if risk_score >= 0.70:
            if accuracy_m is not None:
                from geolocation.location_event import generate_webcam_simulated_alert, create_detection_event
                det_cls = "FIRE" if fire_conf >= smoke_conf else "SMOKE"
                loc_type = location_source_type or "LIVE_BROWSER_GEOLOCATION"
                
                simulated_alert = generate_webcam_simulated_alert(
                    risk_score=risk_score,
                    severity=severity,
                    detected_class=det_cls,
                    fire_confidence=fire_conf,
                    smoke_confidence=smoke_conf,
                    latitude=latitude,
                    longitude=longitude,
                    accuracy=accuracy_m,
                    location_name=source_location,
                    location_source_type=loc_type,
                    source=source,
                    satellite_verification=satellite_verification
                )
                detection_event = create_detection_event(
                    risk_score=risk_score,
                    severity=severity,
                    detected_class=det_cls,
                    fire_confidence=fire_conf,
                    smoke_confidence=smoke_conf,
                    latitude=latitude,
                    longitude=longitude,
                    accuracy=accuracy_m,
                    location_name=source_location,
                    location_source_type=loc_type,
                    source=source,
                    satellite_verification=satellite_verification
                )
            else:
                simulated_alert = generate_simulated_alert(
                    risk_score=risk_score,
                    fire_conf=fire_conf,
                    smoke_conf=smoke_conf,
                    area_ratio=area_ratio,
                    source=source,
                    latitude=latitude,
                    longitude=longitude,
                    source_location=source_location
                )

        return {
            "risk_score": risk_score,
            "status": status,
            "severity": severity,
            "severity_info": sev_info,
            "color": color_code,
            "icon": badge_icon,
            "fire_confidence": float(fire_conf),
            "smoke_confidence": float(smoke_conf),
            "area_ratio": float(area_ratio),
            "source": source,
            "latitude": float(latitude) if latitude is not None else None,
            "longitude": float(longitude) if longitude is not None else None,
            "accuracy_m": float(accuracy_m) if accuracy_m is not None else None,
            "location_source_type": location_source_type,
            "source_location": source_location,
            "simulated_alert": simulated_alert,
            "detection_event": detection_event,
            "verifier_available": self.verifier_available,
            "verifier_status": verifier_status_text,
            "verifier_result": verifier_result,
            "gradcam_result": gradcam_result,
            "satellite_verification": satellite_verification,
            "weights": {
                "fire": self.w_fire,
                "smoke": self.w_smoke,
                "area": self.w_area,
                "verifier": self.w_verifier
            }
        }

    def evaluate_yolo_results(
        self,
        results,
        image_bgr=None,
        verifier_result=None,
        gradcam_result=None,
        source="Image",
        latitude=12.9716,
        longitude=77.5946,
        source_location="Bengaluru Demo Location",
        accuracy_m=None,
        location_source_type=None,
        satellite_verification=None
    ):
        fire_conf, smoke_conf, area_ratio = self.extract_metrics_from_results(results)
        
        # If YOLO detected 0 bboxes but CNN verifier identified FIRE or SMOKE in full frame
        if fire_conf == 0.0 and smoke_conf == 0.0 and verifier_result and verifier_result.get("verified"):
            v_class = str(verifier_result.get("result", "")).upper()
            v_conf = verifier_result.get("confidence", 0.0) or 0.0
            if v_class == "FIRE":
                fire_conf = v_conf * 0.85
                area_ratio = 0.10
            elif v_class == "SMOKE":
                smoke_conf = v_conf * 0.85
                area_ratio = 0.10

        return self.calculate_risk(
            fire_conf,
            smoke_conf,
            area_ratio,
            verifier_result=verifier_result,
            gradcam_result=gradcam_result,
            source=source,
            latitude=latitude,
            longitude=longitude,
            source_location=source_location,
            accuracy_m=accuracy_m,
            location_source_type=location_source_type,
            satellite_verification=satellite_verification
        )


def render_risk_dashboard(risk_data, container=None, render_map=True, is_live_stream=False):
    """
    Renders TERRAFLARE Risk Assessment, AI Verification, Grad-CAM, Satellite Cross-Verification, Map Localization, and Simulated Alert Dashboard.
    """
    import streamlit as st
    from satellite.satellite_ui import render_satellite_panel
    if container is not None:
        container.empty()
    target = container if container is not None else st

    with target.container():
        st.markdown("---")
        st.markdown("### 🔥 Risk Assessment")

        status = risk_data["status"]
        severity = risk_data["severity"]
        score = risk_data["risk_score"]
        color = risk_data["color"]
        icon = risk_data["icon"]

        # Visual Status & Severity Banner
        st.markdown(
            f'''
            <div style="
                background-color: {color}1F;
                border: 2px solid {color};
                border-radius: 10px;
                padding: 14px 20px;
                margin-bottom: 15px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                flex-wrap: wrap;
                gap: 10px;
            ">
                <div style="font-size: 1.2rem; font-weight: 700; color: {color};">
                    {icon} STATUS: {status} &nbsp;|&nbsp; SEVERITY: {severity}
                </div>
                <div style="font-size: 1.4rem; font-weight: 700; color: #FFFFFF;">
                    Risk Score: {score:.2f} <span style="font-size: 1.0rem; opacity: 0.8;">({score * 100:.1f}%)</span>
                </div>
            </div>
            ''',
            unsafe_allow_html=True
        )

        # FIRE DETECTED NEAR [LOCATION] Banner for ALERT status
        if status == "ALERT":
            lat_disp = risk_data.get("latitude")
            lon_disp = risk_data.get("longitude")
            acc_disp = risk_data.get("accuracy_m")
            acc_text = f"±{acc_disp} m" if acc_disp is not None else "N/A"
            loc_name = risk_data.get("source_location") or "Camera Location"

            fire_c = risk_data.get("fire_confidence", 0.0)
            smoke_c = risk_data.get("smoke_confidence", 0.0)
            ver_res = risk_data.get("verifier_result") or {}
            ver_cls = ver_res.get("result")
            det_cls = ver_cls if ver_cls in ["FIRE", "SMOKE"] else ("FIRE" if fire_c >= smoke_c else "SMOKE")

            sat_ver = risk_data.get("satellite_verification") or {}
            sat_status = sat_ver.get("status", "UNAVAILABLE")
            sat_name = sat_ver.get("satellite", "N/A")
            sat_dist = sat_ver.get("distance_km")
            sat_dist_str = f"{sat_dist:.2f} km" if sat_dist is not None else "N/A"

            st.markdown(
                f'''
                <div style="
                    background-color: #450A0A;
                    border: 2px solid #EF4444;
                    border-radius: 8px;
                    padding: 14px 18px;
                    margin-bottom: 15px;
                    color: #FEE2E2;
                ">
                    <div style="font-size: 1.35rem; font-weight: 800; color: #EF4444; margin-bottom: 6px;">
                        🔊 FIRE ALERT — HIGH RISK DETECTED NEAR {str(loc_name).upper()}
                    </div>
                    <div style="font-size: 0.95rem; line-height: 1.6;">
                        <strong>📍 Camera Location:</strong> {loc_name}<br/>
                        <strong>Latitude:</strong> <code>{lat_disp}</code> | <strong>Longitude:</strong> <code>{lon_disp}</code><br/>
                        <strong>Accuracy:</strong> {acc_text}<br/>
                        <strong>Detected Class:</strong> <span style="font-weight: 700; color: #F87171;">{det_cls}</span> (Fire Conf: {fire_c*100:.1f}%, Smoke Conf: {smoke_c*100:.1f}%)<br/>
                        <strong>AI Risk Score:</strong> {score * 100:.1f}% | <strong>Severity:</strong> {severity}<br/>
                        <strong>Source:</strong> {risk_data.get("source", "Live Webcam")}<br/>
                        <hr style="border-color: #7F1D1D; margin: 8px 0;"/>
                        <strong>🛰️ Satellite Verification:</strong> <span style="font-weight: 700; color: #FCD34D;">{sat_status}</span><br/>
                        <strong>Satellite Product:</strong> {sat_name}<br/>
                        <strong>Nearest Hotspot Distance:</strong> {sat_dist_str}
                    </div>
                    <div style="font-size: 0.8rem; margin-top: 8px; color: #FCA5A5; font-style: italic;">
                        ℹ️ Clarification: The location name comes from browser geolocation and reverse geocoding. Bounding boxes represent detections in camera image space.
                    </div>
                </div>
                ''',
                unsafe_allow_html=True
            )

        # 5 Key Metrics Display
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric(
                label="🔥 Fire Conf (45%)",
                value=f"{risk_data['fire_confidence']:.2f}"
            )
        with col2:
            st.metric(
                label="💨 Smoke Conf (20%)",
                value=f"{risk_data['smoke_confidence']:.2f}"
            )
        with col3:
            st.metric(
                label="📐 Area Ratio (15%)",
                value=f"{risk_data['area_ratio']:.3f}"
            )
        with col4:
            st.metric(
                label="📊 Severity",
                value=severity
            )
        with col5:
            st.metric(
                label="🧠 CNN Verifier (20%)",
                value="ACTIVE" if risk_data["verifier_available"] else "Unavailable",
                delta="Active Verifier" if risk_data["verifier_available"] else "Phase 1 Reserved",
                delta_color="normal" if risk_data["verifier_available"] else "off"
            )

        # 🧠 AI VERIFICATION SECTION (Phase 3 & Phase 11)
        st.markdown("### 🧠 AI Verification")
        v_res = risk_data.get("verifier_result")
        if v_res and v_res.get("available"):
            v_status = "ACTIVE"
            v_class = v_res.get("result", "N/A")
            v_conf = v_res.get("confidence")
            v_conf_str = f"{v_conf * 100:.1f}%" if v_conf is not None else "N/A"
            v_upper = str(v_class).upper()
            
            if v_upper in ["FIRE", "SMOKE"]:
                v_verified = "VERIFIED ✅"
            else:
                v_verified = f"REJECTED ({v_upper}) ✅"

            vc1, vc2, vc3, vc4 = st.columns(4)
            with vc1:
                st.metric("Status", v_status)
            with vc2:
                st.metric("Predicted Class", v_class)
            with vc3:
                st.metric("Confidence", v_conf_str)
            with vc4:
                st.metric("Decision", v_verified)
            st.info(f"ℹ️ {v_res.get('message', '')}")
        else:
            st.warning("🧠 **CNN Verifier Status**: **NOT AVAILABLE**")
            st.caption("ℹ️ *Fine-tuned weights for the multi-class CNN verifier (MobileNetV3) are not currently loaded. Preserving Phase 1 fallback behavior with 20% weight reserved.*")

        # 🛡️ FALSE POSITIVE PROTECTION SUMMARY
        st.markdown("### 🛡️ False Positive Protection")
        st.markdown("""
        <div style="background-color: #0F172A; border: 1px solid #10B98140; border-radius: 8px; padding: 14px; margin-bottom: 15px;">
            <table style="width: 100%; color: #E2E8F0; text-align: left; border-collapse: collapse;">
                <thead>
                    <tr style="border-bottom: 1px solid #334155; color: #94A3B8;">
                        <th style="padding: 6px 10px;">Input Condition</th>
                        <th style="padding: 6px 10px;">CNN Prediction</th>
                        <th style="padding: 6px 10px;">System Result</th>
                    </tr>
                </thead>
                <tbody>
                    <tr><td style="padding: 6px 10px;">Fire Image</td><td style="padding: 6px 10px;">FIRE</td><td style="padding: 6px 10px; color: #FF5722; font-weight: 700;">🚨 ALERT</td></tr>
                    <tr><td style="padding: 6px 10px;">Smoke Image</td><td style="padding: 6px 10px;">SMOKE</td><td style="padding: 6px 10px; color: #FFB300; font-weight: 700;">⚠️ WATCH</td></tr>
                    <tr><td style="padding: 6px 10px;">Cloud Image</td><td style="padding: 6px 10px;">CLOUD</td><td style="padding: 6px 10px; color: #10B981; font-weight: 700;">✅ REJECTED</td></tr>
                    <tr><td style="padding: 6px 10px;">Fog Image</td><td style="padding: 6px 10px;">FOG</td><td style="padding: 6px 10px; color: #10B981; font-weight: 700;">✅ REJECTED</td></tr>
                    <tr><td style="padding: 6px 10px;">Normal Forest Image</td><td style="padding: 6px 10px;">NORMAL_FOREST</td><td style="padding: 6px 10px; color: #10B981; font-weight: 700;">✅ SAFE</td></tr>
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

        # 🔍 EXPLAINABLE AI — GRAD-CAM SECTION (Phase 4)
        st.markdown("### 🔍 Explainable AI — Grad-CAM")
        g_res = risk_data.get("gradcam_result")
        if g_res and g_res.get("available") and g_res.get("overlay_rgb") is not None:
            gc1, gc2 = st.columns(2)
            with gc1:
                st.image(g_res["crop_rgb"], caption="Detected Region Crop", use_container_width=True)
            with gc2:
                st.image(g_res["overlay_rgb"], caption=f"Grad-CAM Heatmap Overlay ({g_res.get('target_class', '')})", use_container_width=True)
            
            conf_val = g_res.get('confidence', 0) or 0.0
            st.success(f"🔍 **CNN Decision**: `{g_res.get('target_class')}` | **Confidence**: `{conf_val*100:.1f}%` — {g_res.get('message')}")
        else:
            msg = g_res.get("message") if g_res else "Grad-CAM unavailable for this detection."
            st.info(f"ℹ️ **Grad-CAM Status**: {msg}")
            st.caption("ℹ️ *Grad-CAM generates spatial heatmap overlays on active bounding box crops when fine-tuned CNN verifier weights are loaded.*")

        # 🛰️ SATELLITE CROSS-VERIFICATION SECTION (Phase 15)
        render_satellite_panel(risk_data.get("satellite_verification"), container=target, render_expander=not is_live_stream)

        # 🗺️ GEOSPATIAL MAP LOCALIZATION SECTION (Phase 5 & 15)
        if render_map:
            lat = risk_data.get("latitude", 12.9716)
            lon = risk_data.get("longitude", 77.5946)
            loc_name = risk_data.get("source_location", "Bengaluru Demo Location")
            render_geospatial_map(lat, lon, risk_data, location_name=loc_name)

        # 🚨 SIMULATED ALERT SECTION (Only displayed when risk >= 0.70 / ALERT state)
        if risk_data.get("simulated_alert") is not None:
            st.markdown("### 🚨 Simulated Alert")
            st.warning("⚠️ **SIMULATION NOTICE**: The payload below represents a simulated forest fire emergency dispatch. No actual emergency services or external notifications are contacted.")
            st.json(risk_data["simulated_alert"])

