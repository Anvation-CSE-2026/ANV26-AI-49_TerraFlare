# TERRAFLARE Core Application — Real-Time AI Fire Intelligence & Satellite Verification
import os
import time
import tempfile
import cv2
import numpy as np
import streamlit as st
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from ultralytics import YOLO

# Import TERRAFLARE Core Modules (Phase 1, 2, 3, 4, 5)
from risk_engine import RiskEngine, render_risk_dashboard
from cnn_verifier import CNNVerifier
from gradcam import GradCAMExplainer
from map_engine import render_geospatial_map
from dataset_manager import render_dataset_manager_ui
from training.training_ui import render_yolo11_training_ui, render_cnn_training_ui
from training.model_evaluator_ui import render_model_evaluation_ui
from email_alerts import (
    get_email_config,
    is_email_configured,
    send_alert_email,
    send_alert_email_async,
    send_test_email,
    should_send_alert_email
)
from sound_notifier import SoundAlertManager
from geolocation.location_manager import LocationManager

# --- 1. EMAIL ALERT UTILITY FUNCTION (Phase 16) ---
def send_fire_alert_email(image_bgr, recipient_emails, alert_payload=None):
    res = send_alert_email(image_bgr, recipient_emails, alert_payload or {})
    return res.get("success", False)

# --- 2. PAGE CONFIG & HERO BACKGROUND STYLING ---
st.set_page_config(page_title="TERRAFLARE — Real-Time Fire Intelligence", page_icon="🔥", layout="wide")

# Custom Hero Banner and Dark Theme Styling
st.markdown('''
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;800;900&family=Inter:wght@300;400;600;700;800&display=swap');

    .stApp {
        background-color: #0B0D12 !important;
        color: #E2E8F0 !important;
        font-family: 'Inter', sans-serif !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #121620 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    /* HERO BANNER MATCHING THE REFERENCE IMAGE */
    .tf-hero-banner {
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.95) 0%, rgba(15, 23, 42, 0.98) 100%),
                    radial-gradient(circle at top right, rgba(239, 68, 68, 0.15), transparent 50%);
        border: 1px solid rgba(255, 87, 34, 0.3);
        border-radius: 12px;
        padding: 24px 30px;
        margin-bottom: 25px;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.8);
        color: #FFFFFF;
        position: relative;
    }

    .tf-hero-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 24px;
    }

    .tf-hero-brand {
        font-family: 'Orbitron', sans-serif;
        font-weight: 800;
        font-size: 1.15rem;
        letter-spacing: 2px;
        color: rgba(255, 255, 255, 0.85);
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .tf-hero-status {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        color: rgba(203, 213, 225, 0.8);
        background: rgba(0, 0, 0, 0.5);
        padding: 6px 16px;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .tf-dot-live {
        width: 8px;
        height: 8px;
        background-color: #FF5722;
        border-radius: 50%;
        box-shadow: 0 0 10px #FF5722;
    }

    .tf-hero-tagline {
        color: rgba(255, 112, 67, 0.85);
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    /* TERRAFLARE TEXT WITH 50% VISIBILITY / OPACITY AS REQUESTED */
    .tf-hero-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 4.8rem;
        font-weight: 900;
        line-height: 0.92;
        letter-spacing: 5px;
        color: rgba(255, 255, 255, 0.50) !important;
        margin: 0 0 20px 0;
        text-transform: uppercase;
        text-shadow: 0 4px 20px rgba(0, 0, 0, 0.6);
        transition: color 0.3s ease;
    }

    .tf-orange-dot {
        color: rgba(255, 87, 34, 0.85);
        font-size: 5rem;
    }

    .tf-hero-sub {
        font-size: 0.88rem;
        font-weight: 700;
        letter-spacing: 2px;
        color: rgba(148, 163, 184, 0.85);
        line-height: 1.6;
    }
    .tf-hero-sub span {
        color: rgba(255, 112, 67, 0.85);
        font-size: 0.78rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(22, 27, 34, 0.75) !important;
        border: 1px solid rgba(255, 87, 34, 0.2) !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
    }
    div[data-testid="stMetricValue"] {
        color: #FF5722 !important;
        font-weight: 700 !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #FF5722 0%, #E64A19 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    /* Complete Dark Theme Override for File Uploaders & Drag-Drop Boxes */
    [data-testid="stFileUploader"], 
    section[data-testid="stFileUploader"],
    .stFileUploader {
        background-color: transparent !important;
        border: none !important;
    }

    [data-testid="stFileUploaderDropzone"],
    div[data-testid="stFileUploadDropzone"],
    .stFileUploader > div {
        background-color: #121620 !important;
        border: 2px dashed rgba(255, 87, 34, 0.4) !important;
        border-radius: 12px !important;
        padding: 24px !important;
        color: #E2E8F0 !important;
        transition: all 0.3s ease !important;
    }

    [data-testid="stFileUploaderDropzone"]:hover,
    div[data-testid="stFileUploadDropzone"]:hover {
        border-color: #FF5722 !important;
        background-color: #1A202C !important;
        box-shadow: 0 0 15px rgba(255, 87, 34, 0.25) !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"],
    [data-testid="stFileUploaderDropzone"] *,
    div[data-testid="stFileUploadDropzone"] *,
    .stFileUploader label,
    .stFileUploader p,
    .stFileUploader span,
    .stFileUploader small,
    .stFileUploader div {
        color: #CBD5E1 !important;
        fill: #CBD5E1 !important;
    }

    /* Uploaded File Pill / Card Container - fixes white box when zip is uploaded */
    [data-testid="stFileUploaderFileData"],
    div[data-testid="stFileUploaderFileData"],
    .uploadedFile,
    .stUploadedFile {
        background-color: #1E293B !important;
        border: 1px solid rgba(255, 87, 34, 0.3) !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        color: #F8FAFC !important;
    }

    [data-testid="stFileUploaderFileData"] *,
    div[data-testid="stFileUploaderFileData"] *,
    .uploadedFile *,
    .stUploadedFile * {
        color: #F8FAFC !important;
        fill: #F8FAFC !important;
    }

    /* File Uploader Button ("Browse files") */
    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stFileUploader"] button,
    .stFileUploader button {
        background: #1E293B !important;
        color: #FF7043 !important;
        border: 1px solid rgba(255, 87, 34, 0.5) !important;
        border-radius: 6px !important;
        padding: 6px 16px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stFileUploaderDropzone"] button:hover,
    [data-testid="stFileUploader"] button:hover,
    .stFileUploader button:hover {
        background: linear-gradient(135deg, #FF5722 0%, #E64A19 100%) !important;
        color: #FFFFFF !important;
        border-color: #FF5722 !important;
        box-shadow: 0 4px 12px rgba(255, 87, 34, 0.4) !important;
    }

    /* Tabs Styling - remove white background on tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #121620 !important;
        border-radius: 8px !important;
        padding: 4px !important;
        gap: 4px !important;
    }

    .stTabs [data-baseweb="tab"] {
        color: #94A3B8 !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
        background-color: transparent !important;
        border: none !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1E293B !important;
        color: #FF7043 !important;
        font-weight: 700 !important;
    }

    /* Inputs & Selectboxes dark theme fix */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background-color: #121620 !important;
        border-color: rgba(255, 255, 255, 0.12) !important;
        color: #F8FAFC !important;
    }

    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(90deg, #FF5722, #FF9800) !important;
    }
</style>
''', unsafe_allow_html=True)

# --- HERO BANNER HEADER (TERRAFLARE WITH 50% VISIBILITY) ---
st.markdown('''
<div class="tf-hero-banner">
    <div class="tf-hero-top">
        <div class="tf-hero-brand">🔥 TERRAFLARE AI</div>
        <div class="tf-hero-status"><div class="tf-dot-live"></div> SYSTEM LIVE &nbsp;/&nbsp; 18:42:55 UTC</div>
    </div>
    <div class="tf-hero-tagline">── LIVE ENVIRONMENTAL INTELLIGENCE</div>
    <div class="tf-hero-title">TERRA<br/>FLARE<span class="tf-orange-dot">.</span></div>
    <div class="tf-hero-sub">
        AI-POWERED REAL-TIME FIRE DETECTION<br/>
        <span>AUTONOMOUS EARLY WARNING • SECTOR 07</span>
    </div>
</div>
''', unsafe_allow_html=True)

# --- 3. SIDEBAR NAVIGATION & PARAMETERS ---
st.sidebar.markdown("## 🛡️ TerraFlare")
main_section = st.sidebar.radio(
    "TerraFlare Section",
    ["🔍 Detection", "🧠 Model Lab"],
    key="terraflare_main_section"
)

if main_section == "🧠 Model Lab":
    lab_module = st.sidebar.radio("Model Lab", ["📂 Dataset Manager", "🚀 YOLO11 Training", "🧠 CNN Training", "📊 Model Evaluation"], key="model_lab_module")
    st.sidebar.markdown("---")
    st.sidebar.info("🧠 **Model Lab Phase 10**: Dataset Upload, Validation, YOLO11 & CNN Training, and Unified Model Evaluation & Comparison.")
    input_mode = None
else:
    st.sidebar.header("Navigation")
    input_mode = st.sidebar.radio("Select Input Mode", ["Image", "Video", "Live Camera Feed"])

    st.sidebar.header("Inference Settings")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
    apply_clahe = st.sidebar.checkbox("Apply CLAHE Preprocessing", value=True)

    st.sidebar.header("Performance Settings")
    frame_skip = st.sidebar.slider("Frame Skip (Higher = Slower)", 1, 20, 2)
    img_size = st.sidebar.select_slider("Inference Resolution", options=[320, 480, 640], value=480)

    # --- SIDEBAR: SOURCE LOCATION CONTROLS (Phase 5) ---
    st.sidebar.header("📍 Source Location")
    location_mode = st.sidebar.radio(
        "Location Mode",
        ["Demo Location (Bengaluru)", "Custom Coordinates"],
        key="location_mode_radio"
    )

    if location_mode == "Demo Location (Bengaluru)":
        source_lat = 12.9716
        source_lon = 77.5946
        source_location_name = "Bengaluru Demo Location"
    else:
        source_lat = st.sidebar.number_input("Latitude", value=12.9716, format="%.4f", step=0.01)
        source_lon = st.sidebar.number_input("Longitude", value=77.5946, format="%.4f", step=0.01)
        source_location_name = st.sidebar.text_input("Location Name", value="Custom Source Location")

    # --- SIDEBAR: SATELLITE CROSS-VERIFICATION CONTROLS (Phase 15) ---
    st.sidebar.header("🛰️ Satellite Cross-Verification")
    sat_radius = st.sidebar.selectbox(
        "🛰️ Satellite Verification Radius",
        [1.0, 2.0, 5.0, 10.0],
        index=2,
        format_func=lambda x: f"{int(x) if isinstance(x, (int, float)) and float(x).is_integer() else x} km",
        key="sb_sat_radius"
    )
    sat_window = st.sidebar.selectbox(
        "Satellite Data Window",
        [6, 12, 24, 48],
        index=2,
        format_func=lambda x: f"{x} hours",
        key="sb_sat_window"
    )
    btn_refresh_sat = st.sidebar.button("🔄 Refresh Satellite Verification", key="btn_refresh_sat_sb")

    from satellite.firms_client import get_firms_map_key
    if get_firms_map_key():
        st.sidebar.success("🛰️ FIRMS MAP_KEY Configured")
    else:
        st.sidebar.info("ℹ️ Satellite data unavailable\nNASA FIRMS MAP_KEY not configured.")

    # Sidebar Email Alert Controls (Phase 16A)
    st.sidebar.header("📧 EMAIL ALERT OPTIONS")
    enable_email_alerts = st.sidebar.checkbox("Enable Automatic Email Alerts", value=True, key="chk_enable_email_alerts")
    
    email_cfg = get_email_config()
    st.sidebar.text_input("Project Sender:", value=email_cfg["sender_email"], disabled=True, key="txt_project_sender")
    custom_user_email = st.sidebar.text_input("Recipient Email:", value=email_cfg["default_recipient"], placeholder="user@example.com", key="txt_custom_user_email")

    if st.sidebar.button("🧪 Send Test Email", key="btn_send_test_email_sb", use_container_width=True):
        with st.spinner("Dispatching test email..."):
            test_target = custom_user_email.strip() if (custom_user_email and "@" in custom_user_email.strip()) else email_cfg["default_recipient"]
            test_res = send_test_email(
                target_email=test_target
            )
            st.session_state["last_email_test_result"] = test_res

    if "last_email_test_result" in st.session_state:
        t_res = st.session_state["last_email_test_result"]
        t_status = t_res.get("status")
        if t_status == "SENT":
            st.sidebar.success(f"🟢 SENT: Test email delivered to {', '.join(t_res.get('recipients', []))}")
        elif t_status in ("FAILED", "AUTH_FAILED", "CONN_FAILED", "TLS_FAILED", "RECIPIENT_REJECTED"):
            st.sidebar.error(f"🔴 FAILED: {t_res.get('message')}")
        else:
            st.sidebar.info("ℹ️ Email delivery is not configured. Administrator authentication is required.")

    # Sidebar Sound Alert Controls (Phase 17)
    sound_mgr = SoundAlertManager()
    sound_mgr.render_sidebar_controls()

# Helper function for CLAHE
def preprocess_frame(img):
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        return img
    if 'apply_clahe' in globals() and apply_clahe:
        try:
            img_c = img.copy()
            if img_c.dtype != np.uint8:
                img_c = (img_c * 255.0).clip(0, 255).astype(np.uint8) if img_c.max() <= 1.0 else img_c.clip(0, 255).astype(np.uint8)
            
            if len(img_c.shape) == 2:
                img_c = cv2.cvtColor(img_c, cv2.COLOR_GRAY2BGR)
            elif len(img_c.shape) == 3 and img_c.shape[2] == 4:
                img_c = cv2.cvtColor(img_c, cv2.COLOR_BGRA2BGR)
            elif len(img_c.shape) == 3 and img_c.shape[2] != 3:
                return img
                
            lab = cv2.cvtColor(img_c, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        except Exception as err:
            print(f"[Preprocess Warning] CLAHE failed: {err}")
            return img
    return img

@st.cache_resource
def load_model():
    return YOLO('weights/best.pt')

@st.cache_resource
def load_cnn_verifier():
    return CNNVerifier()

@st.cache_resource
def load_risk_engine(_verifier):
    return RiskEngine(cnn_verifier=_verifier)

@st.cache_resource
def load_gradcam_explainer(_verifier):
    return GradCAMExplainer(_verifier)

model = load_model()
cnn_verifier = load_cnn_verifier()
risk_engine = load_risk_engine(cnn_verifier)
gradcam_explainer = load_gradcam_explainer(cnn_verifier)

if main_section == "🧠 Model Lab":
    if lab_module == "📂 Dataset Manager":
        render_dataset_manager_ui()
    elif lab_module == "🚀 YOLO11 Training":
        render_yolo11_training_ui()
    elif lab_module == "🧠 CNN Training":
        render_cnn_training_ui()
    elif lab_module == "📊 Model Evaluation":
        render_model_evaluation_ui()
# --- MODE 1: IMAGE ---
elif input_mode == "Image":
    st.subheader("📷 Image & ZIP Archive Detection")
    uploaded_file = st.file_uploader("Choose an image or ZIP archive...", type=["jpg", "jpeg", "png", "bmp", "webp", "zip"])

    if uploaded_file is not None:
        images_to_process = []
        is_zip = uploaded_file.name.lower().endswith(".zip")

        if is_zip:
            from dataset_validator import extract_zip_safely, IMAGE_EXTENSIONS
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
                tmp_zip.write(uploaded_file.getbuffer())
                tmp_zip_path = tmp_zip.name

            extracted_temp_dir = tempfile.mkdtemp(prefix="terraflare_zip_")
            try:
                target_root = extract_zip_safely(tmp_zip_path, extracted_temp_dir)
                for root, _, files in os.walk(target_root):
                    for f in sorted(files):
                        if not f.startswith(".") and not f.startswith("__MACOSX"):
                            ext = os.path.splitext(f)[1].lower()
                            if ext in IMAGE_EXTENSIONS:
                                images_to_process.append(os.path.join(root, f))
            except Exception as zip_err:
                st.error(f"❌ Failed to extract ZIP archive: {zip_err}")
            finally:
                if os.path.exists(tmp_zip_path):
                    os.remove(tmp_zip_path)

            if not images_to_process:
                st.error("❌ No valid image files (.jpg, .png, etc.) were found inside the uploaded ZIP file.")
            else:
                st.success(f"📦 Extracted {len(images_to_process)} image(s) from ZIP archive.")
                selected_img_path = st.selectbox(
                    "Select an image from the ZIP archive to inspect:",
                    images_to_process,
                    format_func=lambda x: os.path.basename(x)
                )
                image = cv2.imread(selected_img_path, cv2.IMREAD_COLOR)
        else:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if 'image' in locals() and image is not None:
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

            processed_image = preprocess_frame(image.copy())

            with st.spinner("Running Detection..."):
                results = model.predict(source=processed_image, conf=conf_threshold, imgsz=img_size)

            annotated_bgr = results[0].plot()
            annotated_frame = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)
            with col2:
                st.image(annotated_frame, caption="Detection Output", use_container_width=True)

            # Risk Engine, CNN Verifier, Grad-CAM, Satellite & Map evaluation (Phase 1 to 5 & 15)
            crops = cnn_verifier.crop_detection_regions(processed_image, results)
            best_crop = crops[0] if crops else None
            verifier_result = cnn_verifier.verify_yolo_detections(processed_image, results)
            target_class = verifier_result.get("result") if verifier_result else None
            gradcam_result = gradcam_explainer.generate_gradcam(best_crop, target_class_name=target_class) if best_crop is not None else None

            from satellite.satellite_cache import get_cached_satellite_verification
            fire_conf, smoke_conf, _ = risk_engine.extract_metrics_from_results(results)
            is_fire = (fire_conf > 0.0 or smoke_conf > 0.0 or (verifier_result and verifier_result.get("result") in ["FIRE", "SMOKE"]))
            sat_verification = get_cached_satellite_verification(
                camera_lat=source_lat,
                camera_lon=source_lon,
                search_radius_km=sat_radius,
                time_window_hours=sat_window,
                ai_status="ALERT" if is_fire else "NORMAL",
                is_fire_event=is_fire,
                force_refresh=btn_refresh_sat
            )

            risk_data = risk_engine.evaluate_yolo_results(
                results,
                verifier_result=verifier_result,
                gradcam_result=gradcam_result,
                source="ZIP Archive Image" if is_zip else "Image",
                latitude=source_lat,
                longitude=source_lon,
                source_location=source_location_name,
                satellite_verification=sat_verification
            )
            sound_mgr = SoundAlertManager()
            sound_mgr.process_risk_event(risk_data)
            render_risk_dashboard(risk_data)

            if enable_email_alerts and risk_data.get("status") == "ALERT":
                recipients = [email_cfg["default_recipient"]]
                if custom_user_email and "@" in custom_user_email.strip():
                    recipients.append(custom_user_email.strip())
                send_alert_email_async(
                    image_bgr=annotated_bgr,
                    recipient_emails=recipients,
                    alert_payload=risk_data
                )
        elif uploaded_file is not None and not is_zip:
            st.error("❌ Failed to decode uploaded image file. Please ensure it is a valid JPG, PNG, or WEBP image.")
    else:
        sound_mgr = SoundAlertManager()
        sound_mgr.reset_idle_state()

# --- MODE 2: VIDEO ---
elif input_mode == "Video":
    st.subheader("🎥 Video File Detection")
    uploaded_video = st.file_uploader("Upload a video...", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video is not None:
        import imageio

        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='_h264.mp4').name
        writer = imageio.get_writer(output_path, fps=fps, codec='libx264', format='FFMPEG')

        progress_bar = st.progress(0)
        status_text = st.empty()
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_count = 0
        last_annotated_rgb = None
        max_risk_data = None
        max_crop = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % frame_skip == 0 or last_annotated_rgb is None:
                processed_frame = preprocess_frame(frame.copy())
                results = model.predict(source=processed_frame, conf=conf_threshold, imgsz=img_size, verbose=False)
                last_annotated_rgb = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)

                crops = cnn_verifier.crop_detection_regions(processed_frame, results)
                verifier_result = cnn_verifier.verify_yolo_detections(processed_frame, results)

                from satellite.satellite_cache import get_cached_satellite_verification
                fire_conf, smoke_conf, _ = risk_engine.extract_metrics_from_results(results)
                is_fire = (fire_conf > 0.0 or smoke_conf > 0.0 or (verifier_result and verifier_result.get("result") in ["FIRE", "SMOKE"]))
                sat_verification = get_cached_satellite_verification(
                    camera_lat=source_lat,
                    camera_lon=source_lon,
                    search_radius_km=sat_radius,
                    time_window_hours=sat_window,
                    ai_status="ALERT" if is_fire else "NORMAL",
                    is_fire_event=is_fire,
                    force_refresh=btn_refresh_sat
                )

                current_risk_data = risk_engine.evaluate_yolo_results(
                    results,
                    verifier_result=verifier_result,
                    source="Video",
                    latitude=source_lat,
                    longitude=source_lon,
                    source_location=source_location_name,
                    satellite_verification=sat_verification
                )
                if max_risk_data is None or current_risk_data["risk_score"] > max_risk_data["risk_score"]:
                    max_risk_data = current_risk_data
                    max_crop = crops[0] if crops else None

            writer.append_data(last_annotated_rgb)

            if total_frames > 0:
                progress_bar.progress(min(frame_count / total_frames, 1.0))
                status_text.text(f"Processing frame {frame_count}/{total_frames}...")

        cap.release()
        writer.close()

        status_text.success("Processing complete! Rendering video playback...")

        with open(output_path, 'rb') as video_file:
            st.video(video_file.read())

        if max_risk_data is not None:
            sound_mgr = SoundAlertManager()
            sound_mgr.process_risk_event(max_risk_data)
            if max_crop is not None and cnn_verifier.is_available():
                target_class = max_risk_data.get("verifier_result", {}).get("result") if max_risk_data.get("verifier_result") else None
                max_risk_data["gradcam_result"] = gradcam_explainer.generate_gradcam(max_crop, target_class_name=target_class)
            render_risk_dashboard(max_risk_data)
    else:
        sound_mgr = SoundAlertManager()
        sound_mgr.reset_idle_state()

# --- MODE 3: LIVE CAMERA FEED ---
elif input_mode == "Live Camera Feed":
    st.subheader("📹 Real-Time Continuous Camera Stream & Geolocation")
    
    # Phase 13 Location Manager Integration
    loc_mgr = LocationManager()
    loc_state = loc_mgr.render_location_ui()
    
    cam_lat = loc_state["latitude"]
    cam_lon = loc_state["longitude"]
    cam_acc = loc_state["accuracy"]
    cam_loc_name = loc_state["source_location_name"]
    cam_loc_source_type = loc_state["source_type"]
    
    webcam_type = st.radio("Select Webcam Input Method", ["Continuous Stream (OpenCV)", "Browser Snapshot (st.camera_input)"], horizontal=True)

    if webcam_type == "Browser Snapshot (st.camera_input)":
        st.info("💡 **Browser Snapshot Mode**: Use your browser camera to capture a snapshot (e.g. candle flame, fire, or forest scene) and analyze it instantly with YOLO11 + CNN Verifier + Grad-CAM.")
        img_file_buffer = st.camera_input("Take a snapshot from webcam")
        if img_file_buffer is not None:
            bytes_data = img_file_buffer.getvalue()
            image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                st.error("❌ Could not decode image from camera buffer. Please try taking another snapshot.")
            else:
                processed_image = preprocess_frame(image.copy())

                with st.spinner("Running Detection & AI Verification..."):
                    results = model.predict(source=processed_image, conf=conf_threshold, imgsz=img_size)

                annotated_frame = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)

                col1, col2 = st.columns(2)
                with col1:
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Webcam Snapshot", use_container_width=True)
                with col2:
                    st.image(annotated_frame, caption="YOLO11 Detection Output", use_container_width=True)

                crops = cnn_verifier.crop_detection_regions(processed_image, results)
                verifier_result = cnn_verifier.verify_yolo_detections(processed_image, results)
                target_class = verifier_result.get("result") if verifier_result else None
                
                # Determine best crop for Grad-CAM explanation:
                # If YOLO detected bounding boxes, use top crop; if no boxes were detected, use full image frame
                if crops:
                    best_crop = crops[0]
                elif verifier_result and verifier_result.get("available") and target_class not in [None, "NO_DETECTIONS"]:
                    best_crop = processed_image
                else:
                    best_crop = None

                gradcam_result = gradcam_explainer.generate_gradcam(best_crop, target_class_name=target_class) if best_crop is not None else None

                from satellite.satellite_cache import get_cached_satellite_verification
                fire_conf, smoke_conf, _ = risk_engine.extract_metrics_from_results(results)
                is_fire = (fire_conf > 0.0 or smoke_conf > 0.0 or (verifier_result and verifier_result.get("result") in ["FIRE", "SMOKE"]))
                sat_verification = get_cached_satellite_verification(
                    camera_lat=cam_lat,
                    camera_lon=cam_lon,
                    search_radius_km=sat_radius,
                    time_window_hours=sat_window,
                    ai_status="ALERT" if is_fire else "NORMAL",
                    is_fire_event=is_fire,
                    force_refresh=btn_refresh_sat
                )

                risk_data = risk_engine.evaluate_yolo_results(
                    results,
                    verifier_result=verifier_result,
                    gradcam_result=gradcam_result,
                    source="Live Camera Feed",
                    latitude=cam_lat,
                    longitude=cam_lon,
                    source_location=cam_loc_name,
                    accuracy_m=cam_acc,
                    location_source_type=cam_loc_source_type,
                    satellite_verification=sat_verification
                )
                sound_mgr = SoundAlertManager()
                sound_mgr.process_risk_event(risk_data)
                render_risk_dashboard(risk_data)
                render_geospatial_map(cam_lat, cam_lon, risk_data, location_name=cam_loc_name, accuracy_m=cam_acc, key_suffix="snapshot_geo")

    else:
        st.info("🎥 **Continuous OpenCV Stream Mode**: Live local video stream with real-time YOLO11 bounding box inference, CNN verification, and Risk Engine analysis.")
        
        c_col1, c_col2 = st.columns([1, 4])
        with c_col1:
            run_cam = st.checkbox("Start Camera Stream", key="chk_start_cam")
        with c_col2:
            if run_cam:
                st.caption("🟢 Camera stream active. Uncheck 'Start Camera Stream' to stop.")

        if not run_cam:
            sound_mgr = SoundAlertManager()
            sound_mgr.reset_idle_state()

        st_frame = st.empty()
        st_risk = st.empty()
        st_map = st.empty()

        if "last_email_time" not in st.session_state:
            st.session_state.last_email_time = 0

        if run_cam:
            # Helper function for opening webcam across backends & indices
            def get_working_cap():
                for idx in [0, 1, 2]:
                    # Try AVFoundation backend first on macOS
                    try:
                        cap_test = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
                        if cap_test.isOpened():
                            ret_w, fr_w = cap_test.read()
                            if ret_w and fr_w is not None and fr_w.size > 0:
                                return cap_test, idx
                            cap_test.release()
                    except Exception:
                        pass
                    
                    # Fallback to default backend
                    try:
                        cap_test = cv2.VideoCapture(idx)
                        if cap_test.isOpened():
                            ret_w, fr_w = cap_test.read()
                            if ret_w and fr_w is not None and fr_w.size > 0:
                                return cap_test, idx
                            cap_test.release()
                    except Exception:
                        pass
                return None, -1

            cap, used_idx = get_working_cap()

            if cap is None or not cap.isOpened():
                st.error("❌ Failed to access webcam hardware. Please check your camera connection & permissions, or switch to 'Browser Snapshot' mode above.")
            else:
                # Optimize capture resolution for fast real-time inference
                try:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass

                frame_count = 0
                consecutive_failures = 0
                last_annotated_frame = None
                last_risk_data = None
                last_rendered_status = None
                last_rendered_risk_score = -1.0
                fps_counter = 0.0
                sat_verification = None
                last_sat_check_t = 0.0

                try:
                    while run_cam:
                        loop_start_t = time.time()
                        ret, frame = cap.read()
                        if not ret or frame is None or frame.size == 0:
                            consecutive_failures += 1
                            if consecutive_failures > 10:
                                st.warning("⚠️ Stream interrupted or camera disconnected.")
                                break
                            time.sleep(0.05)
                            continue

                        consecutive_failures = 0
                        frame_count += 1

                        if frame_count % frame_skip == 0 or last_annotated_frame is None:
                            try:
                                processed_frame = preprocess_frame(frame.copy())
                                
                                # 1. YOLO11 Inference (Fast Bounding Box Detection)
                                t_yolo_0 = time.time()
                                results = model.predict(source=processed_frame, conf=conf_threshold, imgsz=img_size, verbose=False)
                                t_yolo_ms = (time.time() - t_yolo_0) * 1000.0

                                annotated_bgr = results[0].plot()
                                last_annotated_frame = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                                
                                # 2. CNN Verifier (MobileNetV3 with torch.no_grad)
                                t_cnn_0 = time.time()
                                crops = cnn_verifier.crop_detection_regions(processed_frame, results)
                                verifier_result = cnn_verifier.verify_yolo_detections(processed_frame, results)
                                t_cnn_ms = (time.time() - t_cnn_0) * 1000.0

                                target_class = verifier_result.get("result") if verifier_result else None
                                if crops:
                                    best_crop = crops[0]
                                elif verifier_result and verifier_result.get("available") and target_class not in [None, "NO_DETECTIONS"]:
                                    best_crop = processed_frame
                                else:
                                    best_crop = None

                                fire_conf, smoke_conf, _ = risk_engine.extract_metrics_from_results(results)
                                is_fire = (fire_conf > 0.0 or smoke_conf > 0.0 or (verifier_result and verifier_result.get("result") in ["FIRE", "SMOKE"]))

                                # 3. Grad-CAM Optimization (Only run backward pass when fire/smoke is detected)
                                t_gcam_0 = time.time()
                                if is_fire and best_crop is not None:
                                    gradcam_result = gradcam_explainer.generate_gradcam(best_crop, target_class_name=target_class)
                                    t_gcam_ms = (time.time() - t_gcam_0) * 1000.0
                                else:
                                    gradcam_result = None
                                    t_gcam_ms = 0.0

                                # 4. Satellite Verification Triggering (Event-Driven & Non-Blocking)
                                # Trigger ONLY when fire is detected (ALERT) or manual refresh requested or cache expired (5 min)
                                curr_t = time.time()
                                should_check_sat = (
                                    btn_refresh_sat or 
                                    (is_fire and (sat_verification is None or (curr_t - last_sat_check_t) > 300.0))
                                )
                                if should_check_sat:
                                    from satellite.satellite_cache import get_cached_satellite_verification
                                    sat_verification = get_cached_satellite_verification(
                                        camera_lat=cam_lat,
                                        camera_lon=cam_lon,
                                        search_radius_km=sat_radius,
                                        time_window_hours=sat_window,
                                        ai_status="ALERT" if is_fire else "NORMAL",
                                        is_fire_event=is_fire,
                                        force_refresh=btn_refresh_sat,
                                        is_async=True
                                    )
                                    last_sat_check_t = curr_t
                                elif not is_fire and sat_verification is not None:
                                    from satellite.satellite_models import create_empty_satellite_verification
                                    sat_verification = create_empty_satellite_verification(
                                        status="UNAVAILABLE",
                                        status_message="Satellite verification not required for normal non-fire detections.",
                                        search_radius_km=sat_radius,
                                        time_window_hours=sat_window
                                    )

                                # 5. Risk Engine Evaluation
                                last_risk_data = risk_engine.evaluate_yolo_results(
                                    results,
                                    verifier_result=verifier_result,
                                    gradcam_result=gradcam_result,
                                    source="Live Camera Feed",
                                    latitude=cam_lat,
                                    longitude=cam_lon,
                                    source_location=cam_loc_name,
                                    accuracy_m=cam_acc,
                                    location_source_type=cam_loc_source_type,
                                    satellite_verification=sat_verification
                                )

                                # Sound Alert Event Evaluation (Phase 17)
                                sound_mgr = SoundAlertManager()
                                sound_mgr.process_risk_event(last_risk_data)

                                # 6. Non-Blocking Event-Driven Email Alert Dispatch (Phase 16)
                                if enable_email_alerts and last_risk_data:
                                    last_email_t = st.session_state.get("last_alert_email_time", 0.0)
                                    last_event_id = st.session_state.get("last_alert_event_id", None)

                                    should_send, send_reason = should_send_alert_email(
                                        last_risk_data,
                                        last_sent_time=last_email_t,
                                        last_sent_event_id=last_event_id,
                                        cooldown_seconds=60.0
                                    )

                                    if should_send:
                                        curr_t = time.time()
                                        st.session_state.last_alert_email_time = curr_t
                                        event_id = (last_risk_data.get("simulated_alert", {}).get("alert_id") or 
                                                    last_risk_data.get("detection_event", {}).get("event_id"))
                                        st.session_state.last_alert_event_id = event_id
                                        
                                        st.toast("🚨 Fire ALERT! Dispatching email notification in background...", icon="📧")
                                        
                                        recipients = [email_cfg["default_recipient"]]
                                        if custom_user_email and "@" in custom_user_email.strip():
                                            recipients.append(custom_user_email.strip())

                                        send_alert_email_async(
                                            image_bgr=annotated_bgr if 'annotated_bgr' in locals() else None,
                                            recipient_emails=recipients,
                                            alert_payload=last_risk_data
                                        )
                            except Exception as infer_err:
                                print(f"[Continuous Stream] Inference Warning: {infer_err}")

                        loop_dt = time.time() - loop_start_t
                        fps_counter = (0.9 * fps_counter) + (0.1 * (1.0 / max(0.001, loop_dt)))

                        # High-Speed Video Frame Render (30+ FPS)
                        if last_annotated_frame is not None:
                            st_frame.image(
                                last_annotated_frame,
                                caption=f"Live OpenCV Stream (Camera #{used_idx}) | ⚡ {fps_counter:.1f} FPS | YOLO: {t_yolo_ms:.1f}ms | CNN: {t_cnn_ms:.1f}ms | Grad-CAM: {'Skipped' if t_gcam_ms == 0 else f'{t_gcam_ms:.1f}ms'} | FIRMS: Async Background",
                                use_container_width=True
                            )

                        # Throttled Dashboard UI Render (Updates every 10 frames or on Status Change to prevent browser lockup)
                        if last_risk_data is not None:
                            curr_status = last_risk_data.get("status")
                            curr_score = round(last_risk_data.get("risk_score", 0.0), 2)
                            status_changed = (last_rendered_status != (curr_status, cam_lat, cam_lon))
                            score_changed = (abs(curr_score - last_rendered_risk_score) >= 0.05)

                            if frame_count % 10 == 0 or status_changed or score_changed:
                                render_risk_dashboard(last_risk_data, container=st_risk, render_map=False, is_live_stream=True)
                                last_rendered_risk_score = curr_score

                            if status_changed:
                                last_rendered_status = (curr_status, cam_lat, cam_lon)
                                try:
                                    st_map.empty()
                                    render_geospatial_map(
                                        cam_lat, cam_lon, last_risk_data,
                                        location_name=cam_loc_name, accuracy_m=cam_acc,
                                        container=st_map, key_suffix="live_geo_static"
                                    )
                                except Exception as map_err:
                                    print(f"[Continuous Stream] Map render warning: {map_err}")

                        time.sleep(0.01)
                except Exception as stream_err:
                    st.warning(f"⚠️ Stream encountered an error or was stopped: {stream_err}")
                finally:
                    if cap is not None and cap.isOpened():
                        cap.release()
