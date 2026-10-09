"""
TerraFlare - Dataset Manager UI Module
Streamlit interface for dataset upload, validation, analysis, statistics, class distribution, and dataset registry.
"""

import os
import tempfile
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from dataset_registry import (
    get_next_dataset_id,
    get_dataset_dir,
    save_dataset_metadata,
    list_registered_datasets,
    get_dataset_metadata
)
from dataset_validator import (
    extract_zip_safely,
    validate_yolo_dataset,
    validate_classification_dataset
)
from dataset_analyzer import (
    analyze_dataset_metrics,
    fetch_sample_previews
)

def render_dataset_manager_ui():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%); padding: 20px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 25px;">
        <h2 style="color: #F8FAFC; margin: 0 0 8px 0; font-weight: 700;">📂 TerraFlare — Dataset Manager & Analyzer</h2>
        <p style="color: #94A3B8; margin: 0;">Upload, validate, inspect, and register computer vision datasets for TerraFlare AI models.</p>
    </div>
    """, unsafe_allow_html=True)

    tab_upload, tab_registry = st.tabs(["📤 Upload & Validate New Dataset", "📚 Dataset Registry & Analysis"])

    # ---------------------------------------------------------
    # TAB 1: UPLOAD & VALIDATE NEW DATASET
    # ---------------------------------------------------------
    with tab_upload:
        st.subheader("1. Dataset Upload Configuration")
        
        col_type, col_name = st.columns([1, 1])
        with col_type:
            dataset_type_choice = st.radio(
                "Select Dataset Type",
                ["🔥 Object Detection", "🧠 Image Classification"],
                key="dataset_type_radio"
            )
            dataset_type = "Object Detection" if "Object Detection" in dataset_type_choice else "Classification"

        with col_name:
            dataset_name = st.text_input("Dataset Name", placeholder="e.g., Forest Fire YOLO v1", key="dataset_name_input")
            
        st.markdown("---")
        
        if dataset_type == "Object Detection":
            st.info("ℹ️ **Object Detection Format**: Upload a `.zip` containing a YOLO structure (`data.yaml`, `images/`, `labels/`).")
            file_types = ["zip"]
        else:
            st.info("ℹ️ **Classification Format**: Upload a `.zip` containing class folders (`train/fire`, `train/smoke`, etc.) OR a `.csv` image-class mapping file.")
            file_types = ["zip", "csv"]

        uploaded_file = st.file_uploader("Choose a dataset file (.zip or .csv)", type=file_types, key="dataset_file_uploader")

        if uploaded_file is not None:
            if not dataset_name.strip():
                st.warning("⚠️ Please provide a Dataset Name before processing.")
            else:
                if st.button("🚀 Process & Validate Dataset", key="process_dataset_btn", use_container_width=True):
                    process_dataset_upload(uploaded_file, dataset_name.strip(), dataset_type)

    # ---------------------------------------------------------
    # TAB 2: DATASET REGISTRY & ANALYSIS
    # ---------------------------------------------------------
    with tab_registry:
        render_registry_and_analysis_view()

def process_dataset_upload(uploaded_file, dataset_name, dataset_type):
    """Handles dataset extraction, validation, metadata saving, and registry update."""
    next_id = get_next_dataset_id()
    dataset_dir = get_dataset_dir(next_id)
    extracted_dir = os.path.join(dataset_dir, "extracted")
    os.makedirs(extracted_dir, exist_ok=True)

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    status_text.info("Extracting and checking archive security...")

    try:
        # Save uploaded file to temporary location
        suffix = f".{uploaded_file.name.split('.')[-1]}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        csv_path = None
        target_dataset_root = extracted_dir

        if uploaded_file.name.endswith(".zip"):
            target_dataset_root = extract_zip_safely(tmp_path, extracted_dir)
        elif uploaded_file.name.endswith(".csv"):
            # Save CSV file inside extracted_dir
            csv_path = os.path.join(extracted_dir, uploaded_file.name)
            with open(csv_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            target_dataset_root = extracted_dir

        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        def update_progress(ratio, msg):
            progress_bar.progress(min(1.0, max(0.0, ratio)))
            status_text.text(msg)

        status_text.info("Validating dataset structure and image integrity...")

        if dataset_type == "Object Detection":
            val_report = validate_yolo_dataset(target_dataset_root, progress_callback=update_progress)
        else:
            val_report = validate_classification_dataset(target_dataset_root, csv_file_path=csv_path, progress_callback=update_progress)

        if not val_report.get("valid_structure"):
            progress_bar.empty()
            status_text.empty()
            err_msg = "\n".join(val_report.get("errors", ["Invalid dataset structure."]))
            st.error(f"❌ Dataset Validation Failed:\n{err_msg}")
            return

        quality = analyze_dataset_metrics(val_report)

        metadata = {
            "dataset_id": next_id,
            "dataset_name": dataset_name,
            "dataset_type": dataset_type,
            "classes": val_report.get("classes", []),
            "total_images": val_report.get("total_images", 0),
            "train_count": val_report.get("train_count", 0),
            "validation_count": val_report.get("validation_count", 0),
            "test_count": val_report.get("test_count", 0),
            "class_distribution": val_report.get("class_distribution", {}),
            "missing_files": val_report.get("missing_files", 0),
            "corrupted_files": val_report.get("corrupted_images", 0),
            "missing_labels": val_report.get("missing_labels", 0),
            "invalid_labels": val_report.get("invalid_labels", 0),
            "dataset_path": target_dataset_root,
            "validation_status": val_report.get("status", "Valid"),
            "quality_summary": quality
        }

        save_dataset_metadata(next_id, metadata)

        progress_bar.progress(1.0)
        status_text.empty()
        st.success(f"✅ Dataset processed successfully! Registered as **{next_id}**.")
        
        # Automatically select new dataset in session state
        st.session_state["selected_dataset_id"] = next_id
        st.rerun()

    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Error processing dataset: {str(e)}")

def render_registry_and_analysis_view():
    datasets = list_registered_datasets()
    
    st.subheader("📚 Dataset Registry")
    if not datasets:
        st.info("No registered datasets found. Upload a new dataset in the 'Upload & Validate' tab above.")
        return

    # Create Registry Dataframe
    reg_data = []
    for d in datasets:
        reg_data.append({
            "Dataset ID": d.get("dataset_id"),
            "Name": d.get("dataset_name"),
            "Type": d.get("dataset_type"),
            "Images": d.get("total_images"),
            "Status": d.get("validation_status", "Valid"),
            "Uploaded": d.get("upload_timestamp")
        })
        
    df_reg = pd.DataFrame(reg_data)
    st.dataframe(df_reg, use_container_width=True, hide_index=True)

    dataset_options = [f"{d.get('dataset_id')} — {d.get('dataset_name')} ({d.get('dataset_type')})" for d in datasets]
    
    # Determine default selection index
    default_idx = 0
    if "selected_dataset_id" in st.session_state:
        for idx, d in enumerate(datasets):
            if d.get("dataset_id") == st.session_state["selected_dataset_id"]:
                default_idx = idx
                break

    selected_option = st.selectbox("Select a Dataset to Inspect Details", dataset_options, index=default_idx, key="dataset_selector_box")
    selected_id = selected_option.split(" — ")[0]
    
    selected_metadata = get_dataset_metadata(selected_id)
    if not selected_metadata:
        st.error("Selected dataset metadata could not be loaded.")
        return

    render_dataset_details(selected_metadata)

def render_dataset_details(meta):
    st.markdown("---")
    st.markdown(f"### 🔍 Detailed Analysis: **{meta.get('dataset_name')}** (`{meta.get('dataset_id')}`)")
    
    # ---------------------------------------------------------
    # 1. DATASET OVERVIEW METRICS
    # ---------------------------------------------------------
    st.markdown("#### 📊 DATASET OVERVIEW")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Images", meta.get("total_images", 0))
    col2.metric("Training", meta.get("train_count", 0))
    col3.metric("Validation", meta.get("validation_count", 0))
    col4.metric("Testing", meta.get("test_count", 0))

    st.markdown("<br/>", unsafe_allow_html=True)
    
    col_cls, col_issues = st.columns([1, 1])
    
    with col_cls:
        st.markdown("**Classes Breakdown:**")
        class_dist = meta.get("class_distribution", {})
        if class_dist:
            for cname, count in class_dist.items():
                cl = cname.lower()
                if "fire" in cl and "non" not in cl:
                    icon = "🔥"
                elif "smoke" in cl:
                    icon = "💨"
                elif "cloud" in cl:
                    icon = "☁️"
                elif "fog" in cl:
                    icon = "🌫️"
                elif "forest" in cl or "tree" in cl or "non_fire" in cl or "normal" in cl:
                    icon = "🌳"
                else:
                    icon = "🏷️"
                st.write(f"{icon} **{cname}**: {count} instances")
        else:
            st.write("No class annotations found.")

    # ---------------------------------------------------------
    # HARD NEGATIVE ANALYSIS SECTION
    # ---------------------------------------------------------
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("#### 🎯 Hard Negative Analysis")
    st.markdown("""
    <div style="background-color: #0F172A; border: 1px solid #334155; border-radius: 8px; padding: 14px; margin-bottom: 15px;">
        <p style="margin: 0 0 8px 0; color: #F8FAFC;"><strong>Hard Negatives</strong> are visually similar environmental conditions that should <strong>NOT</strong> trigger a forest-fire alert.</p>
        <ul style="color: #CBD5E1; margin: 0; padding-left: 20px; line-height: 1.6;">
            <li>☁️ <strong>SMOKE ↔ CLOUD:</strong> Cloud formations visually resembling smoke columns.</li>
            <li>🌫️ <strong>SMOKE ↔ FOG:</strong> Low-hanging fog mist mimicking smoke haze.</li>
            <li>🌅 <strong>FIRE ↔ SUNSET / BRIGHT LIGHT:</strong> Bright orange/red reflections mimicking flames.</li>
            <li>🌳 <strong>FIRE ↔ RED/ORANGE VEGETATION:</strong> Autumn foliage mimicking fire colors.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    with col_issues:
        st.markdown("**Quality Check Metrics:**")
        st.write(f"📁 **Missing Files:** {meta.get('missing_files', 0)}")
        st.write(f"⚠️ **Corrupted Images:** {meta.get('corrupted_files', 0)}")
        st.write(f"🏷️ **Missing Labels:** {meta.get('missing_labels', 0)}")
        st.write(f"❌ **Invalid Labels:** {meta.get('invalid_labels', 0)}")

    # ---------------------------------------------------------
    # 2. CLASS DISTRIBUTION CHART
    # ---------------------------------------------------------
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("####  Class Distribution")
    
    if class_dist:
        df_dist = pd.DataFrame(list(class_dist.items()), columns=["Class", "Count"]).set_index("Class")
        st.bar_chart(df_dist)
        
        # Check Class Imbalance Warning
        quality = meta.get("quality_summary", {})
        warning_msg = quality.get("imbalance_warning")
        if warning_msg:
            st.warning(warning_msg)
        else:
            st.success("✅ Class distribution is balanced.")
    else:
        st.info("No class distribution data available.")

    # ---------------------------------------------------------
    # 3. DATASET QUALITY REPORT
    # ---------------------------------------------------------
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("#### 🔎 DATASET QUALITY")
    
    q = meta.get("quality_summary", {})
    quality_table = [
        {"Metric": "Dataset Structure", "Status / Value": q.get("structure", "✅ Valid")},
        {"Metric": "Images Integrity", "Status / Value": q.get("images", "✅ Valid")},
        {"Metric": "Labels Format", "Status / Value": q.get("labels", "✅ Valid")},
        {"Metric": "Corrupted Images", "Status / Value": str(meta.get("corrupted_files", 0))},
        {"Metric": "Missing Labels", "Status / Value": str(meta.get("missing_labels", 0))},
        {"Metric": "Invalid Labels", "Status / Value": str(meta.get("invalid_labels", 0))},
        {"Metric": "Class Balance Status", "Status / Value": q.get("class_balance_status", "Balanced")}
    ]
    st.table(pd.DataFrame(quality_table))

    # ---------------------------------------------------------
    # 4. DATASET PREVIEW
    # ---------------------------------------------------------
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("#### 🖼️ Dataset Preview")
    
    if st.button("Show Sample Images", key=f"preview_btn_{meta.get('dataset_id')}"):
        dataset_path = meta.get("dataset_path")
        if dataset_path and os.path.exists(dataset_path):
            with st.spinner("Fetching and annotating sample images..."):
                samples = fetch_sample_previews(
                    dataset_root=dataset_path,
                    dataset_type=meta.get("dataset_type", "Object Detection"),
                    class_names=meta.get("classes", []),
                    max_samples=6
                )
            if samples:
                cols = st.columns(3)
                for idx, sample in enumerate(samples):
                    col = cols[idx % 3]
                    with col:
                        st.image(sample["image"], caption=f"{sample['filename']} ({sample['label']})", use_container_width=True)
            else:
                st.warning("No sample images found to display.")
        else:
            st.error("Dataset folder not found on disk.")
