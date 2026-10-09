"""
TerraFlare - Training Manager UI Module
Streamlit interface for YOLO11 Training Manager and CNN MobileNetV3 Training Manager.
"""

import os
import shutil
import json
import pandas as pd
import streamlit as st

from dataset_registry import list_registered_datasets, get_dataset_metadata
from training.training_config import (
    detect_compute_device,
    get_available_yolo_models,
    validate_pre_training
)
from training.yolo_trainer import run_yolo_training, IS_TRAINING_RUNNING
from training.experiment_manager import list_all_experiments, get_experiment_metadata
from training.model_registry import list_registered_models, register_trained_model
from training.cnn_trainer import train_cnn_mobilenetv3
from training.cnn_evaluator import evaluate_cnn_experiment

# ---------------------------------------------------------
# 1. YOLO11 TRAINING UI
# ---------------------------------------------------------
def render_yolo11_training_ui():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1E100A 0%, #0F172A 100%); padding: 20px; border-radius: 12px; border: 1px solid #FF572240; margin-bottom: 25px;">
        <h2 style="color: #F8FAFC; margin: 0 0 8px 0; font-weight: 700;">🚀 TerraFlare — YOLO11 Training Manager</h2>
        <p style="color: #94A3B8; margin: 0;">Configure, pre-validate, and train state-of-the-art YOLO11 object detection models on validated datasets.</p>
    </div>
    """, unsafe_allow_html=True)

    tab_train, tab_experiments, tab_registry = st.tabs([
        "🚀 Train New Model",
        "📊 Training Experiments Log",
        "🏆 Trained Model Registry"
    ])

    with tab_train:
        render_training_form()

    with tab_experiments:
        render_experiments_log()

    with tab_registry:
        render_model_registry_view(model_filter="YOLO")

def render_training_form():
    st.subheader("1. Select Training Dataset")

    all_datasets = list_registered_datasets()
    od_datasets = [d for d in all_datasets if d.get("dataset_type") == "Object Detection"]

    if not od_datasets:
        st.warning("⚠️ No validated Object Detection datasets found in the Dataset Registry. Please upload an Object Detection dataset in the Dataset Manager first.")
        return

    ds_options = [f"{d.get('dataset_id')} — {d.get('dataset_name')} ({d.get('total_images')} images)" for d in od_datasets]
    selected_ds_str = st.selectbox("Dataset:", ds_options, key="train_ds_select")
    selected_ds_id = selected_ds_str.split(" — ")[0]
    selected_ds_meta = get_dataset_metadata(selected_ds_id)

    st.markdown("---")
    st.subheader("2. Training Hyperparameters Configuration")

    col_exp, col_model = st.columns([1, 1])
    with col_exp:
        exp_name = st.text_input("Experiment Name:", value="FireSmoke_YOLO11_001", key="exp_name_input")
    with col_model:
        avail_models = get_available_yolo_models()
        model_display = st.selectbox("Model Architecture:", list(avail_models.keys()), key="model_variant_select")
        chosen_weight_file = avail_models[model_display]

    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        epochs = st.number_input("Epochs:", min_value=1, max_value=500, value=50, step=1, key="epochs_input")
    with col_p2:
        imgsz = st.select_slider("Image Size (px):", options=[320, 480, 640, 800], value=640, key="imgsz_input")
    with col_p3:
        batch_size = st.selectbox("Batch Size:", options=[2, 4, 8, 16, 32, 64], index=3, key="batch_input")
    with col_p4:
        devices = detect_compute_device()
        device_choice = st.selectbox("Hardware Device:", devices, key="device_input")

    col_lr, col_dummy = st.columns([1, 1])
    with col_lr:
        lr0 = st.number_input("Initial Learning Rate (lr0):", value=0.01, format="%.4f", step=0.001, key="lr_input")

    st.markdown("---")
    st.subheader("3. Pre-Training Validation Confirmation")

    is_valid, summary, err_msg = validate_pre_training(selected_ds_meta)

    if not is_valid:
        st.error(f"❌ Pre-training Validation Failed: {err_msg}")
        return

    st.markdown(f"""
    <div style="background-color: #0F172A; border: 1px solid #1E293B; border-radius: 8px; padding: 16px; margin-bottom: 20px;">
        <p style="margin: 0 0 6px 0; font-size: 1rem; color: #F8FAFC;"><strong>Dataset:</strong> {summary['dataset_name']} (<code>{summary['dataset_id']}</code>)</p>
        <p style="margin: 0 0 6px 0; color: #CBD5E1;"><strong>Classes:</strong> {', '.join(summary['classes']) if summary['classes'] else 'None'}</p>
        <p style="margin: 0 0 6px 0; color: #CBD5E1;"><strong>Training Images:</strong> {summary['train_count']} &nbsp;|&nbsp; <strong>Validation Images:</strong> {summary['val_count']} &nbsp;|&nbsp; <strong>Test Images:</strong> {summary['test_count']}</p>
        <p style="margin: 0 0 6px 0; color: #CBD5E1;"><strong>Model Architecture:</strong> {model_display} (<code>{chosen_weight_file}</code>)</p>
        <p style="margin: 0 0 10px 0; color: #CBD5E1;"><strong>Hardware Device:</strong> {device_choice}</p>
        <div style="color: #10B981; font-weight: 700; font-size: 1.05rem;">{summary['status']}</div>
    </div>
    """, unsafe_allow_html=True)

    if IS_TRAINING_RUNNING:
        st.warning("⚠️ A training task is currently running. Please wait for it to complete.")
        return

    if st.button("🚀 START TRAINING", key="start_training_btn", use_container_width=True):
        st.markdown("---")
        st.markdown(f"### 🚀 Training: **{exp_name}**")
        status_box = st.empty()
        
        def update_status(msg):
            status_box.info(f"⏳ {msg}")

        try:
            with st.spinner("Training YOLO11 model in progress... Please wait."):
                results = run_yolo_training(
                    dataset_metadata=selected_ds_meta,
                    experiment_name=exp_name.strip(),
                    model_variant=chosen_weight_file,
                    epochs=int(epochs),
                    imgsz=int(imgsz),
                    batch_size=int(batch_size),
                    lr0=float(lr0),
                    device_choice=device_choice,
                    progress_callback=update_status
                )
            status_box.empty()
            st.success(f"🎉 Training Completed Successfully! Experiment ID: **{results.get('experiment_id')}**")
            display_training_results(results)
        except Exception as e:
            status_box.empty()
            st.error(f"❌ Training Execution Failed: {str(e)}")

def display_training_results(exp):
    st.markdown("---")
    st.markdown(f"### 🏆 YOLO11 Model Results: `{exp.get('experiment_name')}`")
    
    metrics = exp.get("metrics", {})
    p = metrics.get("precision", 0.0) * 100
    r = metrics.get("recall", 0.0) * 100
    f1 = metrics.get("f1", 0.0) * 100
    map50 = metrics.get("map50", 0.0) * 100
    map50_95 = metrics.get("map50_95", 0.0) * 100

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Precision", f"{p:.2f}%")
    col2.metric("Recall", f"{r:.2f}%")
    col3.metric("F1 Score", f"{f1:.2f}%")
    col4.metric("mAP@50", f"{map50:.2f}%")
    col5.metric("mAP@50-95", f"{map50_95:.2f}%")

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("#### 📈 Training Plots & Curves")
    
    exp_dir = exp.get("experiment_dir")
    if exp_dir and os.path.exists(exp_dir):
        plot_files = [f for f in os.listdir(exp_dir) if f.endswith('.png') or f.endswith('.jpg')]
        if plot_files:
            cols = st.columns(min(len(plot_files), 3))
            for idx, pfile in enumerate(plot_files[:6]):
                col = cols[idx % len(cols)]
                with col:
                    img_path = os.path.join(exp_dir, pfile)
                    st.image(img_path, caption=pfile, use_column_width=True)

def render_experiments_log():
    st.subheader("📊 Training Experiments History")
    exps = list_all_experiments()
    
    if not exps:
        st.info("No training experiments recorded yet.")
        return

    exp_rows = []
    for e in exps:
        m = e.get("metrics", {})
        exp_rows.append({
            "EXP ID": e.get("experiment_id"),
            "Name": e.get("experiment_name"),
            "Dataset": e.get("dataset_name"),
            "Model": e.get("model_name"),
            "Epochs": e.get("epochs"),
            "mAP@50": format_metric_val(m.get('map50')),
            "mAP@50-95": format_metric_val(m.get('map50_95')),
            "Status": e.get("status"),
            "Started": e.get("training_start_time")
        })

    st.dataframe(pd.DataFrame(exp_rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------
# 2. CNN MOBILENETV3 TRAINING UI
# ---------------------------------------------------------
def render_cnn_training_ui():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 100%); padding: 20px; border-radius: 12px; border: 1px solid #6366F140; margin-bottom: 25px;">
        <h2 style="color: #F8FAFC; margin: 0 0 8px 0; font-weight: 700;">🧠 TerraFlare — MobileNetV3-Small CNN Training Manager</h2>
        <p style="color: #94A3B8; margin: 0;">Train lightweight MobileNetV3 false-positive verification models with transfer learning and class weighting.</p>
    </div>
    """, unsafe_allow_html=True)

    tab_train, tab_registry = st.tabs([
        "🚀 Train CNN Model",
        "🏆 CNN Model Registry"
    ])

    with tab_train:
        render_cnn_training_form()

    with tab_registry:
        render_model_registry_view(model_filter="CNN")

def render_cnn_training_form():
    st.subheader("1. Select Classification Training Dataset")

    all_datasets = list_registered_datasets()
    cls_datasets = [d for d in all_datasets if d.get("dataset_type") == "Classification"]

    if not cls_datasets:
        st.warning("⚠️ No validated Classification datasets found in the Dataset Registry. Please upload an Image Classification dataset in the Dataset Manager first.")
        return

    ds_options = [f"{d.get('dataset_id')} — {d.get('dataset_name')} ({d.get('total_images')} images)" for d in cls_datasets]
    selected_ds_str = st.selectbox("Classification Dataset:", ds_options, key="cnn_ds_select")
    selected_ds_id = selected_ds_str.split(" — ")[0]
    selected_ds_meta = get_dataset_metadata(selected_ds_id)

    st.markdown("---")
    st.subheader("2. Hyperparameters Configuration")

    col1, col2 = st.columns([1, 1])
    with col1:
        exp_name = st.text_input("Experiment Name:", value="MobileNetV3_FireSmoke_001", key="cnn_exp_name_input")
    with col2:
        st.text_input("Model Architecture:", value="MobileNetV3-Small", disabled=True, key="cnn_model_arch_input")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        epochs = st.number_input("Epochs:", min_value=1, max_value=300, value=20, step=1, key="cnn_epochs_input")
    with c2:
        batch_size = st.selectbox("Batch Size:", options=[8, 16, 32, 64, 128], index=2, key="cnn_batch_input")
    with c3:
        lr0 = st.number_input("Learning Rate:", value=0.001, format="%.4f", step=0.0005, key="cnn_lr_input")
    with c4:
        devices = detect_compute_device()
        device_choice = st.selectbox("Hardware Device:", devices, key="cnn_device_input")

    imgsz = st.select_slider("Image Size (px):", options=[128, 224, 256, 384, 512], value=224, key="cnn_imgsz_input")
    use_class_weighting = st.checkbox("Enable Class-Weighted Loss (Handles Class Imbalance)", value=True, key="cnn_class_weights_check")

    # Detected Classes Summary
    detected_classes = selected_ds_meta.get("classes", [])
    st.markdown("---")
    st.subheader("3. Pre-Training Dataset Confirmation")

    st.markdown(f"""
    <div style="background-color: #0F172A; border: 1px solid #1E293B; border-radius: 8px; padding: 16px; margin-bottom: 20px;">
        <p style="margin: 0 0 6px 0; font-size: 1rem; color: #F8FAFC;"><strong>Dataset:</strong> {selected_ds_meta.get('dataset_name')} (<code>{selected_ds_meta.get('dataset_id')}</code>)</p>
        <p style="margin: 0 0 6px 0; color: #CBD5E1;"><strong>Detected Classes ({len(detected_classes)}):</strong> <code>{', '.join(detected_classes)}</code></p>
        <p style="margin: 0 0 6px 0; color: #CBD5E1;"><strong>Training Images:</strong> {selected_ds_meta.get('train_count', 0)} &nbsp;|&nbsp; <strong>Validation Images:</strong> {selected_ds_meta.get('validation_count', 0)} &nbsp;|&nbsp; <strong>Test Images:</strong> {selected_ds_meta.get('test_count', 0)}</p>
        <p style="margin: 0 0 6px 0; color: #CBD5E1;"><strong>Class Weighting:</strong> {'Enabled ✅' if use_class_weighting else 'Disabled ❌'}</p>
        <div style="color: #10B981; font-weight: 700; font-size: 1.05rem;">Ready to train ✅</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 START CNN TRAINING", key="start_cnn_training_btn", use_container_width=True):
        st.markdown("---")
        st.markdown(f"### 🚀 Training MobileNetV3: **{exp_name}**")
        status_box = st.empty()

        def update_status(msg):
            status_box.info(f"⏳ {msg}")

        try:
            with st.spinner("Training MobileNetV3-Small model in progress..."):
                exp_config = train_cnn_mobilenetv3(
                    dataset_metadata=selected_ds_meta,
                    experiment_name=exp_name.strip(),
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                    learning_rate=float(lr0),
                    image_size=int(imgsz),
                    device_choice=device_choice,
                    use_class_weighting=use_class_weighting,
                    progress_callback=update_status
                )
            
            update_status("Evaluating best model checkpoint on test dataset...")
            metrics = evaluate_cnn_experiment(exp_config, selected_ds_meta)
            
            # Register in Model Registry
            reg_entry = register_trained_model(
                experiment_id=exp_config["experiment_id"],
                experiment_name=exp_config["experiment_name"],
                dataset_id=selected_ds_meta.get("dataset_id"),
                architecture="MobileNetV3-Small",
                best_weights_path=exp_config["best_model_path"],
                metrics=metrics,
                classes=selected_ds_meta.get("classes", [])
            )
            
            status_box.empty()
            st.success(f"🎉 MobileNetV3 Training & Evaluation Completed! Model ID: **{reg_entry['model_id']}**")

            display_cnn_results(exp_config, metrics)
        except Exception as e:
            status_box.empty()
            st.error(f"❌ CNN Training Failed: {str(e)}")

def display_cnn_results(exp_config, metrics):
    st.markdown("---")
    st.markdown(f"### 🧠 MobileNetV3-Small PERFORMANCE: `{exp_config.get('experiment_name')}`")

    acc = metrics.get("accuracy", 0.0) * 100
    p = metrics.get("precision", 0.0) * 100
    r = metrics.get("recall", 0.0) * 100
    f1 = metrics.get("f1", 0.0) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.2f}%")
    col2.metric("Precision", f"{p:.2f}%")
    col3.metric("Recall", f"{r:.2f}%")
    col4.metric("F1 Score", f"{f1:.2f}%")

    st.markdown("<br/>", unsafe_allow_html=True)
    col_per_class, col_cm = st.columns([1, 1])

    with col_per_class:
        st.markdown("#### 📊 Per-Class Performance")
        per_cls = metrics.get("per_class", {})
        if per_cls:
            rows = []
            for cname, mdict in per_cls.items():
                rows.append({
                    "Class": cname,
                    "Precision": f"{mdict.get('precision', 0.0)*100:.1f}%",
                    "Recall": f"{mdict.get('recall', 0.0)*100:.1f}%",
                    "F1 Score": f"{mdict.get('f1', 0.0)*100:.1f}%"
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with col_cm:
        st.markdown("#### 🎯 Confusion Matrix Plot")
        cm_path = metrics.get("confusion_matrix_path")
        if cm_path and os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix", use_column_width=True)

# ---------------------------------------------------------
# 3. SHARED MODEL REGISTRY VIEW & DEPLOY ACTION
def format_metric_val(val):
    if val is None or val == "N/A":
        return "N/A"
    if isinstance(val, str):
        if val.endswith("%"):
            return val
        try:
            val = float(val)
        except ValueError:
            return val
    if isinstance(val, (int, float)):
        val_pct = val * 100.0 if 0.0 <= val <= 1.0 else val
        return f"{val_pct:.2f}%"
    return str(val)

# ---------------------------------------------------------
def render_model_registry_view(model_filter="ALL"):
    st.subheader("🏆 Trained Model Registry")
    models = list_registered_models()

    if model_filter == "YOLO":
        models = [m for m in models if "YOLO" in m.get("architecture", "").upper()]
    elif model_filter == "CNN":
        models = [m for m in models if "MOBILENET" in m.get("architecture", "").upper() or "CNN" in m.get("architecture", "").upper()]

    if not models:
        st.info("No trained models registered under this category yet.")
        return

    reg_rows = []
    for m in models:
        metrics = m.get("metrics", {})
        raw_m = m.get("raw_metrics", {})
        
        acc_val = raw_m.get("accuracy") if "accuracy" in raw_m else raw_m.get("map50", metrics.get("accuracy", metrics.get("map50")))
        prec_val = raw_m.get("precision", metrics.get("precision"))
        rec_val = raw_m.get("recall", metrics.get("recall"))
        f1_val = raw_m.get("f1", metrics.get("f1"))

        reg_rows.append({
            "Model ID": m.get("model_id"),
            "Experiment": m.get("experiment_name"),
            "Dataset ID": m.get("dataset_id"),
            "Architecture": m.get("architecture"),
            "Accuracy / mAP50": format_metric_val(acc_val),
            "Precision": format_metric_val(prec_val),
            "Recall": format_metric_val(rec_val),
            "F1": format_metric_val(f1_val),
            "Status": m.get("status")
        })

    st.dataframe(pd.DataFrame(reg_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Model Actions")
    model_opts = [f"{m.get('model_id')} — {m.get('experiment_name')} ({m.get('architecture')})" for m in models]
    sel_mod_str = st.selectbox("Select Model to Action", model_opts, key=f"reg_model_select_{model_filter}")
    sel_mod_id = sel_mod_str.split(" — ")[0]
    selected_model = next((m for m in models if m.get("model_id") == sel_mod_id), None)

    col_act1, col_act2, col_act3 = st.columns(3)
    with col_act1:
        if st.button("🔍 View Model Details", key=f"view_btn_{sel_mod_id}"):
            if selected_model:
                st.json(selected_model)

    with col_act2:
        if st.button("📊 Evaluate Metrics", key=f"eval_btn_{sel_mod_id}"):
            if selected_model:
                st.success(f"Model `{sel_mod_id}` metrics verified.")
                st.write(selected_model.get("metrics"))

    with col_act3:
        if "CNN" in selected_model.get("architecture", "").upper() or "MOBILENET" in selected_model.get("architecture", "").upper():
            if st.button("🚀 Deploy as CNN Verifier", key=f"deploy_cnn_{sel_mod_id}"):
                deploy_cnn_model(selected_model)
        else:
            if st.button("🚀 Deploy Candidate", key=f"deploy_candidate_{sel_mod_id}"):
                st.info(f"ℹ️ Model `{sel_mod_id}` marked as deployment candidate. Production model remains untouched.")

def deploy_cnn_model(model_entry):
    """
    Copies trained best_model.pth to weights/cnn_verifier.pt and class_mapping.json to weights/cnn_class_mapping.json.
    """
    weights_src = model_entry.get("best_weights_path")
    if not weights_src or not os.path.exists(weights_src):
        st.error("Model weights file not found on disk.")
        return

    exp_dir = os.path.dirname(weights_src)
    cmap_src = os.path.join(exp_dir, "class_mapping.json")

    target_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
    os.makedirs(target_dir, exist_ok=True)

    target_weights = os.path.join(target_dir, "cnn_verifier.pt")
    target_cmap = os.path.join(target_dir, "cnn_class_mapping.json")

    shutil.copy2(weights_src, target_weights)
    if os.path.exists(cmap_src):
        shutil.copy2(cmap_src, target_cmap)

    st.success(f"🎉 Model **{model_entry.get('model_id')}** deployed successfully as active TerraFlare CNN Verifier (`weights/cnn_verifier.pt`)!")
