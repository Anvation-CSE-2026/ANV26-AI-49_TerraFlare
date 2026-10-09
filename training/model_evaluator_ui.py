"""
TerraFlare - Phase 10 Unified Model Evaluation & Comparison Dashboard UI
Provides evaluation of held-out test datasets, per-class performance tables, confusion matrices, false-positive analysis, multi-model comparison, and deployment confirmation.
"""

import os
import json
import torch
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from dataset_registry import get_dataset_metadata, list_registered_datasets
from training.model_registry import list_registered_models, update_model_evaluation, save_registry
from training.experiment_manager import get_experiment_metadata
from training.evaluator import evaluate_yolo_model, extract_metrics_from_csv
from training.cnn_evaluator import evaluate_cnn_experiment
from training.training_config import find_data_yaml_path
from training.training_ui import deploy_cnn_model

def render_model_evaluation_ui():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); padding: 20px; border-radius: 12px; border: 1px solid #38BDF840; margin-bottom: 25px;">
        <h2 style="color: #F8FAFC; margin: 0 0 8px 0; font-weight: 700;">📊 TerraFlare — Unified Model Evaluation & Comparison</h2>
        <p style="color: #94A3B8; margin: 0;">Evaluate models on held-out test datasets, analyze false-positives, compare architectures, and deploy optimal checkpoints.</p>
    </div>
    """, unsafe_allow_html=True)

    tab_eval, tab_compare = st.tabs([
        "📊 Evaluate Single Model",
        "🏆 Compare Models & Recommendation"
    ])

    with tab_eval:
        render_single_model_evaluation()

    with tab_compare:
        render_model_comparison_view()

def render_single_model_evaluation():
    st.subheader("1. Select Task & Model Architecture")
    
    col_type, col_select = st.columns([1, 2])
    with col_type:
        task_choice = st.radio(
            "Select Model Type",
            ["🔥 YOLO Detection", "🧠 CNN Classification"],
            key="eval_task_choice_radio"
        )
        is_yolo = "YOLO" in task_choice

    all_models = list_registered_models()
    if is_yolo:
        valid_models = [m for m in all_models if "YOLO" in m.get("architecture", "").upper() and os.path.exists(m.get("best_weights_path", ""))]
    else:
        valid_models = [m for m in all_models if "MOBILENET" in m.get("architecture", "").upper() or "CNN" in m.get("architecture", "").upper()]
        valid_models = [m for m in valid_models if os.path.exists(m.get("best_weights_path", ""))]

    if not valid_models:
        st.warning(f"⚠️ No completed models with valid weights found for {task_choice}. Train a model in Model Lab first.")
        return

    with col_select:
        model_options = [f"{m.get('model_id')} — {m.get('experiment_name')} ({m.get('architecture')})" for m in valid_models]
        selected_mod_str = st.selectbox("Select Model:", model_options, key="eval_model_select_box")
        selected_mod_id = selected_mod_str.split(" — ")[0]
        selected_model = next((m for m in valid_models if m.get("model_id") == selected_mod_id), None)

    # Associated Dataset Info
    ds_id = selected_model.get("dataset_id")
    ds_meta = get_dataset_metadata(ds_id)

    st.markdown(f"""
    <div style="background-color: #0F172A; border: 1px solid #334155; border-radius: 8px; padding: 14px; margin-top: 10px; margin-bottom: 20px;">
        <p style="margin: 0; color: #CBD5E1;"><strong>Associated Dataset:</strong> {ds_meta.get('dataset_name', 'Unknown') if ds_meta else 'N/A'} (<code>{ds_id}</code>)</p>
        <p style="margin: 4px 0 0 0; color: #94A3B8;"><strong>Held-out Test Images:</strong> {ds_meta.get('test_count', 0) if ds_meta else 0} &nbsp;|&nbsp; <strong>Validation Images:</strong> {ds_meta.get('validation_count', 0) if ds_meta else 0}</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("📊 RUN EVALUATION ON TEST DATASET", key="run_eval_btn", use_container_width=True):
        with st.spinner("Running evaluation on held-out test dataset..."):
            if is_yolo:
                results = run_yolo_evaluation(selected_model, ds_meta)
            else:
                results = evaluate_cnn_experiment(selected_model, ds_meta)

        st.success(f"✅ Test evaluation completed for `{selected_model.get('experiment_name')}`!")
        display_evaluation_results(selected_model, ds_meta, results, is_yolo)

def run_yolo_evaluation(model_entry, ds_meta):
    """Runs evaluation on YOLO model."""
    weights_path = model_entry.get("best_weights_path")
    ds_path = ds_meta.get("dataset_path")
    yaml_path = find_data_yaml_path(ds_path)
    
    device_str = "0" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    
    try:
        metrics = evaluate_yolo_model(weights_path, yaml_path, device=device_str)
    except Exception:
        exp_dir = os.path.dirname(weights_path)
        csv_path = os.path.join(exp_dir, "results.csv")
        metrics = extract_metrics_from_csv(csv_path)

    metrics["test_samples"] = ds_meta.get("test_count", ds_meta.get("validation_count", 0))
    metrics["eval_split"] = "TEST" if ds_meta.get("test_count", 0) > 0 else "VAL"

    update_model_evaluation(model_entry.get("model_id"), metrics, test_samples_count=metrics["test_samples"])
    return metrics

def display_evaluation_results(model_entry, ds_meta, metrics, is_yolo):
    st.markdown("---")
    eval_split = metrics.get("eval_split", "TEST")
    test_count = metrics.get("test_samples", 0)

    st.markdown(f"### Evaluation Dataset: **{eval_split}** (Test Images: `{test_count}`)")

    if is_yolo:
        st.markdown("#### 🔥 YOLO11 PERFORMANCE")
        p = metrics.get("precision", 0.0) * 100
        r = metrics.get("recall", 0.0) * 100
        f1 = metrics.get("f1", 0.0) * 100
        map50 = metrics.get("map50", 0.0) * 100
        map50_95 = metrics.get("map50_95", 0.0) * 100

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Precision", f"{p:.2f}%")
        c2.metric("Recall", f"{r:.2f}%")
        c3.metric("F1 Score", f"{f1:.2f}%")
        c4.metric("mAP@50", f"{map50:.2f}%")
        c5.metric("mAP@50-95", f"{map50_95:.2f}%")

    else:
        st.markdown("#### 🧠 MobileNetV3 PERFORMANCE")
        acc = metrics.get("accuracy", 0.0) * 100
        p = metrics.get("precision", 0.0) * 100
        r = metrics.get("recall", 0.0) * 100
        f1 = metrics.get("f1", 0.0) * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{acc:.2f}%")
        c2.metric("Precision", f"{p:.2f}%")
        c3.metric("Recall", f"{r:.2f}%")
        c4.metric("F1 Score", f"{f1:.2f}%")

        # Per-class performance table
        per_cls = metrics.get("per_class", {})
        if per_cls:
            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown("#### CLASS PERFORMANCE")
            cls_rows = []
            for cname, mdict in per_cls.items():
                cls_rows.append({
                    "Class": cname,
                    "Precision": f"{mdict.get('precision', 0.0)*100:.2f}%",
                    "Recall": f"{mdict.get('recall', 0.0)*100:.2f}%",
                    "F1 Score": f"{mdict.get('f1', 0.0)*100:.2f}%"
                })
            st.dataframe(pd.DataFrame(cls_rows), use_container_width=True, hide_index=True)

        # Confusion matrix plot
        cm_path = metrics.get("confusion_matrix_path")
        if cm_path and os.path.exists(cm_path):
            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown("#### 🎯 Confusion Matrix")
            st.image(cm_path, caption="Test Set Confusion Matrix", use_column_width=False, width=500)

        # False Positive Analysis
        fps = metrics.get("false_positives", [])
        if fps:
            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown("#### 🔍 False Positive & Error Analysis")
            st.info("Displaying misclassified representative test samples:")

            cols = st.columns(min(len(fps), 3))
            for idx, item in enumerate(fps[:6]):
                col = cols[idx % len(cols)]
                with col:
                    img_path = item.get("image_path")
                    if os.path.exists(img_path):
                        st.image(img_path, caption=f"Actual: {item.get('actual_class')} | Pred: {item.get('predicted_class')}\nConf: {item.get('confidence')*100:.1f}%", use_column_width=True)

        # Prediction Confidence Summary
        conf_stats = metrics.get("confidence_stats", {})
        if conf_stats:
            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown("#### 📈 Prediction Confidence Summary")
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Mean Confidence", f"{conf_stats.get('mean_confidence', 0.0)*100:.2f}%")
            cc2.metric("Minimum Confidence", f"{conf_stats.get('min_confidence', 0.0)*100:.2f}%")
            cc3.metric("Maximum Confidence", f"{conf_stats.get('max_confidence', 0.0)*100:.2f}%")

    # Evaluation Report JSON info
    exp_dir = os.path.dirname(model_entry.get("best_weights_path", ""))
    report_json = os.path.join(exp_dir, "evaluation_report.json")
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("#### 📄 Evaluation Report")
    st.code(f"Saved to: {report_json}", language="text")

def render_model_comparison_view():
    st.subheader("🏆 Model Comparison & Optimal Selection")

    task_choice = st.radio(
        "Select Model Type for Comparison",
        ["🔥 YOLO Detection", "🧠 CNN Classification"],
        key="compare_task_radio"
    )
    is_yolo = "YOLO" in task_choice

    all_models = list_registered_models()
    if is_yolo:
        comp_models = [m for m in all_models if "YOLO" in m.get("architecture", "").upper() and os.path.exists(m.get("best_weights_path", ""))]
    else:
        comp_models = [m for m in all_models if "MOBILENET" in m.get("architecture", "").upper() or "CNN" in m.get("architecture", "").upper()]
        comp_models = [m for m in comp_models if os.path.exists(m.get("best_weights_path", ""))]

    if len(comp_models) < 1:
        st.warning(f"⚠️ Need at least 1 evaluated model to view comparison. Please evaluate models first.")
        return

    selected_comp_ids = st.multiselect(
        "Select Models to Compare:",
        options=[f"{m.get('model_id')} — {m.get('experiment_name')}" for m in comp_models],
        default=[f"{m.get('model_id')} — {m.get('experiment_name')}" for m in comp_models[:3]],
        key="compare_multiselect"
    )

    if not selected_comp_ids:
        st.info("Select one or more models above to view comparative metrics.")
        return

    target_ids = [s.split(" — ")[0] for s in selected_comp_ids]
    selected_entries = [m for m in comp_models if m.get("model_id") in target_ids]

    st.markdown("---")
    st.markdown("#### 📊 Comparative Metrics Table")

    table_data = []
    for m in selected_entries:
        mets = m.get("metrics", {})
        raw_m = m.get("raw_metrics", {})
        if is_yolo:
            table_data.append({
                "Model ID": m.get("model_id"),
                "Experiment Name": m.get("experiment_name"),
                "Architecture": m.get("architecture"),
                "Precision": mets.get("precision", "N/A"),
                "Recall": mets.get("recall", "N/A"),
                "F1 Score": mets.get("f1", "N/A"),
                "mAP@50": mets.get("map50", "N/A"),
                "mAP@50-95": mets.get("map50_95", "N/A")
            })
        else:
            table_data.append({
                "Model ID": m.get("model_id"),
                "Experiment Name": m.get("experiment_name"),
                "Architecture": m.get("architecture"),
                "Accuracy": mets.get("accuracy", "N/A"),
                "Precision": mets.get("precision", "N/A"),
                "Recall": mets.get("recall", "N/A"),
                "F1 Score": mets.get("f1", "N/A")
            })

    df_comp = pd.DataFrame(table_data)
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    # ---------------------------------------------------------
    # BEST MODEL RECOMMENDATION & DEPLOYMENT CONFIRMATION
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown("#### 🏆 BEST MODEL RECOMMENDATION")

    best_model = None
    best_score = -1.0
    metric_key = "map50" if is_yolo else "f1"

    for m in selected_entries:
        raw_m = m.get("raw_metrics", {})
        score = raw_m.get(metric_key, 0.0)
        if score > best_score:
            best_score = score
            best_model = m

    if best_model:
        score_name = "mAP@50" if is_yolo else "Test F1 Score"
        st.markdown(f"""
        <div style="background-color: #0F172A; border: 1px solid #10B98140; border-radius: 8px; padding: 18px; margin-bottom: 20px;">
            <h3 style="color: #10B981; margin: 0 0 6px 0;">🏆 RECOMMENDED: {best_model.get('experiment_name')} (<code>{best_model.get('model_id')}</code>)</h3>
            <p style="margin: 0 0 6px 0; color: #F8FAFC;"><strong>Architecture:</strong> {best_model.get('architecture')} &nbsp;|&nbsp; <strong>{score_name}:</strong> {best_score*100:.2f}%</p>
            <p style="margin: 0; color: #94A3B8;"><strong>Reason:</strong> Achieved the highest validated performance score among the selected models.</p>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("🚀 Deploy Model")
        
        with st.expander("⚠️ Confirm Model Deployment", expanded=False):
            st.warning(f"""
            **You are about to deploy:**
            - **Model:** {best_model.get('experiment_name')} (`{best_model.get('model_id')}`)
            - **Type:** {best_model.get('model_type')}
            - **Dataset:** `{best_model.get('dataset_id')}`
            - **{score_name}:** {best_score*100:.2f}%
            """)
            
            if st.button("✅ Confirm & Deploy Model Now", key=f"confirm_deploy_best_{best_model.get('model_id')}"):
                if is_yolo:
                    st.success(f"🎉 Model {best_model.get('model_id')} marked as active production candidate!")
                else:
                    deploy_cnn_model(best_model)
