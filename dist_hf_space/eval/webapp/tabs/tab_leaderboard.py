"""
tab_leaderboard.py
Public Leaderboard Tab with interactive filters and multi-axis radar charts.
"""

import json
import streamlit as st
import pandas as pd
from pathlib import Path
from utils.visualizer import create_radar_chart

def render_leaderboard_tab(db_path: Path):
    st.markdown("## 🏆 Public Benchmark Leaderboard")
    st.markdown(
        "A standardized multi-modal benchmark ranking 3D Chest CT models across "
        "**Diagnostic Classification (Axis A)**, **Volumetric Localization (Axis B)**, "
        "and **Clinical Explainability (Axis C)**."
    )
    
    # Load database
    if not db_path.exists():
        st.error("Leaderboard database not found.")
        return
    
    with open(db_path, "r") as f:
        db = json.load(f)
    
    df = pd.DataFrame(db)
    
    # Top Controls / Filters
    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        track_filter = st.selectbox(
            "Filter by Track:",
            ["All Tracks", "Unified / Multi-Task", "Classification (Axis A)", "Segmentation (Axis B)"]
        )
    with col2:
        paradigm_filter = st.selectbox(
            "Filter by Paradigm:",
            ["All Paradigms", "3D Vision-Language", "Specialized 3D CNN", "Promptable 3D Segmenter", "Dictionary Learning", "Dense 3D Segmenter"]
        )
    with col3:
        sort_by = st.selectbox(
            "Sort Leaderboard By:",
            ["Rank", "Axis A AUROC", "Axis B Dice", "Axis C Pointing Hit", "Inference Latency (Fastest)"]
        )
    
    # Filter logic
    filtered_df = df.copy()
    if track_filter != "All Tracks":
        if "Classification" in track_filter:
            filtered_df = filtered_df[filtered_df["axis_a_auroc"].notnull()]
        elif "Segmentation" in track_filter:
            filtered_df = filtered_df[filtered_df["axis_b_dice"].notnull()]
        elif "Unified" in track_filter:
            filtered_df = filtered_df[filtered_df["track"] == "Unified"]
            
    if paradigm_filter != "All Paradigms":
        filtered_df = filtered_df[filtered_df["paradigm"] == paradigm_filter]
        
    # Sort logic
    if sort_by == "Axis A AUROC":
        filtered_df = filtered_df.sort_values(by="axis_a_auroc", ascending=False)
    elif sort_by == "Axis B Dice":
        filtered_df = filtered_df.sort_values(by="axis_b_dice", ascending=False)
    elif sort_by == "Axis C Pointing Hit":
        filtered_df = filtered_df.sort_values(by="axis_c_pointing_hit", ascending=False)
    elif sort_by == "Inference Latency (Fastest)":
        filtered_df = filtered_df.sort_values(by="latency_sec", ascending=True)
    else:
        filtered_df = filtered_df.sort_values(by="rank", ascending=True)

    # Display Table with formatted values
    display_df = filtered_df[[
        "rank", "model_name", "institution", "paradigm",
        "axis_a_auroc", "axis_a_f1", "axis_a_ece",
        "axis_b_dice", "axis_b_instance_f1", "axis_b_assd_mm",
        "axis_c_pointing_hit", "latency_sec"
    ]].copy()
    
    display_df.columns = [
        "Rank", "Model", "Institution", "Paradigm",
        "Axis A AUROC ↑", "Axis A F1 ↑", "Axis A ECE ↓",
        "Axis B Dice ↑", "Instance F1 ↑", "ASSD (mm) ↓",
        "Pointing Hit ↑", "Latency (s) ↓"
    ]
    
    # Format floats
    for col in ["Axis A AUROC ↑", "Axis A F1 ↑", "Axis A ECE ↓", "Axis B Dice ↑", "Instance F1 ↑", "Pointing Hit ↑"]:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "—")
    for col in ["ASSD (mm) ↓", "Latency (s) ↓"]:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "—")
        
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=320
    )
    
    st.markdown("---")
    
    # Head-to-Head Radar Comparison Section
    st.markdown("### 🕸️ Head-to-Head Radar Profile Comparison")
    st.caption("Select up to 4 models from the database to generate an interactive comparative spider chart across all evaluation dimensions.")
    
    all_model_names = [m["model_name"] for m in db]
    default_selected = [m for m in ["Merlin (RadLLaMA-7b)", "CT-CLIP (CT-LiPro)", "nnU-Net (3D Full-Res)", "LC-KSVD2 (GPU-OMP)"] if m in all_model_names]
    
    selected_models = st.multiselect(
        "Select Models to Compare:",
        options=all_model_names,
        default=default_selected
    )
    
    if selected_models:
        selected_data = [m for m in db if m["model_name"] in selected_models]
        fig = create_radar_chart(selected_data)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select at least one model above to render the comparison radar chart.")

