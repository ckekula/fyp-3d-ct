"""
tab_inspector.py
Interactive 3D CT Slice Inspector & Model Overlay Comparison Tab.
"""

import streamlit as st
from utils.visualizer import render_sample_slice

def render_inspector_tab():
    st.markdown("## 🔍 Interactive 3D CT Slice Inspector")
    st.markdown(
        "Scrub through 3D axial CT slices to visually compare model predictions against "
        "expert radiological ground truth annotations."
    )
    
    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        case_choice = st.selectbox(
            "Select Reference Test Case:",
            [
                "train_10033_a_2 (Ground Glass Opacity / Cat 2c)",
                "train_10000_a_1 (Pulmonary Nodule / Cat 2a)",
                "train_10094_a_2 (Consolidation & Opacity / Cat 2b+2c)"
            ]
        )
    with col2:
        overlay_model = st.selectbox(
            "Select Model Overlay:",
            [
                "CT-CLIP Grad-CAM",
                "Merlin Attention",
                "BiomedParse Soft Mask",
                "nnU-Net Binary Mask",
                "LC-KSVD2 Patch Coverage"
            ]
        )
    with col3:
        show_gt = st.checkbox("Show Expert Ground Truth Contour (Green)", value=True)
        
    slice_idx = st.slider("3D Axial Slice Index (Inferior ↔ Superior)", min_value=0, max_value=240, value=112, step=1)
    
    fig = render_sample_slice(
        slice_idx=slice_idx,
        show_gt=show_gt,
        overlay_type=overlay_model
    )
    
    st.pyplot(fig, use_container_width=True)
    
    # Clinical Case Annotation Summary
    st.info(
        f"**Case Details**: `{case_choice.split()[0]}` | "
        f"**Active Overlay**: `{overlay_model}` | "
        f"**Ground Truth**: Green outline indicates expert radiologist segmentation boundary."
    )

