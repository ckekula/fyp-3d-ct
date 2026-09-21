"""
tab_docs.py
Submission Schemas, Mathematical Formats & API Docs.
"""

import streamlit as st
from pathlib import Path

def render_docs_tab(static_dir: Path):
    st.markdown("## 📖 Benchmark Documentation & Submission Formats")
    
    st.markdown("### 1. Download Starter Submission Templates")
    st.markdown("Use these pre-formatted starter files to structure your model predictions:")
    
    col1, col2 = st.columns(2)
    with col1:
        sample_csv_path = static_dir / "sample_classification.csv"
        if sample_csv_path.exists():
            with open(sample_csv_path, "rb") as f:
                st.download_button(
                    label="📥 Download Starter Classification CSV Template",
                    data=f,
                    file_name="sample_classification.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    with col2:
        st.button("📦 Download Starter Segmentation ZIP Template", disabled=True, use_container_width=True)
        
    st.markdown("---")
    st.markdown("### 2. Standardized Submission File Formats")
    
    tab1, tab2, tab3 = st.tabs(["Track A: Classification", "Track B: Segmentation", "Track C: Unified"])
    
    with tab1:
        st.markdown("""
        **Format**: A comma-separated values (`.csv`) table containing continuous prediction probabilities $\in [0.0, 1.0]$.
        ```csv
        volume_name,lung_nodule,lung_opacity,consolidation,atelectasis
        test_0001.nii.gz,0.892,0.124,0.045,0.081
        test_0002.nii.gz,0.052,0.912,0.840,0.110
        test_0003.nii.gz,0.120,0.945,0.112,0.098
        ```
        """)
        
    with tab2:
        st.markdown("""
        **Format**: A compressed `.zip` archive containing 3D binary NIfTI (`.nii.gz`) or NumPy (`.npz`) arrays matching original scan dimensions $(D, H, W)$.
        ```
        masks.zip
        ├── test_0001.nii.gz
        ├── test_0002.nii.gz
        └── ...
        ```
        """)
        
    with tab3:
        st.markdown("""
        **Format**: A unified `.zip` archive containing both `predictions.csv` and a `masks/` subdirectory.
        ```
        unified_submission.zip
        ├── predictions.csv
        └── masks/
            ├── test_0001.npz
            └── test_0002.npz
        ```
        """)

    st.markdown("---")
    st.markdown("### 3. Open-Source Python Package (`cteval3d`)")
    st.markdown("""
    You can evaluate your models locally before submission using our official pip package:
    ```bash
    pip install cteval3d
    ```
    ```python
    import cteval3d as cteval

    # Axis A: Classification & Calibration
    metrics = cteval.compute_classification_metrics(
        samples, class_names=["lung_nodule", "lung_opacity", "consolidation", "atelectasis"]
    )
    print("Macro AUROC:", metrics["macro"]["auroc"])
    ```
    """)

