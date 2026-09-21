"""
visualizer.py
Interactive Plotly Charts & 2D Slice Visualizer for CT-Eval3D Webapp.
"""

import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional

def create_radar_chart(models_data: List[Dict[str, Any]]) -> go.Figure:
    """
    Generate an interactive Plotly Spider / Radar chart comparing models across:
    1. Axis A: Diagnostic AUROC (0..1)
    2. Axis B: Spatial Dice (0..1)
    3. Axis C: Explainability Pointing Hit (0..1)
    4. Calibration: 1 - ECE (0..1)
    5. Speed: Inverted Latency Score (0..1)
    """
    categories = [
        "Diagnostic AUROC (Axis A)",
        "Spatial Dice (Axis B)",
        "Explainability Hit % (Axis C)",
        "Calibration (1 - ECE)",
        "Inference Speed Score"
    ]
    
    fig = go.Figure()
    
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"]
    
    for idx, model in enumerate(models_data):
        auroc = model.get("axis_a_auroc") or 0.0
        dice = model.get("axis_b_dice") or 0.0
        pointing = model.get("axis_c_pointing_hit") or 0.0
        ece = model.get("axis_a_ece")
        calib = (1.0 - ece) if (ece is not None and not np.isnan(ece)) else 0.5
        latency = model.get("latency_sec") or 5.0
        speed_score = float(np.clip(1.0 - (latency / 15.0), 0.1, 1.0))
        
        values = [auroc, dice, pointing, calib, speed_score]
        values.append(values[0])  # Close loop
        
        color = colors[idx % len(colors)]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=model.get("model_name", f"Model {idx+1}"),
            line=dict(color=color, width=2),
            opacity=0.6
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1.0],
                tickfont=dict(size=10)
            )
        ),
        showlegend=True,
        title=dict(
            text="<b>Multi-Axis Benchmark Comparison (Radar Profile)</b>",
            font=dict(size=16)
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        height=450
    )
    
    return fig

def render_sample_slice(
    slice_idx: int = 112,
    show_gt: bool = True,
    overlay_type: str = "CT-CLIP Grad-CAM"
) -> plt.Figure:
    """Render a synthetic/representative CT axial slice with lesion annotations."""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Generate realistic lung CT background
    x = np.linspace(-1, 1, 256)
    y = np.linspace(-1, 1, 256)
    xx, yy = np.meshgrid(x, y)
    
    # Body contour & lungs
    body = (xx**2 + yy**2) < 0.85
    left_lung = ((xx + 0.35)**2 / 0.15 + yy**2 / 0.5) < 0.7
    right_lung = ((xx - 0.35)**2 / 0.15 + yy**2 / 0.5) < 0.7
    
    ct_img = np.zeros_like(xx)
    ct_img[body] = 0.4
    ct_img[left_lung | right_lung] = 0.15
    
    # Add subtle lung texture
    noise = np.random.normal(0, 0.02, xx.shape)
    ct_img = np.clip(ct_img + noise, 0, 1)
    
    # Lesion location (right lower lobe)
    lesion_mask = ((xx - 0.35)**2 + (yy - 0.25)**2) < 0.015
    ct_img[lesion_mask] += 0.35
    
    ax.imshow(ct_img, cmap="gray")
    
    # Overlay GT
    if show_gt:
        ax.contour(lesion_mask, colors="#00FF00", linewidths=2.5)
    
    # Overlay Model Predictions
    if overlay_type == "CT-CLIP Grad-CAM":
        # Continuous smooth Gaussian heatmap
        heatmap = np.exp(-(((xx - 0.34)**2 + (yy - 0.26)**2) / 0.04))
        ax.imshow(heatmap, cmap="jet", alpha=0.45)
    elif overlay_type == "Merlin Attention":
        heatmap = np.exp(-(((xx - 0.35)**2 + (yy - 0.24)**2) / 0.05))
        ax.imshow(heatmap, cmap="magma", alpha=0.45)
    elif overlay_type == "BiomedParse Soft Mask":
        soft_mask = np.exp(-(((xx - 0.35)**2 + (yy - 0.25)**2) / 0.02))
        ax.imshow(soft_mask, cmap="Blues", alpha=0.5)
    elif overlay_type == "nnU-Net Binary Mask":
        pred_binary = ((xx - 0.345)**2 + (yy - 0.252)**2) < 0.014
        ax.contour(pred_binary, colors="#FF0055", linewidths=2.5)
    elif overlay_type == "LC-KSVD2 Patch Coverage":
        pred_blocks = ((xx - 0.35)**2 + (yy - 0.25)**2) < 0.018
        ax.imshow(pred_blocks, cmap="autumn", alpha=0.4)
    
    ax.set_title(f"Axial Slice {slice_idx} | {overlay_type} (GT in Green)", fontsize=11, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    
    return fig

