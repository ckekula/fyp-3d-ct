---
title: "ThorAxis: Unified 3D Chest CT Benchmark & Leaderboard"
emoji: "🫁"
colorFrom: "indigo"
colorTo: "red"
sdk: "gradio"
sdk_version: "5.0.0"
app_file: "app.py"
pinned: false
license: "mit"
---

# ThorAxis / CT-Eval3D: Unified 3D Chest CT Benchmark & Leaderboard

A standardized public benchmarking platform for evaluating 3D Chest CT Artificial Intelligence models across:
1. **Axis A (Diagnostic Triage)**: Multi-label Classification, AUROC, PR-AUC, and Expected Calibration Error (ECE).
2. **Axis B (Lesion Segmentation)**: 3D Volumetric Dice, IoU, 3D Connected-Component Instance Matching (ReXRank Protocol), and Physical Spacing Distances (ASSD in mm).
3. **Axis C (Clinical Explainability)**: 3D Saliency / Grad-CAM Pointing Game Hit Rate and Energy-Inside-Mask (EIM %).
4. **Axis D (Clinical Efficiency)**: Latency per 3D scan, GPU VRAM requirements, and throughput.

## Quickstart
```bash
pip install -r requirements.txt
python app.py
```
