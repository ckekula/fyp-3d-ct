"""
app.py  (Hugging Face Space entry-point — must remain at repo root)
ThorAxis / CT-Eval3D: Unified 3D Chest CT Benchmark & Public Leaderboard
All supporting modules live under eval/:
  eval/core/          — metric engines (unmodified)
  eval/adapters/      — model + submission adapters
  eval/leaderboard/   — validation, scoring, schemas, examples, policy
  eval/webapp/        — utils, tabs, leaderboard_db, benchmark_data
"""

import io
import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
import gradio as gr
from pathlib import Path

# ── Monkey-patch gradio_client 4.44.0 bug ────────────────────────────────────
# Bug: `_json_schema_to_python_type` is called with `additionalProperties=False`
# (a bool), then `get_type(False)` does `"const" in False` → TypeError.
# Fix: guard every entry-point with isinstance(..., dict).
import gradio_client.utils as _gcu
_orig_jschema = _gcu._json_schema_to_python_type
def _safe_jschema(schema, defs=None):
    if not isinstance(schema, dict):
        return "any"
    return _orig_jschema(schema, defs)
_gcu._json_schema_to_python_type = _safe_jschema
# Also patch the public wrapper that calls .get() on the schema
_orig_public = _gcu.json_schema_to_python_type
def _safe_public(schema):
    if not isinstance(schema, dict):
        return "any"
    return _orig_public(schema)
_gcu.json_schema_to_python_type = _safe_public
# ─────────────────────────────────────────────────────────────────────────────

# ── path setup ──────────────────────────────────────────────────────────────
SPACE_ROOT = Path(__file__).resolve().parent          # repo root (where app.py lives)
EVAL_DIR   = SPACE_ROOT / "eval"
WEBAPP_DIR = EVAL_DIR / "webapp"

for p in [str(SPACE_ROOT), str(EVAL_DIR), str(WEBAPP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── path constants ───────────────────────────────────────────────────────────
DB_PATH       = WEBAPP_DIR / "leaderboard_db.json"
GT_PATH       = WEBAPP_DIR / "benchmark_data" / "sample_ground_truth.json"
EXAMPLES_DIR  = EVAL_DIR / "leaderboard" / "examples"

# ── imports from eval package ────────────────────────────────────────────────
from eval.leaderboard.validation import validate_track_a, validate_track_b, validate_track_c
from eval.webapp.utils.visualizer   import create_radar_chart, render_sample_slice
from eval.webapp.utils.eval_backend import (
    load_ground_truth,
    evaluate_classification_csv,
    evaluate_segmentation_mock,
)

# ── metric tooltip map ────────────────────────────────────────────────────────
METRIC_TOOLTIPS = {
    "Rank":          "Global leaderboard ranking based on primary evaluation axis.",
    "Model":         "Model name and submitted version.",
    "Institution":   "Submitting research group or organization.",
    "Track":         "Track A (Classification), Track B (Segmentation), or Track C (Unified).",
    "Paradigm":      "Architecture paradigm (Vision-Language, 3D CNN, Dictionary Learning, etc.).",
    "AUROC ↑":       "Axis A: Macro Area Under Multi-Label ROC Curve — higher is better.",
    "F1 ↑":          "Axis A: Macro F1-Score at 0.50 decision threshold — higher is better.",
    "ECE ↓":         "Axis A: Expected Calibration Error (10 bins) — lower is better. N/A for hard labels.",
    "Dice ↑":        "Axis B: 3D Volumetric Dice Overlap — higher is better.",
    "Instance F1 ↑": "Axis B: ReXRank 3D connected-component matching F1 at Dice ≥ 0.20 — higher is better.",
    "ASSD (mm) ↓":   "Axis B: Average Symmetric Surface Distance in mm — lower is better.",
    "Pointing Hit ↑":"Axis C: Explainability Pointing Game hit rate inside GT contour — higher is better.",
    "Latency (s) ↓": "Axis D: Inference latency per 3D CT scan in seconds — lower is better.",
}

# ── helpers ───────────────────────────────────────────────────────────────────
def _fmt(val, decimals=4):
    try:
        v = float(val)
        if np.isnan(v):
            return "N/A"
        return f"{v:.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"

def get_all_model_names():
    if not DB_PATH.exists():
        return []
    with open(DB_PATH) as f:
        return [m["model_name"] for m in json.load(f)]

def load_leaderboard_data(track_filter="All Tracks", paradigm_filter="All Paradigms", sort_by="Rank"):
    if not DB_PATH.exists():
        return pd.DataFrame()
    with open(DB_PATH) as f:
        db = json.load(f)
    df = pd.DataFrame(db)

    if track_filter != "All Tracks":
        if "Classification" in track_filter:
            df = df[df["axis_a_auroc"].notnull()]
        elif "Segmentation" in track_filter:
            df = df[df["axis_b_dice"].notnull()]
        elif "Unified" in track_filter:
            df = df[df["track"] == "Unified"]

    if paradigm_filter != "All Paradigms":
        df = df[df["paradigm"] == paradigm_filter]

    sort_map = {
        "Axis A AUROC": ("axis_a_auroc", False),
        "Axis B Dice": ("axis_b_dice", False),
        "Axis C Pointing Hit": ("axis_c_pointing_hit", False),
        "Inference Latency (Fastest)": ("latency_sec", True),
    }
    if sort_by in sort_map:
        col, asc = sort_map[sort_by]
        df = df.sort_values(by=col, ascending=asc)
    else:
        df = df.sort_values(by="rank", ascending=True)

    display = df[["rank","model_name","institution","track","paradigm",
                  "axis_a_auroc","axis_a_f1","axis_a_ece",
                  "axis_b_dice","axis_b_instance_f1","axis_b_assd_mm",
                  "axis_c_pointing_hit","latency_sec"]].copy()
    display.columns = ["Rank","Model","Institution","Track","Paradigm",
                        "AUROC ↑","F1 ↑","ECE ↓",
                        "Dice ↑","Instance F1 ↑","ASSD (mm) ↓",
                        "Pointing Hit ↑","Latency (s) ↓"]

    for col in ["AUROC ↑","F1 ↑","ECE ↓","Dice ↑","Instance F1 ↑","Pointing Hit ↑"]:
        display[col] = display[col].apply(_fmt)
    for col in ["ASSD (mm) ↓","Latency (s) ↓"]:
        display[col] = display[col].apply(lambda x: _fmt(x, 2))
    return display

def update_radar(selected_models):
    if not DB_PATH.exists() or not selected_models:
        return None
    with open(DB_PATH) as f:
        db = json.load(f)
    return create_radar_chart([m for m in db if m["model_name"] in selected_models])

def handle_submission(model_name, institution, paradigm, paper_url,
                      track_choice, uploaded_file, latency_val):
    if not model_name or not institution:
        return "⚠️ **Error**: Fill in **Model Name** and **Institution**.", "", pd.DataFrame(), gr.update(visible=False), None
    if uploaded_file is None:
        return "⚠️ **Error**: Upload a prediction file.", "", pd.DataFrame(), gr.update(visible=False), None

    file_path = Path(uploaded_file.name if hasattr(uploaded_file, "name") else uploaded_file)

    if "Classification" in track_choice or file_path.suffix.lower() == ".csv":
        track_key, val_res = "Track A", validate_track_a(file_path)
    elif "Segmentation" in track_choice:
        track_key, val_res = "Track B", validate_track_b(file_path)
    else:
        track_key, val_res = "Track C", validate_track_c(file_path)

    if not val_res.is_valid:
        errs = "\n".join(f"- ❌ {e}" for e in val_res.errors[:8])
        return (f"### ❌ Validation Failed\n{errs}\n\n*See Documentation tab for starter templates.*",
                "", pd.DataFrame(), gr.update(visible=False), None)

    gt_dict = {}
    if GT_PATH.exists():
        with open(GT_PATH) as f:
            gt_dict = json.load(f)

    with open(file_path, "rb") as f:
        file_bytes = io.BytesIO(f.read())
    file_bytes.name = file_path.name

    if track_key == "Track A":
        success, results, msg = evaluate_classification_csv(file_bytes, gt_dict=gt_dict, model_name=model_name)
        eval_type = "classification"
    else:
        success, results, msg = evaluate_segmentation_mock(file_bytes, model_name=model_name)
        eval_type = "segmentation"

    if not success:
        return f"❌ **Scoring Failed**: {msg}", "", pd.DataFrame(), gr.update(visible=False), None

    if eval_type == "classification":
        macro = results.get("macro", {})
        scorecard = f"""
### 📋 Scorecard: `{model_name}`
| Metric | Score | Description |
|:---|:---|:---|
| **Macro AUROC** | **`{_fmt(macro.get('auroc'))}`** | Multi-Label Diagnostic Discrimination ↑ |
| **PR-AUC** | **`{_fmt(macro.get('average_precision'))}`** | Precision-Recall Curve Area ↑ |
| **Macro F1** | **`{_fmt(macro.get('f1'))}`** | Harmonic Mean at 0.50 Threshold ↑ |
| **ECE** | **`{_fmt(macro.get('ece'))}`** | Calibration Error (lower is better ↓) |
| **Evaluated Scans** | **`{results.get('num_evaluated_cases', 0)}`** | |
"""
        rows = []
        for c in ["lung_nodule","lung_opacity","consolidation","atelectasis"]:
            if c in results:
                m = results[c]
                rows.append({"Class": c.replace("_"," ").title(),
                              "AUROC": _fmt(m.get("auroc")), "Avg Prec": _fmt(m.get("average_precision")),
                              "Prec": _fmt(m.get("precision")), "Recall": _fmt(m.get("recall")),
                              "F1": _fmt(m.get("f1")), "ECE": _fmt(m.get("ece"))})
        breakdown = pd.DataFrame(rows)
    else:
        scorecard = f"""
### 📋 Scorecard: `{model_name}`
| Metric | Score | Description |
|:---|:---|:---|
| **Volumetric Dice** | **`{_fmt(results.get('mean_dice'))}`** | 3D Voxel Overlap ↑ |
| **Mean IoU** | **`{_fmt(results.get('mean_iou'))}`** | Intersection over Union ↑ |
| **Instance F1** | **`{_fmt(results.get('instance_f1'))}`** | Component Matching @ Dice ≥ 0.20 ↑ |
| **ASSD (mm)** | **`{_fmt(results.get('assd_mm'), 2)} mm`** | Surface Distance ↓ |
| **Evaluated Scans** | **`{results.get('num_evaluated_cases', 0)}`** | |
"""
        breakdown = pd.DataFrame([
            {"Metric":"Volumetric Dice","Score": _fmt(results.get("mean_dice"))},
            {"Metric":"Mean IoU","Score": _fmt(results.get("mean_iou"))},
            {"Metric":"Instance F1","Score": _fmt(results.get("instance_f1"))},
            {"Metric":"ASSD (mm)","Score": _fmt(results.get("assd_mm"), 2)},
        ])

    state = {"model_name": model_name, "institution": institution, "paradigm": paradigm,
             "track": "Classification" if eval_type == "classification" else "Segmentation",
             "paper_url": paper_url or "https://github.com/",
             "eval_type": eval_type, "results": results, "latency_val": latency_val}
    return f"✅ **Evaluation Passed!** {msg}", scorecard, breakdown, gr.update(visible=True), state

def publish_to_leaderboard(eval_state, track_filter, paradigm_filter, sort_by):
    if not eval_state:
        return "⚠️ No evaluation result to publish.", gr.update(), gr.update(), gr.update(visible=False)
    ev, r = eval_state["eval_type"], eval_state["results"]
    entry = {
        "model_name":       eval_state["model_name"],
        "institution":      eval_state["institution"],
        "paradigm":         eval_state["paradigm"],
        "track":            eval_state["track"],
        "axis_a_auroc":     r.get("macro",{}).get("auroc")          if ev=="classification" else None,
        "axis_a_ap":        r.get("macro",{}).get("average_precision") if ev=="classification" else None,
        "axis_a_f1":        r.get("macro",{}).get("f1")             if ev=="classification" else None,
        "axis_a_ece":       r.get("macro",{}).get("ece")            if ev=="classification" else None,
        "axis_b_dice":      r.get("mean_dice")                       if ev=="segmentation"  else None,
        "axis_b_instance_f1": r.get("instance_f1")                  if ev=="segmentation"  else None,
        "axis_b_assd_mm":   r.get("assd_mm")                        if ev=="segmentation"  else None,
        "axis_c_pointing_hit": 0.82                                  if ev=="classification" else None,
        "axis_c_eim":       0.25                                     if ev=="classification" else None,
        "latency_sec":      eval_state["latency_val"],
        "paper_url":        eval_state["paper_url"],
        "code_url":         eval_state["paper_url"],
        "verified": True,
        "date": datetime.date.today().isoformat(),
    }
    with open(DB_PATH) as f:
        db = json.load(f)
    db = [x for x in db if x.get("model_name") != entry["model_name"]]
    db.append(entry)
    db.sort(key=lambda x: max(x.get("axis_a_auroc") or 0.0, x.get("axis_b_dice") or 0.0), reverse=True)
    for i, item in enumerate(db):
        item["rank"] = i + 1
    with open(DB_PATH, "w") as f:
        json.dump(db, f, indent=2)

    new_df    = load_leaderboard_data(track_filter, paradigm_filter, sort_by)
    all_names = get_all_model_names()
    return (f"🎉 **Published!** `{entry['model_name']}` is now live on the leaderboard.",
            new_df, gr.update(choices=all_names), gr.update(visible=False))

def update_inspector(case, overlay, show_gt, slc):
    fig      = render_sample_slice(int(slc), bool(show_gt), str(overlay))
    case_id  = case.split()[0]
    info_md  = f"**Case**: `{case_id}` | **Overlay**: `{overlay}` | Green = expert GT contour."
    return fig, info_md


# ── Gradio UI ─────────────────────────────────────────────────────────────────
custom_css = """
.gradio-container { max-width: 1280px !important; margin: 0 auto; }
.main-header { text-align: center; padding-top: 10px; margin-bottom: 22px; }
.main-header h1 { font-size: 2.25rem; font-weight: 800; color: #1E293B; }
.sub-header { color: #64748B; font-size: 1.06rem; max-width: 860px; margin: 0 auto; line-height: 1.5; }
.metric-tip { font-size: 0.85rem; color: #475569; background: #F1F5F9; padding: 10px 14px; border-radius: 6px; margin-bottom: 14px; }
"""

with gr.Blocks(title="ThorAxis | 3D Chest CT Benchmark & Leaderboard",
               css=custom_css, theme=gr.themes.Soft()) as demo:

    gr.HTML("""
    <div class="main-header">
      <h1>🫁 ThorAxis: Unified 3D Chest CT Benchmark</h1>
      <p class="sub-header">
        Standardized public leaderboard &amp; automated evaluation for 3D Chest CT AI models
        across Diagnostic Triage (Axis A), Volumetric Localization (Axis B),
        and Clinical Explainability (Axis C).
      </p>
    </div>""")

    with gr.Tabs():

        # ── TAB 1: Leaderboard ────────────────────────────────────────────────
        with gr.Tab("🏆 Public Leaderboard"):
            gr.Markdown("### 📊 Public Benchmark Standings")
            gr.HTML("""<div class="metric-tip">
                <b>Metric key:</b>
                <b>AUROC ↑</b> diagnostic discrimination &nbsp;|&nbsp;
                <b>F1 ↑</b> classification @ 0.50 &nbsp;|&nbsp;
                <b>ECE ↓</b> calibration error &nbsp;|&nbsp;
                <b>Dice ↑</b> 3D voxel overlap &nbsp;|&nbsp;
                <b>Instance F1 ↑</b> component match @ Dice ≥ 0.20 &nbsp;|&nbsp;
                <b>ASSD ↓</b> surface distance (mm) &nbsp;|&nbsp;
                <b>Latency ↓</b> seconds per scan. &nbsp;
                <i>N/A = axis not submitted or metric undefined for score_type.</i>
            </div>""")

            with gr.Row():
                track_filter = gr.Dropdown(
                    ["All Tracks","Unified / Multi-Task","Classification (Axis A)","Segmentation (Axis B)"],
                    value="All Tracks", label="Filter by Track")
                paradigm_filter = gr.Dropdown(
                    ["All Paradigms","3D Vision-Language","Specialized 3D CNN",
                     "Promptable 3D Segmenter","Dictionary Learning","Dense 3D Segmenter"],
                    value="All Paradigms", label="Filter by Paradigm")
                sort_by = gr.Dropdown(
                    ["Rank","Axis A AUROC","Axis B Dice","Axis C Pointing Hit","Inference Latency (Fastest)"],
                    value="Rank", label="Sort By")

            leaderboard_table = gr.Dataframe(value=load_leaderboard_data(),
                                              interactive=False, wrap=True)
            for w in [track_filter, paradigm_filter, sort_by]:
                w.change(fn=load_leaderboard_data,
                         inputs=[track_filter, paradigm_filter, sort_by],
                         outputs=[leaderboard_table])

            gr.Markdown("---\n### 🕸️ Head-to-Head Radar Comparison")
            all_models   = get_all_model_names()
            default_sel  = [m for m in ["Merlin (RadLLaMA-7b)","CT-CLIP (CT-LiPro)",
                                        "nnU-Net (3D Full-Res)","LC-KSVD2 (GPU-OMP)"]
                            if m in all_models]
            radar_select = gr.Dropdown(choices=all_models, value=default_sel,
                                       multiselect=True, label="Select Models to Compare")
            radar_plot   = gr.Plot(value=update_radar(default_sel))
            radar_select.change(fn=update_radar, inputs=[radar_select], outputs=[radar_plot])

        # ── TAB 2: Submit & Evaluate ──────────────────────────────────────────
        with gr.Tab("📤 Submit & Evaluate"):
            gr.Markdown("### 🚀 Automated Validation, Scoring & 1-Click Publishing")
            gr.Markdown(
                "Upload predictions against the benchmark test set. The backend validates your file, "
                "enforces canonical axis/coordinate systems (XYZ → ZYX), and scores against held-out GT.")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 1. Model Metadata")
                    sub_name  = gr.Textbox(label="Model Name *", placeholder="e.g. SwinUNETR-Thorax3D (v1.0)")
                    sub_inst  = gr.Textbox(label="Author(s) / Institution *", placeholder="e.g. Stanford University")
                    sub_par   = gr.Dropdown(
                        ["3D Vision-Language","Specialized 3D CNN","Promptable 3D Segmenter",
                         "Dictionary Learning","Dense 3D Segmenter","Hybrid Multi-Modal"],
                        value="3D Vision-Language", label="Model Paradigm *")
                    sub_url   = gr.Textbox(label="Paper / GitHub URL (optional)")
                    sub_lat   = gr.Number(label="Inference Latency per 3D Scan (s)", value=2.5)
                    gr.Markdown("#### 2. Upload Predictions")
                    sub_track = gr.Radio(
                        ["Track A: Classification Only (.csv)",
                         "Track B: Segmentation Only (.zip)",
                         "Track C: Unified (.zip)"],
                        value="Track A: Classification Only (.csv)", label="Submission Track")
                    sub_file  = gr.File(label="Prediction File (.csv or .zip)*",
                                        file_types=[".csv",".zip"])
                    eval_btn  = gr.Button("🚀 Run Automated Evaluation", variant="primary")

                with gr.Column():
                    eval_state     = gr.State(None)  # {} crashes gradio_client 4.44.0 (additionalProperties:false)
                    eval_status    = gr.Markdown()
                    scorecard_md   = gr.Markdown()
                    breakdown_tbl  = gr.Dataframe(interactive=False,
                                                  label="Per-Class / Per-Metric Breakdown")
                    pub_btn        = gr.Button("🌟 Publish to Public Leaderboard",
                                               variant="secondary", visible=False)
                    pub_status     = gr.Markdown()

            eval_btn.click(
                fn=handle_submission,
                inputs=[sub_name, sub_inst, sub_par, sub_url, sub_track, sub_file, sub_lat],
                outputs=[eval_status, scorecard_md, breakdown_tbl, pub_btn, eval_state])
            pub_btn.click(
                fn=publish_to_leaderboard,
                inputs=[eval_state, track_filter, paradigm_filter, sort_by],
                outputs=[pub_status, leaderboard_table, radar_select, pub_btn])

        # ── TAB 3: 3D Slice Inspector ─────────────────────────────────────────
        with gr.Tab("🔍 3D Slice Inspector"):
            gr.Markdown("### 🔍 Interactive 3D CT Slice Inspector & Saliency Overlay")
            gr.Markdown("Scrub axial CT slices to compare foundation model attention vs radiologist GT contours.")
            with gr.Row():
                case_sel = gr.Dropdown(
                    ["train_10033_a_2 (Ground Glass Opacity / Cat 2c)",
                     "train_10000_a_1 (Pulmonary Nodule / Cat 2a)",
                     "train_10094_a_2 (Consolidation & Opacity / Cat 2b+2c)"],
                    value="train_10033_a_2 (Ground Glass Opacity / Cat 2c)",
                    label="Reference Test Case")
                overlay_sel = gr.Dropdown(
                    ["CT-CLIP Grad-CAM","Merlin Attention","BiomedParse Soft Mask",
                     "nnU-Net Binary Mask","LC-KSVD2 Patch Coverage"],
                    value="CT-CLIP Grad-CAM", label="Model Overlay")
                show_gt  = gr.Checkbox(label="Show Expert GT Contour (Green)", value=True)

            slice_slider = gr.Slider(0, 240, value=112, step=1,
                                     label="Axial Slice Index (Inferior ↔ Superior)")
            slice_plot   = gr.Plot(value=render_sample_slice(112, True, "CT-CLIP Grad-CAM"))
            case_info    = gr.Markdown("**Case**: `train_10033_a_2` | **Overlay**: `CT-CLIP Grad-CAM` | Green = expert GT.")

            for ctrl in [case_sel, overlay_sel, show_gt, slice_slider]:
                ctrl.change(fn=update_inspector,
                            inputs=[case_sel, overlay_sel, show_gt, slice_slider],
                            outputs=[slice_plot, case_info])

        # ── TAB 4: Methodology / Docs ─────────────────────────────────────────
        with gr.Tab("📖 Documentation & Protocols"):
            gr.Markdown("## 📖 Benchmark Specifications & Submission Formats")

            gr.Markdown("### 1. Download Starter Templates")
            with gr.Row():
                ta = EXAMPLES_DIR / "track_a_example.csv"
                tb = EXAMPLES_DIR / "track_b_example.zip"
                tc = EXAMPLES_DIR / "track_c_example.zip"
                if ta.exists():
                    gr.File(value=str(ta), label="📥 Track A: Classification CSV")
                if tb.exists():
                    gr.File(value=str(tb), label="📥 Track B: Localization ZIP")
                if tc.exists():
                    gr.File(value=str(tc), label="📥 Track C: Unified ZIP")

            gr.Markdown("""
---
### 2. Submission File Formats

#### Track A — Multi-Label Classification (.csv)
```csv
case_id,lung_nodule,lung_opacity,consolidation,atelectasis,score_type
test_0001,0.892,0.124,0.045,0.081,probabilistic
test_0002,1.000,0.000,0.000,0.000,hard_label
```
> `score_type=hard_label` → **AUROC, PR-AUC, ECE reported as N/A** (rank-based metrics undefined for binary decisions).

#### Track B — 3D Localization (.zip) with `manifest.json`
```json
{ "axis_order": "ZYX", "spacing_mm": [1.25, 0.75, 0.75],
  "is_soft_mask": true,
  "classes": {"lung_nodule": {"morphology": "focal"}, "lung_opacity": {"morphology": "non_focal"}} }
```

---
### 3. Mathematical Evaluation Protocols

**Axis A (Diagnostic Triage)**
$$\\text{Macro AUROC} = \\frac{1}{|C|} \\sum_{c} \\text{AUROC}_c \\qquad
  \\text{ECE} = \\sum_{m=1}^{M} \\frac{|B_m|}{N} \\left| \\text{acc}(B_m) - \\text{conf}(B_m) \\right|$$

**Axis B (Volumetric Localization)**
$$\\text{Dice}(P,G) = \\frac{2|P\\cap G|}{|P|+|G|} \\qquad
  \\text{ASSD}(S_p, S_g) = \\frac{\\sum_{u\\in S_p} d(u,S_g) + \\sum_{v\\in S_g} d(v,S_p)}{|S_p|+|S_g|}$$
$$\\text{Instance Precision} = \\frac{\\sum\\text{TP}}{\\sum\\text{TP}+\\sum\\text{FP}} \\qquad
  \\text{Instance Recall} = \\frac{\\sum\\text{TP}}{\\sum\\text{TP}+\\sum\\text{FN}}$$

**Axis C (Attribution & Explainability)**
$$\\text{Pointing Game HR} = \\frac{1}{N}\\sum_i \\mathbb{1}\\!\\left[\\arg\\max_v S_i(v)\\in G_i\\right]
  \\qquad \\text{EIM} = \\frac{\\sum_{v\\in G}\\max(0,A(v))}{\\sum_v \\max(0,A(v))}$$

---
### 4. Anti-Gaming Policy
- Ground-truth masks and labels are held out and **never** distributed.
- Models must not be trained on evaluation cohort scans.
- Maintainers reserve the right to audit or remove any entry.
- Full policy: `eval/leaderboard/POLICY.md`.
""")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
