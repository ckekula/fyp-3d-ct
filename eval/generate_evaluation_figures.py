#!/usr/bin/env python3
"""
Generate publication-ready visualization figures for BiomedParse localization evaluation.
Creates PNG figures suitable for LaTeX inclusion.

Usage:
    python generate_evaluation_figures.py
    
Output:
    Saves figures to documents/Progress Report/Images/
"""

import json
import csv
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Setup
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
METRICS_FILE = REPO_ROOT / 'eval' / 'outputs' / 'localization' / 'biomed_parse_localization_metrics_progress.json'
PERCASE_FILE = REPO_ROOT / 'eval' / 'outputs' / 'localization' / 'biomed_parse_localization_per_case_progress.csv'
FIGURES_DIR = REPO_ROOT / 'documents' / 'Progress Report' / 'Images'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load data
if not METRICS_FILE.exists():
    raise FileNotFoundError(f"Metrics file not found: {METRICS_FILE}")
if not PERCASE_FILE.exists():
    raise FileNotFoundError(f"Per-case CSV file not found: {PERCASE_FILE}")

with open(METRICS_FILE) as f:
    metrics = json.load(f)

class_dice = defaultdict(list)
class_iou = defaultdict(list)
with open(PERCASE_FILE) as f:
    reader = csv.DictReader(f)
    for row in reader:
        class_name = row['class_name']
        class_dice[class_name].append(float(row['dice']))
        class_iou[class_name].append(float(row['iou']))

# Define color scheme
COLORS = {
    'lung_nodule': '#FF6B6B',
    'lung_opacity': '#4ECDC4',
    'consolidation': '#45B7D1',
    'atelectasis': '#FFA07A'
}
FOCAL_COLOR = '#FF6B6B'
NONFOCAL_COLOR = '#95A5A6'

# ============================================================================
# Figure 1: Dice Score Distribution by Class (4-panel histogram)
# ============================================================================
print("Generating Figure 1: Dice Score Distributions...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Dice Score Distribution by Class (BiomedParse Localization)', 
             fontsize=16, fontweight='bold', y=0.995)

classes = sorted(class_dice.keys())
axes_flat = axes.flatten()

for idx, class_name in enumerate(classes):
    ax = axes_flat[idx]
    dice_values = class_dice[class_name]
    
    # Histogram
    counts, bins, patches = ax.hist(dice_values, bins=50, color=COLORS[class_name], 
                                     alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Statistics
    mean_dice = np.mean(dice_values)
    median_dice = np.median(dice_values)
    std_dice = np.std(dice_values)
    zero_pct = 100 * sum(1 for v in dice_values if v < 0.001) / len(dice_values)
    
    ax.axvline(mean_dice, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_dice:.4f}')
    ax.axvline(median_dice, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_dice:.4f}')
    
    ax.set_title(f'{class_name.replace("_", " ").title()} (n={len(dice_values)})\nZeros: {zero_pct:.1f}%', 
                 fontweight='bold')
    ax.set_xlabel('Dice Score', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'Fig1_Dice_Distributions.png', dpi=300, bbox_inches='tight')
print(f"[OK] Saved: {FIGURES_DIR / 'Fig1_Dice_Distributions.png'}")
plt.close()

# ============================================================================
# Figure 2: Mean Metrics Comparison Bar Chart
# ============================================================================
print("Generating Figure 2: Mean Metrics Comparison...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('BiomedParse Localization: Per-Class Performance Metrics', 
             fontsize=16, fontweight='bold')

by_class = metrics['by_class']

# Extract metrics for each class
class_names_clean = []
dice_scores = []
iou_scores = []
dice_gt_005 = []

for class_key in sorted(by_class.keys()):
    if class_key != 'all':
        class_data = by_class[class_key]
        summary = class_data.get('summary', {})
        class_names_clean.append(class_key.replace('_', ' ').title())
        dice_scores.append(summary.get('mean_dice', 0))
        iou_scores.append(summary.get('mean_iou', 0))
        dice_gt_005.append(summary.get('frac_dice_gt_0.05', 0))

x = np.arange(len(class_names_clean))
width = 0.25

# Dice scores
ax = axes[0]
bars1 = ax.bar(x - width, dice_scores, width, label='Mean Dice', 
               color=[COLORS[cls.lower().replace(' ', '_')] for cls in class_names_clean],
               alpha=0.8, edgecolor='black')
ax.set_ylabel('Dice Score', fontweight='bold', fontsize=12)
ax.set_title('Mean Dice Score', fontweight='bold', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(class_names_clean, rotation=45, ha='right')
ax.grid(alpha=0.3, axis='y')
ax.set_ylim(0, max(dice_scores) * 1.2)
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# IoU scores
ax = axes[1]
bars2 = ax.bar(x - width, iou_scores, width, label='Mean IoU',
               color=[COLORS[cls.lower().replace(' ', '_')] for cls in class_names_clean],
               alpha=0.8, edgecolor='black')
ax.set_ylabel('IoU Score', fontweight='bold', fontsize=12)
ax.set_title('Mean IoU Score', fontweight='bold', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(class_names_clean, rotation=45, ha='right')
ax.grid(alpha=0.3, axis='y')
ax.set_ylim(0, max(iou_scores) * 1.2)
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Fraction of findings with Dice > 0.05
ax = axes[2]
bars3 = ax.bar(x - width, dice_gt_005, width, label='Dice > 0.05',
               color=[COLORS[cls.lower().replace(' ', '_')] for cls in class_names_clean],
               alpha=0.8, edgecolor='black')
ax.set_ylabel('Fraction of Findings', fontweight='bold', fontsize=12)
ax.set_title('Findings with Dice > 0.05', fontweight='bold', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(class_names_clean, rotation=45, ha='right')
ax.grid(alpha=0.3, axis='y')
ax.set_ylim(0, max(dice_gt_005) * 1.2)
for bar in bars3:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'Fig2_Metrics_Comparison.png', dpi=300, bbox_inches='tight')
print(f"[OK] Saved: {FIGURES_DIR / 'Fig2_Metrics_Comparison.png'}")
plt.close()

# ============================================================================
# Figure 3: Performance Gap: Focal vs Non-Focal
# ============================================================================
print("Generating Figure 3: Focal vs Non-Focal Analysis...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Critical Finding: Model Performs Dramatically Better on Focal Lesions', 
             fontsize=14, fontweight='bold', color='darkred')

by_morph = metrics['by_morphology']
focal_data = by_morph['focal']['summary']
nonfocal_data = by_morph['non_focal']['summary']

metrics_names = ['Dice', 'IoU', 'Dice>0.05', 'Global Hit Rate (Dice>=0.10)']
focal_values = [
    focal_data.get('mean_dice', 0),
    focal_data.get('mean_iou', 0),
    focal_data.get('frac_dice_gt_0.05', 0),
    focal_data.get('global_hit_rate', 0)
]
nonfocal_values = [
    nonfocal_data.get('mean_dice', 0),
    nonfocal_data.get('mean_iou', 0),
    nonfocal_data.get('frac_dice_gt_0.05', 0),
    nonfocal_data.get('global_hit_rate', 0)
]

# Side-by-side bars
ax = axes[0]
x = np.arange(len(metrics_names))
width = 0.35
bars1 = ax.bar(x - width/2, focal_values, width, label='Focal (Nodules)', 
               color=FOCAL_COLOR, alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, nonfocal_values, width, label='Non-Focal (Opacity, etc.)', 
               color=NONFOCAL_COLOR, alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_ylabel('Score', fontweight='bold', fontsize=12)
ax.set_title('Morphology-Based Performance Comparison', fontweight='bold', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend(fontsize=11, loc='upper right')
ax.grid(alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)

# Performance ratio
ax = axes[1]
ratios = [focal_values[i] / (nonfocal_values[i] + 1e-6) for i in range(len(metrics_names))]
colors_ratio = ['darkred' if r > 1 else 'darkgreen' for r in ratios]
bars = ax.bar(metrics_names, ratios, color=colors_ratio, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.axhline(y=1, color='black', linestyle='--', linewidth=2, label='Equal Performance (ratio=1)')
ax.set_ylabel('Performance Ratio (Focal / Non-Focal)', fontweight='bold', fontsize=12)
ax.set_title('Performance Multiplier Gap', fontweight='bold', fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis='y')

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}×', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'Fig3_Focal_vs_NonFocal.png', dpi=300, bbox_inches='tight')
print(f"[OK] Saved: {FIGURES_DIR / 'Fig3_Focal_vs_NonFocal.png'}")
plt.close()

# ============================================================================
# Figure 4: Cumulative Distribution Function (CDF)
# ============================================================================
print("Generating Figure 4: Cumulative Distribution Functions...")
fig, ax = plt.subplots(figsize=(12, 6))

for class_name in sorted(class_dice.keys()):
    values = sorted(class_dice[class_name])
    cdf = np.arange(1, len(values) + 1) / len(values)
    ax.plot(values, cdf, linewidth=2.5, label=class_name.replace('_', ' ').title(),
            color=COLORS[class_name], marker='o', markersize=3, markevery=20)

ax.set_xlabel('Dice Score', fontweight='bold', fontsize=12)
ax.set_ylabel('Cumulative Probability', fontweight='bold', fontsize=12)
ax.set_title('CDF of Dice Scores: Performance Distribution by Class', 
             fontweight='bold', fontsize=14)
ax.legend(fontsize=11, loc='lower right')
ax.grid(alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Add reference lines
ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, label='Moderate (0.5)')
ax.axvline(x=0.7, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, label='Good (0.7)')
ax.legend(fontsize=11, loc='lower right')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'Fig4_CDF_Dice.png', dpi=300, bbox_inches='tight')
print(f"[OK] Saved: {FIGURES_DIR / 'Fig4_CDF_Dice.png'}")
plt.close()

# ============================================================================
# Figure 5: Confusion Pattern Analysis (Zero Case Breakdown)
# ============================================================================
print("Generating Figure 5: Confusion Pattern Analysis...")
fig, ax = plt.subplots(figsize=(14, 7))

# Categorize dice scores
categories = {
    'Zero Dice (0%)': [],
    'Near Zero (0.1-1%)': [],
    'Very Low (1-10%)': [],
    'Low (10-30%)': [],
    'Medium (30-60%)': [],
    'High (>60%)': []
}

category_colors = ['#d62728', '#ff7f0e', '#ffcc00', '#2ca02c', '#1f77b4', '#9467bd']

for class_name in sorted(class_dice.keys()):
    dice_values = class_dice[class_name]
    counts = [
        sum(1 for v in dice_values if v < 0.001),
        sum(1 for v in dice_values if 0.001 <= v < 0.01),
        sum(1 for v in dice_values if 0.01 <= v < 0.1),
        sum(1 for v in dice_values if 0.1 <= v < 0.3),
        sum(1 for v in dice_values if 0.3 <= v < 0.6),
        sum(1 for v in dice_values if v >= 0.6),
    ]
    
    # Stacked bar
    bottom = 0
    for idx, (cat, count) in enumerate(zip(categories.keys(), counts)):
        pct = 100 * count / len(dice_values)
        ax.bar(class_name.replace('_', ' ').title(), pct, bottom=bottom, 
               label=cat if class_name == 'lung_nodule' else '',
               color=category_colors[idx], edgecolor='black', linewidth=0.8)
        
        # Add percentage labels for major segments
        if pct > 6:
            ax.text(list(class_dice.keys()).index(class_name), bottom + pct / 2, 
                   f'{pct:.0f}%', ha='center', va='center', fontweight='bold', fontsize=11)
        bottom += pct

ax.set_ylabel('Percentage of Cases (%)', fontweight='bold', fontsize=14)
ax.set_title('Dice Score Distribution Breakdown: Where Does the Model Fail?', 
             fontweight='bold', fontsize=18)
ax.set_ylim(0, 100)
ax.grid(alpha=0.3, axis='y')
ax.set_xticklabels([name.replace('_', ' ').title() for name in sorted(class_dice.keys())], fontsize=13)
ax.tick_params(axis='y', labelsize=13)

# Create legend with unique entries and place outside the figure
handles = [mpatches.Patch(facecolor=color, edgecolor='black', label=cat) 
           for cat, color in zip(categories.keys(), category_colors)]
ax.legend(handles=handles, fontsize=12, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0.)

plt.tight_layout(rect=[0, 0, 0.88, 1])
plt.savefig(FIGURES_DIR / 'Fig5_Confusion_Patterns.png', dpi=300, bbox_inches='tight')
print(f"[OK] Saved: {FIGURES_DIR / 'Fig5_Confusion_Patterns.png'}")
plt.close()

# ============================================================================
# Generate LaTeX code for easy inclusion
# ============================================================================
# Numbers below are pulled live from `metrics` (loaded from METRICS_FILE at
# the top of this script) rather than hardcoded, so this section can't go
# stale relative to the actual evaluation run the way a hand-typed table can.
print("\nGenerating LaTeX inclusion code...")

overall_summary = metrics['overall']['summary']
n_total_cases = overall_summary['num_cases']

def _class_row(cls_key, summary):
    return r"{} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {} \\".format(
        cls_key.replace('_', ' ').title(),
        summary['mean_dice'],
        summary['mean_iou'],
        summary['frac_dice_gt_0.05'],
        summary['global_hit_rate'],
        summary['num_cases'],
    )

class_rows_latex = "\n".join(
    _class_row(cls, metrics['by_class'][cls]['summary'])
    for cls in sorted(metrics['by_class'].keys())
    if cls != 'all'
)

morphology_rows_latex = "\n".join(
    _class_row(morph, metrics['by_morphology'][morph]['summary'])
    for morph in sorted(metrics['by_morphology'].keys())
    if morph != 'unknown'
)

# Per-class zero-Dice fraction / max / median, computed from the same
# per-case Dice values used elsewhere in this script (class_dice), not
# hand-typed -- so these narrative numbers can't drift from the actual data.
class_stats = {}
for cls, values in class_dice.items():
    arr = np.asarray(values, dtype=float)
    class_stats[cls] = {
        "zero_frac": float(np.mean(arr < 0.001)) if len(arr) else float("nan"),
        "max": float(np.max(arr)) if len(arr) else float("nan"),
        "median": float(np.median(arr)) if len(arr) else float("nan"),
    }

focal_dice = metrics['by_morphology'].get('focal', {}).get('summary', {}).get('mean_dice', float('nan'))
nonfocal_dice = metrics['by_morphology'].get('non_focal', {}).get('summary', {}).get('mean_dice', float('nan'))
focal_multiplier = focal_dice / nonfocal_dice if nonfocal_dice else float('nan')

nodule_stats = class_stats.get('lung_nodule', {})
opacity_stats = class_stats.get('lung_opacity', {})
consolidation_stats = class_stats.get('consolidation', {})
atelectasis_stats = class_stats.get('atelectasis', {})
nodule_hit_rate = metrics['by_class'].get('lung_nodule', {}).get('summary', {}).get('frac_dice_gt_0.05', float('nan'))

latex_code = rf"""
% ============================================================================
% EVALUATION RESULTS SECTION - Add to Chapter 4
% ============================================================================

\section{{Grounded Localization Evaluation Results}}

The BiomedParse model was evaluated on volumetric CT localization using four lesion classes from
the RexGrounding-CT dataset. The following subsections present the quantitative analysis
across {n_total_cases} test cases.

\subsection{{Overall Performance Summary}}

The model achieves limited localization accuracy across the dataset:
\begin{{itemize}}
    \item Mean Dice Score: {overall_summary['mean_dice']:.4f}
    \item Mean IoU: {overall_summary['mean_iou']:.4f}
    \item Dice > 0.05: {overall_summary['frac_dice_gt_0.05']:.4f} ({overall_summary['frac_dice_gt_0.05']*100:.1f}\% of findings exceed a Dice of 0.05)
    \item Global Hit Rate (Dice $\geq$ 0.10, ReXGroundingCT's official definition): {overall_summary['global_hit_rate']:.4f} ({overall_summary['global_hit_rate']*100:.1f}\% of findings)
\end{{itemize}}

\subsection{{Per-Class Performance}}

\begin{{table}}[h]
\centering
\caption{{Localization Performance by Lesion Class}}
\begin{{tabular}}{{lccccr}}
\hline
\textbf{{Class}} & \textbf{{Dice}} & \textbf{{IoU}} & \textbf{{Dice>0.05}} & \textbf{{Hit Rate}} & \textbf{{Samples}} \\
\hline
{class_rows_latex}
\hline
\end{{tabular}}
\label{{tab:perclass-localization}}
\end{{table}}

\begin{{figure}}[h]
\centering
\includegraphics[width=0.95\linewidth]{{Images/Fig2_Metrics_Comparison.png}}
\caption{{Per-class localization performance across three key metrics.
The model shows substantially better performance on lung nodules (focal lesions)
compared to other lesion types.}}
\label{{fig:metrics-comparison}}
\end{{figure}}

\subsection{{Distribution Analysis}}

The Dice score distributions reveal a stark pattern: while lung nodules show some successful detections
(max: {nodule_stats.get('max', float('nan')):.2f}, median: {nodule_stats.get('median', float('nan')):.2f}), non-focal lesions remain largely undetected.

\begin{{figure}}[h]
\centering
\includegraphics[width=0.95\linewidth]{{Images/Fig1_Dice_Distributions.png}}
\caption{{Histogram of Dice scores for each lesion class. Red and blue dashed lines indicate
mean and median respectively. Zero-Dice percentages shown in subtitle. Note the dramatic difference
between lung nodules (dispersed distribution) and other classes (peaked at zero).}}
\label{{fig:dice-distributions}}
\end{{figure}}

\subsection{{Critical Finding: Focal vs. Non-Focal Lesion Bias}}

The evaluation reveals a fundamental model bias: focal lesions (nodules) are detected at dramatically
higher rates than non-focal lesions (diffuse patterns).

\begin{{table}}[h]
\centering
\caption{{Morphology-Based Performance Analysis}}
\begin{{tabular}}{{lccccr}}
\hline
\textbf{{Morphology}} & \textbf{{Dice}} & \textbf{{IoU}} & \textbf{{Dice>0.05}} & \textbf{{Hit Rate}} & \textbf{{Samples}} \\
\hline
{morphology_rows_latex}
\hline
\end{{tabular}}
\label{{tab:morphology-localization}}
\end{{table}}

\begin{{figure}}[h]
\centering
\includegraphics[width=0.95\linewidth]{{Images/Fig3_Focal_vs_NonFocal.png}}
\caption{{Critical performance gap between focal (lung nodules) and non-focal (diffuse) lesions.
Left panel shows direct metric comparison. Right panel visualizes the performance multiplier gap:
the model achieves {focal_multiplier:.1f}\texttimes{{}} higher Dice on nodules versus non-focal lesions.}}
\label{{fig:focal-nonfocal-gap}}
\end{{figure}}

\subsection{{Score Distribution and Failure Modes}}

A cumulative distribution analysis shows how performance varies across cases.

\begin{{figure}}[h]
\centering
\includegraphics[width=0.95\linewidth]{{Images/Fig4_CDF_Dice.png}}
\caption{{Cumulative Distribution Function of Dice scores. Shows the proportion of cases
achieving each Dice threshold. Lung nodules reach higher thresholds, while other classes
remain concentrated near zero.}}
\label{{fig:cdf-dice}}
\end{{figure}}

The failure-mode distribution reveals the severity of non-focal lesion detection:

\begin{{figure}}[h]
\centering
\includegraphics[width=0.95\linewidth]{{Images/Fig5_Confusion_Patterns.png}}
\caption{{Stacked bar chart showing the fraction of cases in each performance category.
Consolidation and atelectasis show ${{>}}{max(consolidation_stats.get('zero_frac', 0), atelectasis_stats.get('zero_frac', 0))*100:.0f}\%$ zero-Dice cases (complete failure to localize).
Lung opacity shows ${opacity_stats.get('zero_frac', float('nan'))*100:.0f}\%$ zero-Dice. Only lung nodules show meaningful detection diversity.}}
\label{{fig:confusion-patterns}}
\end{{figure}}

\subsection{{Conclusions from Localization Evaluation}}

The evaluation demonstrates that BiomedParse:
\begin{{enumerate}}
    \item Effectively localizes focal lesions (lung nodules) with moderate success ({nodule_hit_rate*100:.1f}\% of findings exceed Dice 0.05)
    \item Fails dramatically on non-focal (diffuse) lesions:
    \begin{{itemize}}
        \item Atelectasis: {atelectasis_stats.get('zero_frac', float('nan'))*100:.0f}\% zero-Dice (essentially not detected)
        \item Consolidation: {consolidation_stats.get('zero_frac', float('nan'))*100:.0f}\% zero-Dice (essentially not detected)
        \item Lung Opacity: {opacity_stats.get('zero_frac', float('nan'))*100:.0f}\% zero-Dice (mostly not detected)
    \end{{itemize}}
    \item Shows architectural bias toward focal patterns, suggesting the model's feature representations
    may not generalize to diffuse pathology
\end{{enumerate}}

This focal-bias is a critical finding for understanding model limitations in clinical deployment,
where many important lesions manifest as diffuse patterns rather than discrete nodules.
"""

with open(FIGURES_DIR / 'LaTeX_Evaluation_Section.txt', 'w') as f:
    f.write(latex_code)

print(f"[OK] Saved LaTeX template: {FIGURES_DIR / 'LaTeX_Evaluation_Section.txt'}")

# ============================================================================
# Summary Table
# ============================================================================
print("\n" + "="*70)
print("GENERATION COMPLETE!")
print("="*70)
print("\nGenerated Files:")
print("  1. Fig1_Dice_Distributions.png         - Per-class histograms")
print("  2. Fig2_Metrics_Comparison.png         - Bar chart of all metrics")
print("  3. Fig3_Focal_vs_NonFocal.png          - Critical focal/non-focal gap")
print("  4. Fig4_CDF_Dice.png                   - Cumulative distributions")
print("  5. Fig5_Confusion_Patterns.png         - Failure mode breakdown")
print("  6. LaTeX_Evaluation_Section.txt        - Ready-to-use LaTeX code")

print(f"\nAll files saved to: {FIGURES_DIR.absolute()}")
print("\nNext steps for report integration:")
print("  1. Copy LaTeX_Evaluation_Section.txt into Chapters/ch_4.tex")
print("  2. All PNG figures are already in the Images/ folder (auto-linked in LaTeX)")
print("  3. Compile Report.tex to generate PDF with figures")
