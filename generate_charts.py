"""
Generate Thesis Visualization Charts
Reads experiment results from data/experiments/ and produces
publication-quality figures for the thesis.

Usage:
    python generate_charts.py

Output:
    figures/*.png (7 charts, 300 DPI)
"""

import sys
import os
import json
from pathlib import Path
from glob import glob

# Fix Windows encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EXPERIMENTS_DIR = Path("data/experiments")
FIGURES_DIR = Path("figures")
DPI = 300
FIG_WIDTH = 10
FIG_HEIGHT = 6

# Thesis-quality style
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 100,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})
sns.set_theme(style="whitegrid", font_scale=1.1)

# Color palette
CLEAN_COLOR = '#2196F3'       # Blue
POISONED_COLOR = '#F44336'    # Red
STRATEGY_COLORS = {
    'injection': '#F44336',
    'contradiction': '#FF9800',
    'subtle': '#9C27B0',
    'entity_swap': '#4CAF50',
}
ABLATION_COLORS = ['#2196F3', '#FF9800', '#4CAF50', '#F44336', '#9C27B0']


# ---------------------------------------------------------------------------
# Data Loading Helpers
# ---------------------------------------------------------------------------

def find_latest_experiment(name_pattern: str) -> Path:
    """Find the latest experiment file matching a name pattern."""
    pattern = str(EXPERIMENTS_DIR / f"experiment_{name_pattern}_*.json")
    files = sorted(glob(pattern))
    if not files:
        return None
    return Path(files[-1])


def load_experiment(path: Path) -> dict:
    """Load an experiment JSON file."""
    if path is None or not path.exists():
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_sample_scores(data: dict) -> dict:
    """Extract per-sample scores grouped by clean/poisoned."""
    samples = data.get('sample_results', [])
    clean = [s for s in samples if not s.get('is_poisoned_set', False)]
    poisoned = [s for s in samples if s.get('is_poisoned_set', False)]
    return {'clean': clean, 'poisoned': poisoned}


# ---------------------------------------------------------------------------
# Figure 1: Per-Strategy Detection Rate
# ---------------------------------------------------------------------------

def fig1_strategy_detection_rate():
    """Bar chart of detection rates for each poisoning strategy."""
    strategies = ['injection', 'contradiction', 'subtle', 'entity_swap']
    labels = ['Injection', 'Contradiction', 'Subtle', 'Entity Swap']
    detection_rates = []
    colors = []

    for strategy in strategies:
        path = find_latest_experiment(f"truthfulqa_{strategy}")
        data = load_experiment(path)
        if data:
            metrics = data['metrics']
            # Per-strategy metrics may use 'unknown' key for single-strategy runs
            pstrat = metrics.get('per_strategy_metrics', {})
            if 'unknown' in pstrat:
                rate = pstrat['unknown']['detection_rate']
            elif strategy in pstrat:
                rate = pstrat[strategy]['detection_rate']
            else:
                rate = metrics.get('recall', 0)
            detection_rates.append(rate * 100)
        else:
            detection_rates.append(0)
        colors.append(STRATEGY_COLORS[strategy])

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    bars = ax.bar(labels, detection_rates, color=colors, edgecolor='white', linewidth=1.5)

    # Add value labels on bars
    for bar, val in zip(bars, detection_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=13)

    ax.set_ylabel('Detection Rate (%)')
    ax.set_title('Poison Detection Rate by Attack Strategy (TruthfulQA)')
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())

    save_figure(fig, 'fig1_strategy_detection_rate.png')


# ---------------------------------------------------------------------------
# Figure 2: Trust Score Distribution - Clean vs Poisoned
# ---------------------------------------------------------------------------

def fig2_trust_score_distribution():
    """Box plot of trust scores: clean vs poisoned."""
    path = find_latest_experiment("truthfulqa_poison_detection")
    data = load_experiment(path)
    if not data:
        print("  [SKIP] No TruthfulQA main experiment found")
        return

    groups = extract_sample_scores(data)
    clean_scores = [s['trust_score'] for s in groups['clean']]
    poisoned_scores = [s['trust_score'] for s in groups['poisoned']]

    fig, ax = plt.subplots(figsize=(8, FIG_HEIGHT))

    bp = ax.boxplot(
        [clean_scores, poisoned_scores],
        tick_labels=['Clean Documents', 'Poisoned Documents'],
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
    )

    bp['boxes'][0].set_facecolor(CLEAN_COLOR)
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor(POISONED_COLOR)
    bp['boxes'][1].set_alpha(0.7)

    # Add mean markers
    clean_mean = np.mean(clean_scores)
    poisoned_mean = np.mean(poisoned_scores)
    ax.scatter([1], [clean_mean], color='white', edgecolor='black', s=100, zorder=5, marker='D', label=f'Mean: {clean_mean:.3f}')
    ax.scatter([2], [poisoned_mean], color='white', edgecolor='black', s=100, zorder=5, marker='D', label=f'Mean: {poisoned_mean:.3f}')

    # Threshold line
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, label='Trust Threshold (0.5)')

    # Separation annotation
    mid_y = (clean_mean + poisoned_mean) / 2
    ax.annotate('', xy=(2.6, clean_mean), xytext=(2.6, poisoned_mean),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text(2.75, mid_y, f'Separation\n{clean_mean - poisoned_mean:.3f}',
            ha='left', va='center', fontsize=10, fontweight='bold')

    ax.set_ylabel('Trust Score')
    ax.set_title('Trust Score Distribution: Clean vs Poisoned (TruthfulQA)')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower left', fontsize=10)

    save_figure(fig, 'fig2_trust_score_distribution.png')


# ---------------------------------------------------------------------------
# Figure 3: Confusion Matrices (TruthfulQA and FEVER)
# ---------------------------------------------------------------------------

def fig3_confusion_matrices():
    """Side-by-side confusion matrix heatmaps."""
    tqa_path = find_latest_experiment("truthfulqa_poison_detection")
    fever_path = find_latest_experiment("fever_poison_detection")
    tqa = load_experiment(tqa_path)
    fever = load_experiment(fever_path)

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH + 2, FIG_HEIGHT))

    datasets = [
        ('TruthfulQA', tqa, axes[0]),
        ('FEVER', fever, axes[1]),
    ]

    for name, data, ax in datasets:
        if data is None:
            ax.text(0.5, 0.5, f'{name}\nNo data', ha='center', va='center', transform=ax.transAxes)
            continue

        m = data['metrics']
        cm = np.array([
            [m['true_negatives'], m['false_positives']],
            [m['false_negatives'], m['true_positives']]
        ])

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Trusted', 'Untrusted'],
                    yticklabels=['Clean', 'Poisoned'],
                    annot_kws={'size': 16, 'fontweight': 'bold'},
                    linewidths=2, linecolor='white',
                    cbar=False)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'{name}\n(Acc: {m["accuracy"]:.0%}, F1: {m["f1_score"]:.2f})')

    plt.suptitle('Confusion Matrices: Poison Detection', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, 'fig3_confusion_matrices.png')


# ---------------------------------------------------------------------------
# Figure 4: TruthfulQA vs FEVER Comparison
# ---------------------------------------------------------------------------

def fig4_dataset_comparison():
    """Grouped bar chart comparing metrics across datasets."""
    tqa_path = find_latest_experiment("truthfulqa_poison_detection")
    fever_path = find_latest_experiment("fever_poison_detection")
    tqa = load_experiment(tqa_path)
    fever = load_experiment(fever_path)

    if not tqa or not fever:
        print("  [SKIP] Need both TruthfulQA and FEVER experiments")
        return

    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metric_keys = ['accuracy', 'precision', 'recall', 'f1_score']

    tqa_vals = [tqa['metrics'][k] * 100 for k in metric_keys]
    fever_vals = [fever['metrics'][k] * 100 for k in metric_keys]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    bars1 = ax.bar(x - width / 2, tqa_vals, width, label='TruthfulQA', color=CLEAN_COLOR, edgecolor='white')
    bars2 = ax.bar(x + width / 2, fever_vals, width, label='FEVER', color=POISONED_COLOR, edgecolor='white')

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                    f'{h:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Score (%)')
    ax.set_title('Detection Performance: TruthfulQA vs FEVER')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend()

    save_figure(fig, 'fig4_dataset_comparison.png')


# ---------------------------------------------------------------------------
# Figure 5: Ablation Study
# ---------------------------------------------------------------------------

def fig5_ablation_study():
    """Grouped bar chart of ablation study results."""
    configs = [
        ('default', 'Default\n(0.4/0.35/0.25)'),
        ('factuality_heavy', 'Factuality\n(0.6/0.2/0.2)'),
        ('consistency_heavy', 'Consistency\n(0.2/0.6/0.2)'),
        ('poison_heavy', 'Poison\n(0.2/0.2/0.6)'),
        ('equal', 'Equal\n(0.33/0.34/0.33)'),
    ]

    accuracies = []
    recalls = []
    separations = []

    for name, _ in configs:
        path = find_latest_experiment(f"ablation_{name}")
        data = load_experiment(path)
        if data:
            m = data['metrics']
            accuracies.append(m['accuracy'] * 100)
            recalls.append(m['recall'] * 100)
            separations.append(m['trust_score_separation'] * 100)
        else:
            accuracies.append(0)
            recalls.append(0)
            separations.append(0)

    labels = [lbl for _, lbl in configs]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(FIG_WIDTH + 1, FIG_HEIGHT))
    bars1 = ax.bar(x - width, accuracies, width, label='Accuracy', color='#2196F3')
    bars2 = ax.bar(x, recalls, width, label='Recall', color='#F44336')
    bars3 = ax.bar(x + width, separations, width, label='Trust Separation (x100)', color='#4CAF50')

    # Value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                        f'{h:.1f}', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Value (%)')
    ax.set_title('Ablation Study: Effect of Trust Index Weights')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right')

    save_figure(fig, 'fig5_ablation_study.png')


# ---------------------------------------------------------------------------
# Figure 6: Component Score Breakdown
# ---------------------------------------------------------------------------

def fig6_component_breakdown():
    """Violin/strip plots for factuality, consistency, poison probability."""
    path = find_latest_experiment("truthfulqa_poison_detection")
    data = load_experiment(path)
    if not data:
        print("  [SKIP] No TruthfulQA main experiment found")
        return

    samples = data.get('sample_results', [])

    # Build data for plotting
    import pandas as pd

    rows = []
    for s in samples:
        group = 'Poisoned' if s.get('is_poisoned_set', False) else 'Clean'
        rows.append({
            'Group': group,
            'Factuality': s.get('factuality_score', 0),
            'Consistency': s.get('consistency_score', 0),
            'Poison Probability': s.get('poison_probability', 0),
        })

    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH + 4, FIG_HEIGHT))
    components = ['Factuality', 'Consistency', 'Poison Probability']
    palette = {'Clean': CLEAN_COLOR, 'Poisoned': POISONED_COLOR}

    for ax, comp in zip(axes, components):
        sns.boxplot(data=df, x='Group', y=comp, hue='Group', ax=ax, palette=palette,
                    width=0.5, linewidth=1.5, legend=False)
        sns.stripplot(data=df, x='Group', y=comp, hue='Group', ax=ax, palette=palette,
                      size=4, alpha=0.4, jitter=True, legend=False)
        ax.set_title(comp)
        ax.set_xlabel('')
        ax.set_ylim(-0.05, 1.05)

    plt.suptitle('Evaluation Component Scores: Clean vs Poisoned', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'fig6_component_breakdown.png')


# ---------------------------------------------------------------------------
# Figure 7: Per-Strategy Trust Score Comparison
# ---------------------------------------------------------------------------

def fig7_strategy_trust_scores():
    """Bar chart of average trust scores per strategy (clean vs poisoned)."""
    strategies = ['injection', 'contradiction', 'subtle', 'entity_swap']
    labels = ['Injection', 'Contradiction', 'Subtle', 'Entity Swap']

    clean_means = []
    poisoned_means = []

    for strategy in strategies:
        path = find_latest_experiment(f"truthfulqa_{strategy}")
        data = load_experiment(path)
        if data:
            m = data['metrics']
            clean_means.append(m['avg_trust_clean'])
            poisoned_means.append(m['avg_trust_poisoned'])
        else:
            clean_means.append(0)
            poisoned_means.append(0)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    bars1 = ax.bar(x - width / 2, clean_means, width, label='Clean', color=CLEAN_COLOR, edgecolor='white')
    bars2 = ax.bar(x + width / 2, poisoned_means, width, label='Poisoned', color=POISONED_COLOR, edgecolor='white')

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=10)

    # Trust threshold line
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Trust Threshold')

    ax.set_ylabel('Average Trust Score')
    ax.set_title('Trust Score by Attack Strategy: Clean vs Poisoned')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.legend()

    save_figure(fig, 'fig7_strategy_trust_scores.png')


# ---------------------------------------------------------------------------
# Save Helper
# ---------------------------------------------------------------------------

def save_figure(fig, filename):
    """Save a figure to the figures directory."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [SAVED] {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("THESIS CHART GENERATION")
    print("=" * 60)
    print(f"Reading experiments from: {EXPERIMENTS_DIR}")
    print(f"Saving figures to: {FIGURES_DIR}")
    print()

    # List available experiments
    all_experiments = sorted(glob(str(EXPERIMENTS_DIR / "*.json")))
    print(f"Found {len(all_experiments)} experiment files\n")

    print("[1/7] Per-Strategy Detection Rate...")
    fig1_strategy_detection_rate()

    print("[2/7] Trust Score Distribution...")
    fig2_trust_score_distribution()

    print("[3/7] Confusion Matrices...")
    fig3_confusion_matrices()

    print("[4/7] Dataset Comparison (TruthfulQA vs FEVER)...")
    fig4_dataset_comparison()

    print("[5/7] Ablation Study...")
    fig5_ablation_study()

    print("[6/7] Component Score Breakdown...")
    fig6_component_breakdown()

    print("[7/7] Per-Strategy Trust Scores...")
    fig7_strategy_trust_scores()

    print()
    print("=" * 60)
    print("ALL CHARTS GENERATED!")
    print("=" * 60)

    # List generated files
    generated = sorted(glob(str(FIGURES_DIR / "*.png")))
    print(f"\nGenerated {len(generated)} figures:")
    for f in generated:
        size_kb = Path(f).stat().st_size / 1024
        print(f"  - {Path(f).name} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
