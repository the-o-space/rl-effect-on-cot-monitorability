"""Visualize evaluation results with per-model and comparative plots"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from metrics import load_traces, load_cue_traces, load_judge_results

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)


def get_unique_models(traces, cue_traces, judge):
    """Extract unique model names from dataframes"""
    models = set()
    for df in [traces, cue_traces, judge]:
        if 'model' in df.columns:
            models.update(df['model'].unique())
    return sorted(list(models))


def plot_score_by_difficulty_single(traces, model_name, out):
    """Plot score by difficulty for a single model"""
    grouped = traces.groupby('difficulty')['score'].agg(['mean', 'std', 'count'])

    fig, ax = plt.subplots()
    ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                marker='o', linewidth=2, markersize=8, capsize=5, capthick=2)
    ax.set_xlabel('Difficulty', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Score by Difficulty: {model_name}', fontsize=14, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    # Add count annotations
    for diff, row in grouped.iterrows():
        ax.text(diff, row['mean'] + 0.05, f"n={int(row['count'])}",
                ha='center', fontsize=9, alpha=0.7)

    plt.tight_layout()
    safe_name = model_name.replace('/', '_')
    plt.savefig(out / f'score_by_difficulty_{safe_name}.png', dpi=300)
    plt.close()


def plot_score_by_difficulty_comparison(traces, models, out):
    """Comparative plot of score by difficulty across models"""
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    for i, model in enumerate(models):
        model_traces = traces[traces['model'] == model]
        grouped = model_traces.groupby('difficulty')['score'].agg(['mean', 'std'])

        ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                   marker='o', linewidth=2, markersize=8, capsize=5, capthick=2,
                   label=model, color=colors[i])

    ax.set_xlabel('Difficulty', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Score by Difficulty: Model Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out / 'score_by_difficulty_comparison.png', dpi=300)
    plt.close()


def plot_score_vs_reveal_single(cue, model_name, out):
    """Plot score vs reveal ratio for a single model"""
    valid = cue[cue['reveal_ratio'].notna() & cue['score'].notna()]
    if len(valid) < 3:
        return

    fig, ax = plt.subplots()
    ax.scatter(valid['reveal_ratio'], valid['score'], alpha=0.6, s=50)

    if len(valid) >= 5:
        z = np.polyfit(valid['reveal_ratio'], valid['score'], 1)
        x_line = np.linspace(0, 1, 100)
        ax.plot(x_line, np.poly1d(z)(x_line), 'r--', linewidth=2,
                label=f'y={z[0]:.2f}x+{z[1]:.2f}')
        ax.legend()

    ax.set_xlabel('Reveal Ratio', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Score vs Reveal Ratio: {model_name}', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    safe_name = model_name.replace('/', '_')
    plt.savefig(out / f'score_vs_reveal_{safe_name}.png', dpi=300)
    plt.close()


def plot_score_vs_reveal_comparison(cue, models, out):
    """Comparative plot of score vs reveal ratio across models"""
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    for i, model in enumerate(models):
        model_cue = cue[cue['model'] == model]
        valid = model_cue[model_cue['reveal_ratio'].notna() & model_cue['score'].notna()]

        if len(valid) < 3:
            continue

        ax.scatter(valid['reveal_ratio'], valid['score'], alpha=0.6, s=50,
                  label=model, color=colors[i])

        if len(valid) >= 5:
            z = np.polyfit(valid['reveal_ratio'], valid['score'], 1)
            x_line = np.linspace(0, 1, 100)
            ax.plot(x_line, np.poly1d(z)(x_line), '--', linewidth=2, color=colors[i])

    ax.set_xlabel('Reveal Ratio', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Score vs Reveal Ratio: Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out / 'score_vs_reveal_comparison.png', dpi=300)
    plt.close()


def plot_cue_effectiveness_single(traces, cue, model_name, out):
    """Plot baseline vs cue effectiveness for a single model"""
    if len(cue) == 0:
        return

    fig, ax = plt.subplots()
    data = [traces['score'].values, cue['score'].values]
    labels = [f'Baseline\n(n={len(traces)})', f'Cued\n(n={len(cue)})']

    bp = ax.boxplot(data, labels=labels, widths=0.5, patch_artist=True, showmeans=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Baseline vs Cued: {model_name}', fontsize=14, fontweight='bold')
    ax.set_ylim(-0.1, 1.1)
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    safe_name = model_name.replace('/', '_')
    plt.savefig(out / f'cue_effectiveness_{safe_name}.png', dpi=300)
    plt.close()


def plot_cue_effectiveness_comparison(traces, cue, models, out):
    """Comparative plot of baseline vs cue across models"""
    fig, ax = plt.subplots(figsize=(12, 6))

    positions = []
    data = []
    labels = []
    colors = []

    pos = 1
    for model in models:
        model_traces = traces[traces['model'] == model]
        model_cue = cue[cue['model'] == model]

        if len(model_cue) == 0:
            continue

        # Baseline
        data.append(model_traces['score'].values)
        positions.append(pos)
        labels.append(f'{model}\nBaseline')
        colors.append('lightblue')

        # Cued
        data.append(model_cue['score'].values)
        positions.append(pos + 1)
        labels.append(f'{model}\nCued')
        colors.append('lightgreen')

        pos += 3

    bp = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True, showmeans=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9, rotation=45, ha='right')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Baseline vs Cued: Model Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(-0.1, 1.1)
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(out / 'cue_effectiveness_comparison.png', dpi=300)
    plt.close()


def plot_verbalization_vs_reveal_single(cue, judge, model_name, out):
    """Plot verbalization vs reveal ratio for a single model"""
    merged = cue.merge(judge[['problem_id', 'verbalization_score']], on='problem_id', how='left')
    valid = merged[merged['reveal_ratio'].notna() & merged['verbalization_score'].notna()]

    if len(valid) < 3:
        return

    fig, ax = plt.subplots()
    ax.scatter(valid['reveal_ratio'], valid['verbalization_score'], alpha=0.6, s=50, c='green')

    if len(valid) >= 5:
        z = np.polyfit(valid['reveal_ratio'], valid['verbalization_score'], 1)
        x_line = np.linspace(0, 1, 100)
        ax.plot(x_line, np.poly1d(z)(x_line), 'r--', linewidth=2,
                label=f'y={z[0]:.2f}x+{z[1]:.2f}')
        ax.legend()

    ax.set_xlabel('Reveal Ratio', fontsize=12)
    ax.set_ylabel('Verbalization Score', fontsize=12)
    ax.set_title(f'Verbalization vs Reveal: {model_name}', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    safe_name = model_name.replace('/', '_')
    plt.savefig(out / f'verbalization_vs_reveal_{safe_name}.png', dpi=300)
    plt.close()


def plot_judge_agreement_single(traces, judge, model_name, out):
    """Plot judge agreement for a single model"""
    merged = traces.merge(judge[['problem_id', 'reasoning_score']], on='problem_id', how='inner')
    merged['score_bin'] = (merged['score'] >= 0.5).astype(int)
    merged['judge_bin'] = (merged['reasoning_score'] >= 0.5).astype(int)
    valid = merged[merged['reasoning_score'].notna()]

    if len(valid) < 3:
        return

    import pandas as pd
    cm = pd.crosstab(valid['score_bin'], valid['judge_bin'])

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax)
    ax.set_ylabel('Actual Score (≥0.5)', fontsize=11)
    ax.set_xlabel('Judge Reasoning (≥0.5)', fontsize=11)
    ax.set_title(f'Judge Agreement: {model_name}', fontsize=14, fontweight='bold')

    plt.tight_layout()
    safe_name = model_name.replace('/', '_')
    plt.savefig(out / f'judge_agreement_{safe_name}.png', dpi=300)
    plt.close()


def plot_quality_scores_single(judge, model_name, out):
    """Plot quality scores for a single model"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, field in enumerate(['verifiability_score', 'completeness_score']):
        if field not in judge.columns:
            axes[i].text(0.5, 0.5, 'No Data', ha='center', va='center')
            axes[i].set_title(field.replace('_score', '').title())
            continue

        scores = judge[field].dropna()
        if len(scores) == 0:
            axes[i].text(0.5, 0.5, 'No Data', ha='center', va='center')
            axes[i].set_title(field.replace('_score', '').title())
            continue

        color = 'steelblue' if i == 0 else 'seagreen'
        axes[i].hist(scores, bins=min(20, max(5, len(scores)//2)), color=color, alpha=0.7, edgecolor='black')
        axes[i].axvline(scores.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'μ={scores.mean():.3f}')
        axes[i].set_xlabel(field.replace('_', ' ').title(), fontsize=12)
        axes[i].set_ylabel('Frequency', fontsize=12)
        axes[i].set_title(field.replace('_score', '').title(), fontsize=13, fontweight='bold')
        axes[i].legend()
        axes[i].grid(alpha=0.3, axis='y')

    fig.suptitle(f'Quality Scores: {model_name}', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    safe_name = model_name.replace('/', '_')
    plt.savefig(out / f'quality_scores_{safe_name}.png', dpi=300)
    plt.close()


def plot_quality_scores_comparison(judge, models, out):
    """Comparative plot of quality scores across models"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, field in enumerate(['verifiability_score', 'completeness_score']):
        data = []
        labels = []

        for model in models:
            model_judge = judge[judge['model'] == model]
            if field in model_judge.columns:
                scores = model_judge[field].dropna()
                if len(scores) > 0:
                    data.append(scores.values)
                    labels.append(f'{model}\n(n={len(scores)})')

        if len(data) == 0:
            axes[i].text(0.5, 0.5, 'No Data', ha='center', va='center')
            axes[i].set_title(field.replace('_score', '').title())
            continue

        color = 'steelblue' if i == 0 else 'seagreen'
        bp = axes[i].boxplot(data, labels=labels, patch_artist=True, showmeans=True)
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        axes[i].set_ylabel('Score', fontsize=12)
        axes[i].set_title(field.replace('_score', '').title(), fontsize=13, fontweight='bold')
        axes[i].set_ylim(-0.05, 1.05)
        axes[i].grid(alpha=0.3, axis='y')
        axes[i].tick_params(axis='x', rotation=45)

    fig.suptitle('Quality Scores: Model Comparison', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out / 'quality_scores_comparison.png', dpi=300)
    plt.close()


def generate_all_plots(results_dir="eval_results"):
    """Generate all plots: per-model and comparative"""
    print("Loading data...")
    traces = load_traces(results_dir)
    cue = load_cue_traces(results_dir)
    judge = load_judge_results(results_dir)

    out = Path(results_dir) / "plots"
    out.mkdir(parents=True, exist_ok=True)

    # Get unique models
    models = get_unique_models(traces, cue, judge)
    print(f"Found {len(models)} model(s): {', '.join(models)}")

    if len(models) == 0:
        print("No model information found. Skipping plots.")
        return

    # Generate per-model plots
    print("\nGenerating per-model plots...")
    for model in models:
        print(f"\n  Model: {model}")
        model_traces = traces[traces['model'] == model] if 'model' in traces.columns else traces
        model_cue = cue[cue['model'] == model] if 'model' in cue.columns else cue
        model_judge = judge[judge['model'] == model] if 'model' in judge.columns else judge

        plots = [
            ("Score by difficulty", lambda: plot_score_by_difficulty_single(model_traces, model, out)),
            ("Score vs reveal", lambda: plot_score_vs_reveal_single(model_cue, model, out)),
            ("Cue effectiveness", lambda: plot_cue_effectiveness_single(model_traces, model_cue, model, out)),
            ("Verbalization vs reveal", lambda: plot_verbalization_vs_reveal_single(model_cue, model_judge, model, out)),
            ("Judge agreement", lambda: plot_judge_agreement_single(model_traces, model_judge, model, out)),
            ("Quality scores", lambda: plot_quality_scores_single(model_judge, model, out)),
        ]

        for name, func in plots:
            try:
                print(f"    - {name}")
                func()
            except Exception as e:
                print(f"    - {name}: Skipped ({e})")

    # Generate comparative plots if multiple models
    if len(models) > 1:
        print(f"\nGenerating comparative plots ({len(models)} models)...")

        comp_plots = [
            ("Score by difficulty", lambda: plot_score_by_difficulty_comparison(traces, models, out)),
            ("Score vs reveal", lambda: plot_score_vs_reveal_comparison(cue, models, out)),
            ("Cue effectiveness", lambda: plot_cue_effectiveness_comparison(traces, cue, models, out)),
            ("Quality scores", lambda: plot_quality_scores_comparison(judge, models, out)),
        ]

        for name, func in comp_plots:
            try:
                print(f"  - {name}")
                func()
            except Exception as e:
                print(f"  - {name}: Skipped ({e})")

    print(f"\nPlots saved to {out}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='eval_results')
    args = parser.parse_args()

    generate_all_plots(args.results_dir)
