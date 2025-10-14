"""Visualize evaluation results focused on verbalization metrics"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr
from metrics import load_traces, load_cue_traces, load_judge_results

# Set up plotting style with better layout
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.constrained_layout.use'] = True  # Fix title cutoff
plt.rcParams['font.size'] = 10


def get_unique_models(traces, cue_traces, judge):
    """Extract unique model names from dataframes"""
    models = set()
    for df in [traces, cue_traces, judge]:
        if 'model' in df.columns:
            models.update(df['model'].unique())
    return sorted(list(models))


def plot_verbalization_rate_by_model(cue_traces, judge, models, out):
    """Plot verbalization rate by model (main verbalization plot)"""
    rates = []
    model_names = []
    n_values = []
    
    for model in models:
        model_cue = cue_traces[cue_traces['model'] == model] if 'model' in cue_traces.columns else cue_traces
        model_judge = judge[judge['model'] == model] if 'model' in judge.columns else judge
        
        if len(model_cue) == 0:
            continue
            
        # Match cued traces with verbalization scores
        merged = model_cue.merge(model_judge[['problem_id', 'verbalization_score']], 
                               on='problem_id', how='inner')
        verb_scores = merged['verbalization_score'].dropna()
        
        if len(verb_scores) > 0:
            rate = (verb_scores >= 0.5).mean()
            rates.append(rate)
            model_names.append(model)
            n_values.append(len(verb_scores))
    
    if not rates:
        print("No verbalization data found")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(rates)))
    bars = ax.bar(range(len(rates)), rates, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, rate, n) in enumerate(zip(bars, rates, n_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.1%}\n(n={n})', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Verbalization Rate')
    ax.set_title('Verbalization Rate by Model\n(Cued Traces Only)', fontweight='bold', pad=20)
    ax.set_xticks(range(len(rates)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.savefig(out / 'verbalization_rate_by_model.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_judge_metrics_comparison(traces, judge, models, out):
    """Plot judge agreement, completeness, and reasoning scores by model"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Prepare data for all three metrics
    metrics = ['reasoning_score', 'completeness_score']
    metric_names = ['Reasoning Correctness', 'Completeness']
    axes = [ax1, ax2]
    
    # Plot reasoning correctness and completeness
    for ax, metric, name in zip(axes, metrics, metric_names):
        data = []
        labels = []
        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
        
        for i, model in enumerate(models):
            model_traces = traces[traces['model'] == model] if 'model' in traces.columns else traces
            model_judge = judge[judge['model'] == model] if 'model' in judge.columns else judge
            
            if len(model_judge) == 0 or metric not in model_judge.columns:
                continue
            
            # Merge traces with judge scores
            merged = model_traces.merge(model_judge[['problem_id', metric]], 
                                      on='problem_id', how='inner')
            scores = merged[metric].dropna()
            
            if len(scores) > 0:
                data.append(scores.values)
                labels.append(f'{model.split("_")[-1]}\n(n={len(scores)})')
        
        if data:
            bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True)
            for patch, color in zip(bp['boxes'], colors[:len(data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_ylabel('Score')
            ax.set_title(name, fontweight='bold')
            ax.set_ylim(-0.05, 1.05)
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(name, fontweight='bold')
    
    # Plot judge agreement (correlation)
    agreement_data = []
    agreement_labels = []
    for model in models:
        model_traces = traces[traces['model'] == model] if 'model' in traces.columns else traces
        model_judge = judge[judge['model'] == model] if 'model' in judge.columns else judge
        
        if len(model_judge) == 0 or 'reasoning_score' not in model_judge.columns:
            continue
        
        merged = model_traces.merge(model_judge[['problem_id', 'reasoning_score']], 
                                  on='problem_id', how='inner')
        valid = merged[merged['reasoning_score'].notna() & merged['score'].notna()]
        
        if len(valid) > 5:  # Need sufficient data for correlation
            try:
                corr, _ = pearsonr(valid['score'], valid['reasoning_score'])
                agreement_data.append(corr)
                agreement_labels.append(f'{model.split("_")[-1]}\n(r={corr:.3f})')
            except:
                continue
    
    if agreement_data:
        colors = plt.cm.Set2(np.linspace(0, 1, len(agreement_data)))
        bars = ax3.bar(range(len(agreement_data)), agreement_data, 
                      color=colors, alpha=0.7, edgecolor='black')
        
        # Add correlation values on bars
        for bar, corr in zip(bars, agreement_data):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02 if height > 0 else height - 0.05,
                    f'{corr:.3f}', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')
        
        ax3.set_ylabel('Correlation (r)')
        ax3.set_title('Model-Judge Agreement', fontweight='bold')
        ax3.set_xticks(range(len(agreement_data)))
        ax3.set_xticklabels([label.split('\n')[0] for label in agreement_labels], rotation=45)
        ax3.set_ylim(-1, 1)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.grid(axis='y', alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Model-Judge Agreement', fontweight='bold')
    
    plt.suptitle('Judge Evaluation Metrics by Model', fontsize=16, fontweight='bold')
    plt.savefig(out / 'judge_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_score_uplift_by_reveal_ratio(traces, cue_traces, models, out):
    """Plot score uplift (improvement) as a function of reveal ratio"""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    
    for i, model in enumerate(models):
        model_traces = traces[traces['model'] == model] if 'model' in traces.columns else traces
        model_cue = cue_traces[cue_traces['model'] == model] if 'model' in cue_traces.columns else cue_traces
        
        if len(model_cue) == 0 or 'base_problem_id' not in model_cue.columns:
            continue
        
        # Calculate uplift for each reveal ratio
        reveal_ratios = sorted(model_cue['reveal_ratio'].unique())
        mean_uplifts = []
        std_uplifts = []
        
        for ratio in reveal_ratios:
            cued_subset = model_cue[model_cue['reveal_ratio'] == ratio]
            
            # Match with original traces
            if 'base_problem_id' in model_traces.columns:
                matched = model_traces.merge(
                    cued_subset[['base_problem_id', 'score']], 
                    on='base_problem_id', 
                    suffixes=('_orig', '_cued')
                )
            else:
                # Fallback: extract base_problem_id from problem_id
                traces_with_base = model_traces.copy()
                traces_with_base['base_problem_id'] = traces_with_base['problem_id'].str.split('_').str[0]
                matched = traces_with_base.merge(
                    cued_subset[['base_problem_id', 'score']], 
                    on='base_problem_id', 
                    suffixes=('_orig', '_cued')
                )
            
            if len(matched) > 0:
                uplifts = matched['score_cued'] - matched['score_orig']
                mean_uplifts.append(uplifts.mean())
                std_uplifts.append(uplifts.std())
        
        if len(mean_uplifts) >= 2:
            ax.errorbar(reveal_ratios, mean_uplifts, yerr=std_uplifts,
                       marker='o', linewidth=2, markersize=8, capsize=5, 
                       label=model, color=colors[i])
    
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)  # Zero line
    ax.set_xlabel('Reveal Ratio')
    ax.set_ylabel('Score Uplift (Cued - Original)')
    ax.set_title('Score Uplift vs Reveal Ratio', fontweight='bold', pad=20)
    ax.set_xlim(-0.05, 1.05)
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    
    plt.savefig(out / 'score_uplift_by_reveal_ratio.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_verbalization_summary(cue_traces, judge, models, out):
    """Create a summary plot showing verbalization patterns"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    
    # 1. Verbalization rate by model
    rates = []
    model_names = []
    for model in models:
        model_cue = cue_traces[cue_traces['model'] == model] if 'model' in cue_traces.columns else cue_traces
        model_judge = judge[judge['model'] == model] if 'model' in judge.columns else judge
        
        if len(model_cue) == 0:
            continue
            
        merged = model_cue.merge(model_judge[['problem_id', 'verbalization_score']], 
                               on='problem_id', how='inner')
        verb_scores = merged['verbalization_score'].dropna()
        
        if len(verb_scores) > 0:
            rate = (verb_scores >= 0.5).mean()
            rates.append(rate)
            model_names.append(model.split('_')[-1])  # Shorter names for display
    
    if rates:
        bars = ax1.bar(range(len(rates)), rates, color=colors[:len(rates)], alpha=0.7)
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{rate:.1%}', 
                    ha='center', va='bottom', fontweight='bold')
        ax1.set_title('Verbalization Rate by Model', fontweight='bold')
        ax1.set_ylabel('Verbalization Rate')
        ax1.set_xticks(range(len(rates)))
        ax1.set_xticklabels(model_names, rotation=45)
        ax1.set_ylim(0, 1.1)
        ax1.grid(axis='y', alpha=0.3)
    
    # 2. Verbalization score distribution
    for i, model in enumerate(models):
        model_cue = cue_traces[cue_traces['model'] == model] if 'model' in cue_traces.columns else cue_traces
        model_judge = judge[judge['model'] == model] if 'model' in judge.columns else judge
        
        if len(model_cue) == 0:
            continue
            
        merged = model_cue.merge(model_judge[['problem_id', 'verbalization_score']], 
                               on='problem_id', how='inner')
        verb_scores = merged['verbalization_score'].dropna()
        
        if len(verb_scores) > 5:
            ax2.hist(verb_scores, bins=10, alpha=0.6, label=model.split('_')[-1], 
                    color=colors[i], edgecolor='black')
    
    ax2.set_title('Verbalization Score Distribution', fontweight='bold')
    ax2.set_xlabel('Verbalization Score')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Verbalization vs Reveal Ratio (aggregated)
    reveal_ratios = [0.1, 0.25, 0.5, 0.8, 1.0]
    mean_verb_by_reveal = []
    std_verb_by_reveal = []
    
    for ratio in reveal_ratios:
        all_scores = []
        for model in models:
            model_cue = cue_traces[cue_traces['model'] == model] if 'model' in cue_traces.columns else cue_traces
            model_judge = judge[judge['model'] == model] if 'model' in judge.columns else judge
            
            if len(model_cue) == 0:
                continue
            
            merged = model_cue.merge(model_judge[['problem_id', 'verbalization_score']], 
                                   on='problem_id', how='inner')
            subset = merged[merged['reveal_ratio'] == ratio]
            if len(subset) > 0:
                all_scores.extend(subset['verbalization_score'].dropna().tolist())
        
        if all_scores:
            mean_verb_by_reveal.append(np.mean(all_scores))
            std_verb_by_reveal.append(np.std(all_scores))
        else:
            mean_verb_by_reveal.append(0)
            std_verb_by_reveal.append(0)
    
    if any(mean_verb_by_reveal):
        ax3.errorbar(reveal_ratios, mean_verb_by_reveal, yerr=std_verb_by_reveal,
                    marker='o', linewidth=2, markersize=8, capsize=5, color='red')
        ax3.set_title('Verbalization vs Reveal Ratio (All Models)', fontweight='bold')
        ax3.set_xlabel('Reveal Ratio')
        ax3.set_ylabel('Mean Verbalization Score')
        ax3.grid(alpha=0.3)
    
    # 4. Verbalization rate by reveal ratio
    rates_by_reveal = []
    for ratio in reveal_ratios:
        all_scores = []
        for model in models:
            model_cue = cue_traces[cue_traces['model'] == model] if 'model' in cue_traces.columns else cue_traces
            model_judge = judge[judge['model'] == model] if 'model' in judge.columns else judge
            
            if len(model_cue) == 0:
                continue
            
            merged = model_cue.merge(model_judge[['problem_id', 'verbalization_score']], 
                                   on='problem_id', how='inner')
            subset = merged[merged['reveal_ratio'] == ratio]
            if len(subset) > 0:
                all_scores.extend(subset['verbalization_score'].dropna().tolist())
        
        if all_scores:
            rate = np.mean([s >= 0.5 for s in all_scores])
            rates_by_reveal.append(rate)
        else:
            rates_by_reveal.append(0)
    
    if any(rates_by_reveal):
        bars = ax4.bar(reveal_ratios, rates_by_reveal, color='green', alpha=0.7, width=0.08)
        for bar, rate in zip(bars, rates_by_reveal):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{rate:.1%}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax4.set_title('Verbalization Rate by Reveal Ratio', fontweight='bold')
        ax4.set_xlabel('Reveal Ratio')
        ax4.set_ylabel('Verbalization Rate')
        ax4.set_ylim(0, 1.1)
        ax4.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Verbalization Analysis Summary', fontsize=16, fontweight='bold')
    plt.savefig(out / 'verbalization_summary.png', dpi=300, bbox_inches='tight')
    plt.close()




def generate_verbalization_plots(results_dir="eval_results"):
    """Generate focused verbalization analysis plots"""
    print("Loading data...")
    traces = load_traces(results_dir)
    cue_traces = load_cue_traces(results_dir)
    judge = load_judge_results(results_dir)

    out = Path(results_dir) / "plots"
    out.mkdir(parents=True, exist_ok=True)

    # Get unique models
    models = get_unique_models(traces, cue_traces, judge)
    print(f"Found {len(models)} model(s): {', '.join(models)}")

    if len(models) == 0:
        print("No model information found. Skipping plots.")
        return

    # Check if we have judge data
    if len(judge) == 0:
        print("No judge data found. Skipping plots.")
        return

    print("\nGenerating verbalization-focused plots...")
    
    # Core verbalization and judge plots
    plots = [
        ("Verbalization rate by model", lambda: plot_verbalization_rate_by_model(cue_traces, judge, models, out)),
        ("Judge metrics comparison", lambda: plot_judge_metrics_comparison(traces, judge, models, out)),
        ("Score uplift by reveal ratio", lambda: plot_score_uplift_by_reveal_ratio(traces, cue_traces, models, out)),
        ("Verbalization summary", lambda: plot_verbalization_summary(cue_traces, judge, models, out)),
    ]

    for name, func in plots:
        try:
            print(f"  - {name}")
            func()
        except Exception as e:
            print(f"  - {name}: Skipped ({e})")
            import traceback
            print(f"    Error details: {traceback.format_exc()}")

    print(f"\nVerbalization plots saved to {out}/")
    print("\nGenerated plots:")
    for plot_file in sorted(out.glob("*.png")):
        print(f"  - {plot_file.name}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='eval_results')
    args = parser.parse_args()

    generate_verbalization_plots(args.results_dir)
