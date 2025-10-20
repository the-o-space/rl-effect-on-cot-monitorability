#!/usr/bin/env python3
"""
Re-run judges on existing evaluation traces.

This script is useful when:
1. You want to change the judge evaluation mode (e.g., from CoT-only to CoT+output)
2. You need to re-evaluate traces that previously failed
3. You want to use a different judge model
4. You've updated the judge prompts

BACKWARD COMPATIBILITY:
By default, results are saved to a SEPARATE directory (e.g., judge_results_output_only/)
to preserve your original CoT-only results. Original judge_results/ is NOT touched.

Usage:
    # SAFE: Test new evaluation mode (saves to judge_results_output_only/)
    python3 rerun_judges.py --evaluation-mode output_only

    # SAFE: Test CoT+output (saves to judge_results_cot_and_output/)
    python3 rerun_judges.py --evaluation-mode cot_and_output

    # SAFE: Custom suffix
    python3 rerun_judges.py --evaluation-mode output_only --output-suffix "_test"

    # CAREFUL: Overwrite original (creates backup first)
    python3 rerun_judges.py --evaluation-mode output_only --overwrite

    # DANGEROUS: Overwrite without backup
    python3 rerun_judges.py --evaluation-mode output_only --overwrite --no-backup

    # Only re-run specific judges
    python3 rerun_judges.py --verbalization-only
    python3 rerun_judges.py --correctness-only

    # Re-run only on cued traces
    python3 rerun_judges.py --cued-only

    # Filter by model prefix (only Anthropic Claude models)
    python3 rerun_judges.py --model-prefix "anthropic/claude" --cued-only
"""

import argparse
import json
from pathlib import Path

from utils import Config, PromptLoader, OpenRouterClient
from judge import Judge
from cue_generator import CueGenerator


def extract_model_name_from_filename(filename):
    """
    Extract model name from trace filename.

    Examples:
        "anthropic_claude-sonnet-4_traces.json" -> "anthropic/claude-sonnet-4"
        "openai_gpt-5_cue_traces.json" -> "openai/gpt-5"
    """
    stem = filename.stem  # Remove .json
    # Remove common suffixes (most specific first to avoid partial matches)
    for suffix in ['_cue_traces', '_initial', '_retry', '_traces']:
        stem = stem.replace(suffix, '')

    # Replace first underscore with slash (provider/model format)
    parts = stem.split('_', 1)
    if len(parts) == 2:
        return f"{parts[0]}/{parts[1]}"
    return stem


def load_all_traces(results_dir):
    """Load all traces from traces/ and cue_traces/ directories with model names"""
    traces = []

    # Load regular traces
    traces_dir = Path(results_dir) / "traces"
    if traces_dir.exists():
        for f in traces_dir.glob("*.json"):
            with open(f) as file:
                trace_data = json.load(file)
                # Infer model name from filename if not present
                model_name = extract_model_name_from_filename(f)
                for trace in trace_data:
                    if 'model' not in trace or not trace['model']:
                        trace['model'] = model_name
                traces.extend(trace_data)

    # Load cue traces
    cue_dir = Path(results_dir) / "cue_traces"
    if cue_dir.exists():
        for f in cue_dir.glob("*.json"):
            with open(f) as file:
                trace_data = json.load(file)
                # Infer model name from filename if not present
                model_name = extract_model_name_from_filename(f)
                for trace in trace_data:
                    if 'model' not in trace or not trace['model']:
                        trace['model'] = model_name
                traces.extend(trace_data)

    return traces


def rerun_judges(
    results_dir="eval_results",
    config_path="config.yaml",
    prompts_dir="prompts",
    evaluation_mode=None,
    judge_model=None,
    verbalization_only=False,
    correctness_only=False,
    verifiability_only=False,
    cued_only=False,
    model_prefix=None,
    output_suffix=None,
    backup=True,
    overwrite=False
):
    """Re-run judges on existing traces"""

    print("="*60)
    print("RE-RUNNING JUDGES ON EXISTING TRACES")
    print("="*60)

    # Load configuration
    config = Config(config_path)
    prompt_loader = PromptLoader(prompts_dir)
    client = OpenRouterClient(base_url=config.api_base_url)
    cue_generator = CueGenerator()

    # Determine evaluation mode and judge model
    eval_mode = evaluation_mode or config.get('judges.evaluation_mode', 'cot_with_fallback')
    judge_model_name = judge_model or config.get('models.judge_model', 'anthropic/claude-sonnet-4')

    print(f"\nConfiguration:")
    print(f"  Results directory: {results_dir}")
    print(f"  Judge model: {judge_model_name}")
    print(f"  Evaluation mode: {eval_mode}")
    if model_prefix:
        print(f"  Model filter: {model_prefix}*")

    # Initialize judge
    judge = Judge(
        client=client,
        prompt_loader=prompt_loader,
        judge_model=judge_model_name,
        evaluation_mode=eval_mode
    )

    # Determine which judges to run
    run_verbalization = verbalization_only or (not correctness_only and not verifiability_only)
    run_correctness = correctness_only or (not verbalization_only and not verifiability_only)
    run_verifiability = verifiability_only or (not verbalization_only and not correctness_only)

    print(f"\nJudges to run:")
    print(f"  Verbalization: {run_verbalization}")
    print(f"  Correctness: {run_correctness}")
    print(f"  Verifiability: {run_verifiability}")

    # Load all traces
    print(f"\nLoading traces from {results_dir}...")
    traces = load_all_traces(results_dir)
    print(f"Loaded {len(traces)} traces")

    # Filter by model prefix if specified
    if model_prefix:
        original_count = len(traces)
        traces = [t for t in traces if t.get('model', '').startswith(model_prefix)]
        print(f"Filtered to {len(traces)} traces matching model prefix '{model_prefix}*' (from {original_count})")

    # Filter if cued_only
    if cued_only:
        original_count = len(traces)
        traces = [t for t in traces if t.get('cue_type') is not None]
        print(f"Filtered to {len(traces)} cued traces (from {original_count})")

    if len(traces) == 0:
        print("No traces to evaluate!")
        return

    # Re-evaluate using batch processing
    print(f"\nRe-evaluating {len(traces)} traces (batch processing)...")

    # Prepare batch requests
    judge_requests = []
    trace_mapping = []  # Maps request index to (trace_idx, judge_type)

    print(f"Preparing judge requests...")
    for i, trace in enumerate(traces):
        # Verbalization judge (only for cued traces)
        if run_verbalization and trace.get('cue_type'):
            try:
                # Reconstruct cue
                cue_str, _ = cue_generator.generate_cue(
                    solution_path=trace['expected_answer'].split(' → '),
                    cue_type=trace['cue_type'],
                    reveal_ratio=trace['reveal_ratio']
                )

                judge_requests.append({
                    'judge_type': 'verbalization_judge',
                    'kwargs': {
                        'problem': trace['problem_id'],
                        'cue': cue_str,
                        'reasoning': trace.get('reasoning'),
                        'output': trace.get('output')
                    }
                })
                trace_mapping.append((i, 'verbalization'))
            except Exception as e:
                print(f"  Error preparing verbalization for {trace['problem_id']}: {str(e)[:100]}")

        # Correctness judge
        if run_correctness:
            judge_requests.append({
                'judge_type': 'reasoning_correctness_judge',
                'kwargs': {
                    'problem': trace['task_description'],
                    'reasoning': trace.get('reasoning'),
                    'output': trace.get('output'),
                    'answer': trace.get('predicted_answer', ''),
                    'correct_answer': trace['expected_answer']
                }
            })
            trace_mapping.append((i, 'correctness'))

        # Verifiability judge
        if run_verifiability:
            judge_requests.append({
                'judge_type': 'cot_verifiability_judge',
                'kwargs': {
                    'problem': trace['task_description'],
                    'reasoning_without_answer': trace.get('reasoning'),
                    'output': trace.get('output')
                }
            })
            trace_mapping.append((i, 'verifiability'))

    # Run batch evaluation
    max_concurrent = config.get('api.max_concurrent_requests', 8)
    print(f"\nRunning {len(judge_requests)} judge evaluations (max {max_concurrent} concurrent)...")

    batch_results = judge.batch_evaluate(
        judge_requests,
        max_concurrent=max_concurrent
    )

    # Organize results by trace
    results_by_trace = {}
    errors = 0

    for (trace_idx, judge_type), result in zip(trace_mapping, batch_results):
        if trace_idx not in results_by_trace:
            results_by_trace[trace_idx] = {
                'problem_id': traces[trace_idx]['problem_id'],
                'model': traces[trace_idx].get('model', 'unknown'),
                'evaluation_mode': eval_mode  # Add metadata about evaluation mode
            }

        if result.get('error'):
            errors += 1

        results_by_trace[trace_idx][judge_type] = result

    # Convert to list maintaining order
    judge_results = [results_by_trace[i] for i in sorted(results_by_trace.keys())]

    # Determine output directory
    base_dir = Path(results_dir)
    judge_dir = base_dir / "judge_results"

    # Auto-generate suffix based on evaluation mode if not overwriting
    if not overwrite and output_suffix is None:
        # Auto-generate suffix from evaluation mode
        output_suffix = f"_{eval_mode}" if eval_mode != "cot_with_fallback" else ""
    elif overwrite:
        output_suffix = ""

    # Determine output directory
    if output_suffix:
        output_dir = base_dir / f"judge_results{output_suffix}"
        print(f"\n⚠️  Saving to separate directory: {output_dir.name}/")
        print(f"   (Original results in judge_results/ will be preserved)")
    else:
        output_dir = judge_dir
        if not overwrite:
            print(f"\n⚠️  WARNING: Will overwrite existing results in {judge_dir.name}/")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Backup existing results if requested and overwriting
    if backup and output_suffix == "" and judge_dir.exists():
        import shutil
        import time
        backup_dir = base_dir / f"judge_results_backup_{int(time.time())}"
        print(f"\n📦 Creating backup: {backup_dir.name}/")
        shutil.copytree(judge_dir, backup_dir)
        print(f"   Original results backed up to: {backup_dir}/")

    # Group by model
    by_model = {}
    for result in judge_results:
        model = result['model']
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(result)

    # Save per model
    print(f"\nSaving results...")
    for model, results in by_model.items():
        output_file = output_dir / f"{model.replace('/', '_')}_judge_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  ✓ {model}: {len(results)} results → {output_file.name}")

    print(f"\n{'='*60}")
    print(f"RE-EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total traces evaluated: {len(traces)}")
    print(f"Errors: {errors}")
    print(f"Results saved to: {output_dir}/")

    if output_suffix:
        print(f"\n💡 To use these results:")
        print(f"   1. Check results look good in {output_dir.name}/")
        print(f"   2. If satisfied, move to main directory:")
        print(f"      mv {output_dir}/* {judge_dir}/")
        print(f"   3. Regenerate metrics: python3 metrics.py")
        print(f"   4. Regenerate plots: python3 visualize.py")


def main():
    parser = argparse.ArgumentParser(
        description='Re-run judges on existing evaluation traces',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Re-run with current config
  python3 rerun_judges.py

  # Use CoT+output evaluation mode
  python3 rerun_judges.py --evaluation-mode cot_and_output

  # Use only output (for models without reasoning)
  python3 rerun_judges.py --evaluation-mode output_only

  # Re-run only verbalization judge
  python3 rerun_judges.py --verbalization-only

  # Re-run only on cued traces
  python3 rerun_judges.py --cued-only

  # Filter by model prefix (only Anthropic Claude models)
  python3 rerun_judges.py --model-prefix "anthropic/claude"

  # Filter by model prefix and only cued traces
  python3 rerun_judges.py --model-prefix "openai/gpt-5" --cued-only

  # Combine multiple filters
  python3 rerun_judges.py --model-prefix "anthropic" --evaluation-mode cot_and_output --cued-only
        """
    )

    parser.add_argument('--results-dir', default='eval_results',
                       help='Path to evaluation results directory')
    parser.add_argument('--config', default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--prompts-dir', default='prompts',
                       help='Path to prompts directory')

    parser.add_argument('--evaluation-mode',
                       choices=['cot_only', 'output_only', 'cot_with_fallback', 'cot_and_output'],
                       help='Evaluation mode (overrides config)')
    parser.add_argument('--judge-model',
                       help='Judge model to use (overrides config)')

    parser.add_argument('--verbalization-only', action='store_true',
                       help='Only run verbalization judge')
    parser.add_argument('--correctness-only', action='store_true',
                       help='Only run correctness judge')
    parser.add_argument('--verifiability-only', action='store_true',
                       help='Only run verifiability judge')

    parser.add_argument('--cued-only', action='store_true',
                       help='Only evaluate cued traces')

    parser.add_argument('--model-prefix',
                       help='Filter traces by model prefix (e.g., "anthropic/claude", "openai/gpt-5")')

    # Backward compatibility options
    parser.add_argument('--output-suffix',
                       help='Suffix for output directory (e.g., "_output_only"). Auto-generated from evaluation mode if not specified.')
    parser.add_argument('--no-backup', action='store_true',
                       help='Skip backup when overwriting (use with caution)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite original judge_results/ directory (default: save to separate directory)')

    args = parser.parse_args()

    rerun_judges(
        results_dir=args.results_dir,
        config_path=args.config,
        prompts_dir=args.prompts_dir,
        evaluation_mode=args.evaluation_mode,
        judge_model=args.judge_model,
        verbalization_only=args.verbalization_only,
        correctness_only=args.correctness_only,
        verifiability_only=args.verifiability_only,
        model_prefix=args.model_prefix,
        cued_only=args.cued_only,
        output_suffix=args.output_suffix,
        backup=not args.no_backup,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
