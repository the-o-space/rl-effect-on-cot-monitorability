"""
Run progressive difficulty evaluation for Quantum Reasoning

Usage:
    # Use defaults from config.yaml
    python run_experiment.py

    # Override specific parameters
    python run_experiment.py --model anthropic/claude-sonnet-4
    python run_experiment.py --model deepseek/deepseek-r1 --n-problems 3 --n-retries 3
    python run_experiment.py --reasoning-effort high --max-tokens 8000
"""

import argparse
from pathlib import Path
from quantum_eval import QuantumReasoningEval, ComplexityLevel


def main():
    parser = argparse.ArgumentParser(
        description='Run Quantum Reasoning progressive evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use all defaults from config.yaml
  python run_experiment.py

  # Override model
  python run_experiment.py --model anthropic/claude-sonnet-4

  # Override evaluation parameters
  python run_experiment.py --n-problems 5 --n-retries 3 --temperature 0.7

  # Test specific difficulty range
  python run_experiment.py --min-difficulty 1 --max-difficulty 50

  # For reasoning models
  python run_experiment.py --model deepseek/deepseek-r1 --reasoning-effort high
        """
    )

    # Model selection (optional - uses config default if not specified)
    parser.add_argument('--model', type=str,
                       help='Model name (e.g., anthropic/claude-sonnet-4). Default: from config.yaml')

    # Evaluation parameters (optional - use config defaults)
    parser.add_argument('--n-problems', type=int,
                       help='Number of problems per complexity level. Default: from config.yaml')
    parser.add_argument('--n-retries', type=int,
                       help='Number of retry attempts for failed problems. Default: from config.yaml')
    parser.add_argument('--temperature', type=float,
                       help='Sampling temperature. Default: from config.yaml')
    parser.add_argument('--max-tokens', type=int,
                       help='Maximum output tokens. Default: from config.yaml')
    parser.add_argument('--n-cue-examples', type=int,
                       help='Number of cue examples per failed problem. Default: from config.yaml')

    # Reasoning configuration (for reasoning models like DeepSeek R1)
    parser.add_argument('--reasoning-effort', type=str, choices=['low', 'medium', 'high'],
                       help='Reasoning effort level. Default: from config.yaml')
    parser.add_argument('--reasoning-max-tokens', type=int,
                       help='Max reasoning tokens. Default: from config.yaml')
    parser.add_argument('--exclude-reasoning', action='store_true',
                       help='Exclude reasoning from response')

    # Configuration files
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file (default: config.yaml)')
    parser.add_argument('--prompts-dir', type=str, default='prompts',
                       help='Path to prompts directory (default: prompts)')

    # Complexity grid override
    parser.add_argument('--min-difficulty', type=int,
                       help='Minimum difficulty level (filters complexity grid)')
    parser.add_argument('--max-difficulty', type=int,
                       help='Maximum difficulty level (filters complexity grid)')

    args = parser.parse_args()

    # Initialize evaluator
    print("Initializing evaluator...")
    evaluator = QuantumReasoningEval(
        config_path=args.config,
        prompts_dir=args.prompts_dir
    )

    # Build reasoning config if specified via CLI
    reasoning_config = None
    if args.reasoning_effort or args.reasoning_max_tokens or args.exclude_reasoning:
        reasoning_config = {}
        if args.reasoning_effort:
            reasoning_config['effort'] = args.reasoning_effort
        if args.reasoning_max_tokens:
            reasoning_config['max_tokens'] = args.reasoning_max_tokens
        if args.exclude_reasoning:
            reasoning_config['exclude'] = True

    # Custom complexity grid if specified
    complexity_grid = None
    if args.min_difficulty is not None or args.max_difficulty is not None:
        default_grid = evaluator.generate_complexity_grid()

        min_diff = args.min_difficulty if args.min_difficulty is not None else 0
        max_diff = args.max_difficulty if args.max_difficulty is not None else float('inf')

        complexity_grid = [
            level for level in default_grid
            if min_diff <= level.difficulty <= max_diff
        ]

        print(f"\nUsing filtered complexity grid: {len(complexity_grid)} levels")
        print(f"Difficulty range: {min_diff} to {max_diff}")
        for level in complexity_grid:
            print(f"  - {level}")
        print()

    # Run progressive evaluation
    # Only pass parameters that were explicitly set via CLI
    # Others will use config defaults
    kwargs = {}
    if args.model is not None:
        kwargs['model'] = args.model
    if args.n_problems is not None:
        kwargs['n_problems_per_level'] = args.n_problems
    if args.n_retries is not None:
        kwargs['n_retry_attempts'] = args.n_retries
    if args.temperature is not None:
        kwargs['temperature'] = args.temperature
    if args.max_tokens is not None:
        kwargs['max_tokens'] = args.max_tokens
    if args.n_cue_examples is not None:
        kwargs['n_cue_examples'] = args.n_cue_examples
    if reasoning_config is not None:
        kwargs['reasoning_config'] = reasoning_config
    if complexity_grid is not None:
        kwargs['complexity_grid'] = complexity_grid

    evaluator.run_progressive_eval(**kwargs)

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
