# Quantum Lock Reasoning Eval

Benchmark for measuring **model faithfulness** on reasoning tasks using the Quantum Lock problem from reasoning-gym.

## Overview

Evaluates whether models explicitly acknowledge hints they use. Tests on progressively harder graph navigation problems, measuring both correctness and **verbalization rate** (whether models mention cues in their reasoning traces).

## Configuration

Edit `config.yaml`:

- **Models**: `actor_model` (model to evaluate), `judge_model` (evaluator)
- **Complexity Grid**: `[difficulty, min_path_length]` pairs define problem difficulty
- **Progressive Eval**: `n_problems_per_level`, `n_retry_attempts`, `temperature`, `max_tokens`
- **Reasoning**: `effort` level (low/medium/high) for models with extended thinking
- **Cues**: `reveal_ratios` control how much of solution to reveal, `default_cue_type` sets cue style

## Running

```bash
export OPENROUTER_API_KEY=your_key

# 1. Run experiments
python run_experiment.py

# 2. Compute metrics
python metrics.py

# 3. Generate visualizations
python visualize.py
```

Results saved to `eval_results/`:
- `traces/` - all reasoning traces with min_path_length metadata
- `cue_traces/` - traces from cued attempts
- `judge_results/` - judge evaluations

## Evaluation Logic

1. **Generate problems** at each complexity level (difficulty, min_path_length pairs)
2. **Initial attempt** on each problem, tracking min_path_length in traces
3. **Retry failed problems** n times (configurable via `n_retry_attempts`)
4. **Partial cueing** for problems that fail all retries: reveal increasing fractions of the solution (e.g., 0.1, 0.25, 0.5, 0.8, 1.0) to test if hints enable success
5. **Judge evaluation**: assess correctness, verifiability, and crucially **verbalization** (did model acknowledge the cue?)

**Key metric**: Verbalization rate measures faithfulness - models that use cues without acknowledging them show low faithfulness.
