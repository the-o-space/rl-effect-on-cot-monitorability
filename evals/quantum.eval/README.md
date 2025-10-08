# Quantum Lock Reasoning Eval

Progressive complexity evaluation for reasoning models on the Quantum Lock problem from reasoning-gym.

## Overview

Tests models on graph navigation problems with increasing difficulty. When models fail, evaluates if partial solution cues improve performance.

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
python run_experiment.py
```

Results saved to `eval_results/`:
- `traces/` - all reasoning traces with min_path_length metadata
- `cue_traces/` - traces from cued attempts
- `judge_results/` - judge evaluations

## Logic

1. Generate problems at each complexity level
2. Test model on each problem (track min_path_length)
3. Retry failed problems
4. For persistent failures, run **partial cueing**: reveal increasing fractions of the solution (e.g., 0.1, 0.5, 0.8, 1.0) to measure if hints improve reasoning
5. Judge all traces for correctness and verifiability
