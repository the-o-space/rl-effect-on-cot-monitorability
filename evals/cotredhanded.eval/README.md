# CoT Red-Handed Evaluation

Wrapper for running evaluations using the [cotredhanded](https://github.com/cotredhanded/code) package.

## Setup

Install with the optional dependency group:

```bash
uv sync --extra cotredhanded.eval
```

## Usage

Run directly as a module:

```bash
uv run --extra cotredhanded.eval python -m cot_red_handed.tasks.run_tasks \
  --models "openrouter/meta-llama/llama-3.2-3b-instruct" \
  --dataset "bigcodebench" \
  --side-tasks "varname" \
  --attack-policy "dependency" \
  --id-set "TEST" \
  --output-dir "evals/cotredhanded.eval/test_run" \
  --limit 3
```

## Reference

See the [upstream documentation](https://github.com/cotredhanded/code) for detailed parameter descriptions and configuration options.
