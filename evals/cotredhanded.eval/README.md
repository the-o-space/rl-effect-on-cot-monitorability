# CoT Red-Handed Evaluation

Wrapper for running evaluations using the [cotredhanded](https://github.com/cotredhanded/code) package.

## Setup

Install with the optional dependency group:

```bash
uv sync --extra cotredhanded.eval
```

## Usage

Run evaluations using this wrapper script:

```bash
uv run --extra cotredhanded.eval evals/cotredhanded.eval/run.py \
    --models "together/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" \
    --dataset "bigcodebench" \
    --side-tasks "varname,logio,ioexfil" \
    --attack-policy "dependency" \
    --id-set "TEST" \
    --output-dir "evaluation_results" \
    --limit 50
```

Or run directly as a module:

```bash
uv run --extra cotredhanded.eval python -m cot_red_handed.tasks.run_tasks --help
```

## Reference

See the [upstream documentation](https://github.com/cotredhanded/code) for detailed parameter descriptions and configuration options.
