# Evaluations

This directory contains evaluation packages for testing models on various tasks.

## Quick Start

### Run an evaluation

```bash
make eval EVAL=example.eval
```

### With custom config

```bash
make eval EVAL=example.eval CONFIG=path/to/custom_config.yaml
```

## Structure

Each eval is a self-contained package:

```
evals/
├── example.eval/
│   ├── __init__.py
│   ├── run.py              # Main eval logic, must define run_eval(config)
│   ├── config.yaml         # Default configuration
│   ├── requirements.txt    # Eval-specific dependencies
│   └── metrics.py          # Scoring/metrics implementation
```

## Creating a New Eval

1. Copy `example.eval/` directory
2. Customize `config.yaml` with your dataset, model, metrics
3. Implement eval logic in `run.py` (must have `run_eval(config)` function)
4. Add custom metrics in `metrics.py`
5. Run: `make eval EVAL=your.eval`

## Model Providers

Configured via `model.provider` in config.yaml:

- `openrouter`: OpenRouter API (requires `OPENROUTER_API_KEY` env var)
- `runpod`: RunPod serverless endpoints (coming soon)
- `local`: Local model inference (coming soon)

## Environment Setup

```bash
# Install shared package
uv pip install -e shared

# Create .env file from template
cp .env.example .env

# Edit .env and set your tokens:
#   OPENROUTER_API_KEY - Get from OpenRouter dashboard
#   LOGFIRE_TOKEN - Get from Logfire project settings -> Write Tokens
#   LOGFIRE_ENVIRONMENT - Your name or environment name (e.g., "theo-ryzhenkov")

# If using direnv, add to .envrc:
# dotenv .env
```
