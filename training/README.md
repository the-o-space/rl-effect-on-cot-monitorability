# Training Experiments

This directory contains RL training experiments for the project.

## Quick Start

### Run a training experiment

```bash
# Use default config (configs/config.yaml)
make train EXP=example.rl

# Use a specific config from configs/ directory
make train EXP=example.rl CONFIG=mock.yaml
```

## Structure

Each training experiment is a self-contained package:

```
training/
├── example.rl/
│   ├── __init__.py
│   ├── train.py            # Main training entrypoint
│   ├── setup.py            # Model/dataset loading
│   ├── configs/
│   │   ├── config.yaml     # Default configuration
│   │   └── mock.yaml       # Fast test config (mock data)
│   └── requirements.txt    # Experiment-specific dependencies
```

## Creating a New Training Experiment

1. Copy `example.rl/` directory
2. Add configs to `configs/` directory:
   - `config.yaml` (default production config)
   - `mock.yaml` (fast test config with mock data)
3. Implement training logic in `train.py`
4. Update `requirements.txt` with dependencies
5. Run: `make train EXP=your.experiment`

## Config Structure

Configs should include:

```yaml
training:
  episodes: 1000
  learning_rate: 0.001

model:
  path: "models/..."

dataset:
  path: "datasets/..."
```

## Environment Setup

```bash
# Install shared package
uv pip install -e ../../shared

# Install experiment dependencies
cd example.rl
pip install -r requirements.txt
```
