# RL Effect on CoT Monitorability

This repository contains experiments for studying the effect of Reinforcement Learning on the monitorability of Chain-of-Thought reasoning in language models.

## Project Structure

This project is structured as a monorepo containing multiple, self-contained RL experiments. Each experiment is a Python package with its own dependencies, configuration, and Dockerfile, allowing for modular development, testing, and deployment.

```
/
├── training/
│   ├── example.rl/
│   │   ├── __init__.py
│   │   ├── train.py          # Main entrypoint for the experiment
│   │   ├── config.yaml       # Default configuration
│   │   ├── Dockerfile        # Container definition
│   │   ├── requirements.txt  # Python dependencies
│   │   └── test_train.py     # Local unit tests
│   └── ...                   # Other experiments
├── shared/
│   └── ...                   # Shared utilities, installable as a package
├── Makefile                  # Helper commands for managing experiments
└── ...                       # Other project files
```

## Development Workflow

### Creating a New Experiment

To create a new experiment, copy the `training/example.rl` directory and customize its contents.

### Local Testing

Each experiment can be tested locally by running it with a dedicated test configuration. This allows for fast, reproducible test runs using mock assets.

The test runner script (`tests/run_test.py`) invokes the experiment's training script with the test configuration located at `tests/test_config.yaml`. The mock assets for this are in `models/mock/` and `datasets/mock/`.

To run the tests for a specific experiment:
```sh
make test EXP=example.rl
```
This simply provides a different configuration to the same training script, removing the need for special flags or logic within the experiment code itself.

### Building and Deploying

Each experiment is packaged as a Docker image for portability and reproducibility.

To build the Docker image for an experiment:
```sh
make build EXP=example.t-rl
```

(You will be required to authenticate the device with `docker login` if you are launching it for the first time to download the docker python images)

To push the image to a container registry (e.g., Docker Hub), first create a `.env` file from the example and set your registry username:

```sh
cp .env.example .env
# Now edit .env to set your REGISTRY_USER
```

Next, authenticate with your container registry. For local development, the recommended method is `docker login`:

*   **Docker Hub:** `docker login`
*   **GitHub Container Registry:** `docker login ghcr.io` (use a Personal Access Token with `write:packages` scope as your password)

Finally, run the push command:
```sh
make push EXP=example.rl
```

The container can then be deployed to a cloud platform like RunPod.
