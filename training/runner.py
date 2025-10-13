"""
Central training runner that orchestrates training execution.
"""
import typer
import subprocess
import sys
from pathlib import Path

from shared import logger


def main(
    exp_name: str = typer.Argument(..., help="Name of the training experiment to run (e.g., example.rl)"),
    config: str = typer.Option("config.yaml", "--config", "-c", help="Config filename in experiment's configs/ directory"),
):
    """
    Run a training experiment with the specified configuration.
    
    This runner:
    1. Loads the experiment's config from configs/ directory
    2. Executes the experiment's train.py with the config
    3. Handles results logging
    """
    exp_dir = Path(__file__).parent / exp_name

    logger.configure(
        service_name="training-runner",
    )
    
    if not exp_dir.exists():
        logger.error(f"Experiment directory not found: {exp_name}")
        raise typer.Exit(code=1)
    
    # Load config from experiment's configs/ directory
    config_path = exp_dir / "configs" / config
    
    if not config_path.exists():
        logger.error(f"Config not found: {config}")
        raise typer.Exit(code=1)
    
    # Check for train.py
    train_script = exp_dir / "train.py"
    if not train_script.exists():
        logger.error(f"Training script not found: {train_script}")
        raise typer.Exit(code=1)
    
    with logger.span(f"training.{exp_name}"):
        logger.info(f"Starting training experiment: {exp_name}", config=config)
        logger.info(f"Using config: {config_path}")
        
        # Run the training script with the config
        result = subprocess.run(
            [sys.executable, str(train_script), "--config", str(config_path)],
            cwd=Path(__file__).parent.parent,
        )
        
        if result.returncode == 0:
            logger.info("Training completed successfully")
        else:
            logger.error(f"Training failed with exit code {result.returncode}")
            raise typer.Exit(code=result.returncode)


if __name__ == "__main__":
    typer.run(main)
