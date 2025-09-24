import typer
from pathlib import Path
import yaml
import logging

logging.basicConfig(level=logging.INFO)

def main(
    config_path: Path = typer.Option(..., "--config", "-c", help="Path to the config file."),
):
    """
    Main entrypoint for the RL training experiment.
    """
    logging.info(f"Loading config from {config_path}...")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logging.info(f"Config loaded: {config}")

    # The script now always expects model and dataset paths in the config.
    model_path = config.get("model", {}).get("path")
    dataset_path = config.get("dataset", {}).get("path")
    
    if not model_path or not dataset_path:
        logging.error("Config must contain model.path and dataset.path.")
        raise typer.Exit(code=1)
        
    logging.info(f"Loading model from: {model_path}")
    logging.info(f"Loading dataset from: {dataset_path}")

    logging.info("Starting training loop...")
    # Training loop logic goes here
    # for episode in range(config["training"]["episodes"]):
    #     ...
    logging.info("Training finished.")


if __name__ == "__main__":
    typer.run(main)
