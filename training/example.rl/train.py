import typer
from pathlib import Path
import yaml

from shared import logger

def main(
    config_path: Path = typer.Option(..., "--config", "-c", help="Path to the config file."),
):
    """
    Main entrypoint for the RL training experiment.
    """
    logger.info(f"Loading config from {config_path}...")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info(f"Config loaded", config=config)

    # The script now always expects model and dataset paths in the config.
    model_path = config.get("model", {}).get("path")
    dataset_path = config.get("dataset", {}).get("path")
    
    if not model_path or not dataset_path:
        logger.error("Config must contain model.path and dataset.path.")
        raise typer.Exit(code=1)
        
    logger.info(f"Loading model", model_path=model_path)
    logger.info(f"Loading dataset", dataset_path=dataset_path)

    logger.info("Starting training loop...")
    # Training loop logic goes here
    # for episode in range(config["training"]["episodes"]):
    #     ...
    logger.info("Training finished.")


if __name__ == "__main__":
    typer.run(main)
