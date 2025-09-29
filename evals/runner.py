"""
Central eval runner that orchestrates evaluation execution.
"""
import typer
import yaml
import sys
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main(
    eval_name: str = typer.Argument(..., help="Name of the eval to run (e.g., example.eval)"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Override config file path"),
):
    """
    Run an evaluation with the specified configuration.
    
    This runner:
    1. Loads the eval's config (or override config)
    2. Imports and executes the eval's run.py
    3. Handles results logging
    """
    eval_dir = Path(__file__).parent / eval_name
    
    if not eval_dir.exists():
        logger.error(f"Eval directory not found: {eval_dir}")
        raise typer.Exit(code=1)
    
    # Load config
    if config_path is None:
        config_path = eval_dir / "config.yaml"
    
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        raise typer.Exit(code=1)
    
    logger.info(f"Loading config from {config_path}...")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Running eval: {eval_name}")
    logger.info(f"Config: {config}")
    
    # Add eval directory to Python path so we can import it
    sys.path.insert(0, str(eval_dir))
    
    try:
        # Import and run the eval
        import run as eval_module
        
        if hasattr(eval_module, "run_eval"):
            results = eval_module.run_eval(config)
            logger.info("Eval completed successfully")
            logger.info(f"Results: {results}")
        else:
            logger.error(f"Eval module must define a run_eval(config) function")
            raise typer.Exit(code=1)
    
    except ImportError as e:
        logger.error(f"Failed to import eval module: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"Eval execution failed: {e}", exc_info=True)
        raise typer.Exit(code=1)
    finally:
        sys.path.pop(0)


if __name__ == "__main__":
    typer.run(main)
