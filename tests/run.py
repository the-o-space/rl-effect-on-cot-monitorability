import typer
import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

def main(exp: str = typer.Argument(..., help="The experiment to test, e.g., 'example.rl'")):
    """
    Runs the specified experiment in mock mode using a centralized test config.
    """
    project_root = Path(__file__).parent.parent
    exp_dir = project_root / "training" / exp
    train_script = exp_dir / "train.py"
    config_path = project_root / "tests" / "test_config.yaml"

    if not train_script.exists():
        logging.error(f"Error: train.py not found for experiment '{exp}' at {train_script}")
        raise typer.Exit(code=1)

    if not config_path.exists():
        logging.error(f"Error: Test config not found at {config_path}")
        raise typer.Exit(code=1)

    command = [
        "python",
        str(train_script),
        "--config",
        str(config_path),
    ]

    logging.info(f"Running test for '{exp}'...")
    logging.info(f"Command: {' '.join(command)}")

    result = subprocess.run(command, capture_output=True, text=True, check=False)

    print("\n--- STDOUT ---")
    print(result.stdout)
    print("\n--- STDERR ---")
    print(result.stderr)

    if result.returncode != 0:
        logging.error(f"Test failed for experiment '{exp}' with exit code {result.returncode}")
        raise typer.Exit(code=1)
    
    logging.info(f"Test passed for experiment '{exp}'!")

if __name__ == "__main__":
    typer.run(main)
