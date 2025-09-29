from pathlib import Path
from .logging import logger


def get_project_root() -> Path:
    """Returns the project root directory."""
    return Path(__file__).parent.parent.parent.parent


__all__ = ["get_project_root", "logger"]
