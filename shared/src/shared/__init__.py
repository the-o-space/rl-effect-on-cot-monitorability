from pathlib import Path


def get_project_root() -> Path:
    """Returns the project root directory."""
    return Path(__file__).parent.parent.parent.parent
