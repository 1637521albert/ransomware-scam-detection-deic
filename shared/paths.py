from pathlib import Path
import os


def get_required_env_path(var_name: str) -> Path:
    value = os.getenv(var_name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {var_name}")

    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"Path from {var_name} does not exist: {path}")
    return path
