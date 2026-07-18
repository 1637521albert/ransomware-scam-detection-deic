from pathlib import Path
import os


def repo_root_from_file(file_path: str) -> Path:
    return Path(file_path).resolve().parents[1]


def module_dir_from_file(file_path: str) -> Path:
    return Path(file_path).resolve().parent


def module_outputs_dir(file_path: str) -> Path:
    return module_dir_from_file(file_path) / "outputs"


def get_required_env_path(var_name: str) -> Path:
    value = os.getenv(var_name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {var_name}")

    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"Path from {var_name} does not exist: {path}")
    return path


def build_run_name(train_samples, val_samples, test_samples, exp_alg, hops, limit, space, limit_mode):
    return f"{train_samples}-{val_samples}-{test_samples} {exp_alg} {hops} hops {limit}{space}{limit_mode} limit"


def build_run_dir(file_path: str, run_name: str) -> Path:
    return module_outputs_dir(file_path) / run_name


def ensure_split_dirs(run_dir: Path):
    (run_dir / "train").mkdir(parents=True, exist_ok=True)
    (run_dir / "val").mkdir(parents=True, exist_ok=True)
    (run_dir / "test").mkdir(parents=True, exist_ok=True)