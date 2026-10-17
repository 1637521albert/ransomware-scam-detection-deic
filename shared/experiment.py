# Helpers shared by the experiment scripts: training seeds, run metadata,
# configuration snapshots and result files.

import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path


def git_commit(project_root):
    """Commit hash of the code that produced a result, marked if there are uncommitted changes."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(project_root), stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=str(project_root), stderr=subprocess.DEVNULL
        ).decode().strip()
        return commit + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def timestamp():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def write_json(path, data):
    """Writes a JSON file atomically, so an interrupted run never leaves a partial result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=4, default=str)
    os.replace(str(tmp), str(path))


def read_json(path):
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Training seeds
# --------------------------------------------------------------------------

def training_seeds(n_seeds):
    """The first n_seeds training seeds. Beyond config.TRAINING_SEEDS, additional
    seeds are drawn deterministically from DATA_SEED, so a run with more seeds
    always contains the runs with fewer seeds."""
    import config
    seeds = list(config.TRAINING_SEEDS)
    rng = random.Random(config.DATA_SEED)
    while len(seeds) < n_seeds:
        candidate = rng.randint(1, 1000000)
        if candidate not in seeds:
            seeds.append(candidate)
    return seeds[:n_seeds]


def ask_n_seeds(cli_value=None):
    """Number of random-seed runs: given on the command line, or asked to the user."""
    import config
    if cli_value is not None:
        n = cli_value
    elif not sys.stdin.isatty():
        n = config.DEFAULT_N_SEEDS
    else:
        answer = input(f"How many random-seed runs do you want to execute per configuration? [{config.DEFAULT_N_SEEDS}]: ").strip()
        n = int(answer) if answer else config.DEFAULT_N_SEEDS
    if n < 1:
        raise ValueError("The number of random-seed runs must be at least 1.")
    if n < config.MIN_SEEDS_FOR_TESTS:
        print(f"[WARNING] With fewer than {config.MIN_SEEDS_FOR_TESTS} seeds the Wilcoxon tests are not computed.")
    return n


# --------------------------------------------------------------------------
# Configuration snapshots
# --------------------------------------------------------------------------

def library_versions(modules):
    versions = {"python": platform.python_version()}
    for name in modules:
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[name] = "not installed"
    return versions


def config_snapshot():
    """Every setting of config.py, as a JSON-serializable dictionary."""
    import config
    snapshot = {}
    for key in dir(config):
        if not key.isupper():
            continue
        value = getattr(config, key)
        snapshot[key] = str(value) if isinstance(value, Path) else value
    return snapshot


def experiment_snapshot(extra=None, modules=()):
    """Full description of an experiment: configuration, data provenance, code and environment."""
    import config
    snapshot = {
        "created": timestamp(),
        "git_commit": git_commit(config.PROJECT_ROOT),
        "host": platform.node(),
        "cpu_count": config.CPU_COUNT,
        "libraries": library_versions(modules),
        "config": config_snapshot(),
    }
    if config.SPLITS_SUMMARY_FILE.exists():
        snapshot["splits"] = read_json(config.SPLITS_SUMMARY_FILE)
    snapshot.update(extra or {})
    return snapshot
