# Helpers shared by the experiment scripts: training seeds, run metadata,
# configuration snapshots and result files.

import ast
import hashlib
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
# Fingerprints: what a result was computed from
# --------------------------------------------------------------------------

def digest(obj):
    """Short hash of a JSON-serializable value (keys sorted)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def code_fingerprint(code):
    """Hash of the syntax tree (so not the comments) of the given code: path -> None for
    the whole file, or the names of its top-level functions."""
    parts = []
    for path in sorted(code, key=str):
        tree = ast.parse(Path(path).read_text())
        functions = code[path]
        if functions is None:
            parts.append(ast.dump(tree))
        else:
            nodes = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
            parts += [ast.dump(nodes[f]) for f in functions]
    return digest(parts)


def file_fingerprint(paths):
    """Hash of the content of the given files."""
    h = hashlib.sha256()
    for path in paths:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()[:16]


def graph_fingerprints(normalization_path):
    """Fingerprint of the normalized multi-instance graphs (normalization.json: their
    expansion, serialization and normalization), or None if they are not normalized."""
    path = Path(normalization_path)
    return read_json(path).get("fingerprint") if path.exists() else None


def trained_on_current_graphs(result, normalization_path):
    """Whether a GNN result (the content of its result.json) was trained on the graphs that
    are normalized now: results trained on other graphs are not reused."""
    current = graph_fingerprints(normalization_path)
    trained = ((result.get("data") or {}).get("normalization") or {}).get("fingerprint")
    return current is not None and trained == current


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
