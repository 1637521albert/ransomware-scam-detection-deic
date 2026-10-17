# FULL PIPELINE - every step of one or both methodologies with a single command
#
#   python run_pipeline.py                      # both methodologies
#   python run_pipeline.py multi-instance       # Methodology I only
#   python run_pipeline.py ego-centric          # Methodology II only
#   python run_pipeline.py both --n-seeds 10    # without asking the number of seeds
#   python run_pipeline.py --from train-gnn     # start from a given step
#   python run_pipeline.py --list               # show the steps and their commands
#
# Every step runs in its own process with the interpreter of its environment:
# RSD_BLOCKSCI_PYTHON for the BlockSci steps and RSD_TORCH_PYTHON for the
# others (both default to the current interpreter). The number of random-seed
# runs is asked once at the start, so the rest runs unattended. Each step skips
# the work it has already done, so an interrupted pipeline can simply be
# launched again. The pipeline stops at the first failing step.
#
# Output: PIPELINE_LOGS_DIR/<timestamp>.json, with the commands, interpreters,
# status and duration of every step.

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from shared.experiment import ask_n_seeds, git_commit, timestamp, training_seeds, write_json

MULTI_INSTANCE = PROJECT_ROOT / "Inductive Multi-Instance Address Classification"
EGO_CENTRIC = PROJECT_ROOT / "Inductive Ego-Centric Address Classification"

# (name, environment, script, arguments, methodologies the step belongs to)
STEPS = [
    ("splits",           "blocksci", PROJECT_ROOT / "create_splits.py",                   [],              {"multi-instance", "ego-centric"}),
    ("expand-mi",        "blocksci", MULTI_INSTANCE / "expansion.py",                     [],              {"multi-instance"}),
    ("expand-ego",       "blocksci", EGO_CENTRIC / "expansion.py",                        [],              {"ego-centric"}),
    ("serialize",        "torch",    MULTI_INSTANCE / "serialize_graph.py",               [],              {"multi-instance"}),
    ("normalize",        "torch",    MULTI_INSTANCE / "normalize_graphs.py",              [],              {"multi-instance"}),
    ("train-gnn",        "torch",    MULTI_INSTANCE / "run_experiments.py",               ["--n-seeds"],   {"multi-instance"}),
    ("extract-features", "torch",    EGO_CENTRIC / "extract_topological_features.py",     [],              {"ego-centric"}),
    ("train-ego",        "torch",    EGO_CENTRIC / "models.py",                           ["--n-seeds"],   {"ego-centric"}),
    # The training scripts already run the tests of their methodology; with both
    # methodologies they are run again at the end to compare them
    ("wilcoxon",         "torch",    PROJECT_ROOT / "wilcoxon_tests.py",                  [],              {"both"}),
]

# Modules checked in each environment before starting
REQUIRED_MODULES = {
    "blocksci": ["blocksci", "pandas", "numpy", "tqdm"],
    "torch": ["torch", "torch_geometric", "sklearn", "scipy", "pandas", "networkx", "xgboost", "shap", "joblib"],
}


def interpreter(environment):
    value = config.BLOCKSCI_PYTHON if environment == "blocksci" else config.TORCH_PYTHON
    return value or sys.executable


def selected_steps(methodology):
    wanted = {"multi-instance", "ego-centric", "both"} if methodology == "both" else {methodology}
    return [step for step in STEPS if step[4] & wanted]


def command(step, n_seeds):
    name, environment, script, arguments, _ = step
    args = []
    for argument in arguments:
        args += [argument, str(n_seeds)] if argument == "--n-seeds" else [argument]
    return [interpreter(environment), str(script)] + args


def check_environments(steps):
    """Fails early if an interpreter cannot import the modules of its steps."""
    missing = []
    for environment in sorted({step[1] for step in steps}):
        python = interpreter(environment)
        modules = REQUIRED_MODULES[environment]
        check = ("import importlib.util, sys\n"
                 "print(' '.join(m for m in sys.argv[1:] if importlib.util.find_spec(m) is None))")
        result = subprocess.run([python, "-c", check] + modules,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        if result.returncode != 0:
            missing.append(f"{environment} ({python}): cannot run the interpreter: {result.stderr.strip()}")
        elif result.stdout.strip():
            missing.append(f"{environment} ({python}): missing {result.stdout.strip()}")
    if missing:
        variable = {"blocksci": "RSD_BLOCKSCI_PYTHON", "torch": "RSD_TORCH_PYTHON"}
        print("Environment check failed:")
        for line in missing:
            print("  - " + line)
        print(f"Set {variable['blocksci']} and {variable['torch']} to the interpreters of the two environments.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run every step of one or both methodologies.")
    parser.add_argument("methodology", nargs="?", default="both", choices=["both", "multi-instance", "ego-centric"])
    parser.add_argument("--n-seeds", type=int, help="random-seed runs per configuration (asked if not given)")
    parser.add_argument("--from", dest="start", choices=[s[0] for s in STEPS], help="first step to run")
    parser.add_argument("--list", action="store_true", help="show the steps and their commands without running them")
    parser.add_argument("--skip-check", action="store_true", help="do not check the environments before starting")
    args = parser.parse_args()

    steps = selected_steps(args.methodology)
    if args.start:
        names = [s[0] for s in steps]
        if args.start not in names:
            parser.error(f"step '{args.start}' is not part of the {args.methodology} pipeline: {', '.join(names)}")
        steps = steps[names.index(args.start):]

    if args.list:
        for i, step in enumerate(steps, 1):
            print(f"{i}. {step[0]:17s} [{step[1]}] {' '.join(command(step, args.n_seeds or '<asked>'))}")
        return

    if not args.skip_check:
        check_environments(steps)

    trains = any("--n-seeds" in step[3] for step in steps)
    n_seeds = ask_n_seeds(args.n_seeds) if trains else None

    record = {
        "started": timestamp(),
        "git_commit": git_commit(config.PROJECT_ROOT),
        "methodology": args.methodology,
        "n_seeds": n_seeds,
        "seeds": training_seeds(n_seeds) if n_seeds else None,
        "interpreters": {"blocksci": interpreter("blocksci"), "torch": interpreter("torch")},
        "output_dir": str(config.OUTPUT_DIR),
        "steps": [],
    }
    log_path = config.PIPELINE_LOGS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}.json"

    for i, step in enumerate(steps, 1):
        cmd = command(step, n_seeds)
        print(f"\n{'=' * 70}\n[{i}/{len(steps)}] {step[0]} ({step[1]} environment)\n{' '.join(cmd)}\n{'=' * 70}", flush=True)

        start = time.time()
        returncode = subprocess.run(cmd, cwd=str(Path(step[2]).parent)).returncode
        duration = time.time() - start

        record["steps"].append({"name": step[0], "environment": step[1], "command": cmd,
                                "returncode": returncode, "duration_s": round(duration, 1)})
        write_json(log_path, record)

        if returncode != 0:
            print(f"\nStep '{step[0]}' failed (exit code {returncode}). Fix the problem and relaunch with "
                  f"--from {step[0]}; completed work is kept.")
            sys.exit(returncode)
        print(f"\n{step[0]} finished in {duration / 60:.1f} min", flush=True)

    record["finished"] = timestamp()
    write_json(log_path, record)
    print(f"\nPipeline finished. Results in {config.OUTPUT_DIR}, log in {log_path}")


if __name__ == "__main__":
    main()
