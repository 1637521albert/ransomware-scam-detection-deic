# FULL PIPELINE - every step of one or both methodologies with a single command
#
#   python run_pipeline.py --original           # both methodologies, original configuration (config.py)
#   python run_pipeline.py                      # both methodologies, parameters chosen by the user
#   python run_pipeline.py multi-instance       # Methodology I only
#   python run_pipeline.py ego-centric          # Methodology II only
#   python run_pipeline.py --original --n-seeds 10   # without any question
#   python run_pipeline.py --config <file>      # a saved custom configuration
#   python run_pipeline.py --from train-gnn     # start from a given step
#   python run_pipeline.py --list               # show the steps and their commands
#   python run_pipeline.py ego-centric --original --from interpret-ego   # only the ablation and SHAP
#
# --original runs every step with the original configuration (config.py), and
# only the number of random-seed runs is asked. Without it, the parameters of the
# selected methodologies are asked group by group (the original values are the
# defaults); they are saved in OUTPUT_DIR/<configuration id>/configuration.json,
# where all the data and results of that configuration are kept, and can be run
# again with --config <file> (see shared/configuration.py).
#
# Every step runs in its own process with the current interpreter, whose
# environment must have every library of requirements.txt and BlockSci. The
# number of random-seed runs is asked once at the start, so the rest runs
# unattended. Each step skips the work it has already done with the same code,
# parameters and inputs (its fingerprint) and redoes the rest, so an interrupted
# pipeline can simply be launched again. The pipeline stops at the first
# failing step.
#
# Output: PIPELINE_LOGS_DIR/<timestamp>.json, with the commands, interpreter,
# status and duration of every step.

import argparse
import importlib
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from shared import configuration
from shared.experiment import ask_n_seeds, git_commit, timestamp, training_seeds, write_json

MULTI_INSTANCE = PROJECT_ROOT / "Inductive Multi-Instance Address Classification"
EGO_CENTRIC = PROJECT_ROOT / "Inductive Ego-Centric Address Classification"

# (name, script, arguments, methodologies the step belongs to)
STEPS = [
    ("splits",           PROJECT_ROOT / "create_splits.py",                   [],              {"multi-instance", "ego-centric"}),
    ("expand-mi",        MULTI_INSTANCE / "expansion.py",                     [],              {"multi-instance"}),
    ("expand-ego",       EGO_CENTRIC / "expansion.py",                        [],              {"ego-centric"}),
    ("serialize",        MULTI_INSTANCE / "serialize_graph.py",               [],              {"multi-instance"}),
    ("normalize",        MULTI_INSTANCE / "normalize_graphs.py",              [],              {"multi-instance"}),
    ("train-gnn",        MULTI_INSTANCE / "run_experiments.py",               ["--n-seeds"],   {"multi-instance"}),
    ("extract-features", EGO_CENTRIC / "extract_topological_features.py",     [],              {"ego-centric"}),
    ("peeling-ego",      EGO_CENTRIC / "peeling_analysis.py",                 [],              {"ego-centric"}),
    ("tune-ego",         EGO_CENTRIC / "tune_hyperparameters.py",             [],              {"ego-centric"}),
    ("train-ego",        EGO_CENTRIC / "train_models.py",                     ["--n-seeds"],   {"ego-centric"}),
    ("interpret-ego",    EGO_CENTRIC / "interpret_model.py",                  [],              {"ego-centric"}),
    # The training scripts already run the tests of their methodology; with both
    # methodologies they are run again at the end to compare them
    ("wilcoxon",         PROJECT_ROOT / "wilcoxon_tests.py",                  [],              {"both"}),
]

# Modules checked before starting; BlockSci only when a step that uses it is run
REQUIRED_MODULES = ["torch", "torch_geometric", "sklearn", "scipy", "pandas", "numpy", "networkx", "igraph",
                    "xgboost", "shap", "joblib", "matplotlib", "tqdm"]
BLOCKSCI_STEPS = {"splits", "expand-mi", "expand-ego"}


def selected_steps(methodology):
    wanted = {"multi-instance", "ego-centric", "both"} if methodology == "both" else {methodology}
    return [step for step in STEPS if step[3] & wanted]


def command(step, n_seeds):
    name, script, arguments, _ = step
    args = []
    for argument in arguments:
        args += [argument, str(n_seeds)] if argument == "--n-seeds" else [argument]
    return [sys.executable, str(script)] + args


def check_environment(steps):
    """Fails early if the current environment cannot import the modules of the steps."""
    modules = list(REQUIRED_MODULES)
    if any(step[0] in BLOCKSCI_STEPS for step in steps):
        modules.append("blocksci")
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if missing:
        print(f"Environment check failed ({sys.executable}): missing {' '.join(missing)}")
        print("Install requirements.txt and BlockSci in this environment (see the README).")
        sys.exit(1)


def choose_configuration(args):
    """Sets RSD_CONFIG_FILE for every step: none with --original, the given file with
    --config, or the parameters chosen by the user. A custom configuration is saved in
    its output directory."""
    os.environ.pop("RSD_CONFIG_FILE", None)
    importlib.reload(config)
    if args.config:
        os.environ["RSD_CONFIG_FILE"] = str(Path(args.config).expanduser().resolve())
        importlib.reload(config)
    elif not args.original:
        methodologies = {"multi-instance", "ego-centric"} if args.methodology == "both" else {args.methodology}
        params = configuration.ask_configuration(config.ORIGINAL_PARAMETERS, methodologies)
        if params != config.ORIGINAL_PARAMETERS:
            path = config.OUTPUT_DIR / configuration.identifier(params) / "configuration.json"
            configuration.save(params, path)
            os.environ["RSD_CONFIG_FILE"] = str(path)
        importlib.reload(config)

    if config.ORIGINAL_CONFIGURATION:
        os.environ.pop("RSD_CONFIG_FILE", None)
        print("\nConfiguration: original (config.py)")
        return "--original"
    configuration.save(config.PARAMETERS, config.CONFIGURATION_FILE)
    os.environ["RSD_CONFIG_FILE"] = str(config.CONFIGURATION_FILE)
    importlib.reload(config)
    print(f"\nConfiguration: {config.CONFIGURATION_ID}, saved in {config.CONFIGURATION_FILE}")
    return f"--config '{config.CONFIGURATION_FILE}'"


def main():
    parser = argparse.ArgumentParser(description="Run every step of one or both methodologies.")
    parser.add_argument("methodology", nargs="?", default="both", choices=["both", "multi-instance", "ego-centric"])
    parser.add_argument("--n-seeds", type=int, help="random-seed runs per configuration (asked if not given)")
    parser.add_argument("--from", dest="start", choices=[s[0] for s in STEPS], help="first step to run")
    parser.add_argument("--list", action="store_true", help="show the steps and their commands without running them")
    parser.add_argument("--skip-check", action="store_true", help="do not check the environment before starting")
    chosen = parser.add_mutually_exclusive_group()
    chosen.add_argument("--original", action="store_true",
                        help="original configuration of config.py (only the number of seeds is asked)")
    chosen.add_argument("--config", help="JSON file of a custom configuration (e.g. a saved configuration.json)")
    args = parser.parse_args()

    steps = selected_steps(args.methodology)
    if args.start:
        names = [s[0] for s in steps]
        if args.start not in names:
            parser.error(f"step '{args.start}' is not part of the {args.methodology} pipeline: {', '.join(names)}")
        steps = steps[names.index(args.start):]

    if args.list:
        for i, step in enumerate(steps, 1):
            print(f"{i}. {step[0]:17s} {' '.join(command(step, args.n_seeds or '<asked>'))}")
        return

    if not args.skip_check:
        check_environment(steps)

    config_flag = choose_configuration(args)
    trains = any("--n-seeds" in step[2] for step in steps)
    n_seeds = ask_n_seeds(args.n_seeds) if trains else None

    record = {
        "started": timestamp(),
        "git_commit": git_commit(config.PROJECT_ROOT),
        "methodology": args.methodology,
        "splits_file": str(config.SPLITS_FILE),
        "configuration": config.CONFIGURATION_ID,
        "configuration_file": config.CONFIG_FILE,
        "parameters": config.PARAMETERS,
        "n_seeds": n_seeds,
        "seeds": training_seeds(n_seeds) if n_seeds else None,
        "interpreter": sys.executable,
        "output_dir": str(config.RUN_DIR),
        "steps": [],
    }
    log_path = config.PIPELINE_LOGS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}.json"

    for i, step in enumerate(steps, 1):
        cmd = command(step, n_seeds)
        print(f"\n{'=' * 70}\n[{i}/{len(steps)}] {step[0]}\n{' '.join(cmd)}\n{'=' * 70}", flush=True)

        start = time.time()
        returncode = subprocess.run(cmd, cwd=str(Path(step[1]).parent)).returncode
        duration = time.time() - start

        record["steps"].append({"name": step[0], "command": cmd,
                                "returncode": returncode, "duration_s": round(duration, 1)})
        write_json(log_path, record)

        if returncode != 0:
            print(f"\nStep '{step[0]}' failed (exit code {returncode}). Fix the problem and relaunch with "
                  f"{config_flag} --from {step[0]}; completed work is kept.")
            sys.exit(returncode)
        print(f"\n{step[0]} finished in {duration / 60:.1f} min", flush=True)

    record["finished"] = timestamp()
    write_json(log_path, record)
    print(f"\nPipeline finished. Results in {config.RUN_DIR}, log in {log_path}")


if __name__ == "__main__":
    main()
