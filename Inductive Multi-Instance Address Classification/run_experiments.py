# STEP 5 (Methodology I) - ALL GNN EXPERIMENTS (PyTorch environment)
#
# Asks how many random-seed runs to execute, trains every configuration of
# config.GNN_ARCHITECTURES x config.GNN_LAYERS with that many seeds (one
# model.py process per run), aggregates the results and runs the Wilcoxon tests
# (wilcoxon_tests.py):
#
#   python run_experiments.py                  # asks the number of seeds
#   python run_experiments.py --n-seeds 10     # without asking
#   python run_experiments.py --workers 8      # parallel runs (default: as many as fit in memory)
#   python run_experiments.py --aggregate-only # only rebuild the result tables and tests
#
# All the cores are used: as many runs as fit in the available memory train at
# the same time (config.GNN_PARALLEL_RUNS), with CPU_COUNT // runs threads each.
# Completed runs are skipped, so an interrupted execution can simply be relaunched.
#
# Output in MULTI_INSTANCE_RESULTS_DIR:
#   experiment_config.json   full configuration of the experiment (data, expansion,
#                            normalization, training, evaluation, seeds, resources)
#   runs/<run>/              one folder per run (see model.py), with train.log
#   gnn_runs.csv             one row per run
#   gnn_summary.csv          mean, standard deviation and median per configuration

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import wilcoxon_tests
from shared.experiment import ask_n_seeds, experiment_snapshot, read_json, training_seeds, write_json
from shared.stats import summarize_runs

MODEL_SCRIPT = Path(__file__).resolve().parent / "model.py"
GRAPHS_DIR = config.MULTI_INSTANCE_GRAPHS_DIR
RESULTS_DIR = config.MULTI_INSTANCE_RESULTS_DIR
RUNS_DIR = RESULTS_DIR / "runs"

METRICS = ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_auc_roc", "epochs", "best_epoch", "time_per_epoch_s"]


def run_name(arch, layers, seed):
    return f"{arch}-{layers}l-seed{seed}"


def available_memory_bytes():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def parallel_runs(requested):
    """Number of simultaneous training runs: the requested one, or as many as fit in memory."""
    if requested != "auto":
        return max(1, int(requested))
    graphs_size = sum((GRAPHS_DIR / split / "graph_normalized.pth").stat().st_size
                      for split in config.SPLITS if (GRAPHS_DIR / split / "graph_normalized.pth").exists())
    memory = available_memory_bytes()
    if not graphs_size or not memory:
        return 1
    per_run = graphs_size * config.GNN_MEMORY_FACTOR
    runs = int(0.9 * memory // per_run)
    print(f"Available memory {memory / 2**30:.1f} GiB, estimated {per_run / 2**30:.1f} GiB per run")
    return max(1, min(config.CPU_COUNT, runs))


def launch(arch, layers, seed, threads):
    run_dir = RUNS_DIR / run_name(arch, layers, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    with open(run_dir / "train.log", "w") as log:
        process = subprocess.run(
            [sys.executable, str(MODEL_SCRIPT), "--arch", arch, "--layers", str(layers), "--seed", str(seed),
             "--threads", str(threads), "--verbose"],
            stdout=log, stderr=subprocess.STDOUT, env=env,
        )
    return run_name(arch, layers, seed), process.returncode


def train_missing(runs, workers, threads):
    pending = [r for r in runs if not (RUNS_DIR / run_name(*r) / "result.json").exists()]
    print(f"{len(runs) - len(pending)} of {len(runs)} runs already completed, {len(pending)} to train "
          f"({workers} parallel runs x {threads} threads).")

    failed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(launch, *r, threads) for r in pending]
        for i, future in enumerate(as_completed(futures), 1):
            name, code = future.result()
            status = "done" if code == 0 else f"FAILED (exit code {code}, see {RUNS_DIR / name / 'train.log'})"
            print(f"[{i}/{len(pending)}] {name}: {status}")
            if code != 0:
                failed.append(name)
    return failed


def save_experiment_config(archs, layers, seeds, workers, threads):
    data = {}
    for key, path in [("expansion", GRAPHS_DIR / "expansion_config.json"),
                      ("normalization", GRAPHS_DIR / "normalization.json")]:
        data[key] = read_json(path) if path.exists() else None

    write_json(RESULTS_DIR / "experiment_config.json", experiment_snapshot(
        extra={
            "methodology": "inductive multi-instance address classification",
            "data": data,
            "architectures": archs,
            "layers": layers,
            "n_seeds": len(seeds),
            "seeds": seeds,
            "training": config.GNN_TRAINING,
            "evaluation": {
                "selection": "epoch with the highest validation accuracy",
                "test_split": "labeled seed addresses of the test graph",
                "prediction": "illicit if the predicted probability is >= 0.5",
                "metrics": ["accuracy", "precision", "recall", "f1", "auc_roc"],
                "statistical_tests": "wilcoxon_tests.py",
            },
            "resources": {"parallel_runs": workers, "threads_per_run": threads, "cpu_count": config.CPU_COUNT},
        },
        modules=("torch", "torch_geometric", "sklearn", "numpy", "pandas"),
    ))


def aggregate():
    results = [read_json(p) for p in sorted(RUNS_DIR.glob("*/result.json"))]
    if not results:
        print("No completed runs to aggregate.")
        return

    drop = ("history", "hyperparameters", "evaluation", "data", "libraries")
    runs = pd.DataFrame([{k: v for k, v in r.items() if k not in drop} for r in results])
    runs = runs.sort_values(["arch", "layers", "seed"])
    runs.to_csv(RESULTS_DIR / "gnn_runs.csv", index=False)

    summary = summarize_runs(runs, ["arch", "layers"], METRICS)
    summary.to_csv(RESULTS_DIR / "gnn_summary.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n" + summary[["arch", "layers", "n_runs", "test_f1_mean", "test_f1_std", "test_f1_median",
                          "test_auc_roc_mean", "test_auc_roc_median", "epochs_mean"]].round(4).to_string(index=False))
    print(f"\nResults saved at: {RESULTS_DIR}")
    wilcoxon_tests.run_all()


def main():
    parser = argparse.ArgumentParser(description="Train all GNN configurations with several seeds and aggregate the results.")
    parser.add_argument("--n-seeds", type=int, help="random-seed runs per configuration (asked if not given)")
    parser.add_argument("--workers", default=config.GNN_PARALLEL_RUNS, help="parallel training processes, or 'auto'")
    parser.add_argument("--archs", nargs="+", default=config.GNN_ARCHITECTURES, help="subset of architectures")
    parser.add_argument("--layers", nargs="+", type=int, default=config.GNN_LAYERS, help="subset of numbers of layers")
    parser.add_argument("--aggregate-only", action="store_true", help="only rebuild the result tables and tests")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    if not args.aggregate_only:
        seeds = training_seeds(ask_n_seeds(args.n_seeds))
        runs = [(a, l, s) for a in args.archs for l in args.layers for s in seeds]
        workers = min(parallel_runs(args.workers), len(runs))
        threads = max(1, config.CPU_COUNT // workers)
        save_experiment_config(args.archs, args.layers, seeds, workers, threads)
        failed = train_missing(runs, workers, threads)

    aggregate()
    if failed:
        print(f"\n{len(failed)} runs failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
