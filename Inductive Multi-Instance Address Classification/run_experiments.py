# STEP 5 (Methodology I) - ALL GNN EXPERIMENTS
#
# Asks how many random-seed runs to execute, trains every configuration of
# config.GNN_ARCHITECTURES x config.GNN_LAYERS with that many seeds (one
# model.py process per run), aggregates the results and runs the Wilcoxon tests
# (wilcoxon_tests.py):
#
#   python run_experiments.py                  # asks the number of seeds
#   python run_experiments.py --n-seeds 10     # without asking
#   python run_experiments.py --workers 8      # 8 parallel runs (default: scheduled by memory)
#   python run_experiments.py --aggregate-only # only rebuild the result tables and tests
#
# All the cores and the memory are used:
#   - The graphs are loaded once, and every run is a process forked from this one, so
#     all the runs share a single copy of the graphs in memory (they only read them).
#   - The memory of every configuration is measured first with one epoch (saved in
#     memory_probe.json): its peak minus the graphs is what one of its runs needs.
#   - The runs are started in order of cost, the configurations with the longest
#     epoch first (so the longest runs do not start last), whenever their memory
#     (times GNN_MEMORY_MARGIN) and their threads are available.
#   - Every configuration has a fixed number of threads per run: the cores divided
#     by the runs of that configuration that fit in memory. A run gives the same
#     result with the same threads (recorded in result.json), so its result does
#     not depend on the scheduling.
# With --workers N (or GNN_PARALLEL_RUNS = N), N runs train at a time with
# CPU_COUNT // N threads each, without measuring the memory.
# Completed runs are skipped, so an interrupted execution can simply be relaunched.
#
# Output in MULTI_INSTANCE_RESULTS_DIR:
#   experiment_config.json   full configuration of the experiment (data, expansion,
#                            normalization, training, evaluation, seeds, resources)
#   runs/<run>/              one folder per run (see model.py), with train.log
#   gnn_runs.csv             one row per run
#   gnn_summary.csv          mean, standard deviation and median per configuration

import argparse
import gc
import json
import multiprocessing
import os
import subprocess
import sys
from multiprocessing.connection import wait
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import wilcoxon_tests
from shared.experiment import ask_n_seeds, experiment_snapshot, graph_fingerprints, read_json, training_seeds, write_json
from shared.parallel import available_memory_bytes
from shared.stats import summarize_runs

MODEL_SCRIPT = Path(__file__).resolve().parent / "model.py"
GRAPHS_DIR = config.MULTI_INSTANCE_GRAPHS_DIR
RESULTS_DIR = config.MULTI_INSTANCE_RESULTS_DIR
RUNS_DIR = RESULTS_DIR / "runs"
PROBE_FILE = RESULTS_DIR / "memory_probe.json"
NORMALIZATION_FILE = GRAPHS_DIR / "normalization.json"

METRICS = ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_auc_roc", "epochs", "best_epoch", "time_per_epoch_s"]


def run_name(arch, layers, seed):
    return f"{arch}-{layers}l-seed{seed}"


def completed(arch, layers, seed):
    """A run is completed if it has a result trained on the current graphs, hyperparameters and code."""
    path = RUNS_DIR / run_name(arch, layers, seed) / "result.json"
    import model
    return path.exists() and model.is_current(read_json(path))


def probe_configurations(configurations, seed):
    """Peak memory of one epoch of every configuration, measured once per graphs and saved."""
    probes = read_json(PROBE_FILE) if PROBE_FILE.exists() else {}
    graphs = graph_fingerprints(NORMALIZATION_FILE)
    for arch, layers in configurations:
        key = f"{arch}-{layers}"
        # Probes saved by earlier versions do not have the memory of the graphs, and probes
        # of other graphs do not give the memory of these ones
        if key in probes and "graph_bytes" in probes[key] and probes[key].get("graphs") == graphs:
            continue
        print(f"Measuring the memory of one {key} epoch...", flush=True)
        output = subprocess.run(
            [sys.executable, str(MODEL_SCRIPT), "--arch", arch, "--layers", str(layers), "--seed", str(seed),
             "--threads", str(config.CPU_COUNT), "--probe"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        lines = [l for l in output.stdout.splitlines() if l.startswith("PROBE ")]
        if output.returncode != 0 or not lines:
            print(output.stdout[-2000:])
            raise RuntimeError(f"The memory probe of {key} failed.")
        probes[key] = dict(json.loads(lines[-1][len("PROBE "):]), graphs=graphs)
        print(f"  {key}: {probes[key]['peak_rss_bytes'] / 2**30:.1f} GiB peak, of which "
              f"{probes[key]['graph_bytes'] / 2**30:.1f} GiB of graphs; one epoch in "
              f"{probes[key]['epoch_time_s']:.0f} s with {config.CPU_COUNT} threads", flush=True)
        write_json(PROBE_FILE, probes)
    return probes


# Graphs shared by the forked runs (loaded once by load_shared_graphs)
_SHARED = {"graphs": None}


def load_shared_graphs():
    """Loads the graphs once in this process, before forking the runs. The objects
    created so far are moved out of the garbage collector's reach, so the runs do not
    copy their memory pages just by inspecting them."""
    import torch
    import model
    # This process does no computation: no thread pool is started before forking
    torch.set_num_threads(1)
    _SHARED["graphs"] = model.load_graphs()
    gc.collect()
    gc.freeze()
    return model.graph_bytes(_SHARED["graphs"])


def run_forked(arch, layers, seed, threads, log_path):
    """One run in a forked process: its output goes to its train.log."""
    fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    sys.stdout = os.fdopen(1, "w", buffering=1)
    sys.stderr = os.fdopen(2, "w", buffering=1)
    import model
    model.train_and_save(arch, layers, seed, threads, graphs=_SHARED["graphs"], verbose=True)


def plan(pending, probes, shared_bytes):
    """Order, memory and threads of the pending runs (see the top of this file)."""
    memory = available_memory_bytes()
    budget = 0.9 * memory - shared_bytes
    plans = {}
    for arch, layers in {(a, l) for a, l, _ in pending}:
        probe = probes[f"{arch}-{layers}"]
        need = max(1, probe["peak_rss_bytes"] - probe["graph_bytes"]) * config.GNN_MEMORY_MARGIN
        n_runs = sum(1 for a, l, _ in pending if (a, l) == (arch, layers))
        fit = max(1, min(int(budget // need), n_runs, config.CPU_COUNT))
        plans[(arch, layers)] = {"memory_bytes": need, "threads": max(1, config.CPU_COUNT // fit),
                                 "epoch_time_s": probe["epoch_time_s"], "fit_alone": fit}
    order = sorted(pending, key=lambda r: (-plans[r[:2]]["epoch_time_s"], -plans[r[:2]]["memory_bytes"], r))
    print(f"Available memory {memory / 2**30:.1f} GiB: {shared_bytes / 2**30:.1f} GiB of graphs shared by the runs, "
          f"{max(0, budget) / 2**30:.1f} GiB for the runs", flush=True)
    for (arch, layers), p in sorted(plans.items(), key=lambda item: -item[1]["epoch_time_s"]):
        print(f"  {arch}-{layers}: {p['memory_bytes'] / 2**30:.1f} GiB and {p['threads']} threads per run "
              f"(up to {p['fit_alone']} at a time), one epoch in {p['epoch_time_s']:.0f} s with all the cores", flush=True)
    return order, plans, budget


def launch(arch, layers, seed, threads):
    """A forked run sharing the loaded graphs."""
    run_dir = RUNS_DIR / run_name(arch, layers, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Started {run_name(arch, layers, seed)} with {threads} threads (log: {run_dir / 'train.log'})", flush=True)
    process = multiprocessing.get_context("fork").Process(
        target=run_forked, args=(arch, layers, seed, threads, run_dir / "train.log"))
    process.start()
    return process


def train_missing(runs, workers, probes):
    """Trains the runs without a result: scheduled by memory (workers None) or with a
    fixed number of parallel runs."""
    pending = [r for r in runs if not completed(*r)]
    print(f"{len(runs) - len(pending)} of {len(runs)} runs already completed on these graphs, "
          f"{len(pending)} to train.", flush=True)
    if not pending:
        return [], {}
    shared_bytes = load_shared_graphs()
    if workers is None:
        order, plans, budget = plan(pending, probes, shared_bytes)
    else:
        # Fixed number of parallel runs, in the order of the configurations
        threads = max(1, config.CPU_COUNT // workers)
        order = pending
        plans = {r[:2]: {"memory_bytes": 0, "threads": threads} for r in pending}
        budget = float("inf")

    running, failed, done = {}, [], 0
    queue = list(order)
    while queue or running:
        # Starts every queued run (in order) whose memory and threads are available
        used_memory = sum(plans[r[:2]]["memory_bytes"] for r, _ in running.values())
        used_threads = sum(plans[r[:2]]["threads"] for r, _ in running.values())
        slots = workers if workers is not None else float("inf")
        for r in list(queue):
            p = plans[r[:2]]
            fits = (used_memory + p["memory_bytes"] <= budget and used_threads + p["threads"] <= config.CPU_COUNT
                    and len(running) < slots)
            # A run always starts if nothing else is running, so that the queue advances
            if fits or not running:
                process = launch(*r, p["threads"])
                running[process.sentinel] = (r, process)
                queue.remove(r)
                used_memory += p["memory_bytes"]
                used_threads += p["threads"]
        for sentinel in wait(list(running)):
            r, process = running.pop(sentinel)
            process.join()
            done += 1
            succeeded = completed(*r)
            status = "done" if succeeded else f"FAILED (see {RUNS_DIR / run_name(*r) / 'train.log'})"
            print(f"[{done}/{len(pending)}] {run_name(*r)}: {status}", flush=True)
            if not succeeded:
                failed.append(run_name(*r))
    return failed, {f"{a}-{l}": p for (a, l), p in plans.items()}


def save_experiment_config(archs, layers, seeds, workers, plans, probes=None):
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
                "selection": "epoch with the lowest validation loss",
                "test_split": "labeled seed addresses of the test graph",
                "prediction": "illicit if the predicted probability is >= 0.5",
                "metrics": ["accuracy", "precision", "recall", "f1", "auc_roc"],
                "statistical_tests": "wilcoxon_tests.py",
            },
            "resources": {"scheduling": "by memory" if workers is None else f"{workers} parallel runs",
                          "per_configuration": plans, "cpu_count": config.CPU_COUNT,
                          "graphs": "loaded once and shared by the forked runs", "memory_probes": probes},
        },
        modules=("torch", "torch_geometric", "sklearn", "numpy", "pandas"),
    ))


def aggregate():
    # Only the runs trained on the current graphs, hyperparameters and code
    results = [read_json(p) for p in sorted(RUNS_DIR.glob("*/result.json"))]
    import model
    current = [r for r in results if model.is_current(r)]
    if len(current) < len(results):
        print(f"{len(results) - len(current)} runs trained on other graphs are left out of the results.")
    results = current
    if not results:
        # Tables of other graphs would be read by the Wilcoxon tests as if they were current
        for name in ("gnn_runs.csv", "gnn_summary.csv"):
            if (RESULTS_DIR / name).exists():
                (RESULTS_DIR / name).unlink()
        print("No completed runs trained on the current graphs, hyperparameters and code to aggregate.")
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
        pending_configs = sorted({(a, l) for a, l, s in runs if not completed(a, l, s)})
        workers, probes = None, None
        if args.workers != "auto":
            workers = max(1, int(args.workers))
        elif pending_configs:
            probes = probe_configurations(pending_configs, seeds[0])
        failed, plans = train_missing(runs, workers, probes)
        if plans:
            save_experiment_config(args.archs, args.layers, seeds, workers, plans, probes)

    aggregate()
    if failed:
        print(f"\n{len(failed)} runs failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
