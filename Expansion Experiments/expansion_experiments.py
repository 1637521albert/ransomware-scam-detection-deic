# RANSOMWARE ADDRESSES DETECTION - EXPANSION EXPERIMENTS
#
# Reproduces the expansion experiments of Table 2 of the paper, run separately from
# the pipeline (requires BlockSci and RSD_BITCOINHEIST_CSV):
#   1. Configuration: expansion variants with k=2 hops and a random node width
#      limit w=5 -> graph density and share of nodes in the largest SCC
#   2. Hops: fw bw tx whole with w=5 and k = 1, 2, 3
#   3. Width: fw bw tx whole with k=2 and w = 1, 2, 5, 10
# The user chooses which experiments to run (or --experiments 1,2,3). The graphs are
# expanded by the parallel expansion of the multi-instance methodology
# (shared/parallel_expansion.py), with any variant and width-limit mode.
#
# Seed addresses: PER_CLASS (1,500) illicit BitcoinHeist addresses found in the chain
# and as many licit ones (3,000 in total), sampled with DATA_SEED
# (create_splits.select_seeds) and saved in OUTPUTS_DIR/seeds.csv. Every run expands
# all of them into a single graph.
#
# The runs are executed one after another, and every expansion is parallelized over
# all the cores (--workers, default: one per core, limited by the available memory;
# see GRAPH EXPANSION), as are the lookups of the seed addresses in the chain. Every
# run restarts from the same random seed, so each run is reproducible on its own, and
# the graphs and their metrics do not depend on the number of workers; the runtime
# does, so it is saved with the number of workers of the run.
# The metrics of each run are saved in OUTPUTS_DIR/runs/ and reused while they were
# computed by the same code with the same seed addresses (their fingerprint);
# otherwise the run is computed again. Every metric of every run is
# saved in OUTPUTS_DIR/<experiment>.csv, and the tables of the paper in
# OUTPUTS_DIR/tables/<experiment>.csv.

### Import libraries

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# One worker per core: single-threaded numerical libraries (before importing them)
from shared.threads import limit_threads

import argparse
import gc
import json
import random
import time
from array import array
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm

import config
from shared import parallel_expansion
from shared.experiment import code_fingerprint, digest, file_fingerprint, git_commit, timestamp, write_json
from shared.parallel import blocksci_workers
from shared.paths import get_required_env_path
from experiments_config import (
    SEED,
    PER_CLASS,
    LIMIT_MODE,
    EXPERIMENTS,
    OUTPUTS_DIR,
    SEEDS_FILE,
    RUNS_DIR,
    TABLES_DIR,
    run_name,
)

### Experiments selection

def get_experiments(answer=None):
    options = {"1": "configuration", "2": "hops", "3": "width"}

    if answer is None:
        if not sys.stdin.isatty():
            return list(options.values())
        print("\nExperiments to run (any combination, e.g. 1,3):\n---------------------------------")
        print("1. Configuration\n2. Hops\n3. Width")
        answer = input("Option: ")
    selected = [s.strip() for s in answer.split(",") if s.strip()]

    if not selected or any(s not in options for s in selected):
        raise ValueError("Choose one or more of 1, 2 and 3, separated by commas.")
    return list(dict.fromkeys(options[s] for s in selected))

### Collecting addresses

def select_seeds(chain):
    """The seed addresses of the experiments, selected once and saved, and selected again
    if they were selected by other code, BitcoinHeist data, chain or DATA_SEED (their
    fingerprint, create_splits.selection_fingerprint). The runs include the seeds in
    their own fingerprint, so they are computed again with new seeds."""
    import create_splits
    bitcoinheist_csv = get_required_env_path("RSD_BITCOINHEIST_CSV")
    summary_file = OUTPUTS_DIR / "seeds_summary.json"
    fingerprint = create_splits.selection_fingerprint(bitcoinheist_csv, "seeds", per_class=PER_CLASS)
    saved = create_splits.saved_fingerprint(SEEDS_FILE, summary_file)
    if saved == fingerprint:
        print(f"Seed addresses already selected with this code and data: {SEEDS_FILE}")
        return pd.read_csv(SEEDS_FILE)
    if SEEDS_FILE.exists():
        print(f"The saved seed addresses were selected by other code or data ({saved or 'unknown'} instead of "
              f"{fingerprint}): selecting them again")

    seeds, checks, info = create_splits.select_seeds(chain, bitcoinheist_csv, PER_CLASS)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    seeds.to_csv(SEEDS_FILE, index=False)
    # Written last: with its fingerprint, it marks the seeds as complete
    write_json(summary_file, {
        "created": timestamp(), "git_commit": git_commit(config.PROJECT_ROOT), "fingerprint": fingerprint,
        "source": "Expansion Experiments/expansion_experiments.py", **info, "checks": checks,
    })
    return seeds

def load_seeds(chain, seeds_df):
    # Seeds are expanded in the order of the seeds file, so every execution is the same
    seeds = {}
    for addr_str, label in tqdm(list(zip(seeds_df["address"], seeds_df["label"])), desc="Loading seed addresses"):
        addr_obj = chain.address_from_string(addr_str)
        if addr_obj is None:
            raise ValueError(f"Seed address {addr_str} is not in the chain")
        seeds[addr_str] = (addr_obj, int(label))
    n_illicit = sum(label for _, label in seeds.values())
    print(f"Seed addresses: {n_illicit} illicit + {len(seeds) - n_illicit} licit = {len(seeds)}\n")
    return seeds

## GRAPH EXPANSION
#
# Every run expands all the seed addresses into a single graph with the parallel
# expansion of the multi-instance methodology (shared/parallel_expansion.py), so the
# graphs are the ones of the multi-instance expansion with these seeds, any expansion
# variant (shared.expansion.VARIANTS) and width-limit mode (experiments_config), and do
# not depend on the number of workers. The width-limit sampling of every expansion step
# (one node choosing among its candidate transactions with the random node limit, or a
# whole hop with the random hop limit) is counted:
limit_stats = {}

def reset_limit_stats():
    limit_stats.clear()
    limit_stats.update({
        "nonempty_steps": 0,    # steps with at least one candidate
        "activated_steps": 0,   # steps with more candidates than the limit
        "candidates": 0,        # candidates available over all steps
        "retained": 0,          # candidates kept over all steps
    })

def count_sample(n_candidates, n_retained):
    limit_stats["nonempty_steps"] += n_candidates > 0
    limit_stats["activated_steps"] += n_candidates > n_retained
    limit_stats["candidates"] += n_candidates
    limit_stats["retained"] += n_retained

# Workers of every expansion (set in main)
WORKERS = {"n": 1}

def expand(variant, addresses, hops, txs, inputs, outputs, limit):
    # Forward-backward variants use the same number of forward and backward hops
    parallel_expansion.expand(variant, addresses, txs, inputs, outputs, for_hops=hops, back_hops=hops, hops=hops,
                              limit_mode=LIMIT_MODE, limit=limit)

## GRAPH METRICS

def graph_metrics(addresses, txs, inputs, outputs):
    # Same graph as serialize_graph.py: address nodes, transaction nodes, input
    # edges (addr -> tx), output edges (tx -> addr) and spent output edges
    # (spent tx -> spending tx) between transactions of the graph. The graph of
    # shared/parallel_expansion.py only keeps the inputs and outputs with an address.
    addr_index = {addr_str: i for i, addr_str in enumerate(addresses)}
    n_addr = len(addr_index)
    tx_index = {h: n_addr + i for i, h in enumerate(txs)}
    n_nodes = n_addr + len(tx_index)

    # Edges in compact integer arrays (8 bytes per endpoint)
    src = array("q")
    dst = array("q")

    for input_id, (addr_str, spending, *_) in inputs.items():
        a = addr_index.get(addr_str)
        t = tx_index.get(spending)
        if a is not None and t is not None:
            src.append(a)
            dst.append(t)

        # The spent output is part of the graph when it is one of its output edges
        if t is not None and input_id in outputs:
            spent = tx_index.get(input_id[0])
            if spent is not None:
                src.append(spent)
                dst.append(t)

    for (spending, _), (addr_str, *_) in outputs.items():
        t = tx_index.get(spending)
        a = addr_index.get(addr_str)
        if a is not None and t is not None:
            src.append(t)
            dst.append(a)

    n_edges = len(src)
    adjacency = sp.coo_matrix(
        (np.ones(n_edges, dtype=np.int8), (np.frombuffer(src, dtype=np.int64), np.frombuffer(dst, dtype=np.int64))),
        shape=(n_nodes, n_nodes),
    ).tocsr()
    n_tx = len(tx_index)
    del src, dst, addr_index, tx_index
    _, scc_labels = connected_components(adjacency, directed=True, connection='strong')
    largest_scc = int(np.bincount(scc_labels).max()) if n_nodes > 0 else 0

    return {
        "addr_nodes": n_addr,
        "tx_nodes": n_tx,
        "total_nodes": n_nodes,
        "total_edges": n_edges,
        "density": n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0,
        "avg_degree": n_edges / n_nodes if n_nodes > 0 else 0.0,
        "largest_scc": largest_scc,
        "largest_scc_pct": 100 * largest_scc / n_nodes if n_nodes > 0 else 0.0,
    }

## EXPERIMENTS

def run_path(variant, hops, limit):
    return RUNS_DIR / f"{run_name(variant, hops, limit)}.json"

def load_run(variant, hops, limit):
    with open(run_path(variant, hops, limit)) as f:
        return json.load(f)

# A saved run is reused only if it was computed by the same expansion and metrics code,
# with the same variant, hops, width limit, random seed and seed addresses
CODE = {Path(__file__).resolve(): ["expand", "count_sample", "graph_metrics", "run_experiment"],
        PROJECT_ROOT / "shared" / "expansion.py": None,
        PROJECT_ROOT / "shared" / "parallel_expansion.py": None}

def run_fingerprint(variant, hops, limit):
    return digest({"code": code_fingerprint(CODE), "variant": variant, "hops": hops, "limit": limit,
                   "limit_mode": LIMIT_MODE, "seed": SEED, "seeds": file_fingerprint([SEEDS_FILE])})

def run_is_current(variant, hops, limit):
    path = run_path(variant, hops, limit)
    return path.exists() and load_run(variant, hops, limit).get("fingerprint") == run_fingerprint(variant, hops, limit)

def run_experiment(variant, hops, limit, seeds):
    print(f"\nRunning {variant} | {hops} hops | {LIMIT_MODE} limit {limit} ({WORKERS['n']} workers)...", flush=True)

    # Every run starts from the same random state, so it can be reproduced on its own
    random.seed(SEED)
    np.random.seed(SEED)

    # The expansion adds the discovered addresses in place, so it works on a copy of the seeds
    addresses = {addr_str: (("seed", addr_str), label) for addr_str, (_, label) in seeds.items()}
    txs = {}
    inputs = {}
    outputs = {}

    reset_limit_stats()

    # The expansion only creates objects that live until the end of the run, so
    # the garbage collector is paused instead of repeatedly scanning them
    gc.collect()
    gc.disable()
    try:
        start = time.perf_counter()
        expand(variant, addresses, hops, txs, inputs, outputs, limit)
        runtime = time.perf_counter() - start
    finally:
        gc.enable()

    results = {
        "variant": variant,
        "hops": hops,
        "limit": limit,
        "n_seeds": len(seeds),
        "n_illicit_seeds": sum(label for _, label in seeds.values()),
        "runtime_s": runtime,
        "workers": WORKERS["n"],
        **graph_metrics(addresses, txs, inputs, outputs),
        "limit_activation_pct": 100 * limit_stats["activated_steps"] / limit_stats["nonempty_steps"] if limit_stats["nonempty_steps"] else 0.0,
        "candidate_retention_pct": 100 * limit_stats["retained"] / limit_stats["candidates"] if limit_stats["candidates"] else 0.0,
        "width_limit": dict(limit_stats),
        "fingerprint": run_fingerprint(variant, hops, limit),
    }

    write_json(run_path(variant, hops, limit), results)

    print(f"Finished {variant} | {hops} hops | {LIMIT_MODE} limit {limit}: "
          f"{results['total_nodes']} nodes, {results['total_edges']} edges, {runtime:.1f} s", flush=True)

    # The graph of a run is freed before the next one
    del addresses, txs, inputs, outputs
    gc.collect()
    return results

def run_pending(runs, seeds):
    """Runs the given runs that are not saved yet, one after another."""
    pending = [r for r in dict.fromkeys(runs) if not run_is_current(*r)]
    print(f"\n{len(set(runs)) - len(pending)} runs already saved with this code and seeds, {len(pending)} to run, "
          f"{WORKERS['n']} workers per expansion ({config.CPU_COUNT} cores)", flush=True)
    for variant, hops, limit in pending:
        run_experiment(variant, hops, limit, seeds)

def sweep_table(results, column, metrics):
    # Growth metrics are reported relative to the first value of the sweep, as in the paper
    df = pd.DataFrame(results)
    base = df.iloc[0]
    table = {
        "Total nodes": [f"x{v / base['total_nodes']:.2f}" for v in df["total_nodes"]],
        "Total edges": [f"x{v / base['total_edges']:.2f}" for v in df["total_edges"]],
        "Avg. degree": [f"{v:.2f}" for v in df["avg_degree"]],
        "Limit activation": [f"{v:.1f}%" for v in df["limit_activation_pct"]],
        "Candidate retention": [f"{v:.1f}%" for v in df["candidate_retention_pct"]],
        "Largest SCC": [f"{v:.2f}%" for v in df["largest_scc_pct"]],
        "Runtime": [f"x{v / base['runtime_s']:.2f}" for v in df["runtime_s"]],
    }
    return pd.DataFrame({m: table[m] for m in metrics}, index=[f"{column}={v}" for v in df[column]]).T

def all_metrics(results):
    """Every metric of the runs, with the width-limit counters as columns."""
    rows = []
    for r in results:
        row = {k: v for k, v in r.items() if k != "width_limit"}
        row.update({f"width_limit_{k}": v for k, v in r["width_limit"].items()})
        rows.append(row)
    return pd.DataFrame(rows)

def paper_table(experiment, results):
    if experiment == "configuration":
        return pd.DataFrame({
            "Density": [f"{r['density']:.3e}" for r in results],
            "%SCC_max": [f"{r['largest_scc_pct']:.3f}%" for r in results],
        }, index=[r["variant"] for r in results])
    if experiment == "hops":
        return sweep_table(results, "hops",
            ["Total nodes", "Total edges", "Avg. degree", "Largest SCC", "Runtime"])
    return sweep_table(results, "limit",
        ["Total nodes", "Total edges", "Avg. degree", "Limit activation", "Candidate retention", "Largest SCC", "Runtime"])

def main():
    limit_threads()
    parser = argparse.ArgumentParser(description="Expansion experiments of Table 2 of the paper.")
    parser.add_argument("--experiments", help="experiments to run, e.g. 1,3 (1 configuration, 2 hops, 3 width; "
                                              "asked if not given)")
    parser.add_argument("--workers", type=int, help="parallel workers of every expansion (default: one per core, "
                                                    "limited by the available memory; 1 runs every expansion in a "
                                                    "single process)")
    args = parser.parse_args()
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be at least 1")

    if not config.ORIGINAL_CONFIGURATION:
        raise SystemExit("The expansion experiments use the original configuration: unset RSD_CONFIG_FILE.")

    experiments = get_experiments(args.experiments)

    import blocksci
    chain = blocksci.Blockchain(str(get_required_env_path("RSD_BLOCKSCI_CONFIG")))
    seeds = load_seeds(chain, select_seeds(chain))

    # The forked workers inherit the chain; every width-limit sampling is counted
    WORKERS["n"] = args.workers or blocksci_workers()
    parallel_expansion.configure(chain, WORKERS["n"], on_sample=count_sample)
    tx_rebuild = parallel_expansion.detect_tx_by_index(next(tx for addr, _ in seeds.values() for tx in addr.txes))
    print(f"Transactions rebuilt in the workers from their index with {tx_rebuild}")
    write_json(OUTPUTS_DIR / "experiments_config.json", {
        "created": timestamp(),
        "git_commit": git_commit(config.PROJECT_ROOT),
        "experiments": {name: [{"variant": v, "hops": h, "limit": w} for v, h, w in EXPERIMENTS[name]]
                        for name in EXPERIMENTS},
        "limit_mode": LIMIT_MODE,
        "per_class": PER_CLASS,
        "n_seeds": len(seeds),
        "seed": SEED,
        "random_state": "random.seed(SEED) before every run",
        "workers": WORKERS["n"],
        "tx_rebuild": tx_rebuild,
    })

    # The runs shared by several experiments (fw bw tx whole, k=2, w=5) are only run once
    run_pending([run for experiment in experiments for run in EXPERIMENTS[experiment]], seeds)

    tables = {}
    for experiment in experiments:
        results = [load_run(variant, hops, limit) for variant, hops, limit in EXPERIMENTS[experiment]]
        all_metrics(results).to_csv(OUTPUTS_DIR / f"{experiment}.csv", index=False)

        table = paper_table(experiment, results)
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        table.to_csv(TABLES_DIR / f"{experiment}.csv")
        tables[experiment] = table

    n_illicit = sum(label for _, label in seeds.values())
    for experiment, table in tables.items():
        print("\n========================================")
        print(f"{experiment.capitalize()} experiment ({n_illicit} illicit + {len(seeds) - n_illicit} licit seeds)")
        print("========================================")
        print(table.to_string())

    print(f"\nResults saved at: {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
