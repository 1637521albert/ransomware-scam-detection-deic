# RANSOMWARE ADDRESSES DETECTION - EXPANSION EXPERIMENTS
#
# Measures the structure and cost of the expanded graphs, run separately from the
# pipeline (requires BlockSci and RSD_BITCOINHEIST_CSV):
#   1. Configuration: expansion variants with k=2 hops and a random node width
#      limit w=5 -> graph density and share of nodes in the largest SCC and WCC
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
#
# The runtimes of the runs are compared with each other, so they are measured in the
# same context (see TIMING): all the runs of the selected experiments are timed in one
# session (one execution), started on an idle machine, and every run is expanded first
# without timing (warm-up), so it is timed with the blockchain data it reads in the page
# cache, whatever ran before it. The load of the machine during every timed expansion
# is saved with the run.
#
# The metrics of each run are saved in OUTPUTS_DIR/runs/ and reused while they were
# computed by the same code with the same seed addresses (their fingerprint) and all
# the runs of the selected experiments were timed in one session with the current timing
# settings and workers; otherwise all of them are run again. Every metric of every run is
# saved in OUTPUTS_DIR/<experiment>.csv, and a summary table of every experiment in
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
import platform
import random
import statistics
import time
from array import array
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm

import config
from shared import machine, parallel_expansion
from shared.experiment import code_fingerprint, digest, file_fingerprint, git_commit, timestamp, write_json
from shared.parallel import blocksci_workers
from shared.paths import get_required_env_path
from experiments_config import (
    SEED,
    PER_CLASS,
    LIMIT_MODE,
    EXPERIMENTS,
    WARMUP_RUNS,
    TIMED_RUNS,
    IDLE_CHECK_S,
    MAX_OTHER_CORES,
    MIN_AVAILABLE_MEMORY_PCT,
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
    # The seed addresses keep their label; the addresses found by the expansion get -1
    seed_nodes = np.array([addr_index[a] for a, (_, label) in addresses.items() if label != -1], dtype=np.int64)
    del src, dst, addr_index, tx_index
    _, scc_labels = connected_components(adjacency, directed=True, connection='strong')
    largest_scc = int(np.bincount(scc_labels).max()) if n_nodes > 0 else 0
    del scc_labels

    # Weakly connected components: the GNNs also pass messages along the reverse of every
    # relation, so information flows between any two nodes of the same WCC
    n_wcc, wcc_labels = connected_components(adjacency, directed=True, connection='weak')
    wcc_sizes = np.bincount(wcc_labels) if n_nodes > 0 else np.zeros(0, dtype=np.int64)
    largest_wcc = int(wcc_sizes.max()) if n_nodes > 0 else 0
    seed_wccs = wcc_labels[seed_nodes]
    seeds_in_largest_wcc = int((seed_wccs == wcc_sizes.argmax()).sum()) if n_nodes > 0 else 0
    n_seed_wccs = int(np.unique(seed_wccs).size)

    return {
        "addr_nodes": n_addr,
        "tx_nodes": n_tx,
        "total_nodes": n_nodes,
        "total_edges": n_edges,
        "density": n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0,
        "avg_degree": n_edges / n_nodes if n_nodes > 0 else 0.0,
        "largest_scc": largest_scc,
        "largest_scc_pct": 100 * largest_scc / n_nodes if n_nodes > 0 else 0.0,
        "n_wcc": int(n_wcc),
        "largest_wcc": largest_wcc,
        "largest_wcc_pct": 100 * largest_wcc / n_nodes if n_nodes > 0 else 0.0,
        # The seeds in the largest WCC, and the number of WCCs the seeds are spread over
        "seeds_in_largest_wcc_pct": 100 * seeds_in_largest_wcc / len(seed_nodes) if len(seed_nodes) else 0.0,
        "n_seed_wccs": n_seed_wccs,
    }

## EXPERIMENTS

def run_path(variant, hops, limit):
    return RUNS_DIR / f"{run_name(variant, hops, limit)}.json"

def load_run(variant, hops, limit):
    with open(run_path(variant, hops, limit)) as f:
        return json.load(f)

# A saved run is reused only if it was computed by the same expansion and metrics code,
# with the same variant, hops, width limit, random seed and seed addresses
CODE = {Path(__file__).resolve(): ["expand", "count_sample", "graph_metrics", "expand_seeds", "run_experiment"],
        PROJECT_ROOT / "shared" / "expansion.py": None,
        PROJECT_ROOT / "shared" / "parallel_expansion.py": None}

def run_fingerprint(variant, hops, limit):
    return digest({"code": code_fingerprint(CODE), "variant": variant, "hops": hops, "limit": limit,
                   "limit_mode": LIMIT_MODE, "seed": SEED, "seeds": file_fingerprint([SEEDS_FILE])})

def run_is_current(variant, hops, limit):
    path = run_path(variant, hops, limit)
    return path.exists() and load_run(variant, hops, limit).get("fingerprint") == run_fingerprint(variant, hops, limit)

## TIMING
#
# The runtimes of the runs are compared with each other, so they must be measured in the
# same context. Three things would otherwise change between the runs:
#   - the page cache: a run that follows another one finds much of the blockchain data
#     the previous one read already in memory, while the first one reads it from disk.
#     Every run is expanded WARMUP_RUNS times without timing first (the expansions of a
#     run are identical, from the same random state), so every timed expansion finds its
#     own data in the cache, whatever ran before it;
#   - the session: runs timed in different executions ran under different loads and
#     caches. All the runs of the selected experiments are timed in one session, and the
#     saved runs are only reused if all of them were timed in the same session;
#   - other work on the machine: a session only starts on an idle machine (check_idle),
#     and the cores used by the other processes during every timed expansion are saved
#     with the run.

def new_session(workers, idle_check):
    return {"id": timestamp(), "host": platform.node(), "cores": config.CPU_COUNT, "workers": workers,
            "warmup_runs": WARMUP_RUNS, "timed_runs": TIMED_RUNS, "git_commit": git_commit(config.PROJECT_ROOT),
            "idle_check": idle_check}

def session_matches(session, workers):
    # A saved session can be reused with the current timing settings and workers
    return (session.get("host") == platform.node() and session.get("workers") == workers
            and session.get("warmup_runs") == WARMUP_RUNS and session.get("timed_runs") == TIMED_RUNS)

def saved_session(runs, workers):
    """The timing session of the given runs, if all of them are saved with the current code
    and seeds and were timed in one session with the current settings; otherwise None and
    the reason."""
    stale = [run_name(*r) for r in runs if not run_is_current(*r)]
    if stale:
        return None, f"{len(stale)} of {len(runs)} runs are not saved with this code and seeds"
    timings = [load_run(*r).get("timing") for r in runs]
    if any(t is None for t in timings):
        return None, "some runs were saved without the timing of a session"
    sessions = {t["session"]["id"] for t in timings}
    if len(sessions) > 1:
        return None, f"the runs were timed in {len(sessions)} different sessions ({', '.join(sorted(sessions))})"
    session = timings[0]["session"]
    if not session_matches(session, workers):
        return None, (f"the runs were timed on {session.get('host')} with {session.get('workers')} workers, "
                      f"{session.get('warmup_runs')} warm-up and {session.get('timed_runs')} timed expansions "
                      f"(now {platform.node()}, {workers}, {WARMUP_RUNS} and {TIMED_RUNS})")
    return session, None

def check_idle(allow_busy):
    """Measures the load of the machine before a session, which only starts if it is idle
    (or with allow_busy). Returns the measured load (None without /proc)."""
    print(f"\nChecking that the machine is idle ({IDLE_CHECK_S} s)...", flush=True)
    load = machine.machine_load(IDLE_CHECK_S)
    if load is None:
        print("The load of the machine cannot be measured here (no /proc): make sure that nothing else runs "
              "during the session, or the runtimes will not be comparable")
        return None
    print(f"Other processes: {load['other_cores']:.2f} cores on average, "
          f"{load['mem_available_pct']:.0f}% of the memory available ({load['mem_available_gb']:.0f} GB)")
    busy = load["other_cores"] > MAX_OTHER_CORES or load["mem_available_pct"] < MIN_AVAILABLE_MEMORY_PCT
    if busy:
        print(f"The machine is not idle (at most {MAX_OTHER_CORES} cores used by other processes and at least "
              f"{MIN_AVAILABLE_MEMORY_PCT}% of the memory available). Busiest processes:")
        for p in load["busiest_processes"]:
            print(f"  pid {p['pid']}: {p['cores']:.2f} cores, {p['rss_gb']:.1f} GB  {p['command']}")
        if not allow_busy:
            raise SystemExit("The runtimes would not be comparable: wait until these processes finish, or pass "
                             "--allow-busy to time the runs anyway.")
        print("Timing the runs anyway (--allow-busy)")
    load["busy"] = busy
    return load

def expand_seeds(variant, hops, limit, seeds):
    """One expansion of all the seeds from the same random state: the graph and the use of
    the machine during the expansion."""
    # Every expansion starts from the same random state, so it can be reproduced on its own
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
        start = machine.snapshot()
        expand(variant, addresses, hops, txs, inputs, outputs, limit)
        usage = machine.usage_since(start)
    finally:
        gc.enable()
    usage["runtime_s"] = usage.pop("wall_s")
    return (addresses, txs, inputs, outputs), usage

def run_experiment(variant, hops, limit, seeds, session):
    print(f"\nRunning {variant} | {hops} hops | {LIMIT_MODE} limit {limit} ({WORKERS['n']} workers, "
          f"{WARMUP_RUNS} warm-up + {TIMED_RUNS} timed expansions)...", flush=True)

    expansions = []
    sizes = set()
    graph = None
    for i in range(WARMUP_RUNS + TIMED_RUNS):
        warmup = i < WARMUP_RUNS
        # The graph of the previous expansion is freed before the next one
        graph = None
        graph, usage = expand_seeds(variant, hops, limit, seeds)
        sizes.add(tuple(len(part) for part in graph))
        usage["warmup"] = warmup
        expansions.append(usage)
        other = "unknown" if usage["other_cores"] is None else f"{usage['other_cores']:.2f}"
        print(f"{'Warm-up' if warmup else 'Timed'} expansion {i + 1 - (0 if warmup else WARMUP_RUNS)}: "
              f"{usage['runtime_s']:.1f} s (other processes: {other} cores)", flush=True)
    if len(sizes) > 1:
        raise RuntimeError(f"The expansions of the run gave different graphs ({sorted(sizes)}): "
                           f"they are not reproducible, so their runtimes are not comparable")

    timed = [e for e in expansions if not e["warmup"]]
    runtime = statistics.median(e["runtime_s"] for e in timed)
    addresses, txs, inputs, outputs = graph
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
        "timing": {
            "session": session,
            "runtimes_s": [e["runtime_s"] for e in timed],
            "warmup_runtimes_s": [e["runtime_s"] for e in expansions if e["warmup"]],
            "max_other_cores": max((e["other_cores"] for e in timed if e["other_cores"] is not None), default=None),
            "expansions": expansions,
        },
        "fingerprint": run_fingerprint(variant, hops, limit),
    }

    write_json(run_path(variant, hops, limit), results)

    print(f"Finished {variant} | {hops} hops | {LIMIT_MODE} limit {limit}: "
          f"{results['total_nodes']} nodes, {results['total_edges']} edges, {runtime:.1f} s", flush=True)

    # The graph of a run is freed before the next one
    del graph, addresses, txs, inputs, outputs
    gc.collect()
    return results

def run_session(runs, seeds, allow_busy):
    """Times all the given runs in one session, unless all of them are already saved from
    one session with the current code, seeds and timing settings. Returns the session."""
    runs = list(dict.fromkeys(runs))
    session, reason = saved_session(runs, WORKERS["n"])
    if session is not None:
        print(f"\nAll {len(runs)} runs already saved with this code and seeds, timed in session {session['id']} "
              f"({session['workers']} workers)", flush=True)
        return session
    print(f"\nTiming all {len(runs)} runs in a new session: {reason}. {WORKERS['n']} workers per expansion "
          f"({config.CPU_COUNT} cores)", flush=True)
    session = new_session(WORKERS["n"], check_idle(allow_busy))
    for variant, hops, limit in runs:
        run_experiment(variant, hops, limit, seeds, session)
    return session

def sweep_table(results, column, metrics):
    # Growth metrics are reported relative to the first value of the sweep
    df = pd.DataFrame(results)
    base = df.iloc[0]
    table = {
        "Total nodes": [f"x{v / base['total_nodes']:.2f}" for v in df["total_nodes"]],
        "Total edges": [f"x{v / base['total_edges']:.2f}" for v in df["total_edges"]],
        "Avg. degree": [f"{v:.2f}" for v in df["avg_degree"]],
        "Limit activation": [f"{v:.1f}%" for v in df["limit_activation_pct"]],
        "Candidate retention": [f"{v:.1f}%" for v in df["candidate_retention_pct"]],
        "Largest SCC": [f"{v:.2f}%" for v in df["largest_scc_pct"]],
        "Largest WCC": [f"{v:.2f}%" for v in df["largest_wcc_pct"]],
        "Seeds in largest WCC": [f"{v:.1f}%" for v in df["seeds_in_largest_wcc_pct"]],
        "Seed WCCs": [f"{v:,}" for v in df["n_seed_wccs"]],
        "Runtime": [f"x{v / base['runtime_s']:.2f}" for v in df["runtime_s"]],
    }
    return pd.DataFrame({m: table[m] for m in metrics}, index=[f"{column}={v}" for v in df[column]]).T

def all_metrics(results):
    """Every metric of the runs, with the width-limit counters as columns."""
    rows = []
    for r in results:
        row = {k: v for k, v in r.items() if k not in ("width_limit", "timing")}
        row.update({f"width_limit_{k}": v for k, v in r["width_limit"].items()})
        timing = r["timing"]
        row.update({"timing_session": timing["session"]["id"], "timed_runtimes_s": timing["runtimes_s"],
                    "warmup_runtimes_s": timing["warmup_runtimes_s"], "max_other_cores": timing["max_other_cores"]})
        rows.append(row)
    return pd.DataFrame(rows)

def summary_table(experiment, results):
    if experiment == "configuration":
        # Runtime relative to the first variant, as along k and w
        return pd.DataFrame({
            "Density": [f"{r['density']:.3e}" for r in results],
            "%SCC_max": [f"{r['largest_scc_pct']:.3f}%" for r in results],
            "%WCC_max": [f"{r['largest_wcc_pct']:.3f}%" for r in results],
            "%Seeds in WCC_max": [f"{r['seeds_in_largest_wcc_pct']:.1f}%" for r in results],
            "Seed WCCs": [f"{r['n_seed_wccs']:,}" for r in results],
            "Runtime": [f"x{r['runtime_s'] / results[0]['runtime_s']:.2f}" for r in results],
        }, index=[r["variant"] for r in results])
    if experiment == "hops":
        return sweep_table(results, "hops",
            ["Total nodes", "Total edges", "Avg. degree", "Largest SCC", "Largest WCC", "Seeds in largest WCC",
             "Seed WCCs", "Runtime"])
    return sweep_table(results, "limit",
        ["Total nodes", "Total edges", "Avg. degree", "Limit activation", "Candidate retention", "Largest SCC",
         "Largest WCC", "Seeds in largest WCC", "Seed WCCs", "Runtime"])

def main():
    limit_threads()
    parser = argparse.ArgumentParser(description="Expansion experiments: structure and cost of the expanded graphs.")
    parser.add_argument("--experiments", help="experiments to run, e.g. 1,3 (1 configuration, 2 hops, 3 width; "
                                              "asked if not given)")
    parser.add_argument("--workers", type=int, help="parallel workers of every expansion (default: one per core, "
                                                    "limited by the available memory; 1 runs every expansion in a "
                                                    "single process)")
    parser.add_argument("--allow-busy", action="store_true",
                        help="time the runs even if other processes use the machine (their runtimes are then not "
                             "comparable)")
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
        "random_state": "random.seed(SEED) before every expansion",
        "workers": WORKERS["n"],
        "tx_rebuild": tx_rebuild,
        "timing": {"warmup_runs": WARMUP_RUNS, "timed_runs": TIMED_RUNS, "runtime": "median of the timed expansions",
                   "idle_check_s": IDLE_CHECK_S, "max_other_cores": MAX_OTHER_CORES,
                   "min_available_memory_pct": MIN_AVAILABLE_MEMORY_PCT},
    })

    # All the runs of the selected experiments are timed in one session; the runs shared by
    # several experiments (fw bw tx whole, k=2, w=5) are only run once
    session = run_session([run for experiment in experiments for run in EXPERIMENTS[experiment]], seeds,
                          args.allow_busy)

    tables = {}
    for experiment in experiments:
        results = [load_run(variant, hops, limit) for variant, hops, limit in EXPERIMENTS[experiment]]
        all_metrics(results).to_csv(OUTPUTS_DIR / f"{experiment}.csv", index=False)

        table = summary_table(experiment, results)
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        table.to_csv(TABLES_DIR / f"{experiment}.csv")
        tables[experiment] = table

    n_illicit = sum(label for _, label in seeds.values())
    for experiment, table in tables.items():
        print("\n========================================")
        print(f"{experiment.capitalize()} experiment ({n_illicit} illicit + {len(seeds) - n_illicit} licit seeds)")
        print("========================================")
        print(table.to_string())

    print(f"\nRuntimes timed in session {session['id']} on {session['host']} with {session['workers']} workers "
          f"({session['warmup_runs']} warm-up + {session['timed_runs']} timed expansions per run)")
    print(f"\nResults saved at: {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
