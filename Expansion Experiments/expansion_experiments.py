# RANSOMWARE ADDRESSES DETECTION - EXPANSION EXPERIMENTS
#
# Reproduces the expansion experiments of Table 2 of the paper, run separately from
# the pipeline (requires BlockSci and RSD_BITCOINHEIST_CSV):
#   1. Configuration: expansion variants with k=2 hops and a random node width
#      limit w=5 -> graph density and share of nodes in the largest SCC
#   2. Hops: fw bw tx whole with w=5 and k = 1, 2, 3
#   3. Width: fw bw tx whole with k=2 and w = 1, 2, 5, 10
# The user chooses which experiments to run (or --experiments 1,2,3). The expansion
# algorithms are the same as in expansion.py of the multi-instance methodology.
#
# Seed addresses: PER_CLASS (1,500) illicit BitcoinHeist addresses found in the chain
# and as many licit ones, sampled with DATA_SEED and split 60/20/20 per class
# (create_splits.select_splits), saved in OUTPUTS_DIR/splits.csv. Every split is
# expanded on its own, as the multi-instance methodology does.
#
# Every run of a split restarts from the same random seed (SEED + split index), so
# each run is reproducible on its own. The metrics of each run are saved in
# OUTPUTS_DIR/runs/ and reused if they exist: delete the file of a run to compute it
# again. Every metric of every run is saved in OUTPUTS_DIR/<experiment>.csv, and the
# tables of the paper in OUTPUTS_DIR/tables/<experiment>_<split>.csv.

### Import libraries

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import gc
import json
import random
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm

import config
from shared.experiment import git_commit, timestamp, write_json
from shared.paths import get_required_env_path
from experiments_config import (
    SEED,
    PER_CLASS,
    SPLITS,
    LIMIT_MODE,
    EXPERIMENTS,
    OUTPUTS_DIR,
    SPLITS_FILE,
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
    """The seed addresses and splits of the experiments, selected once and saved."""
    if SPLITS_FILE.exists():
        print(f"Seed addresses already selected: {SPLITS_FILE}")
        return pd.read_csv(SPLITS_FILE)

    import create_splits
    bitcoinheist_csv = get_required_env_path("RSD_BITCOINHEIST_CSV")
    splits, checks, info = create_splits.select_splits(chain, bitcoinheist_csv, PER_CLASS)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    splits.to_csv(SPLITS_FILE, index=False)
    write_json(OUTPUTS_DIR / "splits_summary.json", {
        "created": timestamp(), "git_commit": git_commit(config.PROJECT_ROOT),
        "source": "Expansion Experiments/expansion_experiments.py", **info, "checks": checks,
    })
    return splits

def load_seeds(chain, splits):
    # Seeds are expanded in the order of the splits file, so every execution is the same
    seeds = {}
    for split in SPLITS:
        rows = splits[splits["split"] == split]
        seeds[split] = {}
        for addr_str, label in tqdm(list(zip(rows["address"], rows["label"])), desc=f"Loading {split} seed addresses"):
            addr_obj = chain.address_from_string(addr_str)
            if addr_obj is None:
                raise ValueError(f"Seed address {addr_str} is not in the chain")
            seeds[split][addr_str] = (addr_obj, int(label))
        n_illicit = sum(label for _, label in seeds[split].values())
        print(f"  {split:5s}: {n_illicit} illicit + {len(seeds[split]) - n_illicit} licit seed addresses")
    return seeds

## GRAPH EXPANSION
#
# The expansion variants follow expansion.py of the multi-instance methodology
# with a random node width limit, and keep the same candidate order and random
# sampling, so they produce the same graphs. As the experiments only need the
# graph structure, they keep integer identifiers instead of BlockSci objects:
#   addresses: address string -> BlockSci address (to expand it further)
#   txs:       set of tx indices in the graph
#   inputs:    (spent tx index, spent output index) -> (address string, spending tx index)
#   outputs:   (tx index, output index) -> address string
# Transactions are identified by their BlockSci index instead of their hex hash,
# which only differs for the two pairs of duplicated 2010 coinbase txs (BIP30).

# Width-limit counters of the current run. Every call to limit_expansion is one
# expansion step: one node choosing among its candidate transactions.
limit_stats = {}

def reset_limit_stats():
    limit_stats.clear()
    limit_stats.update({
        "nonempty_steps": 0,    # steps with at least one candidate
        "activated_steps": 0,   # steps with more candidates than the limit
        "candidates": 0,        # candidates available over all steps
        "retained": 0,          # candidates kept over all steps
    })

def limit_expansion(candidates, limit):
    # random.sample only depends on the number of candidates, so sampling
    # (index, object) pairs keeps the same elements as sampling the objects
    if not isinstance(candidates, (list, tuple)):
        candidates = list(candidates)
    n_candidates = len(candidates)
    n_retained = min(n_candidates, limit)
    limit_stats["nonempty_steps"] += n_candidates > 0
    limit_stats["activated_steps"] += n_candidates > limit
    limit_stats["candidates"] += n_candidates
    limit_stats["retained"] += n_retained
    return random.sample(candidates, n_retained)

# Encoding an address string is costly and the same addresses appear in many
# transactions, so strings are memoized by address. The cache is cleared before
# every run, so each run pays for its own encodings.
_address_strings = {}

def _address_string_memoized(addr):
    key = (addr.address_num, addr.type)
    addr_str = _address_strings.get(key, _address_strings)
    if addr_str is _address_strings:
        addr_str = _address_strings[key] = getattr(addr, 'address_string', None)
    return addr_str

def _address_string_direct(addr):
    return getattr(addr, 'address_string', None)

_address_string = _address_string_memoized

def address_string(addr):
    # Falls back to encoding every time if addresses cannot be identified
    global _address_string
    try:
        return _address_string(addr)
    except AttributeError:
        _address_string = _address_string_direct
        return _address_string(addr)

def _spending_index_attr(out):
    return out.spending_tx_index if out.is_spent else None

def _spending_index_tx(out):
    tx = out.spending_tx
    return None if tx is None else tx.index

_spending_index = _spending_index_attr

def spending_tx_index(out):
    # Index of the tx spending the output, or None if it is unspent, read
    # without creating the spending tx object when BlockSci allows it
    global _spending_index
    try:
        return _spending_index(out)
    except AttributeError:
        _spending_index = _spending_index_tx
        return _spending_index(out)

def explore_tx(tx, addresses, inputs, outputs, new, explored=None):
    t = tx.index
    new[t] = tx

    # Re-exploring a whole tx rewrites the same entries, so it can be skipped
    if explored is not None:
        if t in explored:
            return
        explored.add(t)

    for inp in tx.inputs:
        addr = inp.address
        addr_str = address_string(addr)

        if addr_str is not None:
            inputs[(inp.spent_tx_index, inp.spent_output.index)] = (addr_str, t)

            if addr_str not in addresses:
                addresses[addr_str] = addr

    for out in tx.outputs:
        addr = out.address
        addr_str = address_string(addr)

        if addr_str is not None:
            outputs[(t, out.index)] = addr_str

            if addr_str not in addresses:
                addresses[addr_str] = addr

### Addr-based dedicated inputs/outputs forward-backward expansion

def addr_dedicated_expand_forward(addresses, prev_addresses, txs, inputs, outputs, limit):
    new_addresses = {}
    candidates = {}

    for addr in prev_addresses.values():
        addr_candidates = {}

        for inp in addr.inputs:
            t = inp.tx_index

            if t not in txs:
                addr_candidates.setdefault(t, []).append(inp)

        addr_candidates = {t: addr_candidates[t] for t in limit_expansion(addr_candidates.keys(), limit)}
        candidates.update(addr_candidates)

    for t, tx_inputs in candidates.items():
        tx = tx_inputs[0].tx
        txs.add(t)

        for inp in tx_inputs:
            inputs[(inp.spent_tx_index, inp.spent_output.index)] = (address_string(inp.address), t)

        for out in tx.outputs:
            addr = out.address
            addr_str = address_string(addr)
            outputs[(t, out.index)] = addr_str
            if addr_str is not None and addr_str not in addresses:
                new_addresses[addr_str] = addr

    return new_addresses

def addr_dedicated_expand_backward(addresses, prev_addresses, txs, inputs, outputs, limit):
    new_addresses = {}
    candidates = {}

    for addr in prev_addresses.values():
        addr_candidates = {}

        for out in addr.outputs:
            t = out.tx_index

            if t not in txs:
                addr_candidates.setdefault(t, []).append(out)

        addr_candidates = {t: addr_candidates[t] for t in limit_expansion(addr_candidates.keys(), limit)}
        candidates.update(addr_candidates)

    for t, tx_outputs in candidates.items():
        tx = tx_outputs[0].tx
        txs.add(t)

        for out in tx_outputs:
            outputs[(t, out.index)] = address_string(out.address)

        for inp in tx.inputs:
            addr = inp.address
            addr_str = address_string(addr)
            inputs[(inp.spent_tx_index, inp.spent_output.index)] = (addr_str, t)
            if addr_str is not None and addr_str not in addresses:
                new_addresses[addr_str] = addr

    return new_addresses

def expand_fw_bw_addr_dedicated(addresses, num_for_hops, num_back_hops, txs, inputs, outputs, limit):
    news = {}

    new = addresses
    for i in range(num_for_hops):
        new = addr_dedicated_expand_forward({**addresses, **news}, new, txs, inputs, outputs, limit)
        news.update(new)

    new = addresses
    for i in range(num_back_hops):
        new = addr_dedicated_expand_backward({**addresses, **news}, new, txs, inputs, outputs, limit)
        news.update(new)

    addresses.update(news)

### Addr-based whole transaction implied addresses forward-backward expansion

def addr_whole_expand_forward(addresses, prev_addresses, txs, inputs, outputs, limit):
    new_addresses = {}
    side_addresses = {}
    candidates = {}

    for addr in prev_addresses.values():
        for in_tx in limit_expansion(list(addr.input_txes), limit):
            t = in_tx.index

            if t not in txs:
                candidates[t] = in_tx

    for t, in_tx in candidates.items():
        txs.add(t)

        for inp in in_tx.inputs:
            addr = inp.address
            addr_str = address_string(addr)

            if addr_str is not None:
                inputs[(inp.spent_tx_index, inp.spent_output.index)] = (addr_str, t)

                if addr_str not in addresses and addr_str not in new_addresses:
                    side_addresses[addr_str] = addr

        for out in in_tx.outputs:
            addr = out.address
            addr_str = address_string(addr)

            if addr_str is not None:
                outputs[(t, out.index)] = addr_str

                if addr_str not in addresses:
                    new_addresses[addr_str] = addr

    return new_addresses, side_addresses

def addr_whole_expand_backward(addresses, prev_addresses, txs, inputs, outputs, limit):
    new_addresses = {}
    side_addresses = {}
    candidates = {}

    for addr in prev_addresses.values():
        for out_tx in limit_expansion(list(addr.output_txes), limit):
            t = out_tx.index

            if t not in txs:
                candidates[t] = out_tx

    for t, out_tx in candidates.items():
        txs.add(t)

        for out in out_tx.outputs:
            addr = out.address
            addr_str = address_string(addr)

            if addr_str is not None:
                outputs[(t, out.index)] = addr_str

                if addr_str not in addresses and addr_str not in new_addresses:
                    side_addresses[addr_str] = addr

        for inp in out_tx.inputs:
            addr = inp.address
            addr_str = address_string(addr)

            if addr_str is not None:
                inputs[(inp.spent_tx_index, inp.spent_output.index)] = (addr_str, t)

                if addr_str not in addresses:
                    new_addresses[addr_str] = addr

    return new_addresses, side_addresses

def expand_fw_bw_addr_whole_same(addresses, num_for_hops, num_back_hops, txs, inputs, outputs, limit):
    # Side addresses are expanded in the same direction as the hop that found them
    new = addresses.copy()
    side = {}
    news = {}
    for i in range(num_for_hops):
        new, side = addr_whole_expand_forward({**addresses, **news}, {**new, **side}, txs, inputs, outputs, limit)
        news.update(new)
        news.update(side)

    new = addresses.copy()
    side = {}
    for i in range(num_back_hops):
        new, side = addr_whole_expand_backward({**addresses, **news}, {**new, **side}, txs, inputs, outputs, limit)
        news.update(new)
        news.update(side)

    addresses.update(news)

### Tx-based whole transaction implied addresses forward-backward expansion

def tx_whole_expand_forward(prev_txs, addresses, txs, inputs, outputs, limit):
    new_txs = {}
    candidates = []
    candidate_txs = set()

    for tx in prev_txs.values():
        tx_candidates = []
        tx_candidate_txs = set()

        for out in tx.outputs:
            t = spending_tx_index(out)

            if t is not None and t not in txs and t not in tx_candidate_txs and t not in candidate_txs:
                tx_candidates.append((t, out))
                tx_candidate_txs.add(t)

        tx_candidates = limit_expansion(tx_candidates, limit)
        candidates += tx_candidates
        candidate_txs.update(t for t, _ in tx_candidates)

    for _, out in candidates:
        explore_tx(out.spending_tx, addresses, inputs, outputs, new_txs)

    return new_txs

def tx_whole_expand_backward(prev_txs, addresses, txs, inputs, outputs, limit):
    new_txs = {}
    candidates = []
    candidate_txs = set()

    for tx in prev_txs.values():
        tx_candidates = []
        tx_candidate_txs = set()

        for inp in tx.inputs:
            t = inp.spent_tx_index

            if t not in txs and t not in tx_candidate_txs and t not in candidate_txs:
                tx_candidates.append((t, inp))
                tx_candidate_txs.add(t)

        tx_candidates = limit_expansion(tx_candidates, limit)
        candidates += tx_candidates
        candidate_txs.update(t for t, _ in tx_candidates)

    for _, inp in candidates:
        explore_tx(inp.spent_tx, addresses, inputs, outputs, new_txs)

    return new_txs

def expand_fw_bw_tx_whole(addresses, num_for_hops, num_back_hops, txs, inputs, outputs, limit):
    new_fw_txs = {}
    new_bw_txs = {}
    fw_candidates = []
    bw_candidates = []

    for addr in list(addresses.values()):
        fw_candidates += limit_expansion(list(addr.input_txes), limit)
        bw_candidates += limit_expansion(list(addr.output_txes), limit)

    explored = set()

    for fw_tx in fw_candidates:
        explore_tx(fw_tx, addresses, inputs, outputs, new_fw_txs, explored=explored)

    for bw_tx in bw_candidates:
        explore_tx(bw_tx, addresses, inputs, outputs, new_bw_txs, explored=explored)

    txs.update(new_fw_txs)
    txs.update(new_bw_txs)

    for i in range(num_for_hops-1):
        new_fw_txs = tx_whole_expand_forward(new_fw_txs, addresses, txs, inputs, outputs, limit)
        txs.update(new_fw_txs)

    for i in range(num_back_hops-1):
        new_bw_txs = tx_whole_expand_backward(new_bw_txs, addresses, txs, inputs, outputs, limit)
        txs.update(new_bw_txs)

### Tx-based all over expansion

def expand_all_over_tx(addresses, num_hops, txs, inputs, outputs, limit):
    new_txs = {}
    explored = set()

    for addr in list(addresses.values()):
        for tx in limit_expansion(list(addr.txes), limit):
            explore_tx(tx, addresses, inputs, outputs, new_txs, explored=explored)

    txs.update(new_txs)

    for i in range(num_hops-1):
        hop_txs = {}
        candidates = []
        candidate_txs = set()

        for tx in new_txs.values():
            tx_candidates = []
            tx_candidate_txs = set()

            for inp in tx.inputs:
                t = inp.spent_tx_index

                if t not in txs and t not in tx_candidate_txs and t not in candidate_txs:
                    tx_candidate_txs.add(t)
                    tx_candidates.append((t, inp, True))

            for out in tx.outputs:
                t = spending_tx_index(out)

                if t is not None and t not in txs and t not in tx_candidate_txs and t not in candidate_txs:
                    tx_candidate_txs.add(t)
                    tx_candidates.append((t, out, False))

            tx_candidates = limit_expansion(tx_candidates, limit)
            candidates += tx_candidates
            candidate_txs.update(t for t, _, _ in tx_candidates)

        for _, edge, is_input in candidates:
            explore_tx(edge.spent_tx if is_input else edge.spending_tx, addresses, inputs, outputs, hop_txs)

        new_txs = hop_txs
        txs.update(new_txs)

def expand(variant, addresses, hops, txs, inputs, outputs, limit):
    # Forward-backward variants use the same number of forward and backward hops
    if LIMIT_MODE != "random node":
        raise ValueError("The expansion experiments only implement the random node width limit.")

    if variant == "all over tx":
        expand_all_over_tx(addresses, hops, txs, inputs, outputs, limit)
    elif variant == "fw bw tx whole":
        expand_fw_bw_tx_whole(addresses, hops, hops, txs, inputs, outputs, limit)
    elif variant == "fw bw addr dedicated":
        expand_fw_bw_addr_dedicated(addresses, hops, hops, txs, inputs, outputs, limit)
    elif variant == "fw bw addr whole same":
        expand_fw_bw_addr_whole_same(addresses, hops, hops, txs, inputs, outputs, limit)
    else:
        raise ValueError(f"Unknown expansion variant: {variant}")

## GRAPH METRICS

def graph_metrics(addresses, txs, inputs, outputs):
    # Same graph as serialize_graph.py: address nodes, transaction nodes, input
    # edges (addr -> tx), output edges (tx -> addr) and spent output edges
    # (spent tx -> spending tx) between transactions of the graph
    addr_index = {addr_str: i for i, addr_str in enumerate(addresses)}
    n_addr = len(addr_index)
    tx_index = {t: n_addr + i for i, t in enumerate(txs)}
    n_nodes = n_addr + len(tx_index)

    src = []
    dst = []

    for input_id, (addr_str, spending) in inputs.items():
        a = addr_index.get(addr_str)
        t = tx_index.get(spending)
        if a is not None and t is not None:
            src.append(a)
            dst.append(t)

        # The spent output is part of the graph when it is one of its output edges
        if addr_str is not None and t is not None and outputs.get(input_id) is not None:
            spent = tx_index.get(input_id[0])
            if spent is not None:
                src.append(spent)
                dst.append(t)

    for (spending, _), addr_str in outputs.items():
        t = tx_index.get(spending)
        a = addr_index.get(addr_str)
        if a is not None and t is not None:
            src.append(t)
            dst.append(a)

    n_edges = len(src)
    adjacency = sp.coo_matrix(
        (np.ones(n_edges, dtype=np.int8), (np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64))),
        shape=(n_nodes, n_nodes),
    ).tocsr()
    _, scc_labels = connected_components(adjacency, directed=True, connection='strong')
    largest_scc = int(np.bincount(scc_labels).max()) if n_nodes > 0 else 0

    return {
        "addr_nodes": n_addr,
        "tx_nodes": len(tx_index),
        "total_nodes": n_nodes,
        "total_edges": n_edges,
        "density": n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0,
        "avg_degree": n_edges / n_nodes if n_nodes > 0 else 0.0,
        "largest_scc": largest_scc,
        "largest_scc_pct": 100 * largest_scc / n_nodes if n_nodes > 0 else 0.0,
    }

## EXPERIMENTS

def run_experiment(variant, hops, limit, split, seeds):
    run_path = RUNS_DIR / f"{run_name(variant, hops, limit, split)}.json"
    if run_path.exists():
        print(f"\nReusing saved results: {run_path}")
        with open(run_path) as f:
            return json.load(f)

    print(f"\nRunning {variant} | {hops} hops | {LIMIT_MODE} limit {limit} | {split}...", flush=True)

    # Every run of a split starts from the same random state, so it can be reproduced on its own
    random.seed(SEED + SPLITS.index(split))
    np.random.seed(SEED + SPLITS.index(split))

    # The expansion adds the discovered addresses in place, so it works on a copy of the seeds
    addresses = {addr_str: addr for addr_str, (addr, _) in seeds.items()}
    txs = set()
    inputs = {}
    outputs = {}

    reset_limit_stats()
    _address_strings.clear()

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
        "split": split,
        "n_seeds": len(seeds),
        "n_illicit_seeds": sum(label for _, label in seeds.values()),
        "runtime_s": runtime,
        **graph_metrics(addresses, txs, inputs, outputs),
        "limit_activation_pct": 100 * limit_stats["activated_steps"] / limit_stats["nonempty_steps"] if limit_stats["nonempty_steps"] else 0.0,
        "candidate_retention_pct": 100 * limit_stats["retained"] / limit_stats["candidates"] if limit_stats["candidates"] else 0.0,
        "width_limit": dict(limit_stats),
    }

    write_json(run_path, results)

    print(f"  {results['total_nodes']} nodes, {results['total_edges']} edges, {runtime:.1f} s")

    del addresses, txs, inputs, outputs
    _address_strings.clear()
    gc.collect()
    return results

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
    parser = argparse.ArgumentParser(description="Expansion experiments of Table 2 of the paper.")
    parser.add_argument("--experiments", help="experiments to run, e.g. 1,3 (1 configuration, 2 hops, 3 width; "
                                              "asked if not given)")
    args = parser.parse_args()

    if not config.ORIGINAL_CONFIGURATION:
        raise SystemExit("The expansion experiments use the original configuration: unset RSD_CONFIG_FILE.")

    experiments = get_experiments(args.experiments)

    import blocksci
    chain = blocksci.Blockchain(str(get_required_env_path("RSD_BLOCKSCI_CONFIG")))
    splits = select_seeds(chain)
    seeds = load_seeds(chain, splits)
    write_json(OUTPUTS_DIR / "experiments_config.json", {
        "created": timestamp(),
        "git_commit": git_commit(config.PROJECT_ROOT),
        "experiments": {name: [{"variant": v, "hops": h, "limit": w} for v, h, w in EXPERIMENTS[name]]
                        for name in EXPERIMENTS},
        "limit_mode": LIMIT_MODE,
        "per_class": PER_CLASS,
        "split_fractions": config.SPLIT_FRACTIONS,
        "seed": SEED,
        "random_state": "random.seed(SEED + split index) before every run of a split",
    })

    tables = {}
    for experiment in experiments:
        results = [run_experiment(variant, hops, limit, split, seeds[split])
                   for split in SPLITS for variant, hops, limit in EXPERIMENTS[experiment]]
        all_metrics(results).to_csv(OUTPUTS_DIR / f"{experiment}.csv", index=False)

        for split in SPLITS:
            table = paper_table(experiment, [r for r in results if r["split"] == split])
            TABLES_DIR.mkdir(parents=True, exist_ok=True)
            table.to_csv(TABLES_DIR / f"{experiment}_{split}.csv")
            tables[(experiment, split)] = table

    for (experiment, split), table in tables.items():
        n_illicit = sum(label for _, label in seeds[split].values())
        print("\n========================================")
        print(f"{experiment.capitalize()} experiment, {split} ({n_illicit} illicit + {len(seeds[split]) - n_illicit} licit seeds)")
        print("========================================")
        print(table.to_string())

    print(f"\nResults saved at: {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
