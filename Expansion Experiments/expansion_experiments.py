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
# The metrics of each run are saved in OUTPUTS_DIR/runs/ and reused if they
# exist: delete the file of a run to compute it again. Every metric of every run is
# saved in OUTPUTS_DIR/<experiment>.csv, and the tables of the paper in
# OUTPUTS_DIR/tables/<experiment>.csv.

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
from array import array
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm

import config
from shared.experiment import git_commit, timestamp, write_json
from shared.parallel import blocksci_workers, parallel_map
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
    """The seed addresses of the experiments, selected once and saved."""
    if SEEDS_FILE.exists():
        print(f"Seed addresses already selected: {SEEDS_FILE}")
        return pd.read_csv(SEEDS_FILE)

    import create_splits
    bitcoinheist_csv = get_required_env_path("RSD_BITCOINHEIST_CSV")
    seeds, checks, info = create_splits.select_seeds(chain, bitcoinheist_csv, PER_CLASS)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    seeds.to_csv(SEEDS_FILE, index=False)
    write_json(OUTPUTS_DIR / "seeds_summary.json", {
        "created": timestamp(), "git_commit": git_commit(config.PROJECT_ROOT),
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
# The expansion variants follow expansion.py of the multi-instance methodology
# with a random node width limit, and keep the same candidate order and random
# sampling, so they produce the same graphs. Every hop is computed in two parallel
# passes over forked workers (shared.parallel.parallel_map, which keeps the order of
# the items), with the order-dependent work in the main process:
#   1. Scan: the workers list the candidate transactions of every node of the
#      frontier (seed addresses, transactions or addresses of the previous hop).
#   2. Selection (main process): the candidates are filtered and the width limit is
#      applied node by node, in the original order. random.sample only depends on
#      the number of candidates, so sampling positions selects the same candidates.
#   3. Explore: the workers read the inputs and outputs of the selected transactions
#      (their address strings, the costliest BlockSci reads).
#   4. Insertion (main process): the records are inserted in the graph in the
#      original order, replaying the loop of the sequential algorithm, so the graph
#      does not depend on the number of workers.
# The workers inherit the chain, the seed addresses and the graph when they are
# forked, and only plain values travel back. The main process keeps plain values only:
#   addresses: address string -> reference to rebuild the address in a worker (only
#              for the address-based variants, which expand addresses further):
#              ("seed", address string) or ("in"/"out", tx index, position) of the
#              input/output where it was found
#   txs:       set of tx indices in the graph
#   inputs:    (spent tx index, spent output index) -> (address string, spending tx index)
#   outputs:   (tx index, output index) -> address string
# Transactions are identified by their BlockSci index, from which the workers rebuild
# them (the BlockSci call that does it is detected and checked at the start).

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
    # Same selection as random.sample(candidates, limit): random.sample only depends
    # on the number of candidates, so the positions are sampled
    n_candidates = len(candidates)
    n_retained = min(n_candidates, limit)
    limit_stats["nonempty_steps"] += n_candidates > 0
    limit_stats["activated_steps"] += n_candidates > limit
    limit_stats["candidates"] += n_candidates
    limit_stats["retained"] += n_retained
    return [candidates[i] for i in random.sample(range(n_candidates), n_retained)]

### Workers
#
# State inherited by the forked workers: the seed addresses, how to rebuild a
# transaction from its index, and the txs of the graph (set before every pass).
# Candidates travel back as compact arrays of tx indices.
_STATE = {"seeds": {}, "tx_by_index": None, "txs": set(), "ids": False}

def tx_of(t):
    return _STATE["tx_by_index"](t)

def address_of(ref):
    if ref[0] == "seed":
        return _STATE["seeds"][ref[1]][0]
    tx = tx_of(ref[1])
    return tx.inputs[ref[2]].address if ref[0] == "in" else tx.outputs[ref[2]].address

def address_id(addr):
    # Identity of an address (not only its string, which a pubkey address shares
    # with its pubkey-hash address)
    return (addr.address_num, str(addr.type))

# Encoding an address string is costly and the same addresses appear in many
# transactions, so strings are memoized by address in every worker.
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

def explore_record(t):
    """Inputs and outputs of a tx: inputs (spent tx index, spent output index, address
    string, address id) and outputs (output index, address string, address id), in the
    order of the tx; the address ids are only read by the dedicated variant."""
    tx = tx_of(t)
    ids = _STATE["ids"]
    ins = [(inp.spent_tx_index, inp.spent_output.index, address_string(inp.address),
            address_id(inp.address) if ids else None) for inp in tx.inputs]
    outs = [(out.index, address_string(out.address), address_id(out.address) if ids else None)
            for out in tx.outputs]
    return ins, outs

# Scans of the tx-based variants: candidate txs of a seed address, or the txs spending
# (forward) or spent by (backward) a tx of the frontier that are not in the graph,
# each tx once, in the order of the inputs/outputs

def scan_seed_input_output_txes(seed_str):
    addr = _STATE["seeds"][seed_str][0]
    return array("q", (tx.index for tx in addr.input_txes)), array("q", (tx.index for tx in addr.output_txes))

def scan_seed_txes(seed_str):
    return array("q", (tx.index for tx in _STATE["seeds"][seed_str][0].txes))

def _new_txs(found):
    txs, seen, new = _STATE["txs"], set(), array("q")
    for t in found:
        if t is not None and t not in txs and t not in seen:
            seen.add(t)
            new.append(t)
    return new

def scan_spending(t):
    return _new_txs(spending_tx_index(out) for out in tx_of(t).outputs)

def scan_spent(t):
    return _new_txs(inp.spent_tx_index for inp in tx_of(t).inputs)

def scan_spent_and_spending(t):
    tx = tx_of(t)
    return _new_txs([inp.spent_tx_index for inp in tx.inputs] + [spending_tx_index(out) for out in tx.outputs])

# Scans of the address-based variants, from the reference of an address of the frontier

def scan_address_inputs(ref):
    # Dedicated forward: txs where the address is an input, not in the graph, in order of appearance
    addr = address_of(ref)
    return address_id(addr), _new_txs(inp.tx_index for inp in addr.inputs)

def scan_address_outputs(ref):
    # Dedicated backward: txs where the address is an output, not in the graph, in order of appearance
    addr = address_of(ref)
    return address_id(addr), _new_txs(out.tx_index for out in addr.outputs)

def scan_address_input_txes(ref):
    # Whole forward: every tx where the address is an input (the limit is applied first)
    return array("q", (tx.index for tx in address_of(ref).input_txes))

def scan_address_output_txes(ref):
    # Whole backward: every tx where the address is an output (the limit is applied first)
    return array("q", (tx.index for tx in address_of(ref).output_txes))

### Main process

# Workers of every parallel pass (set in main)
WORKERS = {"n": 1}

def pmap(func, items, desc):
    return parallel_map(func, items, WORKERS["n"], desc=desc)

def explore(txs_to_explore, desc, ids=False):
    """Records of the given txs, in their order."""
    _STATE["ids"] = ids
    return pmap(explore_record, list(txs_to_explore), f"Exploring {desc}")

def insert_whole(t, record, addresses, inputs, outputs, refs):
    # explore_tx of the sequential version: every input and output with an address
    ins, outs = record
    for k, (spent_t, spent_index, addr_str, _) in enumerate(ins):
        if addr_str is not None:
            addr_str = sys.intern(addr_str)
            inputs[(spent_t, spent_index)] = (addr_str, t)
            if addr_str not in addresses:
                addresses[addr_str] = ("in", t, k) if refs else None
    for j, (index, addr_str, _) in enumerate(outs):
        if addr_str is not None:
            addr_str = sys.intern(addr_str)
            outputs[(t, index)] = addr_str
            if addr_str not in addresses:
                addresses[addr_str] = ("out", t, j) if refs else None

def tx_hop(prev_txs, scan, addresses, txs, inputs, outputs, limit, desc):
    """One hop of the tx-based variants from the txs of the previous hop (in their order)."""
    _STATE["txs"] = txs
    scans = pmap(scan, list(prev_txs), f"Scanning {desc}")
    candidates = []
    candidate_txs = set()
    for found in scans:
        tx_candidates = limit_expansion([t for t in found if t not in candidate_txs], limit)
        candidates += tx_candidates
        candidate_txs.update(tx_candidates)
    del scans
    for t, record in zip(candidates, explore(candidates, desc)):
        insert_whole(t, record, addresses, inputs, outputs, refs=False)
    return dict.fromkeys(candidates)

### Tx-based whole transaction implied addresses forward-backward expansion

def expand_fw_bw_tx_whole(addresses, num_for_hops, num_back_hops, txs, inputs, outputs, limit):
    fw_candidates = []
    bw_candidates = []
    for fw_found, bw_found in pmap(scan_seed_input_output_txes, list(addresses), "Scanning hop 1"):
        fw_candidates += limit_expansion(fw_found, limit)
        bw_candidates += limit_expansion(bw_found, limit)

    # Every tx is explored once, the forward candidates first
    unique = list(dict.fromkeys(fw_candidates + bw_candidates))
    for t, record in zip(unique, explore(unique, "hop 1")):
        insert_whole(t, record, addresses, inputs, outputs, refs=False)

    new_fw_txs = dict.fromkeys(fw_candidates)
    new_bw_txs = dict.fromkeys(bw_candidates)
    txs.update(new_fw_txs)
    txs.update(new_bw_txs)

    for i in range(num_for_hops-1):
        new_fw_txs = tx_hop(new_fw_txs, scan_spending, addresses, txs, inputs, outputs, limit, f"forward hop {i + 2}")
        txs.update(new_fw_txs)

    for i in range(num_back_hops-1):
        new_bw_txs = tx_hop(new_bw_txs, scan_spent, addresses, txs, inputs, outputs, limit, f"backward hop {i + 2}")
        txs.update(new_bw_txs)

### Tx-based all over expansion

def expand_all_over_tx(addresses, num_hops, txs, inputs, outputs, limit):
    new_txs = {}
    for found in pmap(scan_seed_txes, list(addresses), "Scanning hop 1"):
        new_txs.update(dict.fromkeys(limit_expansion(found, limit)))

    # Every tx is explored once
    for t, record in zip(new_txs, explore(new_txs, "hop 1")):
        insert_whole(t, record, addresses, inputs, outputs, refs=False)
    txs.update(new_txs)

    for i in range(num_hops-1):
        new_txs = tx_hop(new_txs, scan_spent_and_spending, addresses, txs, inputs, outputs, limit, f"hop {i + 2}")
        txs.update(new_txs)

### Addr-based dedicated inputs/outputs forward-backward expansion

def addr_dedicated_hop(addresses, news, prev_addresses, txs, inputs, outputs, limit, forward, desc):
    """One hop of the dedicated variant: the txs where every address of the frontier is
    an input (forward) or an output (backward), with only its own inputs (outputs)."""
    _STATE["txs"] = txs
    scan = scan_address_inputs if forward else scan_address_outputs
    candidates = {}
    for frontier_id, found in pmap(scan, list(prev_addresses.values()), f"Scanning {desc}"):
        # A tx selected again by a later address keeps its position and takes the
        # inputs (outputs) of that address, as dict.update does in the sequential version
        for t in limit_expansion(found, limit):
            candidates[t] = frontier_id

    new_addresses = {}
    known = lambda addr_str: addr_str in addresses or addr_str in news
    for (t, frontier_id), (ins, outs) in zip(candidates.items(), explore(candidates, desc, ids=True)):
        txs.add(t)
        if forward:
            for spent_t, spent_index, addr_str, addr_id in ins:
                if addr_id == frontier_id:
                    inputs[(spent_t, spent_index)] = (addr_str and sys.intern(addr_str), t)
            for j, (index, addr_str, _) in enumerate(outs):
                addr_str = addr_str and sys.intern(addr_str)
                outputs[(t, index)] = addr_str
                if addr_str is not None and not known(addr_str):
                    new_addresses[addr_str] = ("out", t, j)
        else:
            for index, addr_str, addr_id in outs:
                if addr_id == frontier_id:
                    outputs[(t, index)] = addr_str and sys.intern(addr_str)
            for k, (spent_t, spent_index, addr_str, _) in enumerate(ins):
                addr_str = addr_str and sys.intern(addr_str)
                inputs[(spent_t, spent_index)] = (addr_str, t)
                if addr_str is not None and not known(addr_str):
                    new_addresses[addr_str] = ("in", t, k)
    return new_addresses

def expand_fw_bw_addr_dedicated(addresses, num_for_hops, num_back_hops, txs, inputs, outputs, limit):
    news = {}

    new = addresses
    for i in range(num_for_hops):
        new = addr_dedicated_hop(addresses, news, new, txs, inputs, outputs, limit, True, f"forward hop {i + 1}")
        news.update(new)

    new = addresses
    for i in range(num_back_hops):
        new = addr_dedicated_hop(addresses, news, new, txs, inputs, outputs, limit, False, f"backward hop {i + 1}")
        news.update(new)

    addresses.update(news)

### Addr-based whole transaction implied addresses forward-backward expansion

def addr_whole_hop(addresses, news, prev_addresses, txs, inputs, outputs, limit, forward, desc):
    """One hop of the whole variant: the txs where every address of the frontier is an
    input (forward) or an output (backward), with all their inputs and outputs. Returns
    the addresses on the other side (new) and on the same side (side) of the txs."""
    _STATE["txs"] = txs
    scan = scan_address_input_txes if forward else scan_address_output_txes
    candidates = {}
    for found in pmap(scan, list(prev_addresses.values()), f"Scanning {desc}"):
        for t in limit_expansion(found, limit):
            if t not in txs:
                candidates[t] = None

    new_addresses = {}
    side_addresses = {}
    known = lambda addr_str: addr_str in addresses or addr_str in news
    for t, (ins, outs) in zip(candidates, explore(candidates, desc)):
        txs.add(t)
        same = [("in", k, (spent_t, spent_index), addr_str) for k, (spent_t, spent_index, addr_str, _) in enumerate(ins)]
        other = [("out", j, (t, index), addr_str) for j, (index, addr_str, _) in enumerate(outs)]
        if not forward:
            same, other = other, same
        for kind, position, edge, addr_str in same:
            if addr_str is not None:
                addr_str = sys.intern(addr_str)
                if kind == "in":
                    inputs[edge] = (addr_str, t)
                else:
                    outputs[edge] = addr_str
                if not known(addr_str) and addr_str not in new_addresses:
                    side_addresses[addr_str] = (kind, t, position)
        for kind, position, edge, addr_str in other:
            if addr_str is not None:
                addr_str = sys.intern(addr_str)
                if kind == "in":
                    inputs[edge] = (addr_str, t)
                else:
                    outputs[edge] = addr_str
                if not known(addr_str):
                    new_addresses[addr_str] = (kind, t, position)
    return new_addresses, side_addresses

def expand_fw_bw_addr_whole_same(addresses, num_for_hops, num_back_hops, txs, inputs, outputs, limit):
    # Side addresses are expanded in the same direction as the hop that found them
    new = addresses.copy()
    side = {}
    news = {}
    for i in range(num_for_hops):
        new, side = addr_whole_hop(addresses, news, {**new, **side}, txs, inputs, outputs, limit, True, f"forward hop {i + 1}")
        news.update(new)
        news.update(side)

    new = addresses.copy()
    side = {}
    for i in range(num_back_hops):
        new, side = addr_whole_hop(addresses, news, {**new, **side}, txs, inputs, outputs, limit, False, f"backward hop {i + 1}")
        news.update(new)
        news.update(side)

    addresses.update(news)

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

def detect_tx_by_index(chain, seeds):
    """The BlockSci call that builds a tx from its index, checked on a tx of a seed address."""
    import blocksci
    sample = next(tx for addr, _ in seeds.values() for tx in addr.txes)
    options = [
        ("blocksci.Tx(index, chain)", lambda i: blocksci.Tx(i, chain)),
        ("chain.tx_with_index(index)", lambda i: chain.tx_with_index(i)),
    ]
    for name, build in options:
        try:
            tx = build(sample.index)
            if tx.index == sample.index and str(tx.hash) == str(sample.hash):
                return build, name
        except Exception:
            continue
    raise RuntimeError("BlockSci does not build a transaction from its index with "
                       + " or ".join(name for name, _ in options))

## GRAPH METRICS

def graph_metrics(addresses, txs, inputs, outputs):
    # Same graph as serialize_graph.py: address nodes, transaction nodes, input
    # edges (addr -> tx), output edges (tx -> addr) and spent output edges
    # (spent tx -> spending tx) between transactions of the graph
    addr_index = {addr_str: i for i, addr_str in enumerate(addresses)}
    n_addr = len(addr_index)
    tx_index = {t: n_addr + i for i, t in enumerate(txs)}
    n_nodes = n_addr + len(tx_index)

    # Edges in compact integer arrays (8 bytes per endpoint)
    src = array("q")
    dst = array("q")

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

def run_experiment(variant, hops, limit, seeds):
    print(f"\nRunning {variant} | {hops} hops | {LIMIT_MODE} limit {limit} ({WORKERS['n']} workers)...", flush=True)

    # Every run starts from the same random state, so it can be reproduced on its own
    random.seed(SEED)
    np.random.seed(SEED)

    # The expansion adds the discovered addresses in place, so it works on a copy of the seeds
    addresses = {addr_str: ("seed", addr_str) for addr_str in seeds}
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
        "n_seeds": len(seeds),
        "n_illicit_seeds": sum(label for _, label in seeds.values()),
        "runtime_s": runtime,
        "workers": WORKERS["n"],
        **graph_metrics(addresses, txs, inputs, outputs),
        "limit_activation_pct": 100 * limit_stats["activated_steps"] / limit_stats["nonempty_steps"] if limit_stats["nonempty_steps"] else 0.0,
        "candidate_retention_pct": 100 * limit_stats["retained"] / limit_stats["candidates"] if limit_stats["candidates"] else 0.0,
        "width_limit": dict(limit_stats),
    }

    write_json(run_path(variant, hops, limit), results)

    print(f"Finished {variant} | {hops} hops | {LIMIT_MODE} limit {limit}: "
          f"{results['total_nodes']} nodes, {results['total_edges']} edges, {runtime:.1f} s", flush=True)

    # The graph of a run is freed before the next one
    del addresses, txs, inputs, outputs
    _STATE["txs"] = set()
    gc.collect()
    return results

def run_pending(runs, seeds):
    """Runs the given runs that are not saved yet, one after another."""
    pending = [r for r in dict.fromkeys(runs) if not run_path(*r).exists()]
    print(f"\n{len(set(runs)) - len(pending)} runs already saved, {len(pending)} to run, "
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

    # The forked workers inherit the chain and the seed addresses
    WORKERS["n"] = args.workers or blocksci_workers()
    tx_by_index, tx_rebuild = detect_tx_by_index(chain, seeds)
    _STATE.update(seeds=seeds, tx_by_index=tx_by_index)
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
