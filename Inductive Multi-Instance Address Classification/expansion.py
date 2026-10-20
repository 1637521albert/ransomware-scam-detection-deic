# STEP 1 (Methodology I) - GRAPH EXPANSION AND FEATURE EXTRACTION (requires BlockSci)
#
# Expands the seed addresses of each split (SPLITS_FILE, created by
# create_splits.py) into one merged heterogeneous graph per split, using the
# expansion configuration MULTI_INSTANCE_EXPANSION of config.py, and exports the
# node and edge features of every graph to MULTI_INSTANCE_GRAPHS_DIR/<split>/:
#   addr_feats.csv, tx_feats.csv, input_feats.csv, output_feats.csv, spent_pairs.csv
# tx_feats.csv also records the hop where every tx was added to the graph (hop,
# hop_direction). The expansion variants are in shared/expansion.py, shared with
# the ego-centric methodology.
#
# Every split is expanded from its own fixed random seed, so a split can be
# rebuilt on its own and gives the same graph in every execution. Splits whose
# files already exist are skipped.
#
# The splits are processed one after another, each with all the cores
# (config.MULTI_INSTANCE_EXPANSION_WORKERS forked processes). Every expansion variant
# runs in parallel (see GRAPH EXPANSION): the workers do every BlockSci access (listing
# the candidate transactions of the frontier as indices, exploring the selected ones and
# computing the features), and the main process keeps plain values only and performs
# the order-dependent steps (random sampling, de-duplication and insertion into the
# graph) in the order of the sequential algorithms of shared/expansion.py, so the graph
# does not depend on the number of workers. The features of an address are computed
# once and reused by the next splits whose graph contains it (see feature_cache_dir),
# and the busiest addresses of every block are computed first.

### Import libraries

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# One worker per core: single-threaded numerical libraries (before importing them)
from shared.threads import limit_threads

from array import array
from collections import Counter

import pandas as pd
import numpy as np
import random
import math
import os
import pickle
import shutil
import blocksci

import config
from shared.expansion import address_string, limit_expansion, spent_tx_hash, tx_hash
from shared.experiment import git_commit, timestamp, write_json
from shared.parallel import blocksci_workers, parallel_map
from shared.paths import get_required_env_path

### Initialize Blocksci

BLOCKSCI_CONFIG = get_required_env_path("RSD_BLOCKSCI_CONFIG")
chain = blocksci.Blockchain(str(BLOCKSCI_CONFIG))

### Parameters

EXPANSION = config.MULTI_INSTANCE_EXPANSION
exp_alg = f"{EXPANSION['direction']}{EXPANSION['graph']}{EXPANSION['address']}{EXPANSION['side_direction']}"
limit_mode = EXPANSION["limit_mode"]
limit = EXPANSION["limit"] if limit_mode != "" else math.inf
for_hops = EXPANSION.get("forward_hops", EXPANSION.get("hops"))
back_hops = EXPANSION.get("backward_hops", EXPANSION.get("hops"))
hops = EXPANSION.get("hops", for_hops)

data_path = config.MULTI_INSTANCE_GRAPHS_DIR
WORKERS = blocksci_workers(config.MULTI_INSTANCE_EXPANSION_WORKERS)

### Collecting addresses

def load_split_seeds():
    """Seed addresses of every split, as a list of (address string, label), in file order."""
    if not config.SPLITS_FILE.exists():
        raise FileNotFoundError(f"Missing {config.SPLITS_FILE}: run create_splits.py first.")

    splits = pd.read_csv(config.SPLITS_FILE)
    return {split: [(addr_str, int(label)) for addr_str, label in
                    zip(splits.loc[splits["split"] == split, "address"], splits.loc[splits["split"] == split, "label"])]
            for split in config.SPLITS}

## GRAPH EXPANSION
#
# Every expansion variant of shared/expansion.py (the reference sequential algorithms,
# also used by the ego-centric methodology) runs here in parallel, giving the same
# graph. Every hop has two parallel passes over forked workers (parallel_map keeps the
# order of the items), with the order-dependent steps in the main process:
#   1. Scan: the workers list the candidate transactions of every node of the frontier,
#      as transaction indices (compact arrays; their hashes are not computed).
#   2. Selection (main process): the candidates are de-duplicated and the width limit is
#      applied node by node, in the order of the sequential algorithm (random.sample
#      only depends on the number of candidates, so the same ones are selected).
#   3. Explore: the workers read the selected transactions: their hash and the inputs and
#      outputs with an address (all of them, or those of the frontier in the dedicated
#      variants), with the values that are exported.
#   4. Insertion (main process): the records are inserted in the graph in the order of
#      the sequential algorithm.
# The workers rebuild transactions from their index (TX_BY_INDEX, detected at the start)
# and inherit the chain and the graph when they are forked. Transactions are identified
# by their index during the expansion and by their hash in the exported files; both only
# differ for the two pairs of duplicated 2010 coinbase transactions (BIP30), which share
# a hash. The main process keeps plain values, with every address string and tx hash
# interned (stored once):
#   addresses: address string -> (reference, label); the reference rebuilds the address:
#              ("seed", address string) or ("in"/"out", tx index, position) of the
#              input/output where it was found
#   txs:       tx hash -> (hop, direction, tx index) of the hop where the tx was added
#   inputs:    (spent tx hash, spent output index) -> (address string, tx hash, age,
#              sequence number, value)
#   outputs:   (tx hash, output index) -> (address string, is spent, value)
# Inputs and outputs without an address are not kept (they are not exported).

# Rebuilds a transaction from its index (set by detect_tx_by_index) and the indices of
# the txs in the graph, read by the scans (set before every scan)
TX_BY_INDEX = {"build": None, "name": None}
_SCAN_STATE = {"txs": set()}

def detect_tx_by_index(sample_tx):
    """The BlockSci call that builds a tx from its index, checked on a known tx."""
    options = [
        ("blocksci.Tx(index, chain)", lambda i: blocksci.Tx(i, chain)),
        ("chain.tx_with_index(index)", lambda i: chain.tx_with_index(i)),
    ]
    for name, build in options:
        try:
            tx = build(sample_tx.index)
            if tx.index == sample_tx.index and str(tx.hash) == str(sample_tx.hash):
                TX_BY_INDEX.update(build=build, name=name)
                return name
        except Exception:
            continue
    raise RuntimeError("BlockSci does not build a transaction from its index with "
                       + " or ".join(name for name, _ in options))

def tx_of(i):
    return TX_BY_INDEX["build"](i)

def resolve_address(ref):
    # Rebuilds a BlockSci address from its reference
    if ref[0] == "seed":
        return chain.address_from_string(ref[1])
    tx = tx_of(ref[1])
    return tx.inputs[ref[2]].address if ref[0] == "in" else tx.outputs[ref[2]].address

def _spending_index_attr(out):
    return out.spending_tx_index if out.is_spent else None

def _spending_index_tx(out):
    tx = out.spending_tx
    return None if tx is None else tx.index

_spending_index = _spending_index_attr

def spending_tx_index(out):
    # Index of the tx spending an output (None if unspent), without building the tx
    # when BlockSci allows it
    global _spending_index
    try:
        return _spending_index(out)
    except AttributeError:
        _spending_index = _spending_index_tx
        return _spending_index(out)

### Workers: scans

def _indices(txes):
    return array("q", (tx.index for tx in txes))

def _new(indices):
    # Distinct indices not in the graph, in order
    txs, seen, new = _SCAN_STATE["txs"], set(), array("q")
    for t in indices:
        if t is not None and t not in txs and t not in seen:
            seen.add(t)
            new.append(t)
    return new

def _groups(pairs, skip_graph=True):
    # (tx index, selector) pairs -> [(tx index, [selectors])], by first appearance, for
    # the txs not in the graph (all of them if skip_graph is False)
    txs, groups = _SCAN_STATE["txs"], {}
    for t, selector in pairs:
        if not skip_graph or t not in txs:
            groups.setdefault(t, []).append(selector)
    return list(groups.items())

def scan_input_output_txes(ref):
    addr = resolve_address(ref)
    return _indices(addr.input_txes), _indices(addr.output_txes)

def scan_txes(ref):
    return _indices(resolve_address(ref).txes)

def scan_input_txes(ref):
    return _indices(resolve_address(ref).input_txes)

def scan_output_txes(ref):
    return _indices(resolve_address(ref).output_txes)

def scan_spending(i):
    return _new(spending_tx_index(out) for out in tx_of(i).outputs)

def scan_spent(i):
    return _new(inp.spent_tx_index for inp in tx_of(i).inputs)

def scan_spent_and_spending(i):
    tx = tx_of(i)
    return _new([inp.spent_tx_index for inp in tx.inputs] + [spending_tx_index(out) for out in tx.outputs])

# Dedicated variants: the inputs are selected by their spent output, the outputs by index

def scan_seed_dedicated(ref):
    addr = resolve_address(ref)
    return (_groups(((inp.tx_index, (inp.spent_tx_index, inp.spent_output.index)) for inp in addr.inputs), False),
            _groups(((out.tx_index, out.index) for out in addr.outputs), False))

def scan_dedicated_spending(i):
    return _groups((t, (i, out.index)) for out in tx_of(i).outputs for t in [spending_tx_index(out)] if t is not None)

def scan_dedicated_spent(i):
    return _groups((inp.spent_tx_index, inp.spent_output.index) for inp in tx_of(i).inputs)

def scan_address_inputs(ref):
    return _groups((inp.tx_index, (inp.spent_tx_index, inp.spent_output.index)) for inp in resolve_address(ref).inputs)

def scan_address_outputs(ref):
    return _groups((out.tx_index, out.index) for out in resolve_address(ref).outputs)

### Workers: exploration

def explore_record(task):
    """Hash, inputs and outputs (with an address) of a tx: all of them, or only the
    selected inputs ('input' mode, in the order of the selectors) or outputs ('output').
    Inputs: (position, spent tx hash, spent output index, address string, age, sequence
    number, value); outputs: (position, index, address string, is spent, value)."""
    i, mode, selectors = task
    tx = tx_of(i)
    if mode == "input":
        by_outpoint = {(inp.spent_tx_index, inp.spent_output.index): (k, inp) for k, inp in enumerate(tx.inputs)}
        in_items = [by_outpoint[s] for s in selectors]
    else:
        in_items = list(enumerate(tx.inputs))
    if mode == "output":
        by_index = {out.index: (j, out) for j, out in enumerate(tx.outputs)}
        out_items = [by_index[s] for s in selectors]
    else:
        out_items = list(enumerate(tx.outputs))
    ins = []
    for k, inp in in_items:
        addr_str = address_string(inp.address)
        if addr_str is not None:
            ins.append((k, spent_tx_hash(inp), inp.spent_output.index, addr_str, inp.age, inp.sequence_num, inp.value))
    outs = []
    for j, out in out_items:
        addr_str = address_string(out.address)
        if addr_str is not None:
            outs.append((j, out.index, addr_str, out.is_spent, out.value))
    return tx_hash(tx), ins, outs

### Main process

intern = sys.intern

def scan(func, items, desc, txs_idx):
    _SCAN_STATE["txs"] = txs_idx
    return parallel_map(func, list(items), WORKERS, desc=f"Scanning {desc}")

def explore(tasks, desc):
    return parallel_map(explore_record, list(tasks), WORKERS, desc=f"Exploring {desc}")

def sample(candidates, limit):
    return limit_expansion(list(candidates), limit)

_OPPOSITE = {"forward": "backward", "backward": "forward"}

class Graph:
    """The graph being expanded (see the top of this section)."""

    def __init__(self, addresses, txs, inputs, outputs):
        self.addresses, self.txs, self.inputs, self.outputs = addresses, txs, inputs, outputs
        self.idx = {value[2] for value in txs.values()}
        self.hash_of = {}

    def add_tx(self, i, hop, direction=None):
        # Same hop records as shared/expansion.py: a tx of the first hop found in both
        # directions is recorded as "both"
        h = self.hash_of[i]
        current = self.txs.get(h)
        if current is None:
            self.txs[h] = (hop, direction, i)
        elif direction in _OPPOSITE and current[:2] == (hop, _OPPOSITE[direction]):
            self.txs[h] = (hop, "both", current[2])
        self.idx.add(i)

    def explored(self, records, indices):
        for i, record in zip(indices, records):
            self.hash_of[i] = intern(record[0])
        return records

    def put_input(self, i, item):
        k, spent_h, spent_index, addr_str, age, sequence_num, value = item
        addr_str = intern(addr_str)
        self.inputs[(intern(spent_h), spent_index)] = (addr_str, self.hash_of[i], age, sequence_num, value)
        return addr_str, ("in", i, k)

    def put_output(self, i, item):
        j, index, addr_str, is_spent, value = item
        addr_str = intern(addr_str)
        self.outputs[(self.hash_of[i], index)] = (addr_str, is_spent, value)
        return addr_str, ("out", i, j)

    def insert_whole(self, i, record):
        # explore_tx of shared/expansion.py: the addresses found are added to the graph
        _, ins, outs = record
        for item in ins:
            addr_str, ref = self.put_input(i, item)
            if addr_str not in self.addresses:
                self.addresses[addr_str] = (ref, -1)
        for item in outs:
            addr_str, ref = self.put_output(i, item)
            if addr_str not in self.addresses:
                self.addresses[addr_str] = (ref, -1)

def explore_and_insert(graph, indices, desc):
    indices = list(indices)
    records = graph.explored(explore([(i, "whole", None) for i in indices], desc), indices)
    for i, record in zip(indices, records):
        graph.insert_whole(i, record)

def select_per_node(scans, limit_mode, limit):
    """Candidates of the frontier nodes in order: each node's new candidates (not among
    the previous nodes' candidates), sampled per node with the random node limit, and
    sampled over the hop with the random hop limit."""
    candidates, chosen = [], set()
    for found in scans:
        node_candidates = [t for t in found if t not in chosen]
        if limit_mode == "random node":
            node_candidates = sample(node_candidates, limit)
        candidates += node_candidates
        chosen.update(node_candidates)
    if limit_mode == "random hop":
        candidates = sample(candidates, limit)
    return candidates

### Tx-based whole transaction implied addresses forward-backward expansion

def tx_whole_hop(graph, frontier, scan_func, hop, direction, limit_mode, limit, desc):
    candidates = select_per_node(scan(scan_func, frontier, desc, graph.idx), limit_mode, limit)
    explore_and_insert(graph, candidates, desc)
    for i in candidates:
        graph.add_tx(i, hop, direction)
    return candidates

def expand_fw_bw_tx_whole(graph, num_for_hops, num_back_hops, limit_mode, limit):
    fw_candidates, bw_candidates = [], []
    for fw_found, bw_found in scan(scan_input_output_txes, [r for r, _ in graph.addresses.values()], "hop 1", graph.idx):
        if limit_mode == "random node":
            fw_found, bw_found = sample(fw_found, limit), sample(bw_found, limit)
        fw_candidates += list(fw_found)
        bw_candidates += list(bw_found)
    if limit_mode == "random hop":
        fw_candidates, bw_candidates = sample(fw_candidates, limit), sample(bw_candidates, limit)

    # Every tx is explored once, the forward candidates first
    explore_and_insert(graph, dict.fromkeys(fw_candidates + bw_candidates), "hop 1")
    new_fw, new_bw = list(dict.fromkeys(fw_candidates)), list(dict.fromkeys(bw_candidates))
    for i in new_fw:
        graph.add_tx(i, 1, "forward")
    for i in new_bw:
        graph.add_tx(i, 1, "backward")

    for i in range(num_for_hops - 1):
        new_fw = tx_whole_hop(graph, new_fw, scan_spending, i + 2, "forward", limit_mode, limit, f"forward hop {i + 2}")
    for i in range(num_back_hops - 1):
        new_bw = tx_whole_hop(graph, new_bw, scan_spent, i + 2, "backward", limit_mode, limit, f"backward hop {i + 2}")

### Tx-based all over expansion

def expand_all_over_tx(graph, num_hops, limit_mode, limit):
    explored = []
    total = []
    for found in scan(scan_txes, [r for r, _ in graph.addresses.values()], "hop 1", graph.idx):
        txes = list(found)
        if limit_mode == "random node":
            txes = sample(txes, limit)
        if limit_mode in ("random node", ""):
            explored += txes
        total += txes
    if limit_mode == "random hop":
        explored = sample(total, limit)

    # Every tx is explored once, in the order in which it is first explored
    new_txs = list(dict.fromkeys(explored))
    explore_and_insert(graph, new_txs, "hop 1")
    for i in new_txs:
        graph.add_tx(i, 1)

    for i in range(num_hops - 1):
        new_txs = tx_whole_hop(graph, new_txs, scan_spent_and_spending, i + 2, None, limit_mode, limit, f"hop {i + 2}")

### Tx-based forward-backward expansion with dedicated inputs/outputs

def select_groups(groups_per_node, limit_mode, limit):
    """Candidates of the dedicated variants: tx index -> selectors, merged node by node
    as dict.update does (a later node replaces the selectors of a tx, which keeps its
    position)."""
    candidates = {}
    for groups in groups_per_node:
        node = dict(groups)
        if limit_mode == "random node":
            node = {t: node[t] for t in sample(node.keys(), limit)}
        candidates.update(node)
    if limit_mode == "random hop":
        candidates = {t: candidates[t] for t in sample(candidates.keys(), limit)}
    return candidates

def explore_dedicated(graph, candidates, mode, desc):
    """Explores the candidates with their selected inputs (mode 'input': all the outputs)
    or outputs (mode 'output': all the inputs), as explore_tx does."""
    indices = list(candidates)
    records = graph.explored(explore([(i, mode, candidates[i]) for i in indices], desc), indices)
    for i, record in zip(indices, records):
        graph.insert_whole(i, record)
    return indices

def expand_fw_bw_tx_dedicated(graph, num_for_hops, num_back_hops, limit_mode, limit):
    # Seed by seed: its forward txs (not chosen by a previous seed), sampled, then its
    # backward txs, sampled (the order of the random draws of the sequential algorithm)
    fw_candidates, bw_candidates = {}, {}
    for fw_groups, bw_groups in scan(scan_seed_dedicated, [r for r, _ in graph.addresses.values()], "hop 1", graph.idx):
        for groups, candidates in ((fw_groups, fw_candidates), (bw_groups, bw_candidates)):
            node = {t: s for t, s in groups if t not in candidates}
            if limit_mode == "random node":
                node = {t: node[t] for t in sample(node.keys(), limit)}
            candidates.update(node)
    if limit_mode == "random hop":
        fw_candidates = {t: fw_candidates[t] for t in sample(fw_candidates.keys(), limit)}
        bw_candidates = {t: bw_candidates[t] for t in sample(bw_candidates.keys(), limit)}

    new_fw = explore_dedicated(graph, fw_candidates, "input", "forward hop 1")
    new_bw = explore_dedicated(graph, bw_candidates, "output", "backward hop 1")
    for i in new_fw:
        graph.add_tx(i, 1, "forward")
    for i in new_bw:
        graph.add_tx(i, 1, "backward")

    for i in range(num_for_hops - 1):
        desc = f"forward hop {i + 2}"
        candidates = select_groups(scan(scan_dedicated_spending, new_fw, desc, graph.idx), limit_mode, limit)
        new_fw = explore_dedicated(graph, candidates, "input", desc)
        for t in new_fw:
            graph.add_tx(t, i + 2, "forward")
    for i in range(num_back_hops - 1):
        desc = f"backward hop {i + 2}"
        candidates = select_groups(scan(scan_dedicated_spent, new_bw, desc, graph.idx), limit_mode, limit)
        new_bw = explore_dedicated(graph, candidates, "output", desc)
        for t in new_bw:
            graph.add_tx(t, i + 2, "backward")

### Addr-based dedicated inputs/outputs forward-backward expansion

def addr_dedicated_hop(graph, known, frontier, forward, hop, limit_mode, limit):
    desc = f"{'forward' if forward else 'backward'} hop {hop}"
    scans = scan(scan_address_inputs if forward else scan_address_outputs, [r for r, _ in frontier.values()], desc, graph.idx)
    candidates = select_groups(scans, limit_mode, limit)
    indices = list(candidates)
    records = graph.explored(explore([(i, "input" if forward else "output", candidates[i]) for i in indices], desc), indices)
    new_addresses = {}
    for i, (_, ins, outs) in zip(indices, records):
        graph.add_tx(i, hop, "forward" if forward else "backward")
        # Forward: the inputs of the frontier, then every output (new addresses);
        # backward: the outputs of the frontier, then every input (new addresses)
        if forward:
            for item in ins:
                graph.put_input(i, item)
            for item in outs:
                addr_str, ref = graph.put_output(i, item)
                if addr_str not in known:
                    new_addresses[addr_str] = (ref, -1)
        else:
            for item in outs:
                graph.put_output(i, item)
            for item in ins:
                addr_str, ref = graph.put_input(i, item)
                if addr_str not in known:
                    new_addresses[addr_str] = (ref, -1)
    return new_addresses

def expand_fw_bw_addr_dedicated(graph, num_for_hops, num_back_hops, limit_mode, limit):
    addresses = graph.addresses
    news = {}
    new = addresses
    for i in range(num_for_hops):
        new = addr_dedicated_hop(graph, {**addresses, **news}, new, True, i + 1, limit_mode, limit)
        news.update(new)
    new = addresses
    for i in range(num_back_hops):
        new = addr_dedicated_hop(graph, {**addresses, **news}, new, False, i + 1, limit_mode, limit)
        news.update(new)
    addresses.update(news)

### Addr-based whole transaction implied addresses forward-backward expansion

def addr_whole_hop(graph, known, frontier, forward, hop, limit_mode, limit):
    """The txs where every frontier address is an input (forward) or an output
    (backward), with all their inputs and outputs. Returns the addresses on the other
    side (new) and on the same side (side) of the txs."""
    desc = f"{'forward' if forward else 'backward'} hop {hop}"
    candidates = {}
    for found in scan(scan_input_txes if forward else scan_output_txes, [r for r, _ in frontier.values()], desc, graph.idx):
        txes = sample(found, limit) if limit_mode == "random node" else found
        for t in txes:
            if t not in graph.idx:
                candidates[t] = None
    if limit_mode == "random hop":
        candidates = dict.fromkeys(sample(candidates.keys(), limit))
    indices = list(candidates)
    records = graph.explored(explore([(i, "whole", None) for i in indices], desc), indices)
    new_addresses, side_addresses = {}, {}
    for i, (_, ins, outs) in zip(indices, records):
        graph.add_tx(i, hop, "forward" if forward else "backward")
        same = [(graph.put_input, item) for item in ins] if forward else [(graph.put_output, item) for item in outs]
        other = [(graph.put_output, item) for item in outs] if forward else [(graph.put_input, item) for item in ins]
        for put, item in same:
            addr_str, ref = put(i, item)
            if addr_str not in known and addr_str not in new_addresses:
                side_addresses[addr_str] = (ref, -1)
        for put, item in other:
            addr_str, ref = put(i, item)
            if addr_str not in known:
                new_addresses[addr_str] = (ref, -1)
    return new_addresses, side_addresses

def expand_fw_bw_addr_whole(graph, num_for_hops, num_back_hops, mode, limit_mode, limit):
    addresses = graph.addresses
    hop = lambda known, frontier, forward, i: addr_whole_hop(graph, known, frontier, forward, i + 1, limit_mode, limit)
    if mode == 'none':
        new_fw, new_bw = addresses.copy(), addresses.copy()
        for i in range(num_for_hops):
            new_fw, side_fw = hop(addresses, new_fw, True, i)
            addresses.update({**new_fw, **side_fw})
            new_bw, side_bw = hop(addresses, new_bw, False, i)
            addresses.update({**new_bw, **side_bw})
    elif mode == 'same':
        new, side, news = addresses.copy(), {}, {}
        for i in range(num_for_hops):
            new, side = hop({**addresses, **news}, {**new, **side}, True, i)
            news.update(new)
            news.update(side)
        new, side = addresses.copy(), {}
        for i in range(num_back_hops):
            new, side = hop({**addresses, **news}, {**new, **side}, False, i)
            news.update(new)
            news.update(side)
        addresses.update(news)
    elif mode == 'opposite':
        new_fw, new_bw = addresses.copy(), addresses.copy()
        for i in range(num_for_hops):
            new_fw, side_fw = hop(addresses, new_fw, True, i)
            addresses.update({**new_fw, **side_fw})
            new_bw, side_bw = hop(addresses, new_bw, False, i)
            addresses.update({**new_bw, **side_bw})
            new_fw.update(side_bw)
            new_bw.update(side_fw)
    else:
        raise ValueError(f"Unknown treatment of side addresses: {mode}")

### Addr-based all over expansion

def expand_all_over_addr(graph, num_hops, limit_mode, limit):
    addresses = graph.addresses
    new_addresses = addresses
    for i in range(num_hops):
        desc = f"hop {i + 1}"
        # Candidates repeat when frontier addresses share a tx: each tx is kept once,
        # before the random hop sampling, so a shared tx is not more likely to be chosen
        candidates = {}
        for found in scan(scan_txes, [r for r, _ in new_addresses.values()], desc, graph.idx):
            txes = sample(found, limit) if limit_mode == "random node" else found
            for t in txes:
                if t not in graph.idx:
                    candidates[t] = None
        if limit_mode == "random hop":
            candidates = dict.fromkeys(sample(candidates.keys(), limit))
        indices = list(candidates)
        records = graph.explored(explore([(t, "whole", None) for t in indices], desc), indices)
        hop_addresses = {}
        for t, (_, ins, outs) in zip(indices, records):
            graph.add_tx(t, i + 1)
            for put, item in [(graph.put_input, item) for item in ins] + [(graph.put_output, item) for item in outs]:
                addr_str, ref = put(t, item)
                if addr_str not in addresses:
                    hop_addresses[addr_str] = (ref, -1)
        new_addresses = hop_addresses
        addresses.update(new_addresses)

def expand(addresses, txs, inputs, outputs):
    """Expands the seed addresses (addresses: address string -> (("seed", string), label))
    with the configured variant, filling the graph in place."""
    graph = Graph(addresses, txs, inputs, outputs)
    print(f"Hop 1, addresses to expand: {len(addresses)}", flush=True)
    if exp_alg == "fw bw tx whole":
        expand_fw_bw_tx_whole(graph, for_hops, back_hops, limit_mode, limit)
    elif exp_alg == "fw bw tx dedicated":
        expand_fw_bw_tx_dedicated(graph, for_hops, back_hops, limit_mode, limit)
    elif exp_alg == "all over tx":
        expand_all_over_tx(graph, hops, limit_mode, limit)
    elif exp_alg.startswith("fw bw addr whole"):
        expand_fw_bw_addr_whole(graph, for_hops, back_hops, exp_alg.split()[-1], limit_mode, limit)
    elif exp_alg == "fw bw addr dedicated":
        expand_fw_bw_addr_dedicated(graph, for_hops, back_hops, limit_mode, limit)
    elif exp_alg == "all over addr":
        expand_all_over_addr(graph, hops, limit_mode, limit)
    else:
        raise ValueError(f"Unknown expansion variant '{exp_alg}'")
    print(f"Final number of addrs: {len(addresses)}, txs: {len(txs)}, inputs: {len(inputs)}, outputs: {len(outputs)}", flush=True)


def tx_wait(heights):
    if len(heights) <= 1:
        return []

    heights = sorted(heights)
    gaps = [b - a for a, b in zip(heights[:-1], heights[1:])]
    return gaps

def summarize(values):
    # min, max, mean and std of a list, or zeros when the list is empty. The values
    # are sorted first, so the result does not depend on the order in which the
    # history of an address is read (floating-point sums depend on the order)
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.sort(np.asarray(values))
    return arr[0].item(), arr[-1].item(), arr.mean(), arr.std()

### Collecting nodes and edges features

# Expanded addresses share most of their txs, so per-tx stats are computed once
# (in every worker process)
in_tx_stats = {}
out_tx_stats = {}

def history_loop(addr):
    """History of an address, read input by input and output by output."""
    income_list = []
    utxo_values = []
    for out in addr.outputs:
        value = int(out.value)
        income_list.append(value)
        if not out.is_spent:
            utxo_values.append(value)

    spent_values = []
    input_ages = []
    for inp in addr.inputs:
        spent_values.append(int(inp.value))
        input_ages.append(inp.age)

    in_txs = ([], [], [], [])
    for in_tx in addr.input_txes:
        stats = in_tx_stats.get(in_tx.index)
        if stats is None:
            stats = in_tx_stats[in_tx.index] = (in_tx.block_height, in_tx.input_count, in_tx.output_count, int(in_tx.input_value))
        for column, value in zip(in_txs, stats):
            column.append(value)

    out_txs = ([], [], [], [])
    for out_tx in addr.output_txes:
        stats = out_tx_stats.get(out_tx.index)
        if stats is None:
            stats = out_tx_stats[out_tx.index] = (out_tx.block_height, out_tx.input_count, out_tx.output_count, int(out_tx.output_value))
        for column, value in zip(out_txs, stats):
            column.append(value)

    return income_list, utxo_values, spent_values, input_ages, in_txs, out_txs


def column(rng, attribute):
    # A whole attribute of a BlockSci range as a list, read in one call
    return np.asarray(getattr(rng, attribute)).tolist()


def tx_columns(tx_range, value_attribute, unique):
    # Block height, input count, output count and input/output value of the txs of a range,
    # optionally keeping only the first occurrence of every tx
    columns = [column(tx_range, a) for a in ("block_height", "input_count", "output_count", value_attribute)]
    if unique:
        seen, keep = set(), []
        for i, tx_index in enumerate(column(tx_range, "index")):
            if tx_index not in seen:
                seen.add(tx_index)
                keep.append(i)
        columns = [[c[i] for i in keep] for c in columns]
    return tuple(columns)


def history_vectorized(addr, unique_txes=False):
    """Same history as history_loop, with every attribute read as one array by BlockSci."""
    outputs = addr.outputs
    income_list = column(outputs, "value")
    spent = column(outputs, "is_spent")
    utxo_values = [value for value, is_spent in zip(income_list, spent) if not is_spent]

    inputs = addr.inputs
    spent_values = column(inputs, "value")
    input_ages = column(inputs, "age")

    in_txs = tx_columns(addr.input_txes, "input_value", unique_txes)
    out_txs = tx_columns(addr.output_txes, "output_value", unique_txes)
    return income_list, utxo_values, spent_values, input_ages, in_txs, out_txs


HISTORY_COMPONENTS = ["output values", "unspent output values", "input values", "input ages",
                      "spending tx heights", "spending tx input counts", "spending tx output counts", "spending tx input values",
                      "funding tx heights", "funding tx input counts", "funding tx output counts", "funding tx output values"]


def flatten(history):
    income_list, utxo_values, spent_values, input_ages, in_txs, out_txs = history
    return [income_list, utxo_values, spent_values, input_ages] + list(in_txs) + list(out_txs)


def history_difference(fast, slow):
    """First history list whose values differ (ignoring their order), or None. Every
    address feature is independent of the order of the history."""
    for name, a, b in zip(HISTORY_COMPONENTS, flatten(fast), flatten(slow)):
        if sorted(a) != sorted(b):
            return f"{name}: {len(a)} values with arrays vs {len(b)} item by item, e.g. {sorted(a)[:5]} vs {sorted(b)[:5]}"
    return None


# The vectorized reading is only used if it gives the same history as the loop on a
# sample of addresses (check_vectorized_history), otherwise the loop is used
VECTORIZED_HISTORY = {"enabled": False, "unique_txes": False}


def check_vectorized_history(sample):
    """Enables the vectorized reading if one of its variants (with the txs as given by
    BlockSci, or with duplicated txs removed) matches the loop on every sampled address."""
    slow = [history_loop(addr) for addr in sample]
    reasons = []
    for unique_txes in (False, True):
        variant = "unique txs" if unique_txes else "txs as given"
        try:
            difference = None
            for addr, expected in zip(sample, slow):
                difference = history_difference(history_vectorized(addr, unique_txes), expected)
                if difference:
                    reasons.append(f"  {variant}: differs for {address_string(addr)} in {difference}")
                    break
        except Exception as error:
            reasons.append(f"  {variant}: not available ({type(error).__name__}: {error})")
            continue
        if difference is None and sample:
            VECTORIZED_HISTORY.update(enabled=True, unique_txes=unique_txes)
            print(f"Vectorized history ({variant}) identical on {len(sample)} sampled addresses: enabled")
            return True
    VECTORIZED_HISTORY.update(enabled=False)
    print("Vectorized history not used, reading addresses one item at a time:\n" + "\n".join(reasons))
    return False


def address_history(addr):
    if VECTORIZED_HISTORY["enabled"]:
        try:
            return history_vectorized(addr, VECTORIZED_HISTORY["unique_txes"])
        except Exception:
            pass
    return history_loop(addr)


def address_features(item):
    addr, label = item
    # spent_values: values of the inputs of the address; input_values: input value of each tx spending from it
    income_list, utxo_values, spent_values, input_ages, in_txs, out_txs = address_history(addr)
    in_heights, in_tx_inputs, in_tx_outputs, input_values = in_txs
    out_heights, out_tx_inputs, out_tx_outputs, output_values = out_txs
    n_utxos = len(utxo_values)

    sorted_incomes = sorted(income_list, reverse=True)
    income = sum(sorted_incomes)

    if income > 0:
        top1_share = float(sum(sorted_incomes[:1])) / float(income)
        top5_share = float(sum(sorted_incomes[:5])) / float(income)
    else:
        top1_share = top5_share = 0.0

    expenditure = sum(spent_values)

    min_input_age, max_input_age, avg_input_age, std_input_age = summarize(input_ages)

    balance = income - expenditure
    outflow_ratio = float(expenditure) / float(income) if income > 0 else 0.0

    n_incoming_txs = len(in_heights)
    n_outgoing_txs = len(out_heights)
    n_total_txs = n_incoming_txs + n_outgoing_txs

    min_utxo_value, max_utxo_value, avg_utxo_value, std_utxo_value = summarize(utxo_values)
    min_out_tx_value, max_out_tx_value, avg_out_tx_value, std_out_tx_value = summarize(output_values)
    min_in_tx_value, max_in_tx_value, avg_in_tx_value, std_in_tx_value = summarize(input_values)
    min_in_tx_inputs, max_in_tx_inputs, avg_in_tx_inputs, std_in_tx_inputs = summarize(in_tx_inputs)
    min_in_tx_outputs, max_in_tx_outputs, avg_in_tx_outputs, std_in_tx_outputs = summarize(in_tx_outputs)
    min_out_tx_inputs, max_out_tx_inputs, avg_out_tx_inputs, std_out_tx_inputs = summarize(out_tx_inputs)
    min_out_tx_outputs, max_out_tx_outputs, avg_out_tx_outputs, std_out_tx_outputs = summarize(out_tx_outputs)

    # addr.txes is the union of addr.input_txes and addr.output_txes
    tx_heights = in_heights + out_heights
    if tx_heights:
        first_tx_block = min(tx_heights)
        last_tx_block = max(tx_heights)
    else:
        first_tx_block = -1
        last_tx_block = -1

    in_tx_wait = tx_wait(in_heights)
    min_in_tx_wait, max_in_tx_wait, avg_in_tx_wait, std_in_tx_wait = summarize(in_tx_wait if len(in_tx_wait) > 1 else [])

    out_tx_wait = tx_wait(out_heights)
    min_out_tx_wait, max_out_tx_wait, avg_out_tx_wait, std_out_tx_wait = summarize(out_tx_wait if len(out_tx_wait) > 1 else [])

    info = {
        'addr_str': addr.address_string,
        'full_type': addr.full_type,
        'class': label,
        'income': income,
        'expenditure': expenditure,
        'balance': balance,
        'outflow_ratio': outflow_ratio,
        'n_incoming_txs': n_incoming_txs,
        'n_outgoing_txs': n_outgoing_txs,
        'n_total_txs': n_total_txs,
        'top1_income_share': top1_share,
        'top5_income_share': top5_share,
        'n_utxos': n_utxos,
        'avg_utxo_value': avg_utxo_value,
        'max_utxo_value': max_utxo_value,
        'min_utxo_value': min_utxo_value,
        'std_utxo_value': std_utxo_value,
        'min_out_tx_value': min_out_tx_value,
        'max_out_tx_value': max_out_tx_value,
        'avg_out_tx_value': avg_out_tx_value,
        'std_out_tx_value': std_out_tx_value,
        'min_in_tx_value': min_in_tx_value,
        'max_in_tx_value': max_in_tx_value,
        'avg_in_tx_value': avg_in_tx_value,
        'std_in_tx_value': std_in_tx_value,
        'min_in_tx_inputs': min_in_tx_inputs,
        'max_in_tx_inputs': max_in_tx_inputs,
        'avg_in_tx_inputs': avg_in_tx_inputs,
        'std_in_tx_inputs': std_in_tx_inputs,
        'min_in_tx_outputs': min_in_tx_outputs,
        'max_in_tx_outputs': max_in_tx_outputs,
        'avg_in_tx_outputs': avg_in_tx_outputs,
        'std_in_tx_outputs': std_in_tx_outputs,
        'min_out_tx_inputs': min_out_tx_inputs,
        'max_out_tx_inputs': max_out_tx_inputs,
        'avg_out_tx_inputs': avg_out_tx_inputs,
        'std_out_tx_inputs': std_out_tx_inputs,
        'min_out_tx_outputs': min_out_tx_outputs,
        'max_out_tx_outputs': max_out_tx_outputs,
        'avg_out_tx_outputs': avg_out_tx_outputs,
        'std_out_tx_outputs': std_out_tx_outputs,
        'first_tx_block': first_tx_block,
        'last_tx_block': last_tx_block,
        'min_in_tx_wait': min_in_tx_wait,
        'max_in_tx_wait': max_in_tx_wait,
        'avg_in_tx_wait': avg_in_tx_wait,
        'std_in_tx_wait': std_in_tx_wait,
        'min_out_tx_wait': min_out_tx_wait,
        'max_out_tx_wait': max_out_tx_wait,
        'avg_out_tx_wait': avg_out_tx_wait,
        'std_out_tx_wait': std_out_tx_wait,
        'min_input_age': min_input_age,
        'max_input_age': max_input_age,
        'avg_input_age': avg_input_age,
        'std_input_age': std_input_age
    }
    return info

def tx_features(item):
    h, tx = item
    return {
        'hash': h,
        'block_height': tx.block_height,
        'fee': tx.fee,
        'is_coinbase': tx.is_coinbase,
        'locktime': tx.locktime,
        'total_size': tx.total_size,
        'version': tx.version
    }

def extract_spent_pairs(inputs_df, outputs_df):
    spent_pairs_df = (
        inputs_df.merge(
            outputs_df,
            left_on  = ['spent_tx_hash', 'spent_output_index'],
            right_on = ['tx_hash',       'index'],
            how      = 'inner',
            suffixes = ('_spend', '_spent')
        )
        [['tx_hash_spend', 'tx_hash_spent']]
    )
    return spent_pairs_df

def address_features_from_ref(item):
    _, (ref, label) = item
    return address_features((resolve_address(ref), label))

def tx_features_from_index(item):
    h, i = item
    return tx_features((h, tx_of(i)))

# Address features are computed and saved in blocks, so an interrupted export
# keeps the completed blocks
ADDRESS_BLOCK = 100000

# The features of an address only depend on its history in the chain, except its class
# (the label of the split), and the graphs of the splits share many addresses (mostly
# the busiest ones, whose features are the costliest). The features computed for a
# split are kept in this directory, block by block, and reused by the next splits with
# their own class; it is removed once every split is exported. Addresses are identified
# by their string and full type (a pubkey address shares its string with its
# pubkey-hash address, but not its type or its history).
def feature_cache_dir():
    return data_path / "_address_features_cache"

def address_key(addr):
    return (addr.address_string, addr.full_type)

def address_key_from_ref(item):
    _, (ref, _) = item
    return address_key(resolve_address(ref))

def cached_feature_index(split):
    """Address key -> (cache file, position) of the features computed for the other splits."""
    index, files = {}, []
    cache_dir = feature_cache_dir()
    if not cache_dir.exists():
        return index, files
    for keys_path in sorted(cache_dir.glob("*.keys.pkl")):
        if keys_path.name.startswith(f"{split}_"):
            continue
        with open(keys_path, "rb") as f:
            keys = pickle.load(f)
        file_id = len(files)
        files.append(keys_path.with_name(keys_path.name.replace(".keys.pkl", ".rows.pkl")))
        for position, key in enumerate(keys):
            index.setdefault(key, (file_id, position))
    return index, files

def cached_rows(needed, files):
    """Feature rows of the given (file id, position), read file by file."""
    rows = {}
    by_file = {}
    for location in set(needed):
        by_file.setdefault(location[0], []).append(location[1])
    for file_id, positions in by_file.items():
        with open(files[file_id], "rb") as f:
            file_rows = pickle.load(f)
        for position in positions:
            rows[(file_id, position)] = file_rows[position]
        del file_rows
    return rows

def save_cached_block(split, b, results):
    cache_dir = feature_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows_path = cache_dir / f"{split}_{b:05d}.rows.pkl"
    keys_path = cache_dir / f"{split}_{b:05d}.keys.pkl"
    # The keys file is written last, so only complete blocks are reused
    for path, data in [(rows_path, results), (keys_path, [(r["addr_str"], r["full_type"]) for r in results])]:
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(data, f, protocol=4)
        os.replace(str(tmp), str(path))

def address_features_in_blocks(items, partial_dir, split, weights):
    """Features of the addresses, computed in parallel block by block. Within a block the
    addresses with most inputs and outputs in the graph (the busiest ones, whose history
    is the longest) are given first to the workers, in small chunks, so a few busy
    addresses do not keep one worker busy at the end of the block; the results are put
    back in the order of the addresses."""
    # Sample spread over all the addresses, to check the vectorized history reading
    sample = [resolve_address(ref) for _, (ref, _) in items[::max(1, len(items) // 200)][:200]]
    check_vectorized_history(sample)

    index, files = cached_feature_index(split)
    partial_dir.mkdir(parents=True, exist_ok=True)
    n_blocks = (len(items) + ADDRESS_BLOCK - 1) // ADDRESS_BLOCK
    blocks = []
    n_reused = 0
    for b in range(n_blocks):
        path = partial_dir / f"block_{b:05d}.pkl"
        if path.exists():
            blocks.append(pd.read_pickle(path))
            continue
        block = items[b * ADDRESS_BLOCK:(b + 1) * ADDRESS_BLOCK]
        desc = f"block {b + 1}/{n_blocks}"

        # Addresses whose features were computed for another split are reused
        locations = {}
        if index:
            keys = parallel_map(address_key_from_ref, block, WORKERS, desc=f"Address keys ({desc})")
            locations = {i: index[key] for i, key in enumerate(keys) if key in index}
        pending = [item for i, item in enumerate(block) if i not in locations]
        order = sorted(range(len(pending)), key=lambda p: -weights.get(pending[p][0], 0))
        rows = parallel_map(address_features_from_ref, [pending[p] for p in order], WORKERS,
                            desc=f"Address features ({desc}, {len(locations)} reused)", chunks_per_worker=64, max_chunk=16)
        in_order = [None] * len(pending)
        for p, row in zip(order, rows):
            in_order[p] = row
        computed = iter(in_order)
        del rows
        reused = cached_rows(locations.values(), files)
        results = []
        for i, (_, (_, label)) in enumerate(block):
            if i in locations:
                row = dict(reused[locations[i]])
                row["class"] = label
            else:
                row = next(computed)
            results.append(row)
        n_reused += len(locations)
        del reused

        save_cached_block(split, b, results)
        df = pd.DataFrame(results)
        tmp = path.with_name(path.name + ".tmp")
        df.to_pickle(tmp)
        os.replace(str(tmp), str(path))
        blocks.append(df)
    if index:
        print(f"Address features reused from other splits: {n_reused} of {len(items)}", flush=True)
    return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()

def export_records(addrs, txs, inputs, outputs, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Inputs and outputs of every address in the graph, to give the busiest ones first
    weights = Counter(addr_str for addr_str, *_ in inputs.values())
    weights.update(addr_str for addr_str, *_ in outputs.values())
    addr_df = address_features_in_blocks(list(addrs.items()), out_dir / "_partial_addr_feats", out_dir.name, weights)
    del weights
    tx_df = pd.DataFrame(parallel_map(tx_features_from_index, [(h, i) for h, (_, _, i) in txs.items()], WORKERS,
                                      desc="Transaction features"))
    # Hop where each tx was added to the graph (not used as a GNN feature)
    tx_df["hop"] = pd.array([hop for hop, _, _ in txs.values()], dtype="Int64")
    tx_df["hop_direction"] = [direction for _, direction, _ in txs.values()]
    inputs_df = pd.DataFrame([
        {'addr_str': addr_str, 'tx_hash': h, 'age': age, 'sequence_num': sequence_num, 'value': value,
         'spent_tx_hash': spent_hash, 'spent_output_index': spent_index}
        for (spent_hash, spent_index), (addr_str, h, age, sequence_num, value) in inputs.items()
    ])
    outputs_df = pd.DataFrame([
        {'tx_hash': h, 'addr_str': addr_str, 'index': index, 'is_spent': is_spent, 'value': value}
        for (h, index), (addr_str, is_spent, value) in outputs.items()
    ])
    spent_pairs_df = extract_spent_pairs(inputs_df, outputs_df)

    addr_df.to_csv(out_dir / "addr_feats.csv", index=False)
    tx_df.to_csv(out_dir / "tx_feats.csv", index=False)
    inputs_df.to_csv(out_dir / "input_feats.csv", index=False)
    outputs_df.to_csv(out_dir / "output_feats.csv", index=False)
    spent_pairs_df.to_csv(out_dir / 'spent_pairs.csv', index=False)

    return addr_df, tx_df, inputs_df, outputs_df

### Exporting data

EXPORT_FILES = ["addr_feats.csv", "tx_feats.csv", "input_feats.csv", "output_feats.csv", "spent_pairs.csv"]

# Format of the expanded graph saved before the export (_expansion_state.pkl)
STATE_FORMAT = 2

def load_state(state_path):
    """The expanded graph saved before the export, or None (also for a graph saved by an
    earlier version, with transactions identified by their hash: it is expanded again)."""
    if not state_path.exists():
        return None
    with open(state_path, "rb") as f:
        state = pickle.load(f)
    if not isinstance(state, dict) or state.get("format") != STATE_FORMAT:
        print(f"{state_path} was saved by an earlier version: the split is expanded again", flush=True)
        state_path.unlink()
        return None
    print(f"Loading the expanded graph saved at {state_path}", flush=True)
    return state["addresses"], state["txs"], state["inputs"], state["outputs"]

def expand_and_export(split_idx, split, seeds):
    print(f"\n{split} set expansion ({len(seeds)} seed addresses, {WORKERS} workers)", flush=True)

    # Each split has its own fixed random state, independent of the other splits
    random.seed(config.DATA_SEED + split_idx)
    np.random.seed(config.DATA_SEED + split_idx)

    # The expanded graph is saved before the export, so an interrupted export does not
    # repeat the expansion
    state_path = data_path / split / "_expansion_state.pkl"
    state = load_state(state_path)
    if state is None:
        addresses = {addr_str: (("seed", addr_str), label) for addr_str, label in seeds}
        txs, inputs, outputs = {}, {}, {}
        expand(addresses, txs, inputs, outputs)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = state_path.with_name(state_path.name + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump({"format": STATE_FORMAT, "addresses": addresses, "txs": txs, "inputs": inputs,
                         "outputs": outputs}, f, protocol=4)
        os.replace(str(tmp), str(state_path))
    else:
        addresses, txs, inputs, outputs = state
    export_records(addresses, txs, inputs, outputs, data_path / split)
    # Checkpoints are removed once the split is completely exported
    state_path.unlink()
    shutil.rmtree(str(data_path / split / "_partial_addr_feats"), ignore_errors=True)
    print(f"{split}: exported to {data_path / split}", flush=True)

def detect_tx_rebuild(split_seeds):
    """Detects how the workers rebuild a transaction from its index, on a tx of a seed."""
    for seeds in split_seeds.values():
        for addr_str, _ in seeds:
            for tx in chain.address_from_string(addr_str).txes:
                return detect_tx_by_index(tx)
    raise RuntimeError("No seed address has a transaction")


if __name__ == "__main__":
    limit_threads()
    split_seeds = load_split_seeds()
    tx_rebuild = detect_tx_rebuild(split_seeds)
    write_json(data_path / "expansion_config.json", {
        "created": timestamp(),
        "git_commit": git_commit(config.PROJECT_ROOT),
        "expansion": EXPANSION,
        "exp_alg": exp_alg,
        "data_seed": config.DATA_SEED,
        "random_state": "random.seed(DATA_SEED + split index) before expanding each split",
        "workers": WORKERS,
        "tx_rebuild": tx_rebuild,
        "splits_file": str(config.SPLITS_FILE),
    })

    for split_idx, split in enumerate(config.SPLITS):
        if all((data_path / split / name).exists() for name in EXPORT_FILES):
            print(f"{split}: already expanded, skipping ({data_path / split})")
        else:
            expand_and_export(split_idx, split, split_seeds[split])

    # The features shared between splits are no longer needed once every split is exported
    shutil.rmtree(str(feature_cache_dir()), ignore_errors=True)
