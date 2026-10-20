# STEP 1 (Methodology I) - GRAPH EXPANSION AND FEATURE EXTRACTION (requires BlockSci)
#
# Expands the seed addresses of each split (SPLITS_FILE, created by
# create_splits.py) into one merged heterogeneous graph per split, using the
# expansion configuration MULTI_INSTANCE_EXPANSION of config.py, and exports the
# node and edge features of every graph to MULTI_INSTANCE_GRAPHS_DIR/<split>/:
#   addr_feats.csv, tx_feats.csv, input_feats.csv, output_feats.csv, spent_pairs.csv
# tx_feats.csv also records the hop where every tx was added to the graph (hop,
# hop_direction). The graph is built by the parallel expansion of
# shared/parallel_expansion.py, with the same variants as shared/expansion.py (used by
# the ego-centric methodology).
#
# Every split is expanded from its own fixed random seed, so a split can be
# rebuilt on its own and gives the same graph in every execution. A split already
# exported by the same code (this file, shared/expansion.py and
# shared/parallel_expansion.py) and parameters is skipped; one built by other code or
# parameters is expanded again (see SPLIT_META).
#
# The splits are processed one after another, each with all the cores
# (config.MULTI_INSTANCE_EXPANSION_WORKERS forked processes): the workers do every
# BlockSci access (the expansion, see shared/parallel_expansion.py, and the features),
# and the main process keeps plain values only and performs the order-dependent steps,
# so the graph does not depend on the number of workers. The features of an address are computed
# once and reused by the next splits whose graph contains it (see feature_cache_dir),
# and the busiest addresses of every block are computed first.

### Import libraries

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# One worker per core: single-threaded numerical libraries (before importing them)
from shared.threads import limit_threads

from collections import Counter

import pandas as pd
import numpy as np
import ast
import hashlib
import json
import random
import math
import os
import pickle
import shutil
import blocksci

import config
from shared import parallel_expansion
from shared.expansion import address_string
from shared.parallel_expansion import detect_tx_by_index, resolve_address, tx_of
from shared.experiment import git_commit, read_json, timestamp, write_json
from shared.parallel import blocksci_workers, forked_pool, parallel_map
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
# The forked workers of the expansion and of the features inherit the chain
parallel_expansion.configure(chain, WORKERS)

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
# The seed addresses of a split are expanded into one graph by the parallel expansion
# shared with the expansion experiments (shared/parallel_expansion.py, which describes
# the graph it builds), with the variant of MULTI_INSTANCE_EXPANSION.

def expand(addresses, txs, inputs, outputs):
    """Expands the seed addresses (addresses: address string -> (("seed", string), label))
    with the configured variant, filling the graph in place."""
    parallel_expansion.expand(exp_alg, addresses, txs, inputs, outputs, for_hops=for_hops, back_hops=back_hops,
                              hops=hops, limit_mode=limit_mode, limit=limit)

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

    in_txs = ([], [], [], [], [])
    for in_tx in addr.input_txes:
        stats = in_tx_stats.get(in_tx.index)
        if stats is None:
            stats = in_tx_stats[in_tx.index] = (in_tx.block_height, in_tx.input_count, in_tx.output_count, int(in_tx.input_value), in_tx.index)
        for column, value in zip(in_txs, stats):
            column.append(value)

    out_txs = ([], [], [], [], [])
    for out_tx in addr.output_txes:
        stats = out_tx_stats.get(out_tx.index)
        if stats is None:
            stats = out_tx_stats[out_tx.index] = (out_tx.block_height, out_tx.input_count, out_tx.output_count, int(out_tx.output_value), out_tx.index)
        for column, value in zip(out_txs, stats):
            column.append(value)

    return income_list, utxo_values, spent_values, input_ages, in_txs, out_txs


def column(rng, attribute):
    # A whole attribute of a BlockSci range as a list, read in one call
    return np.asarray(getattr(rng, attribute)).tolist()


def tx_columns(tx_range, value_attribute, unique):
    # Block height, input count, output count, input/output value and index of the txs of
    # a range, optionally keeping only the first occurrence of every tx
    columns = [column(tx_range, a) for a in ("block_height", "input_count", "output_count", value_attribute, "index")]
    if unique:
        seen, keep = set(), []
        for i, tx_index in enumerate(columns[-1]):
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
                      "spending tx indices",
                      "funding tx heights", "funding tx input counts", "funding tx output counts", "funding tx output values",
                      "funding tx indices"]


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
    # in_txs: the txs where the address is an input (it spends: its outgoing txs);
    # out_txs: the txs where the address is an output (it receives: its incoming txs).
    # The in_tx_* / out_tx_* features keep this BlockSci naming (input/output txs).
    income_list, utxo_values, spent_values, input_ages, in_txs, out_txs = address_history(addr)
    in_heights, in_tx_inputs, in_tx_outputs, input_values, in_indices = in_txs
    out_heights, out_tx_inputs, out_tx_outputs, output_values, out_indices = out_txs
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

    n_incoming_txs = len(out_heights)
    n_outgoing_txs = len(in_heights)
    # Distinct txs: a tx where the address both spends and receives (e.g. its own change)
    # is counted once
    n_total_txs = len(set(in_indices) | set(out_indices))

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

    # Blocks between consecutive txs (at least two txs: one gap or more)
    min_in_tx_wait, max_in_tx_wait, avg_in_tx_wait, std_in_tx_wait = summarize(tx_wait(in_heights))
    min_out_tx_wait, max_out_tx_wait, avg_out_tx_wait, std_out_tx_wait = summarize(tx_wait(out_heights))

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

def address_features_in_blocks(items, partial_dir, split, weights, pool):
    """Features of the addresses, computed in parallel block by block by the workers of
    `pool`. Within a block the addresses with most inputs and outputs in the graph (the
    busiest ones, whose history is the longest) are given first to the workers, in small
    chunks, so a few busy addresses do not keep one worker busy at the end of the block;
    the results are put back in the order of the addresses."""
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
            keys = parallel_map(address_key_from_ref, block, WORKERS, desc=f"Address keys ({desc})", pool=pool)
            locations = {i: index[key] for i, key in enumerate(keys) if key in index}
        pending = [item for i, item in enumerate(block) if i not in locations]
        order = sorted(range(len(pending)), key=lambda p: -weights.get(pending[p][0], 0))
        rows = parallel_map(address_features_from_ref, [pending[p] for p in order], WORKERS,
                            desc=f"Address features ({desc}, {len(locations)} reused)", chunks_per_worker=64, max_chunk=16,
                            pool=pool)
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
    items = list(addrs.items())
    # Sample spread over all the addresses, to check the vectorized history reading (before
    # the workers are forked, so they inherit the choice)
    check_vectorized_history([resolve_address(ref) for _, (ref, _) in items[::max(1, len(items) // 200)][:200]])

    # One pool for every feature pass of the split, so the per-tx stats cached by the
    # workers (in_tx_stats, out_tx_stats) last for the whole split
    with forked_pool(WORKERS) as pool:
        addr_df = address_features_in_blocks(items, out_dir / "_partial_addr_feats", out_dir.name, weights, pool)
        del weights, items
        tx_df = pd.DataFrame(parallel_map(tx_features_from_index, [(h, i) for h, (_, _, i) in txs.items()], WORKERS,
                                          desc="Transaction features", pool=pool))
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
STATE_FORMAT = 3

# Every output of a split (exported files, saved graph, partial address features) and the
# address features shared between splits are only reused if they were built by the same
# code and parameters, identified by a fingerprint: the code of this script and of
# shared/expansion.py (their syntax tree, so comments and formatting do not count), the
# expansion configuration, DATA_SEED, and the seeds of the split. The fingerprint of a
# completely exported split is saved in <split>/SPLIT_META, and is carried by the next
# steps (graph.pth, normalization, training runs), which are redone when it changes.
SPLIT_META = "expansion_meta.json"

def code_fingerprint():
    digest = hashlib.sha256()
    for path in (Path(__file__).resolve(), PROJECT_ROOT / "shared" / "expansion.py",
                 PROJECT_ROOT / "shared" / "parallel_expansion.py"):
        digest.update(ast.dump(ast.parse(path.read_text())).encode())
    return digest.hexdigest()[:16]

def split_fingerprint(code, split_idx, seeds):
    data = {"code": code, "expansion": EXPANSION, "exp_alg": exp_alg, "data_seed": config.DATA_SEED,
            "split_index": split_idx, "seeds": seeds}
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()[:16]

def exported_fingerprint(split):
    """Fingerprint of the completely exported split, or None."""
    meta_path = data_path / split / SPLIT_META
    if not meta_path.exists() or not all((data_path / split / name).exists() for name in EXPORT_FILES):
        return None
    return read_json(meta_path).get("fingerprint")

def check_feature_cache(code):
    # The address features shared between splits only depend on the code
    cache_dir = feature_cache_dir()
    marker = cache_dir / "code_fingerprint.txt"
    if cache_dir.exists() and (not marker.exists() or marker.read_text().strip() != code):
        print(f"Removing {cache_dir}: computed by other code", flush=True)
        shutil.rmtree(str(cache_dir))
    cache_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(code)

def load_state(state_path, fingerprint):
    """The expanded graph saved before the export, or None (also for a graph saved by an
    earlier version or with another fingerprint: the split is expanded again)."""
    if not state_path.exists():
        return None
    with open(state_path, "rb") as f:
        state = pickle.load(f)
    if not isinstance(state, dict) or state.get("format") != STATE_FORMAT or state.get("fingerprint") != fingerprint:
        print(f"{state_path} was saved by other code or parameters: the split is expanded again", flush=True)
        state_path.unlink()
        return None
    print(f"Loading the expanded graph saved at {state_path}", flush=True)
    return state["addresses"], state["txs"], state["inputs"], state["outputs"]

def expand_and_export(split_idx, split, seeds, fingerprint):
    print(f"\n{split} set expansion ({len(seeds)} seed addresses, {WORKERS} workers)", flush=True)
    split_dir = data_path / split
    partial_dir = split_dir / "_partial_addr_feats"
    # The files of an earlier export are no longer valid (until SPLIT_META is written again)
    if (split_dir / SPLIT_META).exists():
        (split_dir / SPLIT_META).unlink()

    # Each split has its own fixed random state, independent of the other splits
    random.seed(config.DATA_SEED + split_idx)
    np.random.seed(config.DATA_SEED + split_idx)

    # The expanded graph is saved before the export, so an interrupted export does not
    # repeat the expansion
    state_path = split_dir / "_expansion_state.pkl"
    state = load_state(state_path, fingerprint)
    if state is None:
        # Partial address features of another graph
        shutil.rmtree(str(partial_dir), ignore_errors=True)
        addresses = {addr_str: (("seed", addr_str), label) for addr_str, label in seeds}
        txs, inputs, outputs = {}, {}, {}
        expand(addresses, txs, inputs, outputs)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = state_path.with_name(state_path.name + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump({"format": STATE_FORMAT, "fingerprint": fingerprint, "addresses": addresses, "txs": txs,
                         "inputs": inputs, "outputs": outputs}, f, protocol=4)
        os.replace(str(tmp), str(state_path))
    else:
        addresses, txs, inputs, outputs = state
    export_records(addresses, txs, inputs, outputs, split_dir)
    # Written last: it marks the split as completely exported with this fingerprint
    write_json(split_dir / SPLIT_META, {"fingerprint": fingerprint, "created": timestamp(),
                                        "git_commit": git_commit(config.PROJECT_ROOT)})
    # Checkpoints are removed once the split is completely exported
    state_path.unlink()
    shutil.rmtree(str(partial_dir), ignore_errors=True)
    print(f"{split}: exported to {split_dir}", flush=True)

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
    code = code_fingerprint()
    fingerprints = {split: split_fingerprint(code, split_idx, split_seeds[split])
                    for split_idx, split in enumerate(config.SPLITS)}
    write_json(data_path / "expansion_config.json", {
        "created": timestamp(),
        "git_commit": git_commit(config.PROJECT_ROOT),
        "code_fingerprint": code,
        "split_fingerprints": fingerprints,
        "expansion": EXPANSION,
        "exp_alg": exp_alg,
        "data_seed": config.DATA_SEED,
        "random_state": "random.seed(DATA_SEED + split index) before expanding each split",
        "workers": WORKERS,
        "tx_rebuild": tx_rebuild,
        "splits_file": str(config.SPLITS_FILE),
    })
    check_feature_cache(code)

    for split_idx, split in enumerate(config.SPLITS):
        exported = exported_fingerprint(split)
        if exported == fingerprints[split]:
            print(f"{split}: already expanded with this code and parameters, skipping ({data_path / split})")
            continue
        if (data_path / split).exists() and any((data_path / split / name).exists() for name in EXPORT_FILES):
            print(f"{split}: the existing graph was built by other code or parameters "
                  f"({exported or 'unknown'} instead of {fingerprints[split]}), expanding it again", flush=True)
        expand_and_export(split_idx, split, split_seeds[split], fingerprints[split])

    # The features shared between splits are no longer needed once every split is exported
    shutil.rmtree(str(feature_cache_dir()), ignore_errors=True)
