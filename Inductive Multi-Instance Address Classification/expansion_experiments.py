# RANSOMWARE ADDRESSES DETECTION - EXPANSION EXPERIMENTS
#
# Runs the "fw bw tx whole" expansion of expansion.py with a fixed
# "random node" width limit over the following experiment grid:
#   - width limit 5, hops 1, 2 and 3
#   - 2 hops, width limits 1, 2, 5 and 10
# Every configuration is saved exactly like expansion.py does, in
# outputs/<run name>/{train,val,test}/ plus graph_config.json.
# Each split folder also gets an expansion_stats.json with the expansion
# runtime, peak memory and width-limit activation/retention counts, which
# experiments_metrics.py reads.

### Import libraries

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import random
import math
import threading
import time
import blocksci

from shared.paths import (
    ensure_split_dirs,
    get_required_env_path,
)
from experiments_config import (
    direction,
    graph,
    address,
    side_direction,
    exp_alg,
    limit_mode,
    train_samples,
    val_samples,
    test_samples,
    EXPERIMENTS,
    experiment_run_dir,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

### Initialize Blocksci

BLOCKSCI_CONFIG = get_required_env_path("RSD_BLOCKSCI_CONFIG")
BITCOINHEIST_CSV = get_required_env_path("RSD_BITCOINHEIST_CSV")
chain = blocksci.Blockchain(str(BLOCKSCI_CONFIG))

### Parameters

def prepare_run_dir(hops, limit):
    for_hops = hops
    back_hops = hops

    data_path = experiment_run_dir(hops, limit)
    ensure_split_dirs(data_path)

    config = {
        "direction": direction,
        "graph": graph,
        "address": address,
        "limit_mode": limit_mode,
        "limit": limit,
        "forward_hops": for_hops,
        "backward_hops": back_hops,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "test_samples": test_samples,
    }

    config_path = data_path / "graph_config.json"
    with open(config_path, "w") as config_file:
        json.dump(config, config_file, indent=4)
    return data_path, for_hops, back_hops

### Collecting addresses

def get_data(file_path):
    df = pd.read_csv(file_path, usecols=['address', 'label']).dropna(subset=['address', 'label'])

    illicit_set = set(df[df['label'] != 'white']['address'].astype(str).str.strip().unique())
    licit_set = set(df[df['label'] == 'white']['address'].astype(str).str.strip().unique()) - illicit_set

    licit_candidates = list(licit_set)
    random.shuffle(licit_candidates)

    addresses = {}
    counts = {}
    target_count = None

    groups = [("illicit", illicit_set, 1), ("licit", licit_candidates, 0)]

    for name, candidates, label in groups:
        count = 0
        limit = target_count
        with tqdm(total=(limit or len(candidates)), desc=f"Extracting {name} addresses") as pbar:
            for addr_str in candidates:
                if limit is not None and count >= limit:
                    break
                try:
                    addr_obj = chain.address_from_string(addr_str)
                    if addr_obj is not None:
                        addresses[addr_str] = (addr_obj, label)
                        count += 1
                        pbar.update(1)
                except Exception:
                    pass
        counts[name] = count
        if target_count is None:
            target_count = count

    print(f"Total unique addresses: {len(addresses)}\n")
    return addresses


# Take the train, val, and test sets and their addresses

def split_addresses(train_samples, val_samples=None, test_samples=None, addresses=None):

    illicit_pool = {k: v for k, v in addresses.items() if v[1] == 1}
    licit_pool = {k: v for k, v in addresses.items() if v[1] == 0}


    # Split illicit addresses
    train_illicit = dict(random.sample(list(illicit_pool.items()), train_samples))
    remaining_illicit = {k: v for k, v in illicit_pool.items() if k not in train_illicit}
    val_illicit = dict(random.sample(list(remaining_illicit.items()), val_samples))
    remaining_illicit = {k: v for k, v in remaining_illicit.items() if k not in val_illicit}
    test_illicit = dict(random.sample(list(remaining_illicit.items()), test_samples))

    # Split licit addresses
    train_licit = dict(random.sample(list(licit_pool.items()), train_samples))
    remaining_licit = {k: v for k, v in licit_pool.items() if k not in train_licit}
    val_licit = dict(random.sample(list(remaining_licit.items()), val_samples))
    remaining_licit = {k: v for k, v in remaining_licit.items() if k not in val_licit}
    test_licit = dict(random.sample(list(remaining_licit.items()), test_samples))

    train_addresses = {**train_illicit, **train_licit}
    val_addresses = {**val_illicit, **val_licit}
    test_addresses = {**test_illicit, **test_licit}

    print("Split sizes (illicit + licit = total):")
    print(f"  Train: {len(train_illicit)} + {len(train_licit)} = {len(train_addresses)}")
    print(f"  Val:   {len(val_illicit)} + {len(val_licit)} = {len(val_addresses)}")
    print(f"  Test:  {len(test_illicit)} + {len(test_licit)} = {len(test_addresses)}\n")

    return train_addresses, val_addresses, test_addresses

### Expansion statistics

# Width-limit counters, reset before expanding each split. Every call to
# limit_expansion is one expansion step: one node choosing among its candidates.
limit_stats = {}

def reset_limit_stats():
    limit_stats.clear()
    limit_stats.update({
        "steps": 0,             # nodes that went through the width limit
        "nonempty_steps": 0,    # ... with at least one candidate
        "activated_steps": 0,   # ... with more candidates than the limit
        "candidates": 0,        # candidates available over all steps
        "retained": 0,          # candidates kept over all steps
    })

def read_memory_kb():
    # VmRSS includes the resident pages of the memory-mapped BlockSci chain;
    # RssAnon only counts memory allocated by the process itself
    memory = {}
    with open("/proc/self/status") as status:
        for line in status:
            key = line.split(":")[0]
            if key in ("VmRSS", "RssAnon"):
                memory[key] = int(line.split()[1])
    return memory

class PeakMemorySampler:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.stop_event = threading.Event()
        self.baseline = read_memory_kb()
        self.peak = dict(self.baseline)

    def sample(self):
        while not self.stop_event.is_set():
            for key, value in read_memory_kb().items():
                self.peak[key] = max(self.peak.get(key, 0), value)
            self.stop_event.wait(self.interval)

    def __enter__(self):
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop_event.set()
        self.thread.join()
        for key, value in read_memory_kb().items():
            self.peak[key] = max(self.peak.get(key, 0), value)

## GRAPH EXPANSION

def limit_expansion(txs, limit):
    n_candidates = len(txs)
    n_retained = min(n_candidates, limit)
    limit_stats["steps"] += 1
    limit_stats["nonempty_steps"] += int(n_candidates > 0)
    limit_stats["activated_steps"] += int(n_candidates > limit)
    limit_stats["candidates"] += n_candidates
    limit_stats["retained"] += n_retained
    return random.sample(txs, n_retained)

def explore_tx(tx, addresses, inputs, outputs, new, mode = 'none', selected_list = []):
    new[str(tx.hash)] = tx

    process_inputs = selected_list if mode == "input" else tx.inputs
    process_outputs = selected_list if mode == "output" else tx.outputs

    for input_tx in process_inputs:
        addr = input_tx.address
        input_id = (str(input_tx.spent_output.tx.hash), input_tx.spent_output.index)

        if hasattr(addr, 'address_string'):
            inputs[input_id] = (input_tx.address, input_tx, input_tx.tx)
            addr_str = addr.address_string

            if addr_str not in addresses:
                addresses[addr_str] = (addr, -1)

    for output_tx in process_outputs:
        addr = output_tx.address
        output_id = (str(output_tx.tx.hash), output_tx.index)

        if hasattr(addr, 'address_string'):
            outputs[output_id] = (output_tx.tx, output_tx, output_tx.address)
            addr_str = addr.address_string

            if addr_str not in addresses:
                addresses[addr_str] = (addr, -1)

### Tx-based whole transaction implied addresses forward-backward expansion

def expand_forward(prev_txs, addresses, txs, inputs, outputs, limit_mode=None, limit=math.inf):
    new_txs = {}
    candidates = []
    candidate_hashes = set()

    for tx in tqdm(prev_txs.values()):
        tx_candidates = []
        tx_candidate_hashes = set()

        for tx_out in tx.outputs:
            out_tx = tx_out.spending_tx

            if out_tx is not None:
                tx_hash = str(out_tx.hash)

                if tx_hash not in txs and tx_hash not in tx_candidate_hashes and tx_hash not in candidate_hashes:
                    tx_candidates.append(out_tx)
                    tx_candidate_hashes.add(tx_hash)

        if limit_mode == "random node":
            tx_candidates = limit_expansion(tx_candidates, limit)

        candidates += tx_candidates
        candidate_hashes.update(str(candidate.hash) for candidate in tx_candidates)

    if limit_mode == "random hop":
        candidates = limit_expansion(candidates, limit)

    for out_tx in candidates:
        explore_tx(out_tx, addresses, inputs, outputs, new_txs)

    return new_txs

def expand_backward(prev_txs, addresses, txs, inputs, outputs, limit_mode=None, limit=math.inf):
    new_txs = {}
    candidates = []
    candidate_hashes = set()

    for tx in tqdm(prev_txs.values()):
        tx_candidates = []
        tx_candidate_hashes = set()

        for tx_in in tx.inputs:
            in_tx = tx_in.spent_tx

            if in_tx is not None:
                tx_hash = str(in_tx.hash)

                if tx_hash not in txs and tx_hash not in tx_candidate_hashes and tx_hash not in candidate_hashes:
                    tx_candidates.append(in_tx)
                    tx_candidate_hashes.add(tx_hash)

        if limit_mode == "random node":
            tx_candidates = limit_expansion(tx_candidates, limit)

        candidates += tx_candidates
        candidate_hashes.update(str(candidate.hash) for candidate in tx_candidates)

    if limit_mode == "random hop":
        candidates = limit_expansion(candidates, limit)

    for in_tx in candidates:
        explore_tx(in_tx, addresses, inputs, outputs, new_txs)

    return new_txs

def expand(addresses, num_for_hops, num_back_hops, txs, inputs, outputs, limit_mode = None, limit = math.inf):
    print("\nInitial number of addrs:", len(addresses))
    print("Initial number of txs:", len(txs))
    print("Initial number of inputs:", len(inputs))
    print("Initial number of outputs:", len(outputs), "\n")

    new_fw_txs = {}
    new_bw_txs = {}
    fw_candidates = []
    bw_candidates = []
    addrs = list(addresses.values()).copy()
    print(f"Hop: 1, addresses to expand:", len(addresses))

    for addr, _ in tqdm(addrs):
        if limit_mode == "random node":
            fw_addr_candidates = limit_expansion(list(addr.input_txes), limit)
            bw_addr_candidates = limit_expansion(list(addr.output_txes), limit)

        else:
            fw_addr_candidates = list(addr.input_txes)
            bw_addr_candidates = list(addr.output_txes)

        fw_candidates += fw_addr_candidates
        bw_candidates += bw_addr_candidates


    if limit_mode == "random hop":
        fw_candidates = limit_expansion(fw_candidates, limit)
        bw_candidates = limit_expansion(bw_candidates, limit)

    for fw_tx in fw_candidates:
        explore_tx(fw_tx, addresses, inputs, outputs, new_fw_txs)

    for bw_tx in bw_candidates:
        explore_tx(bw_tx, addresses, inputs, outputs, new_bw_txs)

    txs.update(new_fw_txs)
    txs.update(new_bw_txs)

    print("\nForward expand:")
    for i in range(num_for_hops-1):
        print(f"Hop: {i+2}, txs to expand: {len(new_fw_txs)}")
        new_fw_txs = expand_forward(new_fw_txs, addresses, txs, inputs, outputs, limit_mode = limit_mode, limit = limit)
        txs.update(new_fw_txs)

    print("\nBackward expand:")
    for i in range(num_back_hops-1):
        print(f"Hop: {i+2}, txs to expand: {len(new_bw_txs)}")
        new_bw_txs = expand_backward(new_bw_txs, addresses, txs, inputs, outputs, limit_mode = limit_mode, limit = limit)
        txs.update(new_bw_txs)

    print("\nFinal number of addrs:", len(addresses))
    print("Final number of txs:", len(txs))
    print("Final number of inputs:", len(inputs))
    print("Final number of outputs:", len(outputs), "\n")

def tx_wait(txes):
    tx_list = list(txes)
    if len(tx_list) <= 1:
        return []

    heights = sorted(tx.block_height for tx in tx_list)
    gaps = [b - a for a, b in zip(heights[:-1], heights[1:])]
    return gaps

### Collecting nodes and edges features

def extract_address_features(addresses):
    results = []
    for addr, label in tqdm(list(addresses.values())):
        out_txs = list(addr.output_txes)
        in_txs = list(addr.input_txes)
        txes = list(addr.txes)

        n_utxos = 0
        utxo_values = []
        income_list = []
        for out in addr.outputs:
            income_list.append(int(out.value))
            if not out.is_spent:
                n_utxos += 1
                utxo_values.append(int(out.value))

        sorted_incomes = sorted(income_list, reverse=True)
        income = sum(sorted_incomes)

        if income > 0:
            top1_income = sum(sorted_incomes[:1])
            top1_share=  float(top1_income) / float(income)
            top5_income = sum(sorted_incomes[:5])
            top5_share=  float(top5_income) / float(income)
        else:
            top1_income = top1_share = top5_income = top5_share = 0.0

        input_ages = []
        expenditure = 0
        for inp in addr.inputs:
            expenditure += int(inp.value)
            input_ages.append(inp.age)

        if input_ages:
            min_input_age = min(input_ages)
            max_input_age = max(input_ages)
            avg_input_age = np.mean(input_ages)
            std_input_age = np.std(input_ages)
        else:
            min_input_age = max_input_age = avg_input_age = std_input_age = 0.0

        balance = income - expenditure
        outflow_ratio = float(expenditure) / float(income) if income > 0 else 0.0

        n_incoming_txs = len(in_txs)
        n_outgoing_txs = len(out_txs)
        n_total_txs = n_incoming_txs + n_outgoing_txs

        if n_utxos > 0:
            max_utxo_value = max(utxo_values)
            min_utxo_value = min(utxo_values)
            avg_utxo_value = np.mean(utxo_values)
            std_utxo_value = np.std(utxo_values)
        else:
            max_utxo_value = min_utxo_value = avg_utxo_value = std_utxo_value = 0.0

        in_tx_inputs = []
        in_tx_outputs = []
        out_tx_inputs = []
        out_tx_outputs = []
        output_values = []
        input_values = []
        for in_tx in in_txs:
            in_tx_inputs.append(len(in_tx.inputs))
            in_tx_outputs.append(len(in_tx.outputs))
            input_values.append(int(in_tx.input_value))
        for out_tx in out_txs:
            out_tx_inputs.append(len(out_tx.inputs))
            out_tx_outputs.append(len(out_tx.outputs))
            output_values.append(int(out_tx.output_value))

        if output_values:
            min_out_tx_value = min(output_values)
            max_out_tx_value = max(output_values)
            avg_out_tx_value = np.mean(output_values)
            std_out_tx_value = np.std(output_values)
        else:
            min_out_tx_value = max_out_tx_value = avg_out_tx_value = std_out_tx_value = 0.0

        if input_values:
            min_in_tx_value = min(input_values)
            max_in_tx_value = max(input_values)
            avg_in_tx_value = np.mean(input_values)
            std_in_tx_value = np.std(input_values)
        else:
            min_in_tx_value = max_in_tx_value = avg_in_tx_value = std_in_tx_value = 0.0

        if in_tx_inputs:
            min_in_tx_inputs = min(in_tx_inputs)
            max_in_tx_inputs = max(in_tx_inputs)
            avg_in_tx_inputs = np.mean(in_tx_inputs)
            std_in_tx_inputs = np.std(in_tx_inputs)
        else:
            min_in_tx_inputs = max_in_tx_inputs = avg_in_tx_inputs = std_in_tx_inputs = 0.0

        if in_tx_outputs:
            min_in_tx_outputs = min(in_tx_outputs)
            max_in_tx_outputs = max(in_tx_outputs)
            avg_in_tx_outputs = np.mean(in_tx_outputs)
            std_in_tx_outputs = np.std(in_tx_outputs)
        else:
            min_in_tx_outputs = max_in_tx_outputs = avg_in_tx_outputs = std_in_tx_outputs = 0.0

        if out_tx_inputs:
            min_out_tx_inputs = min(out_tx_inputs)
            max_out_tx_inputs = max(out_tx_inputs)
            avg_out_tx_inputs = np.mean(out_tx_inputs)
            std_out_tx_inputs = np.std(out_tx_inputs)
        else:
            min_out_tx_inputs = max_out_tx_inputs = avg_out_tx_inputs = std_out_tx_inputs = 0.0

        if out_tx_outputs:
            min_out_tx_outputs = min(out_tx_outputs)
            max_out_tx_outputs = max(out_tx_outputs)
            avg_out_tx_outputs = np.mean(out_tx_outputs)
            std_out_tx_outputs = np.std(out_tx_outputs)
        else:
            min_out_tx_outputs = max_out_tx_outputs = avg_out_tx_outputs = std_out_tx_outputs = 0.0

        block_heights = []
        if len(txes) > 0:
            for tx in txes:
                block_heights.append(tx.block_height)
            first_tx_block = min(block_heights)
            last_tx_block = max(block_heights)
        else:
            first_tx_block = -1
            last_tx_block = -1

        in_tx_wait = tx_wait(addr.input_txes)
        if len(in_tx_wait) > 1:
            min_in_tx_wait = min(in_tx_wait)
            max_in_tx_wait = max(in_tx_wait)
            avg_in_tx_wait = np.mean(in_tx_wait)
            std_in_tx_wait = np.std(in_tx_wait)
        else:
            min_in_tx_wait = max_in_tx_wait = avg_in_tx_wait = std_in_tx_wait = 0.0

        out_tx_wait = tx_wait(addr.output_txes)
        if len(out_tx_wait) > 1:
            min_out_tx_wait = min(out_tx_wait)
            max_out_tx_wait = max(out_tx_wait)
            avg_out_tx_wait = np.mean(out_tx_wait)
            std_out_tx_wait = np.std(out_tx_wait)
        else:
            min_out_tx_wait = max_out_tx_wait = avg_out_tx_wait = std_out_tx_wait = 0.0

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
        results.append(info)

    df = pd.DataFrame(results)
    return df

def extract_tx_features(txs):
    tx_list = list(txs.values())
    features = []
    for tx in tx_list:
        info = {
            'hash': str(tx.hash),
            'block_height': tx.block_height,
            'fee': tx.fee,
            'is_coinbase': tx.is_coinbase,
            'locktime': tx.locktime,
            'total_size': tx.total_size,
            'version': tx.version
        }
        features.append(info)
    return pd.DataFrame(features)

def extract_input_features(inputs):
    features = []
    for input_id, inp in inputs.items():
        inp = inp[1]
        if hasattr(inp.address, 'address_string'):
            info = {
                'addr_str': inp.address.address_string,
                'tx_hash': str(inp.tx.hash),
                'age': inp.age,
                'sequence_num': inp.sequence_num,
                'value': inp.value,
                'spent_tx_hash': input_id[0],
                'spent_output_index': input_id[1]
            }
            features.append(info)
    return pd.DataFrame(features)

def extract_output_features(outputs):
    output_list = list(outputs.values())
    features = []
    for out in output_list:
        out = out[1]
        if hasattr(out.address, 'address_string'):
            info = {
                'tx_hash': str(out.tx.hash),
                'addr_str': out.address.address_string,
                'index': out.index,
                'is_spent': out.is_spent,
                'value': out.value
            }
            features.append(info)
    return pd.DataFrame(features)

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

def export(addrs, txs, inputs, outputs, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    addr_df = extract_address_features(addrs)
    tx_df = extract_tx_features(txs)
    inputs_df = extract_input_features(inputs)
    outputs_df = extract_output_features(outputs)
    spent_pairs_df = extract_spent_pairs(inputs_df, outputs_df)

    addr_df.to_csv(out_dir / "addr_feats.csv", index=False)
    tx_df.to_csv(out_dir / "tx_feats.csv", index=False)
    inputs_df.to_csv(out_dir / "input_feats.csv", index=False)
    outputs_df.to_csv(out_dir / "output_feats.csv", index=False)
    spent_pairs_df.to_csv(out_dir / 'spent_pairs.csv', index=False)

    return addr_df, tx_df, inputs_df, outputs_df

### Running experiments

def run_experiment(hops, limit, seed_splits, rng_state):
    data_path, for_hops, back_hops = prepare_run_dir(hops, limit)

    # Every configuration starts from the same random state that expansion.py
    # would have right after splitting, so each run is reproducible on its own
    random.setstate(rng_state)

    # In "random node" mode val/test use the same limit as train
    limit_test_val = limit

    for split, split_name, seed_addresses, split_limit in [
        ('train', 'Training',   seed_splits[0], limit),
        ('val',   'Validation', seed_splits[1], limit_test_val),
        ('test',  'Test',       seed_splits[2], limit_test_val),
    ]:
        # expand() adds discovered addresses in place, so work on a copy of the seeds
        addresses = dict(seed_addresses)
        txs = {}
        inputs = {}
        outputs = {}

        print("\n----------------------------------------")
        print(f'{split_name} set expansion')
        print("----------------------------------------")
        reset_limit_stats()
        with PeakMemorySampler() as memory:
            start = time.perf_counter()
            expand(addresses, for_hops, back_hops,
                   txs, inputs, outputs,
                   limit_mode=limit_mode, limit=split_limit)
            expansion_time = time.perf_counter() - start

        start = time.perf_counter()
        export(addresses, txs, inputs, outputs, data_path / split)
        export_time = time.perf_counter() - start

        stats = {
            "hops": hops,
            "limit": split_limit,
            "n_seeds": len(seed_addresses),
            "expansion_time_s": expansion_time,
            "expansion_time_per_seed_s": expansion_time / len(seed_addresses),
            "export_time_s": export_time,
            "baseline_rss_mb": memory.baseline.get("VmRSS", 0) / 1024,
            "peak_rss_mb": memory.peak.get("VmRSS", 0) / 1024,
            "baseline_rss_anon_mb": memory.baseline.get("RssAnon", 0) / 1024,
            "peak_rss_anon_mb": memory.peak.get("RssAnon", 0) / 1024,
            "width_limit": dict(limit_stats),
        }
        with open(data_path / split / "expansion_stats.json", "w") as stats_file:
            json.dump(stats, stats_file, indent=4)

        del addresses, txs, inputs, outputs
        gc.collect()

    print(f"\n[DONE] Saved at: {data_path}")


addresses = get_data(str(BITCOINHEIST_CSV))
train_addresses, val_addresses, test_addresses = split_addresses(
    train_samples, val_samples, test_samples, addresses=addresses
)
rng_state = random.getstate()

for i, (hops, limit) in enumerate(EXPERIMENTS, start=1):
    print("\n========================================")
    print(f"Experiment {i}/{len(EXPERIMENTS)}: {exp_alg} | {hops} hops | {limit_mode} limit {limit}")
    print("========================================")
    run_experiment(hops, limit, (train_addresses, val_addresses, test_addresses), rng_state)
