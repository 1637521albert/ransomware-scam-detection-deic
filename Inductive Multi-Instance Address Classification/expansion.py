# STEP 1 (Methodology I) - GRAPH EXPANSION AND FEATURE EXTRACTION (requires BlockSci)
#
# Expands the seed addresses of each split (SPLITS_FILE, created by
# create_splits.py) into one merged heterogeneous graph per split, using the
# expansion configuration MULTI_INSTANCE_EXPANSION of config.py, and exports the
# node and edge features of every graph to MULTI_INSTANCE_GRAPHS_DIR/<split>/:
#   addr_feats.csv, tx_feats.csv, input_feats.csv, output_feats.csv, spent_pairs.csv
# For the selected configuration (fw bw tx whole), tx_feats.csv also records the
# hop where every tx was added to the graph (hop, hop_direction).
#
# Every split is expanded from its own fixed random seed, so a split can be
# rebuilt on its own and gives the same graph in every execution. Splits whose
# files already exist are skipped.
#
# The splits are processed one after another, each with all the cores
# (config.MULTI_INSTANCE_EXPANSION_WORKERS forked processes). For the selected
# configuration (fw bw tx whole) every BlockSci access runs in the workers:
# listing the transactions of the seeds, scanning the frontier of every hop,
# exploring the selected transactions and computing the features. The main
# process only keeps identifiers and plain values and performs the
# order-dependent steps (random sampling, de-duplication and insertion into the
# graph) in the original order, so the result is identical to a single-process
# run. The other expansion variants run in the main process, with parallel
# feature extraction.

### Import libraries

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import math
import os
import pickle
import shutil
import blocksci

import config
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

def limit_expansion(txs, limit):
    # Python >= 3.11 only samples sequences; on older versions random.sample
    # converts dict keys to a tuple in the same order, so the result is the same
    if not isinstance(txs, (list, tuple)):
        txs = list(txs)
    return random.sample(txs, min(len(txs), limit))

# Hex-encoding a tx hash is one of the costliest BlockSci calls, and the same
# transactions are hashed over and over, so hashes are memoized by tx index.
_tx_hashes = {}

def tx_hash(tx):
    idx = tx.index
    h = _tx_hashes.get(idx)
    if h is None:
        h = _tx_hashes[idx] = str(tx.hash)
    return h

def input_tx_hash(inp):
    # Same as str(inp.tx.hash)
    idx = inp.tx_index
    h = _tx_hashes.get(idx)
    if h is None:
        h = _tx_hashes[idx] = str(inp.tx.hash)
    return h

def spent_tx_hash(inp):
    # Same as str(inp.spent_output.tx.hash)
    idx = inp.spent_tx_index
    h = _tx_hashes.get(idx)
    if h is None:
        h = _tx_hashes[idx] = str(inp.spent_tx.hash)
    return h

def output_tx_hash(out):
    # Same as str(out.tx.hash)
    idx = out.tx_index
    h = _tx_hashes.get(idx)
    if h is None:
        h = _tx_hashes[idx] = str(out.tx.hash)
    return h

def address_string(addr):
    # Encodes the address once instead of hasattr() + repeated .address_string
    return getattr(addr, 'address_string', None)

def lookup_tx(tx, tx_id):
    # Same as chain.tx_with_hash(tx_id) for a tx already in hand: only coinbase
    # txs can share a hash (BIP30 duplicates), so the hash index lookup is kept for them
    return chain.tx_with_hash(tx_id) if tx.is_coinbase else tx

def explore_tx(tx, addresses, inputs, outputs, new, mode='none', selected_list=(), explored=None):
    h = tx_hash(tx)
    new[h] = tx

    # Re-exploring a whole tx rewrites the same entries, so it can be skipped
    if explored is not None:
        if tx.index in explored:
            return
        explored.add(tx.index)

    process_inputs = selected_list if mode == "input" else tx.inputs
    process_outputs = selected_list if mode == "output" else tx.outputs

    for input_tx in process_inputs:
        addr = input_tx.address
        addr_str = address_string(addr)

        if addr_str is not None:
            input_id = (spent_tx_hash(input_tx), input_tx.spent_output.index)
            inputs[input_id] = (addr_str, input_tx, tx)

            if addr_str not in addresses:
                addresses[addr_str] = (addr, -1)

    for output_tx in process_outputs:
        addr = output_tx.address
        addr_str = address_string(addr)

        if addr_str is not None:
            outputs[(h, output_tx.index)] = (tx, output_tx, addr_str)

            if addr_str not in addresses:
                addresses[addr_str] = (addr, -1)

### Addr-based dedicated inputs/outputs forward-backward expansion

if exp_alg == 'fw bw addr dedicated':
    
    def expand_forward(addresses, prev_addresses, txs, inputs, outputs, debug=False, limit_mode=None, limit=math.inf):
        new_addresses = {}
        candidates = {}

        for addr, _ in tqdm(prev_addresses.values()):
            addr_candidates = {}

            for inp in addr.inputs:
                tx_id = input_tx_hash(inp)

                if tx_id not in txs:
                    addr_candidates.setdefault(tx_id, []).append(inp)

            if limit_mode == "random node":
                addr_candidates = {tx_id: addr_candidates[tx_id] for tx_id in limit_expansion(addr_candidates.keys(), limit)}

            candidates.update(addr_candidates)

        if limit_mode == "random hop":
            candidates = {tx_id: candidates[tx_id] for tx_id in limit_expansion(candidates.keys(), limit)}

        for tx_id, tx_inputs in candidates.items():
            tx = lookup_tx(tx_inputs[0].tx, tx_id)
            txs[tx_id] = tx

            for inp in tx_inputs:
                input_id = (spent_tx_hash(inp), inp.spent_output.index)
                inputs[input_id] = (address_string(inp.address), inp, tx)

            for out in tx.outputs:
                addr = out.address
                addr_str = address_string(addr)
                outputs[(tx_id, out.index)] = (tx, out, addr_str)
                if addr_str is not None and addr_str not in addresses:
                    new_addresses[addr_str] = (addr, -1)

        return new_addresses
    
    def expand_backward(addresses, prev_addresses, txs, inputs, outputs, debug=False, limit_mode=None, limit=math.inf):
        new_addresses = {}
        candidates = {}

        for addr, _ in tqdm(prev_addresses.values()):
            addr_candidates = {}

            for out in addr.outputs:
                tx_id = output_tx_hash(out)

                if tx_id not in txs:
                    addr_candidates.setdefault(tx_id, []).append(out)

            if limit_mode == "random node":
                addr_candidates = {tx_id: addr_candidates[tx_id] for tx_id in limit_expansion(addr_candidates.keys(), limit)}

            candidates.update(addr_candidates)

        if limit_mode == "random hop":
            candidates = {tx_id: candidates[tx_id] for tx_id in limit_expansion(candidates.keys(), limit)}

        for tx_id, tx_outputs in candidates.items():
            tx = lookup_tx(tx_outputs[0].tx, tx_id)
            txs[tx_id] = tx

            for out in tx_outputs:
                outputs[(tx_id, out.index)] = (tx, out, address_string(out.address))

            for inp in tx.inputs:
                addr = inp.address
                addr_str = address_string(addr)
                input_id = (spent_tx_hash(inp), inp.spent_output.index)
                inputs[input_id] = (addr_str, inp, tx)
                if addr_str is not None and addr_str not in addresses:
                    new_addresses[addr_str] = (addr, -1)

        return new_addresses
    
    def expand(addresses, num_for_hops, num_back_hops, txs, inputs, outputs, limit_mode=None, limit=math.inf):
        print("\nInitial number of addrs:", len(addresses))
        print("Initial number of txs:", len(txs))
        print("Initial number of inputs:", len(inputs))
        print("Initial number of outputs:", len(outputs))
        news = {}

        print("\nForward expand:")
        new = addresses
        for i in range(num_for_hops):
            new = expand_forward({**addresses, **news}, new, txs, inputs, outputs, limit_mode = limit_mode, limit = limit)
            news.update(new)

        print("\nBackward expand:")
        new = addresses
        for i in range(num_back_hops):
            new = expand_backward({**addresses, **news}, new, txs, inputs, outputs, limit_mode = limit_mode, limit = limit)
            news.update(new)

        addresses.update(news)
        print("\nFinal number of addrs:", len(addresses))
        print("Final number of txs:", len(txs))
        print("Final number of inputs:", len(inputs))
        print("Final number of outputs:", len(outputs))

### Addr-based whole transaction implied addresses forward-backward expansion

if exp_alg.startswith('fw bw addr whole'):
    
    def expand_forward(addresses, prev_addresses, txs, inputs, outputs, debug=False, limit_mode=None, limit=math.inf):
        new_addresses = {}
        side_addresses = {}
        candidates = {}

        for addr, _ in tqdm(prev_addresses.values()):
            if limit_mode == "random node":
                in_txs = limit_expansion([in_tx for in_tx in addr.input_txes], limit)
            else:
                in_txs = addr.input_txes

            for in_tx in in_txs:
                tx_id = tx_hash(in_tx)

                if tx_id not in txs:
                    candidates[tx_id] = in_tx


        if limit_mode == "random hop":
            candidates = {tx_id: candidates[tx_id] for tx_id in limit_expansion(candidates.keys(), limit)}

        for tx_id, in_tx in candidates.items():
            txs[tx_id] = in_tx

            for input_tx in in_tx.inputs:
                addr = input_tx.address
                addr_str = address_string(addr)

                if addr_str is not None:
                    input_id = (spent_tx_hash(input_tx), input_tx.spent_output.index)
                    inputs[input_id] = (addr_str, input_tx, in_tx)

                    if addr_str not in addresses and addr_str not in new_addresses:
                        side_addresses[addr_str] = (addr, -1)

            for output_tx in in_tx.outputs:
                addr = output_tx.address
                addr_str = address_string(addr)

                if addr_str is not None:
                    outputs[(tx_id, output_tx.index)] = (in_tx, output_tx, addr_str)

                    if addr_str not in addresses:
                        new_addresses[addr_str] = (addr, -1)

        return new_addresses, side_addresses

    def expand_backward(addresses, prev_addresses, txs, inputs, outputs, debug=False, limit_mode=None, limit=math.inf):
        new_addresses = {}
        side_addresses = {}
        candidates = {}

        for addr, _ in tqdm(prev_addresses.values()):
            if limit_mode == "random node":
                out_txs = limit_expansion([out_tx for out_tx in addr.output_txes], limit)
            else:
                out_txs = addr.output_txes

            for out_tx in out_txs:
                tx_id = tx_hash(out_tx)

                if tx_id not in txs:
                    candidates[tx_id] = out_tx

        if limit_mode == "random hop":
            candidates = {tx_id: candidates[tx_id] for tx_id in limit_expansion(candidates.keys(), limit)}

        for tx_id, out_tx in candidates.items():
            txs[tx_id] = out_tx

            for output_tx in out_tx.outputs:
                addr = output_tx.address
                addr_str = address_string(addr)

                if addr_str is not None:
                    outputs[(tx_id, output_tx.index)] = (out_tx, output_tx, addr_str)

                    if addr_str not in addresses and addr_str not in new_addresses:
                        side_addresses[addr_str] = (addr, -1)

            for inp in out_tx.inputs:
                addr = inp.address
                addr_str = address_string(addr)

                if addr_str is not None:
                    input_id = (spent_tx_hash(inp), inp.spent_output.index)
                    inputs[input_id] = (addr_str, inp, out_tx)

                    if addr_str not in addresses:
                        new_addresses[addr_str] = (addr, -1)

        return new_addresses, side_addresses

    def expand(addresses, num_for_hops, num_back_hops, txs, inputs, outputs, mode=exp_alg.split()[-1], limit_mode=None, limit=math.inf):
        print("\nInitial number of addrs:", len(addresses))
        print("Initial number of txs:", len(txs))
        print("Initial number of inputs:", len(inputs))
        print("Initial number of outputs:", len(outputs))
        
        if mode == 'none':
            new_fw = addresses.copy()
            new_bw = addresses.copy()
            for i in range(num_for_hops):
                print(f"\nHop {i + 1}:")
                print("\nForward phase:")
                new_fw, side_fw = expand_forward(addresses, new_fw, txs, inputs, outputs, limit_mode=limit_mode, limit=limit)
                addresses.update({**new_fw, **side_fw})

                print("\nBackward phase:")
                new_bw, side_bw = expand_backward(addresses, new_bw, txs, inputs, outputs, limit_mode=limit_mode, limit=limit)
                addresses.update({**new_bw, **side_bw})

                print(f"\nTotal number of addrs: {len(addresses)}")
                print(f"Total number of txs: {len(txs)}")

        elif mode == 'same':
            new = addresses.copy()
            side = {}
            news = {}
            print("\nForward expand:")
            for i in range(num_for_hops):
                print(f"Hop {i} addresses to expand:", len(new) + len(side))
                new, side = expand_forward({**addresses, **news}, {**new, **side}, txs, inputs, outputs, limit_mode=limit_mode, limit=limit)
                news.update(new)
                news.update(side)

            print("\nBackward expand:")
            new = addresses.copy()
            side = {}
            for i in range(num_back_hops):
                print(f"Hop {i}, addresses to expand:", len(new) + len(side))
                new, side = expand_backward({**addresses, **news}, {**new, **side}, txs, inputs, outputs, limit_mode=limit_mode, limit=limit)
                news.update(new)
                news.update(side)

            addresses.update(news)

        elif mode == 'opposite':
            new_fw = addresses.copy()
            new_bw = addresses.copy()
            side_fw = {}
            side_bw = {}

            for i in range(num_for_hops):
                print(f"Hop {i}, addresses to expand:", len(new_fw) + len(side_fw) + len(new_bw) + len(side_bw))
                
                print("\nForward phase:")
                new_fw, side_fw = expand_forward(addresses, new_fw, txs, inputs, outputs, limit_mode=limit_mode, limit=limit)
                addresses.update({**new_fw, **side_fw})

                print("\nBackward phase:")
                new_bw, side_bw = expand_backward(addresses, new_bw, txs, inputs, outputs, limit_mode=limit_mode, limit=limit)
                addresses.update({**new_bw, **side_bw})

                new_fw.update(side_bw)
                new_bw.update(side_fw)

        print("\nFinal number of addrs:", len(addresses))
        print("Final number of txs:", len(txs))
        print("Final number of inputs:", len(inputs))
        print("Final number of outputs:", len(outputs))

### Addr-based all over expansion

if exp_alg == "all over addr":

    def expand(addresses, num_hops, txs, inputs, outputs, limit_mode=None, limit=math.inf):
        print("\nInitial number of addrs:", len(addresses))
        print("Initial number of txs:", len(txs))
        print("Initial number of inputs:", len(inputs))
        print("Initial number of outputs:", len(outputs), "\n")

        new_addresses = addresses

        for i in range(num_hops):
            print(f"Hop {i + 1} addresses to expand:", len(new_addresses))
            hop_addresses = {}
            candidates = []

            for address, (addr, label) in tqdm(new_addresses.items()):
                if limit_mode == 'random node':
                    txes = limit_expansion([tx for tx in addr.txes], limit)
                else:
                    txes = addr.txes

                for tx in txes:
                    if tx_hash(tx) not in txs:
                        candidates.append(tx)

            if limit_mode == "random hop":
                candidates = limit_expansion(candidates, limit)

            # Candidates repeat when frontier addresses share a tx; reprocessing
            # one rewrites the same entries, so each tx is processed once
            processed = set()

            for tx in candidates:
                if tx.index in processed:
                    continue
                processed.add(tx.index)

                tx_id = tx_hash(tx)
                txs[tx_id] = tx

                for input_tx in tx.inputs:
                    addr = input_tx.address
                    addr_str = address_string(addr)

                    if addr_str is not None:
                        input_id = (spent_tx_hash(input_tx), input_tx.spent_output.index)
                        inputs[input_id] = (addr_str, input_tx, tx)

                        if addr_str not in addresses:
                            hop_addresses[addr_str] = (addr, -1)

                for output_tx in tx.outputs:
                    addr = output_tx.address
                    addr_str = address_string(addr)

                    if addr_str is not None:
                        outputs[(tx_id, output_tx.index)] = (tx, output_tx, addr_str)

                        if addr_str not in addresses:
                            hop_addresses[addr_str] = (addr, -1)

            new_addresses = hop_addresses
            addresses.update(new_addresses)

        print("\nFinal number of addrs:", len(addresses))
        print("Final number of txs:", len(txs))
        print("Final number of inputs:", len(inputs))
        print("Final number of outputs:", len(outputs))

### Tx-based forward-backward expansion

'''if exp_alg == "fw bw tx": 
    
    def expand_forward(prev_txs, addresses, txs, inputs, outputs, limit_mode=None, limit=math.inf):
        new_txs = {}
        candidates = []

        for tx in tqdm(prev_txs.values()):
            tx_outs = [out for out in tx.outputs]

            if limit_mode == "random node":
                tx_outs = limit_expansion(tx_outs, limit)

            for tx_out in tx_outs:
                out_tx = tx_out.spending_tx

                if out_tx is not None:
                    tx_hash = str(out_tx.hash)

                    if tx_hash not in txs and tx_hash not in new_txs:
                        candidates.append(out_tx)

        if limit_mode == "random hop":
            candidates = limit_expansion(candidates, limit)    

        for out_tx in candidates:
            explore_tx(out_tx, addresses, inputs, outputs, new_txs)

        return new_txs

    def expand_backward(prev_txs, addresses, txs, inputs, outputs, limit_mode=None, limit=math.inf):
        new_txs = {}
        candidates = []

        for tx in tqdm(prev_txs.values()):
            tx_ins = [inp for inp in tx.inputs]

            if limit_mode == "random node":
                tx_ins = limit_expansion(tx_ins, limit)

            for tx_in in tx_ins:
                in_tx = tx_in.spent_tx

                if in_tx is not None:
                    tx_hash = str(in_tx.hash)

                    if tx_hash not in txs and tx_hash not in new_txs:
                        candidates.append(in_tx)

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

        first_fw_txs = {}
        first_bw_txs = {}
        total_txes_inputs = []
        total_txes_outputs = []
        addrs = list(addresses.values()).copy()
        print("Hop: 1, addresses to expand:", len(addresses))

        for addr, _ in tqdm(addrs):
            input_txs = [tx for tx in addr.input_txes]
            output_txs = [tx for tx in addr.output_txes]

            if limit_mode == "random node" or limit_mode == "":

                if limit_mode == "random node":
                    input_txs = limit_expansion(input_txs, limit)
                    output_txs = limit_expansion(output_txs, limit)

                for tx in input_txs:
                    explore_tx(tx, addresses, inputs, outputs, first_fw_txs)

                for tx in output_txs:
                    explore_tx(tx, addresses, inputs, outputs, first_bw_txs)

            total_txes_inputs += input_txs
            total_txes_outputs += output_txs

        if limit_mode == "random hop":
            total_txes_inputs = limit_expansion(total_txes_inputs, limit)
            total_txes_outputs = limit_expansion(total_txes_outputs, limit)

            for tx in total_txes_inputs:
                explore_tx(tx, addresses, inputs, outputs, first_fw_txs)

            for tx in total_txes_outputs:
                explore_tx(tx, addresses, inputs, outputs, first_bw_txs)

        new_fw_txs = first_fw_txs
        new_bw_txs = first_bw_txs
        txs.update(new_fw_txs)
        txs.update(new_bw_txs)

        print("Forward expand:")
        for i in range(num_for_hops-1):
            print(f"Hop: {i+2}, txs to expand:", len(new_fw_txs))
            new_fw_txs = expand_forward(new_fw_txs, addresses, txs, inputs, outputs, limit_mode = limit_mode, limit = limit)
            txs.update(new_fw_txs)

        print("\nBackward expand:")
        for i in range(num_back_hops-1):
            print(f"Hop: {i+2}, txs to expand:", len(new_bw_txs))
            new_bw_txs = expand_backward(new_bw_txs, addresses, txs, inputs, outputs, limit_mode = limit_mode, limit = limit)
            txs.update(new_bw_txs)

        print("\nFinal number of addrs:", len(addresses))
        print("Final number of txs:", len(txs))
        print("Final number of inputs:", len(inputs))
        print("Final number of outputs:", len(outputs), "\n")'''

### Tx-based forward-backward expansion with dedicated inputs/outputs    

if exp_alg == "fw bw tx dedicated":
    def expand_forward(prev_txs, addresses, txs, inputs, outputs, limit_mode=None, limit=math.inf):
        new_txs = {}
        candidates = {}

        for tx in tqdm(prev_txs.values()):
            tx_candidates = {}

            for tx_out in tx.outputs:
                out_tx = tx_out.spending_tx

                if out_tx is not None:
                    tx_id = tx_hash(out_tx)

                    if tx_id not in txs:
                        tx_candidates.setdefault(tx_id, []).append(tx_out.spending_input)

            if limit_mode == "random node":
                tx_candidates = {tx_id: tx_candidates[tx_id] for tx_id in limit_expansion(tx_candidates.keys(), limit)}

            candidates.update(tx_candidates)

        if limit_mode == "random hop":
            candidates = {tx_id: candidates[tx_id] for tx_id in limit_expansion(candidates.keys(), limit)}

        for tx_id, tx_inputs in candidates.items():
            tx = lookup_tx(tx_inputs[0].tx, tx_id)
            explore_tx(tx, addresses, inputs, outputs, new_txs, mode='input', selected_list=tx_inputs)

        return new_txs

    def expand_backward(prev_txs, addresses, txs, inputs, outputs, limit_mode=None, limit=math.inf):
        new_txs = {}
        candidates = {}

        for tx in tqdm(prev_txs.values()):
            tx_candidates = {}

            for tx_in in tx.inputs:
                in_tx = tx_in.spent_tx

                if in_tx is not None:
                    tx_id = tx_hash(in_tx)

                    if tx_id not in txs:
                        tx_candidates.setdefault(tx_id, []).append(tx_in.spent_output)

            if limit_mode == "random node":
                tx_candidates = {tx_id: tx_candidates[tx_id] for tx_id in limit_expansion(tx_candidates.keys(), limit)}

            candidates.update(tx_candidates)

        if limit_mode == "random hop":
            candidates = {tx_id: candidates[tx_id] for tx_id in limit_expansion(candidates.keys(), limit)}

        for tx_id, tx_outputs in candidates.items():
            tx = lookup_tx(tx_outputs[0].tx, tx_id)
            explore_tx(tx, addresses, inputs, outputs, new_txs, mode='output', selected_list=tx_outputs)

        return new_txs
    
    def expand(addresses, num_for_hops, num_back_hops, txs, inputs, outputs, limit_mode = None, limit = math.inf):
        print("\nInitial number of addrs:", len(addresses))
        print("Initial number of txs:", len(txs))
        print("Initial number of inputs:", len(inputs))
        print("Initial number of outputs:", len(outputs), "\n")

        new_fw_txs = {}
        new_bw_txs = {}
        fw_candidates = {}
        bw_candidates = {}
        addrs = list(addresses.values()).copy()
        print("Hop: 1, addresses to expand:", len(addresses))

        for addr, _ in tqdm(addrs):
            fw_addr_candidates = {}
            for inp in addr.inputs:
                tx_id = input_tx_hash(inp)

                if tx_id not in fw_candidates:
                    fw_addr_candidates.setdefault(tx_id, []).append(inp)

            if limit_mode == "random node":
                fw_addr_candidates = {tx_id: fw_addr_candidates[tx_id] for tx_id in limit_expansion(fw_addr_candidates.keys(), limit)}

            fw_candidates.update(fw_addr_candidates)

            bw_addr_candidates = {}
            for out in addr.outputs:
                tx_id = output_tx_hash(out)

                if (tx_id, out.index) not in outputs and tx_id not in bw_candidates:
                    bw_addr_candidates.setdefault(tx_id, []).append(out)

            if limit_mode == "random node":
                bw_addr_candidates = {tx_id: bw_addr_candidates[tx_id] for tx_id in limit_expansion(bw_addr_candidates.keys(), limit)}

            bw_candidates.update(bw_addr_candidates)

        if limit_mode == "random hop":
            fw_candidates = {tx_id: fw_candidates[tx_id] for tx_id in limit_expansion(fw_candidates.keys(), limit)}
            bw_candidates = {tx_id: bw_candidates[tx_id] for tx_id in limit_expansion(bw_candidates.keys(), limit)}

        for fw_tx_id, fw_inputs in fw_candidates.items():
            fw_tx = lookup_tx(fw_inputs[0].tx, fw_tx_id)
            explore_tx(fw_tx, addresses, inputs, outputs, new_fw_txs, mode='input', selected_list=fw_inputs)

        for bw_tx_id, bw_outputs in bw_candidates.items():
            bw_tx = lookup_tx(bw_outputs[0].tx, bw_tx_id)
            explore_tx(bw_tx, addresses, inputs, outputs, new_bw_txs, mode='output', selected_list=bw_outputs)

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

        print("\nInitial number of addrs:", len(addresses))
        print("Initial number of txs:", len(txs))
        print("Initial number of inputs:", len(inputs))
        print("Initial number of outputs:", len(outputs), "\n")

### Tx-based whole transaction implied addresses forward-backward expansion (parallel)
#
# Same algorithm as the sequential fw bw tx whole expansion, with the BlockSci
# work in forked workers. The main process stores:
#   addresses: address string -> (reference to rebuild the address, label), where the
#              reference is ("seed", address string) or ("in"/"out", tx hash, position)
#              of the first input/output where the address was found
#   txs:       tx hash -> (hop, direction) of the hop where the tx was added: hop 1 for
#              the txs of the seeds ("forward" if a seed is an input, "backward" if it
#              is an output, "both"), then 2, 3... of the forward and backward hops
#   inputs:    (spent tx hash, spent output index) -> (address string, tx hash, age, sequence number, value)
#   outputs:   (tx hash, output index) -> (address string, is spent, value)
# Txs are rebuilt from their hash in the workers, which only differs from the
# sequential version for the duplicated 2010 coinbase txs (BIP30).

PARALLEL_PIPELINE = exp_alg == 'fw bw tx whole'

def seed_txs(seed_addr_str):
    # Hashes of the txs where the seed is an input (forward) and an output (backward), in BlockSci order
    addr = chain.address_from_string(seed_addr_str)
    return [tx_hash(tx) for tx in addr.input_txes], [tx_hash(tx) for tx in addr.output_txes]

def explore_record(h):
    # Plain-value version of explore_tx: the inputs and outputs of the tx with an address
    tx = chain.tx_with_hash(h)
    ins = []
    for k, input_tx in enumerate(tx.inputs):
        addr_str = address_string(input_tx.address)
        if addr_str is not None:
            ins.append((spent_tx_hash(input_tx), input_tx.spent_output.index, addr_str, k,
                        input_tx.age, input_tx.sequence_num, input_tx.value))
    outs = []
    for j, output_tx in enumerate(tx.outputs):
        addr_str = address_string(output_tx.address)
        if addr_str is not None:
            outs.append((output_tx.index, addr_str, j, output_tx.is_spent, output_tx.value))
    return ins, outs

def apply_record(h, record, addresses, inputs, outputs):
    # Inserts an explored tx in the graph exactly as explore_tx does
    ins, outs = record
    for spent_h, spent_index, addr_str, k, age, sequence_num, value in ins:
        inputs[(spent_h, spent_index)] = (addr_str, h, age, sequence_num, value)
        if addr_str not in addresses:
            addresses[addr_str] = (("in", h, k), -1)
    for index, addr_str, j, is_spent, value in outs:
        outputs[(h, index)] = (addr_str, is_spent, value)
        if addr_str not in addresses:
            addresses[addr_str] = (("out", h, j), -1)

def explore_all(hashes, addresses, inputs, outputs, desc):
    # Explores every distinct tx once, in parallel, and inserts them in the original order
    unique = list(dict.fromkeys(hashes))
    records = parallel_map(explore_record, unique, WORKERS, desc=desc)
    for h, record in zip(unique, records):
        apply_record(h, record, addresses, inputs, outputs)

_scan_state = {}

def scan_spending_txs(h):
    # (output position, hash) of the distinct txs spending the outputs of tx h that are not in the graph yet
    found, seen = [], set()
    for j, tx_out in enumerate(chain.tx_with_hash(h).outputs):
        out_tx = tx_out.spending_tx
        if out_tx is not None:
            tx_id = tx_hash(out_tx)
            if tx_id not in _scan_state["txs"] and tx_id not in seen:
                seen.add(tx_id)
                found.append((j, tx_id))
    return found

def scan_spent_txs(h):
    # (input position, hash) of the distinct txs spent by the inputs of tx h that are not in the graph yet
    found, seen = [], set()
    for j, tx_in in enumerate(chain.tx_with_hash(h).inputs):
        in_tx = tx_in.spent_tx
        if in_tx is not None:
            tx_id = tx_hash(in_tx)
            if tx_id not in _scan_state["txs"] and tx_id not in seen:
                seen.add(tx_id)
                found.append((j, tx_id))
    return found

def select_candidates(scans, limit_mode, limit):
    # Sequential, in the original order: a tx already selected by a previous frontier tx
    # is not a candidate of the next ones, then each frontier tx samples its candidates.
    # random.sample only depends on the number of candidates, so sampling (position, hash)
    # pairs selects the same txs as sampling the tx objects.
    candidates = []
    candidate_hashes = set()
    for found in scans:
        tx_candidates = [(j, tx_id) for j, tx_id in found if tx_id not in candidate_hashes]
        if limit_mode == "random node":
            tx_candidates = limit_expansion(tx_candidates, limit)
        candidates += [tx_id for _, tx_id in tx_candidates]
        candidate_hashes.update(tx_id for _, tx_id in tx_candidates)
    if limit_mode == "random hop":
        candidates = limit_expansion(candidates, limit)
    return candidates

def tx_whole_hop(prev_txs, addresses, txs, inputs, outputs, scan, desc, limit_mode, limit, hop):
    _scan_state["txs"] = txs
    scans = parallel_map(scan, list(prev_txs), WORKERS, desc=f"Scanning {desc}")
    candidates = select_candidates(scans, limit_mode, limit)
    explore_all(candidates, addresses, inputs, outputs, desc=f"Exploring {desc}")
    return dict.fromkeys(candidates, hop)

def expand_tx_whole_parallel(addresses, num_for_hops, num_back_hops, txs, inputs, outputs, limit_mode=None, limit=math.inf):
    print(f"Hop: 1, addresses to expand: {len(addresses)}", flush=True)
    seed_strs = list(addresses)
    listed = parallel_map(seed_txs, seed_strs, WORKERS, desc="Listing seed txs")

    fw_candidates = []
    bw_candidates = []
    for fw_list, bw_list in listed:
        if limit_mode == "random node":
            fw_list = limit_expansion(fw_list, limit)
            bw_list = limit_expansion(bw_list, limit)
        fw_candidates += fw_list
        bw_candidates += bw_list

    if limit_mode == "random hop":
        fw_candidates = limit_expansion(fw_candidates, limit)
        bw_candidates = limit_expansion(bw_candidates, limit)

    # Every tx is explored once, the forward candidates first
    explore_all(fw_candidates + bw_candidates, addresses, inputs, outputs, desc="Exploring hop 1")
    new_fw_txs = dict.fromkeys(fw_candidates, (1, "forward"))
    new_bw_txs = dict.fromkeys(bw_candidates, (1, "backward"))
    txs.update(new_fw_txs)
    # Updating an existing key keeps its position, so the order of txs does not change
    txs.update({h: (1, "both") if h in new_fw_txs else hop for h, hop in new_bw_txs.items()})

    for i in range(num_for_hops - 1):
        print(f"Forward hop: {i + 2}, txs to expand: {len(new_fw_txs)}", flush=True)
        new_fw_txs = tx_whole_hop(new_fw_txs, addresses, txs, inputs, outputs, scan_spending_txs,
                                  f"forward hop {i + 2}", limit_mode, limit, (i + 2, "forward"))
        txs.update(new_fw_txs)

    for i in range(num_back_hops - 1):
        print(f"Backward hop: {i + 2}, txs to expand: {len(new_bw_txs)}", flush=True)
        new_bw_txs = tx_whole_hop(new_bw_txs, addresses, txs, inputs, outputs, scan_spent_txs,
                                  f"backward hop {i + 2}", limit_mode, limit, (i + 2, "backward"))
        txs.update(new_bw_txs)

    print(f"Final number of addrs: {len(addresses)}, txs: {len(txs)}, inputs: {len(inputs)}, outputs: {len(outputs)}", flush=True)

### Tx-based all over expansion

if exp_alg == "all over tx":

    def expand(addresses, num_hops, txs, inputs, outputs, limit_mode = None, limit = math.inf):
        print("\nInitial number of addrs:", len(addresses))
        print("Initial number of txs:", len(txs))
        print("Initial number of inputs:", len(inputs))
        print("Initial number of outputs:", len(outputs), "\n")

        new_txs = {}
        explored = set()
        addrs = list(addresses.values()).copy()
        total_txes = []
        print("Hop: 0, addresses to expand:", len(addresses))

        for addr in tqdm(addrs):
            txes = [tx for tx in addr[0].txes]

            if limit_mode == "random node" or limit_mode == "":

                if limit_mode == "random node":
                    txes = limit_expansion(txes, limit)

                for tx in txes:
                    explore_tx(tx, addresses, inputs, outputs, new_txs, explored=explored)

            total_txes += txes

        if limit_mode == "random hop":
            total_txes = limit_expansion(total_txes, limit)

        for tx in total_txes:
            explore_tx(tx, addresses, inputs, outputs, new_txs, explored=explored)

        txs.update(new_txs)

        for i in range(num_hops-1):
            print(f"Hop: {i+1}, txs to expand: {len(new_txs)}")
            hop_txs = {}
            candidates = []
            candidate_hashes = set()

            for tx in tqdm(new_txs.values()):
                tx_candidate_hashes = set()

                if limit_mode == 'random node':
                    tx_candidates = []

                for input_tx in tx.inputs:
                    in_tx = input_tx.spent_tx

                    if in_tx is not None:
                        in_hash = tx_hash(in_tx)

                        if in_hash not in txs and in_hash not in tx_candidate_hashes and in_hash not in candidate_hashes:
                            tx_candidate_hashes.add(in_hash)

                            if limit_mode == 'random node':
                                tx_candidates.append(in_tx)

                            else:
                                candidates.append(in_tx)
                                candidate_hashes.add(in_hash)

                for output_tx in tx.outputs:
                    out_tx = output_tx.spending_tx

                    if out_tx is not None:
                        out_hash = tx_hash(out_tx)

                        if out_hash not in txs and out_hash not in tx_candidate_hashes and out_hash not in candidate_hashes:
                            tx_candidate_hashes.add(out_hash)

                            if limit_mode == 'random node':
                                tx_candidates.append(out_tx)

                            else:
                                candidates.append(out_tx)
                                candidate_hashes.add(out_hash)

                if limit_mode == "random node":
                    tx_candidates = limit_expansion(tx_candidates, limit)
                    candidates += tx_candidates
                    candidate_hashes.update(tx_hash(candidate) for candidate in tx_candidates)

            if limit_mode == "random hop":
                candidates = limit_expansion(candidates, limit)

            for tx in candidates:
                explore_tx(tx, addresses, inputs, outputs, hop_txs)

            new_txs = hop_txs
            txs.update(new_txs)

        print("\nFinal number of addrs:", len(addresses))
        print("Final number of txs:", len(txs))
        print("Final number of inputs:", len(inputs))
        print("Final number of outputs:", len(outputs), "\n")

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

def extract_address_features(addresses):
    items = list(addresses.values())
    check_vectorized_history([addr for addr, _ in items[::max(1, len(items) // 200)][:200]])
    results = parallel_map(address_features, items, WORKERS, desc="Address features")
    return pd.DataFrame(results)

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

def extract_tx_features(txs):
    return pd.DataFrame(parallel_map(tx_features, list(txs.items()), WORKERS, desc="Transaction features"))

# inputs map (spent_tx_hash, spent_output_index) -> (addr_str, input, tx)
def input_features(item):
    (spent_hash, spent_index), (addr_str, inp, tx) = item
    if addr_str is None:
        return None
    return {
        'addr_str': addr_str,
        'tx_hash': tx_hash(tx),
        'age': inp.age,
        'sequence_num': inp.sequence_num,
        'value': inp.value,
        'spent_tx_hash': spent_hash,
        'spent_output_index': spent_index
    }

def extract_input_features(inputs):
    features = parallel_map(input_features, list(inputs.items()), WORKERS, desc="Input features")
    return pd.DataFrame([f for f in features if f is not None])

# outputs map (tx_hash, index) -> (tx, output, addr_str)
def output_features(item):
    (h, index), (_, out, addr_str) = item
    if addr_str is None:
        return None
    return {
        'tx_hash': h,
        'addr_str': addr_str,
        'index': index,
        'is_spent': out.is_spent,
        'value': out.value
    }

def extract_output_features(outputs):
    features = parallel_map(output_features, list(outputs.items()), WORKERS, desc="Output features")
    return pd.DataFrame([f for f in features if f is not None])

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

def resolve_address(ref):
    # Rebuilds a BlockSci address from its reference (see the parallel expansion)
    if ref[0] == "seed":
        return chain.address_from_string(ref[1])
    tx = chain.tx_with_hash(ref[1])
    return tx.inputs[ref[2]].address if ref[0] == "in" else tx.outputs[ref[2]].address

def address_features_from_ref(item):
    _, (ref, label) = item
    return address_features((resolve_address(ref), label))

def tx_features_from_hash(h):
    return tx_features((h, chain.tx_with_hash(h)))

# Address features are computed and saved in blocks, so an interrupted export
# keeps the completed blocks
ADDRESS_BLOCK = 100000

def address_features_in_blocks(items, partial_dir):
    # Sample spread over all the addresses, to check the vectorized history reading
    sample = [resolve_address(ref) for _, (ref, _) in items[::max(1, len(items) // 200)][:200]]
    check_vectorized_history(sample)

    partial_dir.mkdir(parents=True, exist_ok=True)
    n_blocks = (len(items) + ADDRESS_BLOCK - 1) // ADDRESS_BLOCK
    blocks = []
    for b in range(n_blocks):
        path = partial_dir / f"block_{b:05d}.pkl"
        if path.exists():
            blocks.append(pd.read_pickle(path))
            continue
        block = items[b * ADDRESS_BLOCK:(b + 1) * ADDRESS_BLOCK]
        df = pd.DataFrame(parallel_map(address_features_from_ref, block, WORKERS,
                                       desc=f"Address features (block {b + 1}/{n_blocks})"))
        tmp = path.with_name(path.name + ".tmp")
        df.to_pickle(tmp)
        os.replace(str(tmp), str(path))
        blocks.append(df)
    return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()

def export_records(addrs, txs, inputs, outputs, out_dir: Path):
    # Export of the parallel expansion: same files and rows as export()
    out_dir.mkdir(parents=True, exist_ok=True)

    addr_df = address_features_in_blocks(list(addrs.items()), out_dir / "_partial_addr_feats")
    tx_df = pd.DataFrame(parallel_map(tx_features_from_hash, list(txs), WORKERS, desc="Transaction features"))
    # Hop where each tx was added to the graph (not used as a GNN feature); empty for an
    # expanded graph saved before the hops were recorded
    hops_of_txs = [value or (None, None) for value in txs.values()]
    tx_df["hop"] = pd.array([hop for hop, _ in hops_of_txs], dtype="Int64")
    tx_df["hop_direction"] = [direction for _, direction in hops_of_txs]
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

### Exporting data

EXPORT_FILES = ["addr_feats.csv", "tx_feats.csv", "input_feats.csv", "output_feats.csv", "spent_pairs.csv"]

def expand_split(seed_addresses, txs, inputs, outputs):
    if exp_alg.startswith('fw bw'):
        expand(seed_addresses, for_hops, back_hops, txs, inputs, outputs,
               limit_mode=limit_mode, limit=limit)
    else:
        expand(seed_addresses, hops, txs, inputs, outputs,
               limit_mode=limit_mode, limit=limit)

def expand_and_export(split_idx, split, seeds):
    print(f"\n{split} set expansion ({len(seeds)} seed addresses, {WORKERS} workers)", flush=True)

    # Each split has its own fixed random state, independent of the other splits
    random.seed(config.DATA_SEED + split_idx)
    np.random.seed(config.DATA_SEED + split_idx)

    txs, inputs, outputs = {}, {}, {}
    if PARALLEL_PIPELINE:
        # The expanded graph is saved before the export, so an interrupted export
        # does not repeat the expansion
        state_path = data_path / split / "_expansion_state.pkl"
        if state_path.exists():
            print(f"Loading the expanded graph saved at {state_path}", flush=True)
            with open(state_path, "rb") as f:
                addresses, txs, inputs, outputs = pickle.load(f)
        else:
            addresses = {addr_str: (("seed", addr_str), label) for addr_str, label in seeds}
            expand_tx_whole_parallel(addresses, for_hops, back_hops, txs, inputs, outputs, limit_mode=limit_mode, limit=limit)
            state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = state_path.with_name(state_path.name + ".tmp")
            with open(tmp, "wb") as f:
                pickle.dump((addresses, txs, inputs, outputs), f, protocol=4)
            os.replace(str(tmp), str(state_path))
        export_records(addresses, txs, inputs, outputs, data_path / split)
        # Checkpoints are removed once the split is completely exported
        state_path.unlink()
        shutil.rmtree(str(data_path / split / "_partial_addr_feats"), ignore_errors=True)
    else:
        addresses = {addr_str: (chain.address_from_string(addr_str), label)
                     for addr_str, label in tqdm(seeds, desc=f"Loading {split} seeds")}
        expand_split(addresses, txs, inputs, outputs)
        export(addresses, txs, inputs, outputs, data_path / split)
    print(f"{split}: exported to {data_path / split}", flush=True)


if __name__ == "__main__":
    write_json(data_path / "expansion_config.json", {
        "created": timestamp(),
        "git_commit": git_commit(config.PROJECT_ROOT),
        "expansion": EXPANSION,
        "exp_alg": exp_alg,
        "data_seed": config.DATA_SEED,
        "random_state": "random.seed(DATA_SEED + split index) before expanding each split",
        "workers": WORKERS,
        "splits_file": str(config.SPLITS_FILE),
    })

    split_seeds = load_split_seeds()
    for split_idx, split in enumerate(config.SPLITS):
        if all((data_path / split / name).exists() for name in EXPORT_FILES):
            print(f"{split}: already expanded, skipping ({data_path / split})")
        else:
            expand_and_export(split_idx, split, split_seeds[split])
