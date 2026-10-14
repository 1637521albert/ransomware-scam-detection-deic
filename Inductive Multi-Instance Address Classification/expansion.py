# RANSOMWARE ADDRESSES DETECTION

### Import libraries

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import random
import math
import blocksci
from pathlib import Path

from shared.paths import (
    build_run_dir,
    build_run_name,
    ensure_split_dirs,
    get_required_env_path,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

### Initialize Blocksci

BLOCKSCI_CONFIG = get_required_env_path("RSD_BLOCKSCI_CONFIG")
BITCOINHEIST_CSV = get_required_env_path("RSD_BITCOINHEIST_CSV")
chain = blocksci.Blockchain(str(BLOCKSCI_CONFIG))

### Parameters

def get_parameters():
    # Legacy seed variants:
    # seed = "licit"
    # seed = "illicit"
    seed = "licit and illicit"

    print("\nDirection of the expansion:\n---------------------------------")
    print("1. Forward Backward\n2. All over")
    direction_options = {"1": "fw bw", "2": "all over"}
    direction = direction_options[input("Option: ")]

    print("\nApproach of the expansion:\n---------------------------------")
    print("1. Transaction-based\n2. Address-based")
    graph_options = {"1": " tx", "2": " addr"}
    graph = graph_options[input("Option: ")]

    if direction == "fw bw":
        print("\nTransaction addresses proportion of the expansion:\n---------------------------------")
        print("1. Whole\n2. Dedicated")
        address_options = {"1": " whole", "2": " dedicated"}
        address = address_options[input("Option: ")]

        if address == " whole" and graph == " addr":
            print("\nSide addresses direction of the expansion:\n---------------------------------")
            print("1. None\n2. Same\n3. Opposite")
            side_direction_options = {"1": " none", "2": " same", "3": " opposite"}
            side_direction = side_direction_options[input("Option: ")]
        else:
            side_direction = ""
    else:
        address = ""
        side_direction = ""

    exp_alg = f'{direction}{graph}{address}{side_direction}'

    print("\nLimit mode:\n---------------------------------")
    print("1. Random node\n2. Random hop\n3. None")
    limit_mode_options = {"1": "random node", "2": "random hop", "3": ""}
    limit_mode = limit_mode_options[input("Option: ")]

    if limit_mode != "":
        print("\nLimit value:\n---------------------------------")
        limit = int(input("Option: "))
        
    else:
        limit = "no"

    if exp_alg.startswith('fw bw'):
        print("\nNumber of forward hops:\n---------------------------------")
        for_hops = int(input("Option: "))
        
        print("\nNumber of backward hops:\n---------------------------------")
        back_hops = int(input("Option: "))
        hops = for_hops

    else:
        print("\nNumber of hops:\n---------------------------------")
        hops = int(input("Option: "))
        for_hops = hops
        back_hops = hops

    print("\nNumber of train samples:\n---------------------------------")
    train_samples = int(input("Option: "))

    print("\nNumber of validation samples:\n---------------------------------")
    val_samples = int(input("Option: "))

    print("\nNumber of test samples:\n---------------------------------")
    test_samples = int(input("Option: "))
    print()

    space = "" if limit_mode == "" else " "

    run_name = build_run_name(
        train_samples,
        val_samples,
        test_samples,
        exp_alg,
        hops,
        limit,
        space,
        limit_mode,
    )
    data_path = build_run_dir(__file__, run_name)
    ensure_split_dirs(data_path)
    
    config = {
        "direction": direction,
        "graph": graph}

    if address != "":
        config["address"] = address

    if side_direction != "":
        config["side_direction"] = side_direction 

    config["limit_mode"] = limit_mode

    if limit_mode != "":
        config["limit"] = limit   

    if exp_alg.startswith('fw bw'):
        config["forward_hops"] = for_hops
        config["backward_hops"] = back_hops

    else:
        config["hops"] = hops

    config["train_samples"] = train_samples
    config["val_samples"] = val_samples
    config["test_samples"] = test_samples

    config_path = data_path / "graph_config.json"
    with open(config_path, "w") as config_file:
        json.dump(config, config_file, indent=4)
    return space, exp_alg, limit_mode, limit, for_hops, back_hops, hops, train_samples, val_samples, test_samples, data_path, seed

space, exp_alg, limit_mode, limit, for_hops, back_hops, hops, train_samples, val_samples, test_samples, data_path, seed = get_parameters()

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

def split_addresses(train_samples, val_samples=None, test_samples=None, seed=None, addresses=None):

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

addresses = get_data(str(BITCOINHEIST_CSV))
train_addresses, val_addresses, test_addresses = split_addresses(
    train_samples, val_samples, test_samples, seed, addresses=addresses
)

## GRAPH EXPANSION

def limit_expansion(txs, limit):
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

def explore_tx(tx, addresses, inputs, outputs, new, mode = 'none', selected_list = [], explored = None):
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
                print(f"\nForward phase:")
                new_fw, side_fw = expand_forward(addresses, new_fw, txs, inputs, outputs, limit_mode=limit_mode, limit=limit)
                addresses.update({**new_fw, **side_fw})

                print(f"\nBackward phase:")
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
        print(f"Hop: 1, addresses to expand:", len(addresses))

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
        print(f"Hop: 1, addresses to expand:", len(addresses))

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

### Tx-based whole transaction implied addresses forward-backward expansion

if exp_alg =='fw bw tx whole':

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
                    tx_id = tx_hash(out_tx)

                    if tx_id not in txs and tx_id not in tx_candidate_hashes and tx_id not in candidate_hashes:
                        tx_candidates.append(out_tx)
                        tx_candidate_hashes.add(tx_id)

            if limit_mode == "random node":
                tx_candidates = limit_expansion(tx_candidates, limit)

            candidates += tx_candidates
            candidate_hashes.update(tx_hash(candidate) for candidate in tx_candidates)

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
                    tx_id = tx_hash(in_tx)

                    if tx_id not in txs and tx_id not in tx_candidate_hashes and tx_id not in candidate_hashes:
                        tx_candidates.append(in_tx)
                        tx_candidate_hashes.add(tx_id)

            if limit_mode == "random node":
                tx_candidates = limit_expansion(tx_candidates, limit)

            candidates += tx_candidates
            candidate_hashes.update(tx_hash(candidate) for candidate in tx_candidates)

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

        explored = set()

        for fw_tx in fw_candidates:
            explore_tx(fw_tx, addresses, inputs, outputs, new_fw_txs, explored=explored)

        for bw_tx in bw_candidates:
            explore_tx(bw_tx, addresses, inputs, outputs, new_bw_txs, explored=explored)

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
        print(f"Hop: 0, addresses to expand:", len(addresses))

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
    # min, max, mean and std of a list, or zeros when the list is empty
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.asarray(values)
    return min(values), max(values), arr.mean(), arr.std()

### Collecting nodes and edges features

def extract_address_features(addresses):
    results = []
    # Expanded addresses share most of their txs, so per-tx stats are computed once
    in_tx_stats = {}
    out_tx_stats = {}

    for addr, label in tqdm(list(addresses.values())):
        income_list = []
        utxo_values = []
        for out in addr.outputs:
            value = int(out.value)
            income_list.append(value)
            if not out.is_spent:
                utxo_values.append(value)
        n_utxos = len(utxo_values)

        sorted_incomes = sorted(income_list, reverse=True)
        income = sum(sorted_incomes)

        if income > 0:
            top1_share = float(sum(sorted_incomes[:1])) / float(income)
            top5_share = float(sum(sorted_incomes[:5])) / float(income)
        else:
            top1_share = top5_share = 0.0

        input_ages = []
        expenditure = 0
        for inp in addr.inputs:
            expenditure += int(inp.value)
            input_ages.append(inp.age)

        min_input_age, max_input_age, avg_input_age, std_input_age = summarize(input_ages)

        balance = income - expenditure
        outflow_ratio = float(expenditure) / float(income) if income > 0 else 0.0

        in_heights = []
        in_tx_inputs = []
        in_tx_outputs = []
        input_values = []
        for in_tx in addr.input_txes:
            stats = in_tx_stats.get(in_tx.index)
            if stats is None:
                stats = in_tx_stats[in_tx.index] = (in_tx.block_height, in_tx.input_count, in_tx.output_count, int(in_tx.input_value))
            in_heights.append(stats[0])
            in_tx_inputs.append(stats[1])
            in_tx_outputs.append(stats[2])
            input_values.append(stats[3])

        out_heights = []
        out_tx_inputs = []
        out_tx_outputs = []
        output_values = []
        for out_tx in addr.output_txes:
            stats = out_tx_stats.get(out_tx.index)
            if stats is None:
                stats = out_tx_stats[out_tx.index] = (out_tx.block_height, out_tx.input_count, out_tx.output_count, int(out_tx.output_value))
            out_heights.append(stats[0])
            out_tx_inputs.append(stats[1])
            out_tx_outputs.append(stats[2])
            output_values.append(stats[3])

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
        results.append(info)

    df = pd.DataFrame(results)
    return df

def extract_tx_features(txs):
    features = []
    for h, tx in txs.items():
        info = {
            'hash': h,
            'block_height': tx.block_height,
            'fee': tx.fee,
            'is_coinbase': tx.is_coinbase,
            'locktime': tx.locktime,
            'total_size': tx.total_size,
            'version': tx.version
        }
        features.append(info)
    return pd.DataFrame(features)

# inputs map (spent_tx_hash, spent_output_index) -> (addr_str, input, tx)
def extract_input_features(inputs):
    features = []
    for (spent_hash, spent_index), (addr_str, inp, tx) in inputs.items():
        if addr_str is not None:
            info = {
                'addr_str': addr_str,
                'tx_hash': tx_hash(tx),
                'age': inp.age,
                'sequence_num': inp.sequence_num,
                'value': inp.value,
                'spent_tx_hash': spent_hash,
                'spent_output_index': spent_index
            }
            features.append(info)
    return pd.DataFrame(features)

# outputs map (tx_hash, index) -> (tx, output, addr_str)
def extract_output_features(outputs):
    features = []
    for (h, index), (_, out, addr_str) in outputs.items():
        if addr_str is not None:
            info = {
                'tx_hash': h,
                'addr_str': addr_str,
                'index': index,
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

### Exporting data

train_txs = {}
train_inputs = {}
train_outputs = {}

val_txs = {}
val_inputs = {}
val_outputs = {}

test_txs = {}
test_inputs = {}
test_outputs = {}

if limit_mode == "random hop":
    limit_test_val = int(limit * (len(val_addresses) / len(train_addresses)))
else:
    limit_test_val = limit


if exp_alg.startswith('fw bw'):
    print("\n----------------------------------------")
    print('Training set expansion')
    print("----------------------------------------")
    expand(train_addresses, for_hops, back_hops,
           train_txs, train_inputs, train_outputs,
           limit_mode=limit_mode, limit=limit)

    print("\n----------------------------------------")
    print('Validation set expansion')
    print("----------------------------------------")
    expand(val_addresses, for_hops, back_hops,
           val_txs, val_inputs, val_outputs,
           limit_mode=limit_mode, limit=limit_test_val)

    print("\n----------------------------------------")
    print('Test set expansion')
    print("----------------------------------------")
    expand(test_addresses, for_hops, back_hops,
           test_txs, test_inputs, test_outputs,
           limit_mode=limit_mode, limit=limit_test_val)

else:
    print("\n----------------------------------------")
    print('Training set expansion')
    print("----------------------------------------")
    expand(train_addresses, hops,
           train_txs, train_inputs, train_outputs,
           limit_mode=limit_mode, limit=limit)

    print("\n----------------------------------------")
    print('Validation set expansion')
    print("----------------------------------------")
    expand(val_addresses, hops,
           val_txs, val_inputs, val_outputs,
           limit_mode=limit_mode, limit=limit_test_val)

    print("\n----------------------------------------")
    print('Test set expansion')
    print("----------------------------------------")
    expand(test_addresses, hops,
           test_txs, test_inputs, test_outputs,
           limit_mode=limit_mode, limit=limit_test_val)

train_addr_df, train_tx_df, train_inputs_df, train_outputs_df = export(
    train_addresses, train_txs, train_inputs, train_outputs,
    data_path / 'train')

val_addr_df, val_tx_df, val_inputs_df, val_outputs_df = export(
    val_addresses, val_txs, val_inputs, val_outputs,
    data_path / 'val')

test_addr_df, test_tx_df, test_inputs_df, test_outputs_df = export(
    test_addresses, test_txs, test_inputs, test_outputs,
    data_path / 'test')
