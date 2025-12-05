# RANSOMWARE ADDRESSES DETECTION

### Import libraries

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import random
import math
import os
import blocksci

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

### Initialize Blocksci

parser_data_directory = '/mnt/data/parsed-data-bitcoin/config.blocksci'
prefix = '/usr/local/src/BlockSci/Notebooks/Ransomware - Mario i Albert/'
chain = blocksci.Blockchain(parser_data_directory)

### Parameters

def get_parameters():
    print("Type of seed addresses:\n---------------------------------")
    print("1. Licit\n2. Illicit\n3. Licit and Illicit (50/50)")
    seed_options = {"1": "licit", "2": "illicit", "3": "licit and illicit"}
    seed = seed_options[input("Option: ")]

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

    data_path = f'{prefix}Heterogeneous/{seed}/{train_samples}-{val_samples}-{test_samples} {exp_alg} {hops} hops {limit}{space}{limit_mode} limit/'
    if not os.path.isdir(data_path):
        os.mkdir(data_path)
        os.mkdir(data_path + 'train/')
        os.mkdir(data_path + 'test/')
        os.mkdir(data_path + 'val/')
    
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

    config_path = os.path.join(data_path, "graph_config.json")
    with open(config_path, "w") as config_file:
        json.dump(config, config_file, indent=4)
    return space, exp_alg, limit_mode, limit, for_hops, back_hops, hops, train_samples, val_samples, test_samples, prefix, data_path, seed

space, exp_alg, limit_mode, limit, for_hops, back_hops, hops, train_samples, val_samples, test_samples, prefix, data_path, seed = get_parameters()

### Collecting addresses

def get_data(file_path):
    df = pd.read_csv(file_path)
    illicit_df = df[df['label'] != 'white']
    illicit_addresses = set(illicit_df['address'].unique())
    addresses = {}
    newer_addresses = {}

    for address in illicit_addresses:
        try:
            addr = chain.address_from_string(address)
            if addr is not None:
                addresses[address] = (addr, 1)
            else:
                newer_addresses[address] = (addr, 1)
        except:
            pass

    print("Number of addresses:", len(addresses))
    return addresses

# Take a single random address from a random transaction in the block where the first transaction of each illicit address appears

def add_random_address_from_block(dictionary, mode):
    licit_addresses = {}
    for address, (addr, label) in tqdm(dictionary.items()):
        for tx in addr.txes:
            first_tx = tx
            break
        block = first_tx.block

        found = False
        while not found:
            random_tx = random.choice(list(block.txes))
            random_address = random.choice([input.address for input in random_tx.inputs] + [output.address for output in random_tx.outputs])
            if hasattr(random_address, 'address_string') and random_address.address_string not in dictionary and random_address.address_string not in licit_addresses:
                licit_addresses[random_address.address_string] = (random_address, 0)
                found = True

    if not os.path.exists(data_path):
        os.makedirs(data_path)
       
    return licit_addresses


# Take the train and test sets and its addresses

def split_addresses(train_samples, val_samples, test_samples, seed):
    if seed == "licit":
        train_addresses = {}
        val_addresses = {}
        test_addresses = {}
        random_train_blocks = random.sample(list(chain), train_samples)
        random_val_blocks = random.sample(list(chain), val_samples)
        random_test_blocks = random.sample(list(chain), test_samples)

        for random_blocks in [random_train_blocks, random_val_blocks, random_test_blocks]:
            for block in tqdm(random_blocks):
                random_tx = random.choice(list(block.txes))
                found = False
                while not found:
                    random_address = random.choice(list(random_tx.inputs) + list(random_tx.outputs)).address
                    if hasattr(random_address, 'address_string') and random_address.address_string not in test_addresses and random_address.address_string not in train_addresses and random_address.address_string not in val_addresses:
                        train_addresses[random_address.address_string] = (random_address, 0)
                        found = True
                        

    elif seed == "illicit":
        addresses = get_data(prefix + "Data/bitcoinheist.csv")
        train_addresses = dict(random.sample(addresses.items(), train_samples))
        remaining_addresses = {k: v for k, v in addresses.items() if k not in train_addresses}
        val_addresses = dict(random.sample(remaining_addresses.items(), val_samples))
        remaining_addresses = {k: v for k, v in remaining_addresses.items() if k not in val_addresses}
        test_addresses = dict(random.sample(remaining_addresses.items(), test_samples))

    elif seed == "licit and illicit":
        addresses = get_data(prefix + "Data/bitcoinheist.csv")
        train_addresses = dict(random.sample(addresses.items(), train_samples))
        remaining_addresses = {k: v for k, v in addresses.items() if k not in train_addresses}
        val_addresses = dict(random.sample(remaining_addresses.items(), val_samples))
        remaining_addresses = {k: v for k, v in remaining_addresses.items() if k not in val_addresses}
        test_addresses = dict(random.sample(remaining_addresses.items(), test_samples))

        licit_train = add_random_address_from_block(train_addresses, 'train')
        licit_val   = add_random_address_from_block(val_addresses, 'val')
        licit_test  = add_random_address_from_block(test_addresses, 'test')

        train_addresses.update(licit_train)
        val_addresses.update(licit_val)
        test_addresses.update(licit_test)


    return train_addresses, val_addresses, test_addresses

train_addresses, val_addresses, test_addresses = split_addresses(train_samples, val_samples, test_samples, seed)
#train_addresses = {'1C9KA8hWUuASCdDq1EPB7PmcnFNqhb1so2': (chain.address_from_string('1C9KA8hWUuASCdDq1EPB7PmcnFNqhb1so2'), 1)}
#test_addresses = {'1C9KA8hWUuASCdDq1EPB7PmcnFNqhb1so2': (chain.address_from_string('1C9KA8hWUuASCdDq1EPB7PmcnFNqhb1so2'), 1)}
#val_addresses = {'1C9KA8hWUuASCdDq1EPB7PmcnFNqhb1so2': (chain.address_from_string('1C9KA8hWUuASCdDq1EPB7PmcnFNqhb1so2'), 1)}

## GRAPH EXPANSION

def limit_expansion(txs, limit):
    return random.sample(txs, min(len(txs), limit))

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

### Addr-based dedicated inputs/outputs forward-backward expansion

if exp_alg == 'fw bw addr dedicated':
    
    def expand_forward(addresses, prev_addresses, txs, inputs, outputs, debug=False, limit_mode=None, limit=math.inf):
        new_addresses = {}
        candidates = {}

        for addr, _ in tqdm(prev_addresses.values()):
            addr_candidates = {}

            for inp in addr.inputs:
                tx_hash = str(inp.tx.hash)
                
                if tx_hash not in txs:
                    if tx_hash not in addr_candidates:
                        addr_candidates[tx_hash] = [inp]
                    else:
                        addr_candidates[tx_hash].append(inp)

            if limit_mode == "random node":
                addr_candidates = {tx_id: addr_candidates[tx_id] for tx_id in limit_expansion(addr_candidates.keys(), limit)}

            candidates.update(addr_candidates)

        if limit_mode == "random hop":
            candidates = {tx_id: candidates[tx_id] for tx_id in limit_expansion(candidates.keys(), limit)}

        for tx_id, tx_inputs in candidates.items():
            tx = chain.tx_with_hash(tx_id)
            txs[str(tx.hash)] = tx

            for inp in tx_inputs:
                input_id = (str(inp.spent_output.tx.hash), inp.spent_output.index)
                inputs[input_id] = (inp.address, inp, inp.tx)

            for out in tx.outputs:
                output_id = (str(out.tx.hash), out.index)
                outputs[output_id] = (out.tx, out, out.address)
                addr = out.address
                if hasattr(addr, 'address_string') and addr.address_string not in addresses:
                    new_addresses[addr.address_string] = (addr, -1)

        return new_addresses
    
    def expand_backward(addresses, prev_addresses, txs, inputs, outputs, debug=False, limit_mode=None, limit=math.inf):
        new_addresses = {}
        candidates = {}

        for addr, _ in tqdm(prev_addresses.values()):
            addr_candidates = {}

            for out in addr.outputs:
                tx_hash = str(out.tx.hash)

                if tx_hash not in txs:
                    if tx_hash not in addr_candidates:
                        addr_candidates[tx_hash] = [out]
                    else:
                        addr_candidates[tx_hash].append(out)

            if limit_mode == "random node":
                addr_candidates = {tx_id: addr_candidates[tx_id] for tx_id in limit_expansion(addr_candidates.keys(), limit)}

            candidates.update(addr_candidates)

        if limit_mode == "random hop":
            candidates = {tx_id: candidates[tx_id] for tx_id in limit_expansion(candidates.keys(), limit)}

        for tx_id, tx_outputs in candidates.items():
            tx = chain.tx_with_hash(tx_id)
            txs[str(tx.hash)] = tx

            for out in tx_outputs:
                output_id = (str(out.tx.hash), out.index)
                outputs[output_id] = (out.tx, out, out.address)

            for inp in tx.inputs:
                input_id = (str(inp.spent_output.tx.hash), inp.spent_output.index)
                inputs[input_id] = (inp.address, inp, inp.tx)
                addr = inp.address
                if hasattr(addr, 'address_string') and addr.address_string not in addresses:
                    new_addresses[addr.address_string] = (addr, -1)

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
                tx_hash = str(in_tx.hash)

                if tx_hash not in txs:
                    candidates[tx_hash] = in_tx


        if limit_mode == "random hop":
            candidates = {tx_id: candidates[tx_id] for tx_id in limit_expansion(candidates.keys(), limit)}

        for tx_hash, in_tx in candidates.items():
            txs[tx_hash] = in_tx

            for input_tx in in_tx.inputs:
                addr = input_tx.address

                if hasattr(addr, 'address_string'):
                    input_id = (str(input_tx.spent_output.tx.hash), input_tx.spent_output.index)
                    inputs[input_id] = (addr, input_tx, input_tx.tx)    

                    if addr.address_string not in addresses and addr.address_string not in new_addresses:
                        side_addresses[addr.address_string] = (addr, -1)

            for output_tx in in_tx.outputs:
                addr = output_tx.address

                if hasattr(addr, 'address_string'):
                    output_id = (str(output_tx.tx.hash), output_tx.index)
                    outputs[output_id] = (output_tx.tx, output_tx, addr)

                    if addr.address_string not in addresses:
                        new_addresses[addr.address_string] = (addr, -1)

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
                tx_hash = str(out_tx.hash)

                if tx_hash not in txs:
                    candidates[tx_hash] = out_tx

        if limit_mode == "random hop":
            candidates = {tx_id: candidates[tx_id] for tx_id in limit_expansion(candidates.keys(), limit)}

        for tx_hash, out_tx in candidates.items():
            txs[tx_hash] = out_tx

            for output_tx in out_tx.outputs:
                addr = output_tx.address
                if hasattr(addr, 'address_string'):
                    output_id = (str(output_tx.tx.hash), output_tx.index)
                    outputs[output_id] = (output_tx.tx, output_tx, )

                    if addr.address_string not in addresses and addr.address_string not in new_addresses:
                        side_addresses[addr.address_string] = (addr, -1)

            for inp in out_tx.inputs:
                addr = inp.address
                if hasattr(addr, 'address_string'):
                    input_id = (str(inp.spent_output.tx.hash), inp.spent_output.index)
                    inputs[input_id] = (addr, inp, inp.tx)

                    if addr.address_string not in addresses:
                        new_addresses[addr.address_string] = (addr, -1)

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
                    tx_hash = str(tx.hash)

                    if tx_hash not in txs and tx is not None:
                        candidates.append(tx)

            if limit_mode == "random hop":
                candidates = limit_expansion(candidates, limit)

            for tx in candidates:
                txs[str(tx.hash)] = tx

                for input_tx in tx.inputs:

                    if hasattr(input_tx.address, 'address_string'):
                        input_id = (str(input_tx.spent_output.tx.hash), input_tx.spent_output.index)
                        inputs[input_id] = (input_tx.address, input_tx, input_tx.tx)
                        addr = input_tx.address

                        if addr.address_string not in addresses:
                            hop_addresses[addr.address_string] = (addr, -1)

                for output_tx in tx.outputs:

                    if hasattr(output_tx.address, 'address_string'):
                        output_id = (str(output_tx.tx.hash), output_tx.index)
                        outputs[output_id] = (output_tx.tx, output_tx, output_tx.address)
                        addr = output_tx.address

                        if addr.address_string not in addresses:
                            hop_addresses[addr.address_string] = (addr, -1)

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

            if limit_mode == "random node" or limit_mode is "":

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
                inp = tx_out.spending_input

                if out_tx is not None:
                    tx_hash = str(out_tx.hash)

                    if str(tx_hash) not in txs:
                        if tx_hash not in tx_candidates:
                            tx_candidates[tx_hash] = [inp]

                        else:
                            tx_candidates[tx_hash].append(inp)

            if limit_mode == "random node":
                tx_candidates = {tx_id: tx_candidates[tx_id] for tx_id in limit_expansion(tx_candidates.keys(), limit)}

            candidates.update(tx_candidates)

        if limit_mode == "random hop":
            candidates = {tx_id: candidates[tx_id] for tx_id in limit_expansion(candidates.keys(), limit)}  

        for tx_id, tx_inputs in candidates.items():
            tx = chain.tx_with_hash(tx_id)
            explore_tx(tx, addresses, inputs, outputs, new_txs, mode='input', selected_list=tx_inputs)

        return new_txs

    def expand_backward(prev_txs, addresses, txs, inputs, outputs, limit_mode=None, limit=math.inf):
        new_txs = {}
        candidates = {}

        for tx in tqdm(prev_txs.values()):
            tx_candidates = {}

            for tx_in in tx.inputs:
                in_tx = tx_in.spent_tx
                out = tx_in.spent_output

                if in_tx is not None:
                    tx_hash = str(in_tx.hash)

                    if tx_hash not in txs:
                        if tx_hash not in tx_candidates:
                            tx_candidates[tx_hash] = [out]

                        else:
                            tx_candidates[tx_hash].append(out)

            if limit_mode == "random node":
                tx_candidates = {tx_id: tx_candidates[tx_id] for tx_id in limit_expansion(tx_candidates.keys(), limit)}

            candidates.update(tx_candidates)

        if limit_mode == "random hop":
            candidates = {tx_id: candidates[tx_id] for tx_id in limit_expansion(candidates.keys(), limit)}

        for tx_id, tx_outputs in candidates.items():
            tx = chain.tx_with_hash(tx_id)  # COMPTE QUE TX_ID ES UN STRING I NO EL HASH, POTSER NO FUNCIONA
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
            ins = addr.inputs
            for inp in ins:
                tx = inp.tx

                if tx.hash not in fw_candidates:
                    if tx.hash not in fw_addr_candidates:
                        fw_addr_candidates[tx.hash] = [inp]
                    
                    else:
                        fw_addr_candidates[tx.hash].append(inp)
            
            if limit_mode == "random node":
                fw_addr_candidates = {tx_id: fw_addr_candidates[tx_id] for tx_id in limit_expansion(fw_addr_candidates.keys(), limit)}

            fw_candidates.update(fw_addr_candidates)

            bw_addr_candidates = {}
            outs = addr.outputs
            for out in outs:
                output_id = (str(out.tx.hash), out.index)

                if output_id not in outputs:
                    tx = out.tx

                    if tx.hash not in bw_candidates:
                        if tx.hash not in bw_addr_candidates:
                            bw_addr_candidates[tx.hash] = [out]

                        else:
                            bw_addr_candidates[tx.hash].append(out)

            if limit_mode == "random node":
                bw_addr_candidates = {tx_id: bw_addr_candidates[tx_id] for tx_id in limit_expansion(bw_addr_candidates.keys(), limit)}

            bw_candidates.update(bw_addr_candidates)
            
        if limit_mode == "random hop":
            fw_candidates = {tx_id: fw_candidates[tx_id] for tx_id in limit_expansion(fw_candidates.keys(), limit)}
            bw_candidates = {tx_id: bw_candidates[tx_id] for tx_id in limit_expansion(bw_candidates.keys(), limit)}

        for fw_tx_id, fw_inputs in fw_candidates.items():
            fw_tx = chain.tx_with_hash(str(fw_tx_id))
            explore_tx(fw_tx, addresses, inputs, outputs, new_fw_txs, mode='input', selected_list=fw_inputs)

        for bw_tx_id, bw_outputs in bw_candidates.items():
            bw_tx = chain.tx_with_hash(str(bw_tx_id))
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

        for tx in tqdm(prev_txs.values()):
            tx_candidates = []

            for tx_out in tx.outputs:
                out_tx = tx_out.spending_tx

                if out_tx is not None:
                    tx_hash = str(out_tx.hash)

                    if tx_hash not in txs and tx_hash not in tx_candidates and tx_hash not in candidates:
                        tx_candidates.append(out_tx)
            
            if limit_mode == "random node":
                tx_candidates = limit_expansion(tx_candidates, limit)
            
            candidates += tx_candidates

        if limit_mode == "random hop":
            candidates = limit_expansion(candidates, limit)

        for out_tx in candidates:
            explore_tx(out_tx, addresses, inputs, outputs, new_txs)
        
        return new_txs

    def expand_backward(prev_txs, addresses, txs, inputs, outputs, limit_mode=None, limit=math.inf):
        new_txs = {}
        candidates = []

        for tx in tqdm(prev_txs.values()):
            tx_candidates = []

            for tx_in in tx.inputs:
                in_tx = tx_in.spent_tx

                if in_tx is not None:
                    tx_hash = str(in_tx.hash)

                    if tx_hash not in txs and tx_hash not in tx_candidates and tx_hash not in candidates:
                        tx_candidates.append(in_tx)

            if limit_mode == "random node":
                tx_candidates = limit_expansion(tx_candidates, limit)

            candidates += tx_candidates

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

### Tx-based all over expansion

if exp_alg == "all over tx":

    def expand(addresses, num_hops, txs, inputs, outputs, limit_mode = None, limit = math.inf):
        print("\nInitial number of addrs:", len(addresses))
        print("Initial number of txs:", len(txs))
        print("Initial number of inputs:", len(inputs))
        print("Initial number of outputs:", len(outputs), "\n")

        new_txs = {}
        addrs = list(addresses.values()).copy()
        total_txes = []
        print(f"Hop: 0, addresses to expand:", len(addresses))

        for addr in tqdm(addrs):
            txes = [tx for tx in addr[0].txes]

            if limit_mode == "random node" or limit_mode is "":

                if limit_mode == "random node":
                    txes = limit_expansion(txes, limit)

                for tx in txes:
                    explore_tx(tx, addresses, inputs, outputs, new_txs)

            total_txes += txes

        if limit_mode == "random hop":
            total_txes = limit_expansion(total_txes, limit)

        for tx in total_txes:
            explore_tx(tx, addresses, inputs, outputs, new_txs)

        txs.update(new_txs)

        for i in range(num_hops-1):
            print(f"Hop: {i+1}, txs to expand: {len(new_txs)}")
            hop_txs = {}
            candidates = []

            for tx_hash, tx in tqdm(new_txs.items()):

                if limit_mode == 'random node':
                    tx_candidates = []

                for input_tx in tx.inputs:
                    in_tx = input_tx.spent_tx

                    if in_tx is not None:
                        tx_hash = str(in_tx.hash)

                        if tx_hash not in txs and tx_hash not in hop_txs:

                            if limit_mode == 'random node':
                                tx_candidates.append(in_tx)

                            else:
                                candidates.append(in_tx)

                for output_tx in tx.outputs:
                    out_tx = output_tx.spending_tx

                    if out_tx is not None:
                        tx_hash = str(out_tx.hash)

                        if tx_hash not in txs and tx_hash not in hop_txs:

                            if limit_mode == 'random node':
                                tx_candidates.append(out_tx)

                            else:
                                candidates.append(out_tx)

                if limit_mode == "random node":
                    candidates += limit_expansion(tx_candidates, limit)

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
        #incoming_counterparties = set()
        #outgoing_counterparties = set()

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
            #outgoing_counterparties.update(set([out.address.address_string for out in in_tx.outputs]))
        for out_tx in out_txs:
            out_tx_inputs.append(len(out_tx.inputs))
            out_tx_outputs.append(len(out_tx.outputs()))
            output_values.append(int(out_tx.output_value))
            #incoming_counterparties.update(set([inp.address.address_string for inp in out_tx.inputs]))

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
            #'incoming_counterparties': len(incoming_counterparties),
            #'outgoing_counterparties': len(outgoing_counterparties)
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

def export(addrs, txs, inputs, outputs, model, prefix = ""):
    addr_df = extract_address_features(addrs)
    tx_df = extract_tx_features(txs)
    inputs_df = extract_input_features(inputs)
    outputs_df = extract_output_features(outputs)
    spent_pairs_df = extract_spent_pairs(inputs_df, outputs_df)
    
    addr_df.to_csv(prefix + "addr_feats.csv", index=False)
    tx_df.to_csv(prefix + "tx_feats.csv", index=False)
    inputs_df.to_csv(prefix + "input_feats.csv", index=False)
    outputs_df.to_csv(prefix + "output_feats.csv", index=False)
    spent_pairs_df.to_csv(prefix + 'spent_pairs.csv', index=False)
    
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
    'hetero', data_path + 'train/')

val_addr_df, val_tx_df, val_inputs_df, val_outputs_df = export(
    val_addresses, val_txs, val_inputs, val_outputs,
    'hetero', data_path + 'val/')

test_addr_df, test_tx_df, test_inputs_df, test_outputs_df = export(
    test_addresses, test_txs, test_inputs, test_outputs,
    'hetero', data_path + 'test/')
