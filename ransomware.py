# RANSOMWARE ADDRESSES DETECTION

### Import libraries

import matplotlib.pyplot as plt
import matplotlib.ticker
import collections
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import random

### Initialize Blocksci

import blocksci
parser_data_directory = "/mnt/data/parsed-data-bitcoin/config.blocksci"
chain = blocksci.Blockchain(parser_data_directory)

### Collecting addresses

chainabuse = []
with open("addresses.txt", "r") as file:
    for line in file:
        chainabuse.append(line[:-1])

chainabuse = list(set(chainabuse))

addresses = {}
newer_addresses = {}

for address in chainabuse:
    try:
        addr = chain.address_from_string(address)
        if addr is not None:
            addresses[address] = addr
        else:
            newer_addresses[address] = addr
    except:
        pass
print("Number of addresses:", len(addresses))
print("Number of newer addresses:", len(newer_addresses))

bitcoinheist = pd.read_csv('BitcoinHeistData.csv')
bitcoinheist = bitcoinheist[bitcoinheist.iloc[:, -1] != "white"]

for _, row in bitcoinheist.iterrows():
    address = row.iloc[0]
    if address not in addresses:
        addr = chain.address_from_string(address)
        if addr is not None:
            addresses[address] = addr
        
print("Number of addresses:", len(addresses))

with open('ransomwhere.json') as f:
    ransomwhere = json.load(f)

for entry in ransomwhere:
    address = entry['address']
    if address not in addresses:
        addr = chain.address_from_string(address)
        if addr:
            addresses[address] = addr
        else:
            if address not in newer_addresses:
                newer_addresses[address] = addr
            
print("Number of addresses:", len(addresses))
print("Number of newer addresses:", len(newer_addresses))

### Neighbors graph expansion

### Whole transaction implied addresses forward-backward expansion

def expand_forward(addresses, prev_addresses, txs, inputs, outputs, debug = False):
    new_addresses = {}
    side_addresses = {}
    i=0
    
    for address in tqdm(prev_addresses.values()):
        if debug:
            if i%10==0:
                print(str(len(new_addresses)) + " new addresses")

        for in_tx in address.input_txes:
            tx_hash = str(in_tx.hash)
            
            if tx_hash not in txs:
                txs[tx_hash] = in_tx

                for input_tx in in_tx.inputs:
                    input_id = (input_tx.spent_output.tx.hash, input_tx.spent_output.index)
                    inputs[input_id] = (input_tx.address, input_tx, input_tx.tx)

                    if hasattr(input_tx.address, 'address_string'):
                        addr = input_tx.address
                        
                        if addr.address_string not in addresses and addr.address_string not in new_addresses and addr.address_string not in side_addresses:
                            side_addresses[addr.address_string] = addr
                    
                for output_tx in in_tx.outputs:
                    output_id = (output_tx.tx.hash, output_tx.index)
                    outputs[output_id] = (output_tx.tx, output_tx, output_tx.address)
                                 
                    if hasattr(output_tx.address, 'address_string'):
                        addr = output_tx.address
                                 
                        if addr.address_string not in addresses and addr.address_string not in new_addresses and addr.address_string not in side_addresses:
                            new_addresses[addr.address_string] = addr
    
    return new_addresses, side_addresses

def expand_backward(addresses, prev_addresses, txs, inputs, outputs, debug = False):
    new_addresses = {}
    side_addresses = {}
    i=0
    
    for address in tqdm(prev_addresses.values()):
        if debug:
            if i%10==0:
                print(str(len(new_addresses)) + " new addresses")

        for out_tx in address.output_txes:
            tx_hash = str(out_tx.hash)
            
            if tx_hash not in txs:
                txs[tx_hash] = out_tx
                
                for output_tx in out_tx.outputs:
                    output_id = (output_tx.tx.hash, output_tx.index)
                    outputs[output_id] = (output_tx.tx, output_tx, output_tx.address)
                                 
                    if hasattr(output_tx.address, 'address_string'):
                        addr = output_tx.address
                                 
                        if addr.address_string not in addresses and addr.address_string not in new_addresses and addr.address_string not in side_addresses:
                            side_addresses[addr.address_string] = addr

                for input_tx in out_tx.inputs:
                    input_id = (input_tx.spent_output.tx.hash, input_tx.spent_output.index)
                    inputs[input_id] = (input_tx.address, input_tx, input_tx.tx)

                    if hasattr(input_tx.address, 'address_string'):
                        addr = input_tx.address
                        
                        if addr.address_string not in addresses and addr.address_string not in new_addresses and addr.address_string not in side_addresses:
                            new_addresses[addr.address_string] = addr
                    
    return new_addresses, side_addresses

### Whole transaction implied addresses forward-backward expansion

def expand(addresses, num_for_hops, num_back_hops, txs, inputs, outputs, mode='same'):
    print("Initial number of addrs:", len(addresses))
    print("Initial number of txs:", len(txs))
    new = addresses
    side = {}
    news = {}
    
    if mode == 'same':
        print("\nForward expand:")
        for i in range(num_for_hops):
            print(f"Hop {i} addresses to expand:", len(new) + len(side))
            new, side = expand_forward({**addresses, **news}, {**new, **side}, txs, inputs, outputs)
            news.update(new)
            news.update(side)

        print("\nBackward expand:")
        new = addresses
        side = {}
        for i in range(num_back_hops):
            print(f"Hop {i} addresses to expand:", len(new) + len(side))
            new, side = expand_backward({**addresses, **news}, {**new, **side}, txs, inputs, outputs)
            news.update(new)
            news.update(side)
            
    elif mode == 'none':
        print("Forward expand:")
        for i in range(num_for_hops):
            new, side = expand_forward({**addresses, **news}, new, txs, inputs, outputs)
            news.update(new)
            news.update(side)

        print("Backward expand:")
        new = addresses
        side = {}
        for i in range(num_back_hops):
            new, side = expand_backward({**addresses, **news}, new, txs, inputs, outputs)
            news.update(new)
            news.update(side)
            
    elif mode == 'opposite':
        pass


    addresses.update(news)
    print("\nFinal number of addrs:", len(addresses))
    print("Final number of txs:", len(txs))

### Collecting nodes and edges features

def extract_address_features(addresses, idx):
    results = []
    
    if isinstance(addresses, dict):
        address_list = list(addresses.values())
    else:
        address_list = addresses
        
    for address in address_list[:idx]:
        info = {
            'addr': address.address_string,
            'full_type': address.full_type,
            'class': 1
        }
        results.append(info)
        
    for address in address_list[idx:]:
        info = {
            'addr': address.address_string,
            'full_type': address.full_type,
            'class': 0
        }
        results.append(info)
    
    df = pd.DataFrame(results)
    
    return df

def extract_tx_features(txs):
    tx_list = list(txs.values())
    features = []
    for tx in tx_list:
        info = {
            'hash': tx.hash,
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
    input_list = list(inputs.values())
    features = []
    for input_tx in input_list:
        input_tx = input_tx[1]
        if hasattr(input_tx.address, 'address_string'):
            addr_string = input_tx.address.address_string
        else:
            addr_string = None

        info = {
            'addr': addr_string,
            'tx': input_tx.tx.hash,
            'age': input_tx.age,
            'block': input_tx.block.height,
            'index': input_tx.index,
            'sequence_num': input_tx.sequence_num,
            'value': input_tx.value
        }
        features.append(info)
    return pd.DataFrame(features)

def extract_output_features(outputs):
    output_list = list(outputs.values())
    features = []
    for output_tx in output_list:
        output_tx = output_tx[1]

        if hasattr(output_tx.address, 'address_string'):
            addr_string = output_tx.address.address_string
        else:
            addr_string = None

        info = {
            'tx': output_tx.tx.hash,
            'addr': addr_string,
            'block': output_tx.block.height,
            'index': output_tx.index,
            'is_spent': output_tx.is_spent,
            'value': output_tx.value
        }
        features.append(info)
    return pd.DataFrame(features)

def export(addrs, txs, inputs, outputs, model, idx, prefix = ""):
    if model == "hetero" or model == "hypergraph":
        addr_df = extract_address_features(addrs, idx)
        tx_df = extract_tx_features(txs)
        inputs_df = extract_input_features(inputs)
        outputs_df = extract_output_features(outputs)
        
        addr_df.to_csv(prefix + "addr_feats.csv", index=False)
        tx_df.to_csv(prefix + "tx_feats.csv", index=False)
        inputs_df.to_csv(prefix + "input_feats.csv", index=False)
        outputs_df.to_csv(prefix + "output_feats.csv", index=False)
        
        return addr_df, tx_df, inputs_df, outputs_df

txs = {}
inputs = {}
outputs = {}
idx = 5000

els = list(addresses.items())
some_addresses = dict(els[:idx])
expand(some_addresses, 1, 1, txs, inputs, outputs)
addr_df, tx_df, inputs_df, outputs_df = export(some_addresses, txs, inputs, outputs, 'hetero', idx, "Heterogeneous/")