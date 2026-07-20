import torch
import numpy as np
import pandas as pd
import json
from torch_geometric.data import HeteroData

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.paths import build_run_dir, build_run_name, ensure_split_dirs


def get_parameters():
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
    return data_path


data_path = get_parameters()

print("\nReading train transactions...")
train_txs_df   = pd.read_csv(data_path / 'train' / 'tx_feats.csv')

print("Reading train addresses...")
train_addrs_df   = pd.read_csv(data_path / 'train' / 'addr_feats.csv')

print("Reading train inputs...")
train_inputs_df   = pd.read_csv(data_path / 'train' / 'input_feats.csv')

print("Reading train outputs...")
train_outputs_df   = pd.read_csv(data_path / 'train' / 'output_feats.csv')

print("Reading train spent pairs...")
train_spent_pairs_df = pd.read_csv(data_path / 'train' / 'spent_pairs.csv')


print("\nReading validation transactions...")
val_txs_df   = pd.read_csv(data_path / 'val' / 'tx_feats.csv')

print("Reading validation addresses...")
val_addrs_df   = pd.read_csv(data_path / 'val' / 'addr_feats.csv')

print("Reading validation inputs...")
val_inputs_df   = pd.read_csv(data_path / 'val' / 'input_feats.csv')

print("Reading validation outputs...")
val_outputs_df   = pd.read_csv(data_path / 'val' / 'output_feats.csv')

print("Reading validation spent pairs...")
val_spent_pairs_df = pd.read_csv(data_path / 'val' / 'spent_pairs.csv')


print("\nReading test transactions...")
test_txs_df   = pd.read_csv(data_path / 'test' / 'tx_feats.csv')

print("Reading test addresses...")
test_addrs_df   = pd.read_csv(data_path / 'test' / 'addr_feats.csv')

print("Reading test inputs...")
test_inputs_df   = pd.read_csv(data_path / 'test' / 'input_feats.csv')

print("Reading test outputs...")
test_outputs_df   = pd.read_csv(data_path / 'test' / 'output_feats.csv')

print("Reading test spent pairs...")
test_spent_pairs_df = pd.read_csv(data_path / 'test' / 'spent_pairs.csv')

def build_graph(txs_df, addrs_df, inputs_df, outputs_df, spent_pairs_df, tx_mapping, addr_mapping):
    data = HeteroData()

    inputs_df['tx_idx'] = inputs_df['tx_hash'].map(tx_mapping).astype(int)
    inputs_df['addr_idx'] = inputs_df['addr_str'].map(addr_mapping).astype(int)
    
    inputs_df['spent_tx_idx'] = inputs_df['spent_tx_hash'].map(tx_mapping)
    missing_spent_tx = inputs_df['spent_tx_idx'].isna()
    inputs_df.loc[missing_spent_tx, 'spent_output_index'] = -1
    
    inputs_df['spent_tx_idx'] = inputs_df['spent_tx_idx'].fillna(-1).astype(int)
    inputs_df['spent_output_index'] = inputs_df['spent_output_index'].astype(int)

    outputs_df['tx_idx'] = outputs_df['tx_hash'].map(tx_mapping).astype(int)
    outputs_df['addr_idx'] = outputs_df['addr_str'].map(addr_mapping).astype(int)

    # Transactions
    data['tx'].x = torch.tensor(txs_df[['block_height', 'fee', 'locktime', 'total_size', 'version']].values, dtype=torch.float)
    bool_features = torch.tensor(txs_df[['is_coinbase']].astype(float).values, dtype=torch.float)
    data['tx'].x = torch.cat([data['tx'].x, bool_features], dim=1)
    data['tx'].id = txs_df.index.values

    # Addresses
    addrs_df['full_type'], _ = pd.factorize(addrs_df['full_type'])
    full_type = torch.tensor(addrs_df[['full_type']].values, dtype=torch.float)
    
    cols = [c for c in list(addrs_df.columns)
        if c not in ["addr_str", "class", "full_type", "incoming_counterparties", "outgoing_counterparties"]]
    aggr_feats = torch.tensor(addrs_df[cols].values, dtype=torch.float)
    
    data['addr'].x = torch.cat([full_type, aggr_feats], dim=1)
    data['addr'].id = addrs_df.index.values
    data['addr'].y = torch.tensor(addrs_df['class'].values, dtype=torch.long)

    # Inputs
    inputs_df['sequence_num'], _ = pd.factorize(inputs_df["sequence_num"])
    data['addr', 'input', 'tx'].edge_index = torch.tensor(inputs_df[['addr_idx', 'tx_idx']].values.T, dtype=torch.long)
    #input_attrs = torch.tensor(inputs_df[['age', 'sequence_num', 'value', 'spent_tx_idx', 'spent_output_index']].values, dtype=torch.float)
    input_attrs = torch.tensor(inputs_df[['age', 'sequence_num', 'value']].values, dtype=torch.float)
    data['addr', 'input', 'tx'].edge_attr = input_attrs

    # Outputs
    data['tx', 'output', 'addr'].edge_index = torch.tensor(outputs_df[['tx_idx', 'addr_idx']].values.T, dtype=torch.long)
    #output_attrs = torch.tensor(outputs_df[['index', 'value']].values, dtype=torch.float)
    output_attrs = torch.tensor(outputs_df[['value']].values, dtype=torch.float)
    bool_spent = torch.tensor(outputs_df[['is_spent']].astype(float).values, dtype=torch.float)
    data['tx', 'output', 'addr'].edge_attr = torch.cat([output_attrs, bool_spent], dim=1)
    
    # Spent Outputs
    spent_edges = spent_pairs_df[['tx_hash_spent', 'tx_hash_spend']].applymap(tx_mapping.get).dropna()
    data['tx', 'spent_output', 'tx'].edge_index = torch.tensor(spent_edges.values.T, dtype=torch.long)

    return data

train_tx_mapping = pd.Series(train_txs_df.index.values, index=train_txs_df['hash']).to_dict()
train_addr_mapping = pd.Series(train_addrs_df.index.values, index=train_addrs_df['addr_str']).to_dict()

val_tx_mapping    = pd.Series(val_txs_df.index.values, index=val_txs_df['hash']).to_dict()
val_addr_mapping  = pd.Series(val_addrs_df.index.values, index=val_addrs_df['addr_str']).to_dict()

test_tx_mapping = pd.Series(test_txs_df.index.values, index=test_txs_df['hash']).to_dict()
test_addr_mapping = pd.Series(test_addrs_df.index.values, index=test_addrs_df['addr_str']).to_dict()

for split, tx_map, addr_map in [
    ('train', train_tx_mapping,  train_addr_mapping),
    ('val',   val_tx_mapping,    val_addr_mapping),
    ('test',  test_tx_mapping,   test_addr_mapping),
]:
    with open(data_path / split / "tx_mapping.json",  "w") as f:
        json.dump(tx_map, f)
    with open(data_path / split / "addr_mapping.json","w") as f:
        json.dump(addr_map, f)
    
train_data = build_graph(
    train_txs_df, train_addrs_df, train_inputs_df, train_outputs_df,
    train_spent_pairs_df, train_tx_mapping, train_addr_mapping
)
val_data = build_graph(
    val_txs_df, val_addrs_df, val_inputs_df, val_outputs_df,
    val_spent_pairs_df, val_tx_mapping, val_addr_mapping
)
test_data = build_graph(
    test_txs_df, test_addrs_df, test_inputs_df, test_outputs_df,
    test_spent_pairs_df, test_tx_mapping, test_addr_mapping
)

torch.save(train_data, data_path / 'train' / 'graph.pth')
torch.save(val_data,   data_path / 'val' / 'graph.pth')
torch.save(test_data,  data_path / 'test' / 'graph.pth')
