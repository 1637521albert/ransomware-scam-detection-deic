# STEP 2 (Methodology I) - GRAPH SERIALIZATION (PyTorch environment)
#
# Converts the feature files of each split (MULTI_INSTANCE_GRAPHS_DIR/<split>/,
# written by expansion.py) into a PyTorch Geometric HeteroData graph:
#   graph.pth          the heterogeneous graph with raw features
#   addr_mapping.json  address string -> address node index
#   tx_mapping.json    transaction hash -> transaction node index
#
# The splits are serialized at the same time in separate processes, and splits
# that are already serialized are skipped.

import json
import multiprocessing
import sys
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import HeteroData

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


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
    input_attrs = torch.tensor(inputs_df[['age', 'sequence_num', 'value']].values, dtype=torch.float)
    data['addr', 'input', 'tx'].edge_attr = input_attrs

    # Outputs
    data['tx', 'output', 'addr'].edge_index = torch.tensor(outputs_df[['tx_idx', 'addr_idx']].values.T, dtype=torch.long)
    output_attrs = torch.tensor(outputs_df[['value']].values, dtype=torch.float)
    bool_spent = torch.tensor(outputs_df[['is_spent']].astype(float).values, dtype=torch.float)
    data['tx', 'output', 'addr'].edge_attr = torch.cat([output_attrs, bool_spent], dim=1)

    # Spent Outputs
    spent_edges = pd.DataFrame({
        col: spent_pairs_df[col].map(tx_mapping) for col in ['tx_hash_spent', 'tx_hash_spend']
    }).dropna()
    data['tx', 'spent_output', 'tx'].edge_index = torch.tensor(spent_edges.values.T, dtype=torch.long)

    return data


def serialize_split(split_path):
    print(f"\nReading {split_path.name} features...")
    txs_df = pd.read_csv(split_path / 'tx_feats.csv')
    addrs_df = pd.read_csv(split_path / 'addr_feats.csv')
    inputs_df = pd.read_csv(split_path / 'input_feats.csv')
    outputs_df = pd.read_csv(split_path / 'output_feats.csv')
    spent_pairs_df = pd.read_csv(split_path / 'spent_pairs.csv')

    tx_mapping = pd.Series(txs_df.index.values, index=txs_df['hash']).to_dict()
    addr_mapping = pd.Series(addrs_df.index.values, index=addrs_df['addr_str']).to_dict()

    with open(split_path / "tx_mapping.json", "w") as f:
        json.dump({k: int(v) for k, v in tx_mapping.items()}, f)
    with open(split_path / "addr_mapping.json", "w") as f:
        json.dump({k: int(v) for k, v in addr_mapping.items()}, f)

    data = build_graph(txs_df, addrs_df, inputs_df, outputs_df, spent_pairs_df, tx_mapping, addr_mapping)
    torch.save(data, split_path / 'graph.pth')
    print(f"Saved: {split_path / 'graph.pth'}")


def main():
    pending = []
    for split in config.SPLITS:
        split_path = config.MULTI_INSTANCE_GRAPHS_DIR / split
        if (split_path / 'graph.pth').exists():
            print(f"\n{split}: already serialized, skipping ({split_path / 'graph.pth'})")
            continue
        if not (split_path / 'addr_feats.csv').exists():
            raise FileNotFoundError(f"Missing features of the {split} split: run expansion.py first.")
        pending.append(split_path)

    if len(pending) > 1:
        with multiprocessing.get_context("fork").Pool(len(pending)) as pool:
            pool.map(serialize_split, pending)
    else:
        for split_path in pending:
            serialize_split(split_path)


if __name__ == "__main__":
    main()
