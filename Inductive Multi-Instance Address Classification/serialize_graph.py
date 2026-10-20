# STEP 2 (Methodology I) - GRAPH SERIALIZATION
#
# Converts the feature files of each split (MULTI_INSTANCE_GRAPHS_DIR/<split>/,
# written by expansion.py) into a PyTorch Geometric HeteroData graph:
#   graph.pth          the heterogeneous graph with raw features
#   addr_mapping.json  address identifier -> address node index
#   tx_mapping.json    transaction hash -> transaction node index
#
# The categorical features are encoded with fixed rules, the same in every split, as
# 0/1 columns (one-hot or flags), so a column means the same in training, validation
# and test and every one is already in [0, 1]:
#   address full_type   one-hot of the script type (P2PK, P2PKH, P2SH, P2WPKH, P2WSH,
#                       other) and two flags: the script involves multisig, and it
#                       involves SegWit (native or nested in P2SH)
#   input sequence_num  one-hot: final (0xFFFFFFFF), 0xFFFFFFFE (locktime enabled,
#                       no replace-by-fee), other (signals replace-by-fee, BIP 125)
#   tx version          flag: version 2 or higher (relative locktimes, BIP 68)
# The feature order is in normalize_graphs.py (ADDR_FEATURES, TX_FEATURES,
# INPUT_FEATURES, OUTPUT_FEATURES).
#
# The splits are serialized at the same time in separate processes, and splits
# already serialized from their current export with this code are skipped
# (graph_meta.json).

import json
import multiprocessing
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from shared.experiment import code_fingerprint, digest

# Script types of the full_type one-hot, by the first component of the BlockSci full
# type (e.g. "scripthash/witness_scripthash/multisig2of3" is P2SH); any other is "other"
SCRIPT_TYPES = {"pubkey": "p2pk", "pubkeyhash": "p2pkh", "scripthash": "p2sh",
                "witness_pubkeyhash": "p2wpkh", "witness_scripthash": "p2wsh"}
FULL_TYPE_COLUMNS = [f"type_{t}" for t in SCRIPT_TYPES.values()] + ["type_other", "type_multisig", "type_segwit"]
SEQUENCE_FINAL = 0xFFFFFFFF
SEQUENCE_NO_RBF = 0xFFFFFFFE
SEQUENCE_COLUMNS = ["sequence_final", "sequence_no_rbf", "sequence_rbf"]


def encode_full_type(full_types):
    """0/1 columns FULL_TYPE_COLUMNS of every BlockSci full type."""
    def encode(full_type):
        parts = str(full_type).split("/")
        script = SCRIPT_TYPES.get(parts[0], "other")
        row = [float(script == t) for t in SCRIPT_TYPES.values()] + [float(script == "other")]
        row.append(float(any(p.startswith("multisig") for p in parts)))
        row.append(float(any(p.startswith("witness") for p in parts)))
        return row
    # The address types are few, so every distinct one is encoded once
    table = {value: encode(value) for value in pd.unique(full_types)}
    return np.array([table[value] for value in full_types], dtype=np.float32).reshape(-1, len(FULL_TYPE_COLUMNS))


def encode_sequence(sequence_nums):
    """0/1 columns SEQUENCE_COLUMNS of every input sequence number."""
    values = sequence_nums.to_numpy()
    final = values == SEQUENCE_FINAL
    no_rbf = values == SEQUENCE_NO_RBF
    return np.stack([final, no_rbf, ~(final | no_rbf)], axis=1).astype(np.float32)


def build_graph(txs_df, addrs_df, inputs_df, outputs_df, spent_pairs_df, tx_mapping, addr_mapping):
    data = HeteroData()

    inputs_df['tx_idx'] = inputs_df['tx_hash'].map(tx_mapping).astype(int)
    inputs_df['addr_idx'] = inputs_df['addr_str'].map(addr_mapping).astype(int)

    outputs_df['tx_idx'] = outputs_df['tx_hash'].map(tx_mapping).astype(int)
    outputs_df['addr_idx'] = outputs_df['addr_str'].map(addr_mapping).astype(int)

    # Transactions: block_height, fee, locktime, total_size, version >= 2, is_coinbase
    tx_x = torch.tensor(txs_df[['block_height', 'fee', 'locktime', 'total_size']].values, dtype=torch.float)
    version_2 = torch.tensor((txs_df['version'] >= 2).astype(float).values, dtype=torch.float).unsqueeze(1)
    is_coinbase = torch.tensor(txs_df['is_coinbase'].astype(float).values, dtype=torch.float).unsqueeze(1)
    data['tx'].x = torch.cat([tx_x, version_2, is_coinbase], dim=1)
    data['tx'].id = txs_df.index.values

    # Addresses: full_type columns, then the address features in the order of addr_feats.csv
    full_type = torch.from_numpy(encode_full_type(addrs_df['full_type'].to_numpy()))
    cols = [c for c in list(addrs_df.columns)
            if c not in ["addr_str", "class", "full_type", "incoming_counterparties", "outgoing_counterparties"]]
    aggr_feats = torch.tensor(addrs_df[cols].values, dtype=torch.float)

    data['addr'].x = torch.cat([full_type, aggr_feats], dim=1)
    data['addr'].id = addrs_df.index.values
    data['addr'].y = torch.tensor(addrs_df['class'].values, dtype=torch.long)

    # Inputs: age, value, sequence columns
    data['addr', 'input', 'tx'].edge_index = torch.tensor(inputs_df[['addr_idx', 'tx_idx']].values.T, dtype=torch.long)
    input_attrs = torch.tensor(inputs_df[['age', 'value']].values, dtype=torch.float)
    sequence = torch.from_numpy(encode_sequence(inputs_df['sequence_num']))
    data['addr', 'input', 'tx'].edge_attr = torch.cat([input_attrs, sequence], dim=1)

    # Outputs: value, is_spent
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


def serialize_split(task):
    split_path, fingerprint = task
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
    # Written last: the graph was serialized from the export with this fingerprint
    with open(split_path / GRAPH_META, "w") as f:
        json.dump(fingerprint, f, indent=2)
    print(f"Saved: {split_path / 'graph.pth'}")


# Fingerprint of a serialized graph: the export it was built from (expansion_meta.json,
# written by expansion.py) and the code of this script. A graph serialized from another
# export or code is serialized again.
GRAPH_META = "graph_meta.json"


def expansion_fingerprint(split_path):
    path = split_path / "expansion_meta.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f).get("fingerprint")


def serialized_fingerprint(split_path):
    path = split_path / GRAPH_META
    if not (split_path / 'graph.pth').exists() or not path.exists():
        return None
    with open(path) as f:
        return json.load(f).get("fingerprint")


def main():
    code = code_fingerprint({Path(__file__).resolve(): None})
    pending = []
    for split in config.SPLITS:
        split_path = config.MULTI_INSTANCE_GRAPHS_DIR / split
        exported = expansion_fingerprint(split_path)
        if exported is None:
            raise FileNotFoundError(f"The {split} split is not completely exported: run expansion.py first.")
        fingerprint = {"expansion_fingerprint": exported, "serialization_code": code}
        fingerprint["fingerprint"] = digest(fingerprint)
        if serialized_fingerprint(split_path) == fingerprint["fingerprint"]:
            print(f"\n{split}: already serialized, skipping ({split_path / 'graph.pth'})")
            continue
        if (split_path / 'graph.pth').exists():
            print(f"\n{split}: graph.pth was serialized from another expansion or code, serializing it again")
        pending.append((split_path, fingerprint))

    if len(pending) > 1:
        with multiprocessing.get_context("fork").Pool(len(pending)) as pool:
            pool.map(serialize_split, pending)
    else:
        for task in pending:
            serialize_split(task)


if __name__ == "__main__":
    main()
