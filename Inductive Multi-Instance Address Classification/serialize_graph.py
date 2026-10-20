# STEP 2 (Methodology I) - GRAPH SERIALIZATION
#
# Converts the feature files of each split (MULTI_INSTANCE_GRAPHS_DIR/<split>/,
# written by expansion.py) into a PyTorch Geometric HeteroData graph:
#   graph.pth          the heterogeneous graph with raw features
#   addr_mapping.json  address string -> address node index
#   tx_mapping.json    transaction hash -> transaction node index
#
# The categorical features (the address full_type and the input sequence number)
# are encoded with one vocabulary shared by every split, so that a code means the
# same value in training, validation and test: the code of a value is its rank
# among the values of all the splits, sorted (CATEGORIES_FILE).
#
# The splits are serialized at the same time in separate processes, and splits
# already serialized from their current export, with this code and vocabulary,
# are skipped (graph_meta.json).

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
from shared.experiment import code_fingerprint, digest, write_json

CATEGORIES_FILE = config.MULTI_INSTANCE_GRAPHS_DIR / "categories.json"
# Categorical columns: feature file and column of every one
CATEGORICAL = {"full_type": ("addr_feats.csv", "full_type"), "sequence_num": ("input_feats.csv", "sequence_num")}


def encode(values, vocabulary):
    """Codes of the values in the shared vocabulary (sorted values of all the splits)."""
    codes = pd.Series(range(len(vocabulary)), index=vocabulary)
    encoded = values.map(codes)
    if encoded.isna().any():
        raise ValueError(f"Values without a code in the shared vocabulary: {sorted(set(values[encoded.isna()]))[:10]}")
    return encoded.astype(int)


def build_graph(txs_df, addrs_df, inputs_df, outputs_df, spent_pairs_df, tx_mapping, addr_mapping, categories):
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
    addrs_df['full_type'] = encode(addrs_df['full_type'], categories['full_type'])
    full_type = torch.tensor(addrs_df[['full_type']].values, dtype=torch.float)

    cols = [c for c in list(addrs_df.columns)
        if c not in ["addr_str", "class", "full_type", "incoming_counterparties", "outgoing_counterparties"]]
    aggr_feats = torch.tensor(addrs_df[cols].values, dtype=torch.float)

    data['addr'].x = torch.cat([full_type, aggr_feats], dim=1)
    data['addr'].id = addrs_df.index.values
    data['addr'].y = torch.tensor(addrs_df['class'].values, dtype=torch.long)

    # Inputs
    inputs_df['sequence_num'] = encode(inputs_df['sequence_num'], categories['sequence_num'])
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


def serialize_split(task):
    split_path, categories, fingerprint = task
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

    data = build_graph(txs_df, addrs_df, inputs_df, outputs_df, spent_pairs_df, tx_mapping, addr_mapping, categories)
    torch.save(data, split_path / 'graph.pth')
    # Written last: the graph was serialized from the export with this fingerprint
    with open(split_path / GRAPH_META, "w") as f:
        json.dump(fingerprint, f, indent=2)
    print(f"Saved: {split_path / 'graph.pth'}")


# Fingerprint of a serialized graph: the export it was built from (expansion_meta.json,
# written by expansion.py), the code of this script and the shared vocabulary. A graph
# serialized from another export, code or vocabulary is serialized again.
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


def shared_categories(split_paths):
    """Vocabulary of every categorical column: its values in all the splits, sorted."""
    categories = {}
    for name, (file_name, column) in CATEGORICAL.items():
        values = set()
        for split_path in split_paths:
            values.update(pd.read_csv(split_path / file_name, usecols=[column])[column].dropna().unique().tolist())
        categories[name] = sorted(values)
    return categories


def main():
    split_paths = [config.MULTI_INSTANCE_GRAPHS_DIR / split for split in config.SPLITS]
    exported = {}
    for split_path in split_paths:
        exported[split_path.name] = expansion_fingerprint(split_path)
        if exported[split_path.name] is None:
            raise FileNotFoundError(f"The {split_path.name} split is not completely exported: run expansion.py first.")

    # The vocabulary only changes with the exports, so it is read again only when they change
    saved = json.loads(CATEGORIES_FILE.read_text()) if CATEGORIES_FILE.exists() else {}
    if saved.get("expansion_fingerprints") == exported:
        categories = saved["categories"]
    else:
        print("Reading the values of the categorical features of every split...")
        categories = shared_categories(split_paths)
        write_json(CATEGORIES_FILE, {"expansion_fingerprints": exported, "categories": categories})
    code = code_fingerprint({Path(__file__).resolve(): None})

    pending = []
    for split_path in split_paths:
        split = split_path.name
        fingerprint = {"expansion_fingerprint": exported[split], "serialization_code": code,
                       "categories": digest(categories)}
        fingerprint["fingerprint"] = digest(fingerprint)
        if serialized_fingerprint(split_path) == fingerprint["fingerprint"]:
            print(f"\n{split}: already serialized, skipping ({split_path / 'graph.pth'})")
            continue
        if (split_path / 'graph.pth').exists():
            print(f"\n{split}: graph.pth was serialized from another expansion, code or vocabulary, serializing it again")
        pending.append((split_path, categories, fingerprint))

    if len(pending) > 1:
        with multiprocessing.get_context("fork").Pool(len(pending)) as pool:
            pool.map(serialize_split, pending)
    else:
        for task in pending:
            serialize_split(task)


if __name__ == "__main__":
    main()
