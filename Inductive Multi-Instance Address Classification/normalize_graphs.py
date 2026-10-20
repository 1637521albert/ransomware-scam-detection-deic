# STEP 3 (Methodology I) - FEATURE NORMALIZATION
#
# Normalizes the node and edge features of the serialized graphs once, so that
# every training run reads the same, already normalized data:
#   1. Clipping of the transaction locktime to the training mean +- 3 std
#   2. Logarithmic compression of heavy-tailed features (negative values, such as
#      the -1 block height of an address without transactions, are set to 0
#      first, as log1p is not defined below -1; how many is reported)
#   3. Min-max scaling, and quantile normalization of selected address features
# The categorical features are 0/1 columns (serialize_graph.py) and are left as they are.
# All statistics are fitted on the training graph only and then applied to the
# validation and test graphs.
#
# Output in MULTI_INSTANCE_GRAPHS_DIR/<split>/:
#   graph_normalized.pth  normalized graph read by model.py
#   labeled_nodes.csv     node index, address and label of the seed addresses
# and MULTI_INSTANCE_GRAPHS_DIR/normalization.json with the fitted parameters.

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from shared.experiment import code_fingerprint, digest, git_commit, read_json, timestamp, write_json
from serialize_graph import FULL_TYPE_COLUMNS, SEQUENCE_COLUMNS

GRAPHS_DIR = config.MULTI_INSTANCE_GRAPHS_DIR

# Names of the node and edge features, in the order of serialize_graph.py: the
# categorical features are 0/1 columns (FULL_TYPE_COLUMNS, SEQUENCE_COLUMNS, version_2,
# is_coinbase, is_spent), already in [0, 1], and are not transformed
ADDR_FEATURES = FULL_TYPE_COLUMNS + \
                ["income", "expenditure", "balance", "outflow_ratio", "n_incoming_txs",
                 "n_outgoing_txs", "n_total_txs", "top1_income_share", "top5_income_share", "n_utxos"] + \
                [f"{stat}_{name}" for name in ["utxo_value"] for stat in ("avg", "max", "min", "std")] + \
                [f"{stat}_{name}" for name in ["out_tx_value", "in_tx_value", "in_tx_inputs", "in_tx_outputs",
                                               "out_tx_inputs", "out_tx_outputs"] for stat in ("min", "max", "avg", "std")] + \
                ["first_tx_block", "last_tx_block"] + \
                [f"{stat}_{name}" for name in ["in_tx_wait", "out_tx_wait", "input_age"] for stat in ("min", "max", "avg", "std")]
TX_FEATURES = ["block_height", "fee", "locktime", "total_size", "version_2", "is_coinbase"]
INPUT_FEATURES = ["age", "value"] + SEQUENCE_COLUMNS
OUTPUT_FEATURES = ["value", "is_spent"]


def indices(features, names):
    return [features.index(name) for name in names]


# Ratios and shares are quantile-normalized; every other numerical address feature is
# log-compressed and min-max scaled
ADDR_RATIOS = ["outflow_ratio", "top1_income_share", "top5_income_share"]
ADDR_NUMERICAL = [f for f in ADDR_FEATURES if f not in FULL_TYPE_COLUMNS]
ADDR_LOG = indices(ADDR_FEATURES, [f for f in ADDR_NUMERICAL if f not in ADDR_RATIOS])
ADDR_MINMAX = indices(ADDR_FEATURES, [f for f in ADDR_NUMERICAL if f not in ADDR_RATIOS])
ADDR_QUANTILE = indices(ADDR_FEATURES, ADDR_RATIOS)
LOCKTIME = TX_FEATURES.index("locktime")
TX_LOG = indices(TX_FEATURES, ["fee", "total_size"])
TX_MINMAX = indices(TX_FEATURES, ["block_height", "fee", "locktime", "total_size"])
INPUT_LOG = indices(INPUT_FEATURES, ["age", "value"])
INPUT_MINMAX = indices(INPUT_FEATURES, ["age", "value"])
OUTPUT_LOG = indices(OUTPUT_FEATURES, ["value"])
OUTPUT_MINMAX = indices(OUTPUT_FEATURES, ["value"])

INPUT = ('addr', 'input', 'tx')
OUTPUT = ('tx', 'output', 'addr')

### 1.- Clip Outliers

def compute_clipping_bounds(x, z_threshold=3, idx=2):
    vals = x[:, idx].cpu().numpy()
    mu, sigma = vals.mean(), vals.std()
    lo = mu - z_threshold * sigma
    hi = mu + z_threshold * sigma
    return lo, hi

def apply_clipping(data, lo, hi, idx=2):
    x = data['tx'].x.clone()
    if x[:, idx].dtype.is_floating_point:
        vals = x[:, idx].cpu().numpy()
        clipped = np.clip(vals, lo, hi)
        with torch.no_grad():
            x[:, idx] = torch.from_numpy(clipped).to(x.device, dtype=x.dtype)
    data['tx'].x = x
    return data

### 2.- Data Normalization

def log_transform(graphs, element_type, element_name, attribute_indices, names, split_names, clipped):
    attribute = 'x' if element_type == 'node' else 'edge_attr'
    for data, split in zip(graphs, split_names):
        values = getattr(data[element_name], attribute)
        for idx in attribute_indices:
            # log1p is only defined above -1, so negative values are set to 0
            negative = int((values[:, idx] < 0).sum())
            if negative:
                values[:, idx] = values[:, idx].clamp(min=0)
                clipped.setdefault(f"{element_type} {element_name}: {names[idx]}", {})[split] = negative
            values[:, idx] = torch.log1p(values[:, idx])


def check_finite(graphs, split_names):
    problems = []
    for data, split in zip(graphs, split_names):
        for element, attribute, names in [('addr', 'x', ADDR_FEATURES), ('tx', 'x', TX_FEATURES),
                                          (INPUT, 'edge_attr', INPUT_FEATURES), (OUTPUT, 'edge_attr', OUTPUT_FEATURES)]:
            values = getattr(data[element], attribute)
            bad = (~torch.isfinite(values)).sum(dim=0)
            for idx in torch.nonzero(bad).flatten().tolist():
                problems.append(f"{split} {element} {names[idx] if idx < len(names) else idx}: {int(bad[idx])} non-finite values")
    if problems:
        raise ValueError("Non-finite normalized features:\n  " + "\n  ".join(problems))

def norm_attr(strategy, train_data, other_graphs, element_type, element_name, attribute_indices):
    attribute = 'x' if element_type == 'node' else 'edge_attr'
    for idx in attribute_indices:
        if strategy == 'minmax':
            scaler = MinMaxScaler()
        elif strategy == 'quantile':
            quantile_count = min(1000, getattr(train_data[element_name], attribute).shape[0])
            # The quantiles are estimated on a random subsample, so the random state is fixed
            scaler = QuantileTransformer(n_quantiles=quantile_count, random_state=config.DATA_SEED)

        values = getattr(train_data[element_name], attribute)
        values[:, idx] = torch.tensor(scaler.fit_transform(values[:, idx].unsqueeze(1).numpy()).flatten(), dtype=torch.float)
        for data in other_graphs:
            values = getattr(data[element_name], attribute)
            values[:, idx] = torch.tensor(scaler.transform(values[:, idx].unsqueeze(1).numpy()).flatten(), dtype=torch.float)


def labeled_nodes(data, split_path):
    with open(split_path / "addr_mapping.json") as f:
        addr_mapping = json.load(f)
    index_to_addr = {int(v): k for k, v in addr_mapping.items()}
    y = data['addr'].y.numpy()
    idx = np.where(y != -1)[0]
    return pd.DataFrame({"node_index": idx, "address": [index_to_addr[i] for i in idx], "label": y[idx]})


def serialized_fingerprints():
    """Fingerprint of the serialized graph of every split (graph_meta.json: its export
    and serialization code)."""
    fingerprints = {}
    for split in config.SPLITS:
        path = GRAPHS_DIR / split / "graph_meta.json"
        if not (GRAPHS_DIR / split / "graph.pth").exists() or not path.exists():
            raise FileNotFoundError(f"Missing the serialized {split} graph: run serialize_graph.py first.")
        fingerprints[split] = read_json(path).get("fingerprint")
    return fingerprints


def main():
    # The graphs are normalized again when they were serialized again or this code changed;
    # the fingerprint of the normalized graphs is what the training runs are reused by
    fingerprints = serialized_fingerprints()
    code = code_fingerprint({Path(__file__).resolve(): None})
    fingerprint = digest({"graphs": fingerprints, "normalization_code": code})
    targets = [GRAPHS_DIR / split / "graph_normalized.pth" for split in config.SPLITS]
    previous = GRAPHS_DIR / "normalization.json"
    if all(t.exists() for t in targets) and previous.exists() \
            and read_json(previous).get("fingerprint") == fingerprint:
        print("Graphs already normalized, nothing to do. Delete graph_normalized.pth to normalize them again.")
        return
    if previous.exists():
        # The normalized graphs are no longer valid (until normalization.json is written again)
        previous.unlink()

    graphs = {}
    for split in config.SPLITS:
        graphs[split] = torch.load(GRAPHS_DIR / split / "graph.pth")

    train_data = graphs["train"]
    others = [graphs[split] for split in config.SPLITS if split != "train"]
    split_names = config.SPLITS
    all_graphs = [graphs[split] for split in split_names]

    lo, hi = compute_clipping_bounds(train_data['tx'].x, z_threshold=3, idx=LOCKTIME)
    for split in config.SPLITS:
        graphs[split] = apply_clipping(graphs[split], lo, hi, idx=LOCKTIME)

    clipped = {}
    log_transform(all_graphs, 'node', 'tx', TX_LOG, TX_FEATURES, split_names, clipped)
    log_transform(all_graphs, 'node', 'addr', ADDR_LOG, ADDR_FEATURES, split_names, clipped)
    log_transform(all_graphs, 'edge', INPUT, INPUT_LOG, INPUT_FEATURES, split_names, clipped)
    log_transform(all_graphs, 'edge', OUTPUT, OUTPUT_LOG, OUTPUT_FEATURES, split_names, clipped)
    for feature, counts in clipped.items():
        print(f"Negative values set to 0 before the log transform in {feature}: {counts}")

    norm_attr("minmax", train_data, others, 'node', 'addr', ADDR_MINMAX)
    norm_attr("minmax", train_data, others, 'node', 'tx', TX_MINMAX)
    norm_attr("minmax", train_data, others, 'edge', INPUT, INPUT_MINMAX)
    norm_attr("minmax", train_data, others, 'edge', OUTPUT, OUTPUT_MINMAX)
    norm_attr("quantile", train_data, others, 'node', 'addr', ADDR_QUANTILE)
    check_finite(all_graphs, split_names)

    for split in config.SPLITS:
        split_path = GRAPHS_DIR / split
        torch.save(graphs[split], split_path / "graph_normalized.pth")
        labeled_nodes(graphs[split], split_path).to_csv(split_path / "labeled_nodes.csv", index=False)
        print(f"Saved: {split_path / 'graph_normalized.pth'}")

    # Written last; the training runs record it and are only reused on the same graphs
    write_json(GRAPHS_DIR / "normalization.json", {
        "created": timestamp(),
        "git_commit": git_commit(config.PROJECT_ROOT),
        "fingerprint": fingerprint,
        "graph_fingerprints": fingerprints,
        "normalization_code": code,
        "locktime_clipping": {"low": float(lo), "high": float(hi)},
        "log_transform": {"tx": TX_LOG, "addr": ADDR_LOG, "input": INPUT_LOG, "output": OUTPUT_LOG},
        "negative_values_set_to_0_before_log": clipped,
        "minmax": {"tx": TX_MINMAX, "addr": ADDR_MINMAX, "input": INPUT_MINMAX, "output": OUTPUT_MINMAX},
        "quantile": {"addr": ADDR_QUANTILE},
    })


if __name__ == "__main__":
    main()
