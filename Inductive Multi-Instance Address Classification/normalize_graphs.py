# STEP 3 (Methodology I) - FEATURE NORMALIZATION (PyTorch environment)
#
# Normalizes the node and edge features of the serialized graphs once, so that
# every training run reads the same, already normalized data:
#   1. Clipping of the transaction locktime to the training mean +- 3 std
#   2. Logarithmic compression of heavy-tailed features
#   3. Min-max scaling, and quantile normalization of selected address features
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
from shared.experiment import git_commit, timestamp, write_json

GRAPHS_DIR = config.MULTI_INSTANCE_GRAPHS_DIR

# Feature indices (see serialize_graph.py for the feature order)
LOCKTIME = 2
TX_LOG = [1, 3]  # fee, total_size
ADDR_LOG = [1, 2, 3, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]
INPUT_LOG = [0, 2]  # age, value
OUTPUT_LOG = [0]  # value
ADDR_MINMAX = [1, 2, 3, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]
TX_MINMAX = [0, 1, 2, 3]  # block_height, fee, locktime, total_size
INPUT_MINMAX = [0, 2]  # age, value
OUTPUT_MINMAX = [0]  # value
ADDR_QUANTILE = [4, 8, 9]

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

def log_transform(graphs, element_type, element_name, attribute_indices):
    for data in graphs:
        for idx in attribute_indices:
            if element_type == 'node':
                data[element_name].x[:, idx] = torch.log1p(data[element_name].x[:, idx])
            elif element_type == 'edge':
                data[element_name].edge_attr[:, idx] = torch.log1p(data[element_name].edge_attr[:, idx])

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


def main():
    targets = [GRAPHS_DIR / split / "graph_normalized.pth" for split in config.SPLITS]
    if all(t.exists() for t in targets):
        print("Graphs already normalized, nothing to do. Delete graph_normalized.pth to normalize them again.")
        return

    graphs = {}
    for split in config.SPLITS:
        path = GRAPHS_DIR / split / "graph.pth"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}: run serialize_graph.py first.")
        graphs[split] = torch.load(path)

    train_data = graphs["train"]
    others = [graphs["val"], graphs["test"]]
    all_graphs = [train_data] + others

    lo, hi = compute_clipping_bounds(train_data['tx'].x, z_threshold=3, idx=LOCKTIME)
    for split in config.SPLITS:
        graphs[split] = apply_clipping(graphs[split], lo, hi, idx=LOCKTIME)

    log_transform(all_graphs, 'node', 'tx', TX_LOG)
    log_transform(all_graphs, 'node', 'addr', ADDR_LOG)
    log_transform(all_graphs, 'edge', INPUT, INPUT_LOG)
    log_transform(all_graphs, 'edge', OUTPUT, OUTPUT_LOG)

    norm_attr("minmax", train_data, others, 'node', 'addr', ADDR_MINMAX)
    norm_attr("minmax", train_data, others, 'node', 'tx', TX_MINMAX)
    norm_attr("minmax", train_data, others, 'edge', INPUT, INPUT_MINMAX)
    norm_attr("minmax", train_data, others, 'edge', OUTPUT, OUTPUT_MINMAX)
    norm_attr("quantile", train_data, others, 'node', 'addr', ADDR_QUANTILE)

    for split in config.SPLITS:
        split_path = GRAPHS_DIR / split
        torch.save(graphs[split], split_path / "graph_normalized.pth")
        labeled_nodes(graphs[split], split_path).to_csv(split_path / "labeled_nodes.csv", index=False)
        print(f"Saved: {split_path / 'graph_normalized.pth'}")

    write_json(GRAPHS_DIR / "normalization.json", {
        "created": timestamp(),
        "git_commit": git_commit(config.PROJECT_ROOT),
        "locktime_clipping": {"low": float(lo), "high": float(hi)},
        "log_transform": {"tx": TX_LOG, "addr": ADDR_LOG, "input": INPUT_LOG, "output": OUTPUT_LOG},
        "minmax": {"tx": TX_MINMAX, "addr": ADDR_MINMAX, "input": INPUT_MINMAX, "output": OUTPUT_MINMAX},
        "quantile": {"addr": ADDR_QUANTILE},
    })


if __name__ == "__main__":
    main()
