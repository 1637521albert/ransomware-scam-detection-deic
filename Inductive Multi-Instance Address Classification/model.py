# STEP 4 (Methodology I) - GNN TRAINING AND EVALUATION (PyTorch environment)
#
# Trains one GNN configuration (architecture + number of layers) with one
# training seed on the normalized graphs (normalize_graphs.py), and evaluates the
# best model on the test graph. All runs are launched by run_experiments.py, but
# a single run can also be executed directly:
#
#   python model.py --arch GAT --layers 2 --seed 42
#
# A run is deterministic: the seed fixes the weight initialization and dropout,
# and PyTorch runs with deterministic algorithms and a fixed number of threads
# (--threads, recorded in the result), so the same arguments always give the
# same result.
#
# Training: Adam (no weight decay), cross-entropy over the labeled seed
# addresses of the training graph, full-graph forward and backward pass per
# epoch. The best epoch is the one with the highest validation accuracy;
# training runs at least min_epochs and then stops after patience epochs
# without improving it (config.GNN_TRAINING).
#
# Output in MULTI_INSTANCE_RESULTS_DIR/runs/<arch>-<layers>l-seed<seed>/:
#   result.json      full configuration (expansion, normalization, training),
#                    training history, best epoch and test metrics
#   predictions.csv  test seed addresses with their label and predicted probability
#   model.pth        weights of the best epoch

import argparse
import copy
import json
import resource
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch_geometric.nn import HeteroConv, Linear, GATConv, LayerNorm, HGTConv, HANConv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from shared.experiment import git_commit, library_versions, read_json, timestamp, write_json

GRAPHS_DIR = config.MULTI_INSTANCE_GRAPHS_DIR
RUNS_DIR = config.MULTI_INSTANCE_RESULTS_DIR / "runs"
HP = config.GNN_TRAINING

## Model Definition

# Number of features of the nodes and edges (see serialize_graph.py) and graph metadata
NODE_DIMS = {'addr': 53, 'tx': 6}
EDGE_DIMS = {'input': 3, 'output': 2}
METADATA = (['addr', 'tx'], [('addr', 'input', 'tx'), ('tx', 'output', 'addr'), ('tx', 'spent_output', 'tx')])


def node_encoders(hidden_channels, dropout_prob):
    """Projection of the raw address and transaction features to hidden_channels, shared by all architectures."""
    return torch.nn.ModuleDict({
        node_type: torch.nn.Sequential(
            Linear(NODE_DIMS[node_type], hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_prob)
        )
        for node_type in ('addr', 'tx')
    })


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, dropout_prob, num_layers):
        super().__init__()

        self.node_enc = node_encoders(hidden_channels, dropout_prob)

        self.norm_addr = LayerNorm(hidden_channels)
        self.norm_tx = LayerNorm(hidden_channels)

        self.convs = torch.nn.ModuleList()
        for layer_idx in range(num_layers):
            conv_config = {
                ('addr', 'input', 'tx'): GATConv((-1, -1), hidden_channels, heads=4, concat=False, edge_dim=EDGE_DIMS['input'], add_self_loops=False, dropout=dropout_prob),
                ('tx', 'output', 'addr'): GATConv((-1, -1), hidden_channels, heads=4, concat=False, edge_dim=EDGE_DIMS['output'], add_self_loops=False, dropout=dropout_prob),
            }
            # A spent_output edge completes a full hop on its own, while the address-transaction
            # edges need two layers, so it is added every two layers (layers 1, 3, ...)
            if layer_idx % 2 == 0:
                conv_config[('tx', 'spent_output', 'tx')] = GATConv(-1, hidden_channels, heads=4, concat=False, add_self_loops=False, dropout=dropout_prob)
            self.convs.append(HeteroConv(conv_config, aggr='sum'))

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        edge_attr_dict = data.edge_attr_dict

        x_dict = {
            'addr': self.node_enc['addr'](x_dict['addr']).squeeze(1),
            'tx':   self.node_enc['tx'](x_dict['tx'])
        }

        for conv in self.convs:
            x_prev = x_dict
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = {
                'addr': self.norm_addr(x_dict['addr'] + x_prev['addr']),
                'tx': self.norm_tx(x_dict['tx'] + x_prev['tx'])
            }

        out = self.lin(x_dict['addr'])
        return out


class HeteroGraphTransformer(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, dropout_prob, num_layers):
        super().__init__()

        self.node_enc = node_encoders(hidden_channels, dropout_prob)

        self.norm_addr = LayerNorm(hidden_channels)
        self.norm_tx = LayerNorm(hidden_channels)

        self.convs = torch.nn.ModuleList([
            HGTConv(hidden_channels, hidden_channels, metadata=METADATA, heads=4)
            for _ in range(num_layers)
        ])

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict

        x_dict = {
            'addr': self.node_enc['addr'](x_dict['addr']).squeeze(1),
            'tx':   self.node_enc['tx'](x_dict['tx'])
        }

        for conv in self.convs:
            x_prev = x_dict
            x_norm = {k: self.norm_addr(v) if k == 'addr' else self.norm_tx(v) for k, v in x_dict.items()}
            x_new = conv(x_norm, edge_index_dict)
            x_new = {key: F.relu(x) for key, x in x_new.items()}
            x_dict = {'addr': x_prev['addr'] + x_new['addr'], 'tx': x_prev['tx'] + x_new['tx']}

        out = self.lin(x_dict['addr'])
        return out


class HeteroAttentionNet(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, dropout_prob, num_layers):
        super().__init__()

        self.node_enc = node_encoders(hidden_channels, dropout_prob)

        self.norm_addr = LayerNorm(hidden_channels)
        self.norm_tx = LayerNorm(hidden_channels)

        self.convs = torch.nn.ModuleList([
            HANConv(hidden_channels, hidden_channels, heads=4, dropout=dropout_prob, metadata=METADATA)
            for _ in range(num_layers)
        ])

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict

        x_dict = {
            'addr': self.node_enc['addr'](x_dict['addr']).squeeze(1),
            'tx':   self.node_enc['tx'](x_dict['tx'])
        }

        for conv in self.convs:
            x_prev = x_dict
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = {
                'addr': self.norm_addr(x_dict['addr'] + x_prev['addr']),
                'tx': self.norm_tx(x_dict['tx'] + x_prev['tx'])
            }


        out = self.lin(x_dict['addr'])
        return out


ARCHITECTURES = {
    "GAT": HeteroGNN,
    "HGT": HeteroGraphTransformer,
    "HAN": HeteroAttentionNet,
}

## Training

def run_name(arch, layers, seed):
    return f"{arch}-{layers}l-seed{seed}"


def set_determinism(seed, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True, warn_only=True)


def data_configuration():
    """How the graphs were built: expansion and normalization settings saved by the previous steps."""
    data = {}
    for key, path in [("expansion", GRAPHS_DIR / "expansion_config.json"),
                      ("normalization", GRAPHS_DIR / "normalization.json"),
                      ("splits", config.SPLITS_SUMMARY_FILE)]:
        data[key] = read_json(path) if path.exists() else None
    return data


def load_graphs():
    graphs = {}
    for split in config.SPLITS:
        path = GRAPHS_DIR / split / "graph_normalized.pth"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}: run normalize_graphs.py first.")
        graphs[split] = torch.load(path)
    return graphs


def classification_metrics(y_true, prob_illicit):
    y_pred = (prob_illicit >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "auc_roc": roc_auc_score(y_true, prob_illicit),
    }


def train_run(arch, layers, seed, threads=1, graphs=None, verbose=False):
    set_determinism(seed, threads)
    graphs = graphs or load_graphs()
    train_data, val_data, test_data = graphs["train"], graphs["val"], graphs["test"]
    masks = {split: graphs[split]['addr'].y != -1 for split in config.SPLITS}

    model = ARCHITECTURES[arch](HP["hidden_channels"], 2, HP["dropout"], layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=HP["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss()

    # Lazy layers are initialized with a first forward pass
    model.train()
    with torch.no_grad():
        model(train_data)

    y_train = train_data['addr'].y[masks["train"]]
    y_val = val_data['addr'].y[masks["val"]]

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    best = {"val_accuracy": -1.0, "epoch": 0, "state": None}
    epochs_no_improve = 0
    start = time.perf_counter()

    if verbose:
        print(f"{run_name(arch, layers, seed)}: training with {threads} threads (started {timestamp()})", flush=True)

    for epoch in range(1, HP["max_epochs"] + 1):
        epoch_start = time.perf_counter()
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(train_data)[masks["train"]], y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_data)[masks["val"]]
            val_loss = criterion(val_logits, y_val).item()
            val_accuracy = (val_logits.argmax(dim=1) == y_val).float().mean().item()

        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        if val_accuracy > best["val_accuracy"]:
            best = {"val_accuracy": val_accuracy, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose:
            # Flushed at every epoch, so the progress can be followed in train.log
            print(f"Epoch {epoch:4d} | train loss {loss.item():.4f} | val loss {val_loss:.4f} | val acc {val_accuracy:.4f} | "
                  f"best {best['val_accuracy']:.4f} (epoch {best['epoch']}) | {time.perf_counter() - epoch_start:.0f} s",
                  flush=True)

        if epoch >= HP["min_epochs"] and epochs_no_improve >= HP["patience"]:
            break

    train_time = time.perf_counter() - start
    epochs = len(history["train_loss"])

    # Test evaluation of the best epoch
    model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        prob_illicit = F.softmax(model(test_data)[masks["test"]], dim=1)[:, 1].numpy()
    y_test = test_data['addr'].y[masks["test"]].numpy()
    test_metrics = classification_metrics(y_test, prob_illicit)

    return {
        "arch": arch,
        "layers": layers,
        "seed": seed,
        "epochs": epochs,
        "best_epoch": best["epoch"],
        "best_val_accuracy": best["val_accuracy"],
        "train_time_s": train_time,
        "time_per_epoch_s": train_time / epochs,
        **{"test_" + k: float(v) for k, v in test_metrics.items()},
        "n_test": int(len(y_test)),
        "threads": threads,
        "hyperparameters": dict(HP),
        "evaluation": {
            "selection": "epoch with the highest validation accuracy",
            "test_split": "labeled seed addresses of the test graph",
            "prediction": "illicit if the predicted probability is >= 0.5",
            "metrics": ["accuracy", "precision", "recall", "f1", "auc_roc"],
        },
        "data": data_configuration(),
        "git_commit": git_commit(config.PROJECT_ROOT),
        "libraries": library_versions(("torch", "torch_geometric", "sklearn", "numpy")),
        "finished": timestamp(),
        "history": history,
    }, best["state"], prob_illicit


def probe_memory(arch, layers, seed, threads):
    """Peak memory of one training epoch (graphs, model, activations and gradients),
    used by run_experiments.py to decide how many runs fit in memory at once."""
    set_determinism(seed, threads)
    graphs = load_graphs()
    train_data, val_data = graphs["train"], graphs["val"]
    train_mask = train_data['addr'].y != -1
    model = ARCHITECTURES[arch](HP["hidden_channels"], 2, HP["dropout"], layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=HP["learning_rate"])
    model.train()
    with torch.no_grad():
        model(train_data)
    start = time.perf_counter()
    optimizer.zero_grad()
    loss = torch.nn.CrossEntropyLoss()(model(train_data)[train_mask], train_data['addr'].y[train_mask])
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        model(val_data)
    epoch_time = time.perf_counter() - start
    # ru_maxrss is in KiB on Linux
    return {"arch": arch, "layers": layers, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "epoch_time_s": epoch_time, "threads": threads}


def save_run(result, state, prob_illicit):
    run_dir = RUNS_DIR / run_name(result["arch"], result["layers"], result["seed"])
    run_dir.mkdir(parents=True, exist_ok=True)

    nodes = pd.read_csv(GRAPHS_DIR / "test" / "labeled_nodes.csv")
    nodes["prob_illicit"] = prob_illicit
    nodes["prediction"] = (prob_illicit >= 0.5).astype(int)
    nodes.to_csv(run_dir / "predictions.csv", index=False)

    torch.save(state, run_dir / "model.pth")
    # Written last: its presence marks the run as completed
    write_json(run_dir / "result.json", result)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate one GNN configuration with one seed.")
    parser.add_argument("--arch", required=True, choices=sorted(ARCHITECTURES))
    parser.add_argument("--layers", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--threads", type=int, default=config.CPU_COUNT, help="PyTorch threads of this run")
    parser.add_argument("--force", action="store_true", help="train again even if the run already exists")
    parser.add_argument("--verbose", action="store_true", help="print the losses and accuracy of every epoch")
    parser.add_argument("--probe", action="store_true", help="only measure the peak memory of one epoch and print it as JSON")
    args = parser.parse_args()

    if args.probe:
        print("PROBE " + json.dumps(probe_memory(args.arch, args.layers, args.seed, args.threads)))
        return

    result_path = RUNS_DIR / run_name(args.arch, args.layers, args.seed) / "result.json"
    if result_path.exists() and not args.force:
        print(f"Already completed: {result_path}")
        return

    result, state, prob_illicit = train_run(args.arch, args.layers, args.seed, threads=args.threads, verbose=args.verbose)
    run_dir = save_run(result, state, prob_illicit)
    print(f"{run_name(args.arch, args.layers, args.seed)}: {result['epochs']} epochs (best {result['best_epoch']}), "
          f"test F1 {result['test_f1']:.4f}, AUC-ROC {result['test_auc_roc']:.4f} -> {run_dir}")


if __name__ == "__main__":
    main()
