# STEP 4 (Methodology I) - GNN TRAINING AND EVALUATION
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
# addresses of the training graph with label smoothing, full-graph forward and
# backward pass per epoch. The model that is evaluated is an exponential moving
# average of the trained weights (see update_average), so the validation loss, the
# best epoch, the saved weights and the test predictions are those of the averaged
# weights. The best epoch is the one with the lowest validation loss (the accuracy
# at the 0.5 threshold is too coarse: it can drop while the loss still improves,
# which stopped runs before they had learned); training runs at least min_epochs
# and then stops after patience epochs
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
from normalize_graphs import ADDR_FEATURES, INPUT_FEATURES, OUTPUT_FEATURES, TX_FEATURES
from shared.experiment import (code_fingerprint, digest, git_commit, library_versions, read_json, timestamp,
                               trained_on_current_graphs, write_json)

GRAPHS_DIR = config.MULTI_INSTANCE_GRAPHS_DIR
RUNS_DIR = config.MULTI_INSTANCE_RESULTS_DIR / "runs"
HP = config.GNN_TRAINING

# What a run is trained with besides its graphs: the training hyperparameters and the code
# of this file (its syntax tree, so comments do not count). A result is only reused if it
# was trained on the current graphs with this fingerprint (is_current).
TRAINING_FINGERPRINT = digest({"hyperparameters": HP, "code": code_fingerprint({Path(__file__).resolve(): None})})


def is_current(result):
    """Whether a result (the content of its result.json) was trained on the current graphs
    with the current training hyperparameters and code."""
    return trained_on_current_graphs(result, GRAPHS_DIR / "normalization.json") \
        and result.get("training_fingerprint") == TRAINING_FINGERPRINT

## Model Definition

# Number of features of the nodes and edges (see normalize_graphs.py) and graph metadata
NODE_DIMS = {'addr': len(ADDR_FEATURES), 'tx': len(TX_FEATURES)}
EDGE_DIMS = {'input': len(INPUT_FEATURES), 'output': len(OUTPUT_FEATURES)}
# The three relations of the serialized graphs follow the flow of funds (an address
# spends into a transaction, a transaction pays an address, a transaction spends the
# output of an earlier one), and messages only travel along an edge. Each relation
# therefore gets a reverse relation (add_reverse_edges), so that every address also
# receives the information of the transactions where it spends and of their future,
# not only of the transactions that paid it and of their past.
REVERSE_RELATIONS = {
    ('addr', 'input', 'tx'): ('tx', 'rev_input', 'addr'),
    ('tx', 'output', 'addr'): ('addr', 'rev_output', 'tx'),
    ('tx', 'spent_output', 'tx'): ('tx', 'rev_spent_output', 'tx'),
}
METADATA = (['addr', 'tx'], list(REVERSE_RELATIONS) + list(REVERSE_RELATIONS.values()))


def add_reverse_edges(data):
    """Adds the reverse relation of every relation, with the same edge features."""
    for relation, reverse in REVERSE_RELATIONS.items():
        store = data[relation]
        data[reverse].edge_index = store.edge_index.flip(0)
        if 'edge_attr' in store:
            data[reverse].edge_attr = store.edge_attr
    return data


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


def layer_norms(hidden_channels, num_layers):
    """One LayerNorm per layer and node type."""
    return torch.nn.ModuleList([
        torch.nn.ModuleDict({node_type: LayerNorm(hidden_channels) for node_type in ('addr', 'tx')})
        for _ in range(num_layers)
    ])


def encode(node_enc, x_dict):
    return {
        'addr': node_enc['addr'](x_dict['addr']).squeeze(1),
        'tx':   node_enc['tx'](x_dict['tx'])
    }


class HeteroGNN(torch.nn.Module):
    """GAT: every layer concatenates its attention heads, of hidden_channels / heads
    dimensions each, into hidden_channels; then ReLU, residual connection and LayerNorm."""

    def __init__(self, hidden_channels, out_channels, dropout_prob, num_layers, heads):
        super().__init__()

        self.node_enc = node_encoders(hidden_channels, dropout_prob)
        self.norms = layer_norms(hidden_channels, num_layers)

        head_channels = hidden_channels // heads
        self.convs = torch.nn.ModuleList()
        for layer_idx in range(num_layers):
            conv_config = {
                ('addr', 'input', 'tx'): GATConv((-1, -1), head_channels, heads=heads, concat=True, edge_dim=EDGE_DIMS['input'], add_self_loops=False, dropout=dropout_prob),
                ('tx', 'output', 'addr'): GATConv((-1, -1), head_channels, heads=heads, concat=True, edge_dim=EDGE_DIMS['output'], add_self_loops=False, dropout=dropout_prob),
                ('tx', 'rev_input', 'addr'): GATConv((-1, -1), head_channels, heads=heads, concat=True, edge_dim=EDGE_DIMS['input'], add_self_loops=False, dropout=dropout_prob),
                ('addr', 'rev_output', 'tx'): GATConv((-1, -1), head_channels, heads=heads, concat=True, edge_dim=EDGE_DIMS['output'], add_self_loops=False, dropout=dropout_prob),
            }
            # A spent_output edge completes a full hop on its own, while the address-transaction
            # edges need two layers, so it is added every two layers (layers 1, 3, ...)
            if layer_idx % 2 == 0:
                conv_config[('tx', 'spent_output', 'tx')] = GATConv(-1, head_channels, heads=heads, concat=True, add_self_loops=False, dropout=dropout_prob)
                conv_config[('tx', 'rev_spent_output', 'tx')] = GATConv(-1, head_channels, heads=heads, concat=True, add_self_loops=False, dropout=dropout_prob)
            self.convs.append(HeteroConv(conv_config, aggr='sum'))

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x_dict = encode(self.node_enc, data.x_dict)

        for conv, norm in zip(self.convs, self.norms):
            x_prev = x_dict
            x_dict = conv(x_dict, data.edge_index_dict, data.edge_attr_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = {key: norm[key](x_dict[key] + x_prev[key]) for key in ('addr', 'tx')}

        return self.lin(x_dict['addr'])


class HeteroGraphTransformer(torch.nn.Module):
    """HGT: the activation and the residual connection are those of HGTConv itself
    (GELU, linear output and a learnable gated skip connection), followed by LayerNorm."""

    def __init__(self, hidden_channels, out_channels, dropout_prob, num_layers, heads):
        super().__init__()

        self.node_enc = node_encoders(hidden_channels, dropout_prob)
        self.norms = layer_norms(hidden_channels, num_layers)

        self.convs = torch.nn.ModuleList([
            HGTConv(hidden_channels, hidden_channels, metadata=METADATA, heads=heads)
            for _ in range(num_layers)
        ])

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x_dict = encode(self.node_enc, data.x_dict)

        for conv, norm in zip(self.convs, self.norms):
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {key: norm[key](x_dict[key]) for key in ('addr', 'tx')}

        return self.lin(x_dict['addr'])


class HeteroAttentionNet(torch.nn.Module):
    """HAN: HANConv already applies ReLU to the output of every relation, so every layer
    only adds the residual connection and LayerNorm."""

    def __init__(self, hidden_channels, out_channels, dropout_prob, num_layers, heads):
        super().__init__()

        self.node_enc = node_encoders(hidden_channels, dropout_prob)
        self.norms = layer_norms(hidden_channels, num_layers)

        self.convs = torch.nn.ModuleList([
            HANConv(hidden_channels, hidden_channels, heads=heads, dropout=dropout_prob, metadata=METADATA)
            for _ in range(num_layers)
        ])

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x_dict = encode(self.node_enc, data.x_dict)

        for conv, norm in zip(self.convs, self.norms):
            x_prev = x_dict
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {key: norm[key](x_dict[key] + x_prev[key]) for key in ('addr', 'tx')}

        return self.lin(x_dict['addr'])


ARCHITECTURES = {
    "GAT": HeteroGNN,
    "HGT": HeteroGraphTransformer,
    "HAN": HeteroAttentionNet,
}

## Training

def build_model(arch, layers):
    """The model of an architecture with its hidden size (hidden_channels_<arch>) and heads."""
    return ARCHITECTURES[arch](HP[f"hidden_channels_{arch}"], 2, HP["dropout"], layers, HP["heads"])


def run_name(arch, layers, seed):
    return f"{arch}-{layers}l-seed{seed}"


@torch.no_grad()
def update_average(averaged, model, step):
    """Moves the averaged weights towards the trained ones after the optimizer step `step`.

    With one full-graph Adam step per epoch and a constant learning rate, the trained
    weights keep moving by about the learning rate near the minimum, so the validation
    loss of the last epochs jumps from epoch to epoch and the best epoch can be a chance
    low. Their exponential moving average (ema_decay: each epoch keeps that share of the
    average) follows the same training without its noise. The decay starts low, so the
    average follows the trained weights in the first epochs instead of lagging behind
    the random initialization; with ema_decay 0 the average is the trained model."""
    decay = min(HP["ema_decay"], (1 + step) / (10 + step))
    for avg, param in zip(averaged.parameters(), model.parameters()):
        avg.lerp_(param, 1 - decay)
    for avg, buffer in zip(averaged.buffers(), model.buffers()):
        avg.copy_(buffer)


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
        graphs[split] = add_reverse_edges(torch.load(path))
    return graphs


def graph_bytes(graphs):
    """Memory of the tensors of the graphs (every storage counted once)."""
    seen, total = set(), 0
    for data in graphs.values():
        for store in data.stores:
            for value in store.values():
                if torch.is_tensor(value):
                    storage = value.storage()
                    if storage.data_ptr() not in seen:
                        seen.add(storage.data_ptr())
                        total += storage.nbytes()
    return total


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

    model = build_model(arch, layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=HP["learning_rate"])
    # Label smoothing only in the training loss: the validation loss that selects the
    # best epoch is the plain cross-entropy
    train_criterion = torch.nn.CrossEntropyLoss(label_smoothing=HP["label_smoothing"])
    criterion = torch.nn.CrossEntropyLoss()

    # Lazy layers are initialized with a first forward pass
    model.train()
    with torch.no_grad():
        model(train_data)
    # Averaged weights (update_average), the model that is evaluated
    averaged = copy.deepcopy(model)

    y_train = train_data['addr'].y[masks["train"]]
    y_val = val_data['addr'].y[masks["val"]]

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    best = {"val_loss": float("inf"), "val_accuracy": None, "epoch": 0, "state": None}
    epochs_no_improve = 0
    start = time.perf_counter()

    if verbose:
        print(f"{run_name(arch, layers, seed)}: training with {threads} threads (started {timestamp()})", flush=True)

    for epoch in range(1, HP["max_epochs"] + 1):
        epoch_start = time.perf_counter()
        model.train()
        optimizer.zero_grad()
        loss = train_criterion(model(train_data)[masks["train"]], y_train)
        loss.backward()
        optimizer.step()
        update_average(averaged, model, epoch)

        averaged.eval()
        with torch.no_grad():
            val_logits = averaged(val_data)[masks["val"]]
            val_loss = criterion(val_logits, y_val).item()
            val_accuracy = (val_logits.argmax(dim=1) == y_val).float().mean().item()

        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "val_accuracy": val_accuracy, "epoch": epoch,
                    "state": copy.deepcopy(averaged.state_dict())}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose:
            # Flushed at every epoch, so the progress can be followed in train.log
            print(f"Epoch {epoch:4d} | train loss {loss.item():.4f} | val loss {val_loss:.4f} | val acc {val_accuracy:.4f} | "
                  f"best val loss {best['val_loss']:.4f} (epoch {best['epoch']}) | {time.perf_counter() - epoch_start:.0f} s",
                  flush=True)

        if epoch >= HP["min_epochs"] and epochs_no_improve >= HP["patience"]:
            break

    train_time = time.perf_counter() - start
    epochs = len(history["train_loss"])

    # Test evaluation of the averaged weights of the best epoch
    averaged.load_state_dict(best["state"])
    averaged.eval()
    with torch.no_grad():
        prob_illicit = F.softmax(averaged(test_data)[masks["test"]], dim=1)[:, 1].numpy()
    y_test = test_data['addr'].y[masks["test"]].numpy()
    test_metrics = classification_metrics(y_test, prob_illicit)

    return {
        "arch": arch,
        "layers": layers,
        "seed": seed,
        "epochs": epochs,
        "best_epoch": best["epoch"],
        "best_val_loss": best["val_loss"],
        "best_val_accuracy": best["val_accuracy"],
        "train_time_s": train_time,
        "time_per_epoch_s": train_time / epochs,
        **{"test_" + k: float(v) for k, v in test_metrics.items()},
        "n_test": int(len(y_test)),
        "threads": threads,
        "hyperparameters": dict(HP),
        "training_fingerprint": TRAINING_FINGERPRINT,
        "evaluation": {
            "selection": "epoch with the lowest validation loss of the averaged weights",
            "weights": "exponential moving average of the trained weights (ema_decay)",
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
    model = build_model(arch, layers)
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
    # The graphs are shared by the runs launched by run_experiments.py, so a run only needs
    # its own memory: the peak minus the graphs
    return {"arch": arch, "layers": layers, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "graph_bytes": graph_bytes(graphs), "epoch_time_s": epoch_time, "threads": threads}


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


def train_and_save(arch, layers, seed, threads, graphs=None, verbose=False):
    """Trains one run and saves its files (graphs: already loaded, e.g. shared by the runs
    forked by run_experiments.py)."""
    result, state, prob_illicit = train_run(arch, layers, seed, threads=threads, graphs=graphs, verbose=verbose)
    run_dir = save_run(result, state, prob_illicit)
    print(f"{run_name(arch, layers, seed)}: {result['epochs']} epochs (best {result['best_epoch']}), "
          f"test F1 {result['test_f1']:.4f}, AUC-ROC {result['test_auc_roc']:.4f} -> {run_dir}", flush=True)
    return result


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
    if result_path.exists() and not args.force and is_current(read_json(result_path)):
        print(f"Already completed: {result_path}")
        return

    train_and_save(args.arch, args.layers, args.seed, args.threads, verbose=args.verbose)


if __name__ == "__main__":
    main()
