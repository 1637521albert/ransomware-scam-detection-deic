# Central configuration of the TRACE experiments.
#
# Every script of the pipeline reads its parameters from this file, so the
# experiments reported in the paper are reproduced by running the scripts in
# the order described in the README, without interactive prompts.
#
# External resources are given through environment variables:
#   RSD_BLOCKSCI_CONFIG   BlockSci config.blocksci file (expansion scripts)
#   RSD_BITCOINHEIST_CSV  BitcoinHeist CSV (create_splits.py)
#   RSD_OUTPUT_DIR        optional, directory for all generated data and results
#                         (default: <repository>/outputs)
#   RSD_BLOCKSCI_PYTHON   optional, Python interpreter of the BlockSci environment
#   RSD_TORCH_PYTHON      optional, Python interpreter of the PyTorch environment
#                         (both used by run_pipeline.py, default: the current one)

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# Randomness
# --------------------------------------------------------------------------

# Seed of every data-generation step: seed selection, splits and expansions.
# It is never changed between training runs, so all runs share the same data.
DATA_SEED = 42

# Training seeds: every model configuration of both methodologies is trained
# once per seed, and the same seeds are used everywhere so that the runs can be
# compared in pairs. The training scripts ask how many seeds to run (default
# DEFAULT_N_SEEDS) and take the first ones of this list; more seeds are drawn
# deterministically from DATA_SEED (see shared/experiment.py).
TRAINING_SEEDS = [10, 42, 185, 205, 436, 475, 639, 649, 892, 963]
DEFAULT_N_SEEDS = 10

# --------------------------------------------------------------------------
# CPU resources
# --------------------------------------------------------------------------

# Every step uses all the cores. Results are deterministic for a given
# configuration; the number of threads of each GNN run is recorded with its
# results, as floating-point sums may differ slightly with another thread count.
CPU_COUNT = os.cpu_count() or 1

# The BlockSci steps (create_splits.py and the expansions) read the blockchain
# from disk, so their workers spend part of the time waiting for it (state D in
# htop). With "auto", they run BLOCKSCI_IO_FACTOR workers per core, so the cores
# stay busy while some workers wait, limited to the number of workers that fit
# in the available memory (BLOCKSCI_WORKER_MEMORY_GB of private memory each,
# the memory pages of the chain are shared). The results do not depend on the
# number of workers.
BLOCKSCI_IO_FACTOR = 3
BLOCKSCI_WORKER_MEMORY_GB = 2.5

# --------------------------------------------------------------------------
# Seed addresses and splits (shared by both methodologies)
# --------------------------------------------------------------------------

# All illicit BitcoinHeist addresses found in the chain are used, together with
# the same number of licit addresses. Per class, VAL_PER_CLASS and
# TEST_PER_CLASS addresses go to validation and test, and the rest to training.
VAL_PER_CLASS = 4000
TEST_PER_CLASS = 4000
SPLITS = ["train", "val", "test"]

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

OUTPUT_DIR = Path(os.getenv("RSD_OUTPUT_DIR", str(PROJECT_ROOT / "outputs"))).expanduser().resolve()

# Interpreters of the two environments, used by run_pipeline.py
BLOCKSCI_PYTHON = os.getenv("RSD_BLOCKSCI_PYTHON")
TORCH_PYTHON = os.getenv("RSD_TORCH_PYTHON")
PIPELINE_LOGS_DIR = OUTPUT_DIR / "pipeline_runs"

# address,label,family,split
SPLITS_FILE = OUTPUT_DIR / "splits.csv"
SPLITS_SUMMARY_FILE = OUTPUT_DIR / "splits_summary.json"

MULTI_INSTANCE_DIR = OUTPUT_DIR / "multi_instance"
MULTI_INSTANCE_GRAPHS_DIR = MULTI_INSTANCE_DIR / "graphs"      # <split>/*.csv, graph.pth, graph_normalized.pth
MULTI_INSTANCE_RESULTS_DIR = MULTI_INSTANCE_DIR / "results"    # runs/, gnn_runs.csv, summaries, tests

EGO_CENTRIC_DIR = OUTPUT_DIR / "ego_centric"
EGO_CENTRIC_EGONETS_DIR = EGO_CENTRIC_DIR / "egonets"          # <split>.jsonl
EGO_CENTRIC_FEATURES_FILE = EGO_CENTRIC_DIR / "features.csv"
EGO_CENTRIC_RESULTS_DIR = EGO_CENTRIC_DIR / "results"          # predictions/, ego_runs.csv, summaries, tests, figures

# --------------------------------------------------------------------------
# Methodology I: inductive multi-instance address classification
# --------------------------------------------------------------------------

# Expansion: forward-backward, transaction-centered, whole scope, k=2, w=5
MULTI_INSTANCE_EXPANSION = {
    "direction": "fw bw",
    "graph": " tx",
    "address": " whole",
    "side_direction": "",
    "limit_mode": "random node",
    "limit": 5,
    "forward_hops": 2,
    "backward_hops": 2,
}

GNN_ARCHITECTURES = ["GAT", "HGT", "HAN"]
GNN_LAYERS = [2, 4]

GNN_TRAINING = {
    "hidden_channels": 32,
    "learning_rate": 0.001,
    "dropout": 0.2,
    "max_epochs": 1000,
    # The best epoch is the one with the highest validation accuracy. Training
    # runs at least MIN_EPOCHS epochs and then stops after PATIENCE epochs
    # without improving it.
    "min_epochs": 50,
    "patience": 20,
}

# Parallel training processes of run_experiments.py, and the cores are split
# among them (CPU_COUNT // runs threads per run). "auto" runs as many as fit in
# the available memory: every run loads the three graphs and keeps the
# activations of a full-graph pass, estimated as GNN_MEMORY_FACTOR times the
# size of the normalized graph files.
GNN_PARALLEL_RUNS = "auto"
GNN_MEMORY_FACTOR = 4

# Parallel processes of the multi-instance expansion (frontier scanning and
# feature extraction of each split)
MULTI_INSTANCE_EXPANSION_WORKERS = "auto"

# --------------------------------------------------------------------------
# Methodology II: inductive ego-centric address classification
# --------------------------------------------------------------------------

# Expansion of an independent egonet per seed: all-over, transaction-centered,
# k=3, w=3
EGO_CENTRIC_EXPANSION = {
    "direction": "all over",
    "graph": " tx",
    "address": "",
    "side_direction": "",
    "limit_mode": "random node",
    "limit": 3,
    "hops": 3,
}

EGO_CENTRIC_FEATURES = {
    "radius": 6,              # BFS radius (address-transaction steps) around the seed
    "betweenness_k": None,    # None = exact betweenness, an integer = sampled approximation
    "save_egonets": False,    # also store every induced egonet as a .gpickle file
    "workers": CPU_COUNT,
}

# Parallel processes expanding the egonets (one seed address at a time each)
EGO_CENTRIC_EXPANSION_WORKERS = "auto"

# Parallel processes training the tabular models (one model and seed each,
# single-threaded, so the results do not depend on the number of processes)
EGO_CENTRIC_MODEL_WORKERS = CPU_COUNT

# Model selected for the feature ablation and SHAP analyses, and the training
# seed of the run that is interpreted
EGO_CENTRIC_BEST_MODEL = "MLP"
EGO_CENTRIC_INTERPRETATION_SEED = TRAINING_SEEDS[0]

# --------------------------------------------------------------------------
# Statistical comparisons
# --------------------------------------------------------------------------

BOOTSTRAP_RESAMPLES = 10000

# Configurations compared with the exact Wilcoxon signed-rank test need at least
# this number of common seeds (with 5 seeds the smallest two-sided p-value is 0.0625)
MIN_SEEDS_FOR_TESTS = 6

STATISTICAL_TESTS_DIR = OUTPUT_DIR / "statistical_tests"
