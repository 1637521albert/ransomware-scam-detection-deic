# Central configuration of the TRACE experiments.
#
# Every script of the pipeline reads its parameters from this file. The values
# below are those of the paper (the original configuration). A custom
# configuration, chosen with run_pipeline.py, is saved in a JSON file and given to
# every script through RSD_CONFIG_FILE: its parameters replace the ones below, and
# its data and results are kept in their own directory (see the end of this file).
#
# External resources are given through environment variables:
#   RSD_BLOCKSCI_CONFIG   BlockSci config.blocksci file (expansion scripts)
#   RSD_BITCOINHEIST_CSV  BitcoinHeist CSV (create_splits.py)
#   RSD_OUTPUT_DIR        optional, directory for all generated data and results
#                         (default: <repository>/outputs)
#   RSD_CONFIG_FILE       optional, JSON file of a custom configuration
#                         (default: the original configuration)

import os
from pathlib import Path

from shared import configuration

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
# htop). With "auto", they run BLOCKSCI_IO_FACTOR workers per core, limited to
# the number of workers that fit in the available memory
# (BLOCKSCI_WORKER_MEMORY_GB of private memory each, the memory pages of the
# chain are shared). More workers than cores only help if the disk is not
# saturated: on the reference machine 3 workers per core were slower than 1, as
# they compete for the disk and for the memory that caches the chain. The
# results do not depend on the number of workers.
BLOCKSCI_IO_FACTOR = 1
BLOCKSCI_WORKER_MEMORY_GB = 2.5

# --------------------------------------------------------------------------
# Seed addresses and splits (shared by both methodologies)
# --------------------------------------------------------------------------

SPLITS = ["train", "val", "test"]

# The seed addresses are all the illicit BitcoinHeist addresses found in the chain,
# together with the same number of randomly sampled licit addresses, selected by
# create_splits.py with DATA_SEED. Each class is split 60/20/20 into training,
# validation and test (validation and test get the rounded fraction of the class,
# training the rest).
SPLIT_FRACTIONS = {"train": 0.6, "val": 0.2, "test": 0.2}

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
    # Training protocol of the paper: the best epoch is the one with the highest
    # validation accuracy, and training stops after PATIENCE epochs without
    # improving it, after at least MIN_EPOCHS epochs (0: no minimum). A longer
    # protocol (e.g. min_epochs 50, patience 20) is left as future work (README).
    "min_epochs": 0,
    "patience": 10,
}

# Parallel training processes of run_experiments.py, and the cores are split
# among them (CPU_COUNT // runs threads per run). "auto" runs as many as fit in
# the available memory: the peak memory of one epoch of every configuration is
# measured first, and every run is assumed to need the largest peak times
# GNN_MEMORY_MARGIN.
GNN_PARALLEL_RUNS = "auto"
GNN_MEMORY_MARGIN = 1.25

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

# Topological features used by the tabular models (extract_topological_features.py)
EGO_CENTRIC_TRAINING_FEATURES = [
    'in_deg', 'out_deg',       # Degrees
    '2star_in', 'merge_2+',    # Aggregation
    '2star_out', 'split_2+',   # Branching
    'peeling_2', 'peeling_3',  # Peeling Chains
    'pagerank', 'betweenness', 'eccentricity', 'density' # Centrality & Graph Metrics
]

# Hyperparameters of the tabular models (models.py)
EGO_CENTRIC_HYPERPARAMETERS = {
    "selector_percentile": 90,
    "scaler": "StandardScaler",

    "logreg_C": 0.3,
    "logreg_penalty_l1": "l1",
    "logreg_penalty_el": "elasticnet",
    "logreg_l1_ratio": 0.5,
    "logreg_max_iter": 5000,

    "rf_n_estimators": 500,
    "rf_max_depth": 20,

    "xgb_n_estimators": 1000,
    "xgb_max_depth": 8,
    "xgb_learning_rate": 0.1,
    "xgb_subsample": 0.9,
    "xgb_colsample_bytree": 0.9,

    "iso_n_estimators": 500,
    "iso_contamination": "auto",
    "svm_C": 1.0,
    "svm_kernel": "rbf",

    "mlp_hidden_layers": (100, 50),
    "mlp_alpha": 0.0001,
    "mlp_max_iter": 1000,
    "mlp_solver": "adam",
    "mlp_learning_rate_init": 0.001
}

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

# --------------------------------------------------------------------------
# Configuration of the run: original or custom
# --------------------------------------------------------------------------

# The configurable parameters (shared/configuration.py) of the original
# configuration, replaced by those of RSD_CONFIG_FILE if it is given. A file with
# the original values runs the original configuration.
ORIGINAL_PARAMETERS = configuration.parameters(globals())
CONFIG_FILE = os.getenv("RSD_CONFIG_FILE") or None
if CONFIG_FILE:
    configuration.apply(globals(), CONFIG_FILE)
PARAMETERS = configuration.parameters(globals())
ORIGINAL_CONFIGURATION = PARAMETERS == ORIGINAL_PARAMETERS
CONFIGURATION_ID = "original" if ORIGINAL_CONFIGURATION else configuration.identifier(PARAMETERS)

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

OUTPUT_DIR = Path(os.getenv("RSD_OUTPUT_DIR", str(PROJECT_ROOT / "outputs"))).expanduser().resolve()

# Everything generated goes to OUTPUT_DIR/ with the original configuration, and to
# OUTPUT_DIR/<configuration id>/ with a custom one
RUN_DIR = OUTPUT_DIR if ORIGINAL_CONFIGURATION else OUTPUT_DIR / CONFIGURATION_ID

# Parameters of a custom configuration, saved by run_pipeline.py
CONFIGURATION_FILE = RUN_DIR / "configuration.json"
PIPELINE_LOGS_DIR = RUN_DIR / "pipeline_runs"

# address,label,family,split
SPLITS_FILE = RUN_DIR / "splits.csv"
SPLITS_SUMMARY_FILE = RUN_DIR / "splits_summary.json"

MULTI_INSTANCE_DIR = RUN_DIR / "multi_instance"
MULTI_INSTANCE_GRAPHS_DIR = MULTI_INSTANCE_DIR / "graphs"      # <split>/*.csv, graph.pth, graph_normalized.pth
MULTI_INSTANCE_RESULTS_DIR = MULTI_INSTANCE_DIR / "results"    # runs/, gnn_runs.csv, summaries, tests

EGO_CENTRIC_DIR = RUN_DIR / "ego_centric"
EGO_CENTRIC_EGONETS_DIR = EGO_CENTRIC_DIR / "egonets"          # <split>.jsonl
EGO_CENTRIC_FEATURES_DIR = EGO_CENTRIC_DIR / "features"      # <split>.csv, <split>.json
EGO_CENTRIC_RESULTS_DIR = EGO_CENTRIC_DIR / "results"          # predictions/, ego_runs.csv, summaries, tests, figures

STATISTICAL_TESTS_DIR = RUN_DIR / "statistical_tests"
