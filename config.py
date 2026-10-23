# Central configuration of the TRACE experiments.
#
# Every script of the pipeline reads its parameters from this file. The values
# below are the original configuration. A custom
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
    # Hidden size of every architecture and attention heads per layer. GAT concatenates
    # its heads (hidden_channels each) and projects them back to its hidden size; HGT
    # and HAN split their hidden size across the heads.
    "hidden_channels_GAT": 32,
    "hidden_channels_HGT": 64,
    "hidden_channels_HAN": 64,
    "heads": 4,
    "learning_rate": 0.001,
    "dropout": 0.2,
    "max_epochs": 1000,
    # Every epoch is one full-graph Adam step. The best epoch is the one with the
    # lowest validation loss, and training stops after PATIENCE epochs without
    # improving it, after at least MIN_EPOCHS epochs. The validation loss is used
    # rather than the validation accuracy (at the 0.5 threshold) because the accuracy
    # can stall for 10-20 epochs early in training while the loss keeps falling,
    # which would stop some runs after a few dozen updates. In a check
    # without early stopping, the epoch of lowest validation loss was between 67 and
    # 168, so MIN_EPOCHS is a safety margin below it.
    "min_epochs": 100,
    "patience": 20,
}

# Training runs of run_experiments.py, forked from a process that loads the graphs
# once (all the runs share them in memory). "auto": the peak memory of one epoch of
# every configuration is measured first; a run of a configuration reserves its peak
# minus the graphs, times GNN_MEMORY_MARGIN; the runs start the configurations with
# the longest epoch first, whenever their memory and threads are available; and a
# configuration has CPU_COUNT // (its runs that fit in memory) threads per run.
# A number N: N runs at a time with CPU_COUNT // N threads each.
GNN_PARALLEL_RUNS = "auto"
GNN_MEMORY_MARGIN = 1.25

# Parallel processes of the multi-instance expansion (frontier scanning and
# feature extraction of each split)
MULTI_INSTANCE_EXPANSION_WORKERS = "auto"

# --------------------------------------------------------------------------
# Methodology II: inductive ego-centric address classification
# --------------------------------------------------------------------------

# Expansion of an independent egonet per seed: all-over, transaction-centered,
# k=3, w=3 (every variant of MULTI_INSTANCE_EXPANSION is available, with the same
# algorithms: shared/expansion.py)
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

# Tabular models (estimators.py, tune_hyperparameters.py, train_models.py). Every
# model is a pipeline: clipping of negative values and log1p of the training features,
# mutual-information feature selection (selector__percentile), scaling (scaler:
# "RobustScaler" or "StandardScaler") and the model. Its hyperparameters are not fixed
# here: they are searched on the training split over the grid of the model, with
# stratified 5-fold cross-validation scored with the F1-score at the 0.5 threshold of
# every evaluation of both methodologies, and searched again whenever the features, the
# grid or the training code change. "fixed" are the settings of the model that are not
# searched; a pipeline step missing from a grid keeps its default (percentile 100,
# RobustScaler). "search": "grid" tries every combination; "halving" (successive
# halving, for the costliest models) tries every combination on a subsample and keeps
# the best third on three times more addresses, until the whole training split.
EGO_CENTRIC_MODELS = {
    "L1 (Lasso)": {
        "search": "grid",
        "fixed": {"penalty": "l1", "solver": "saga", "class_weight": "balanced", "max_iter": 10000, "tol": 1e-3},
        "grid": {"selector__percentile": [70, 85, 100], "scaler": ["RobustScaler", "StandardScaler"],
                 "model__C": [0.01, 0.05, 0.2, 1.0, 5.0]},
    },
    "ElasticNet": {
        "search": "grid",
        "fixed": {"penalty": "elasticnet", "solver": "saga", "class_weight": "balanced", "max_iter": 10000, "tol": 1e-3},
        "grid": {"selector__percentile": [75, 90, 100], "scaler": ["RobustScaler", "StandardScaler"],
                 "model__C": [0.01, 0.1, 0.5, 1.0, 2.0], "model__l1_ratio": [0.2, 0.5, 0.8]},
    },
    "RandomForest": {
        "search": "halving",
        "fixed": {"class_weight": "balanced"},
        "grid": {"selector__percentile": [90, 100], "model__n_estimators": [300, 500],
                 "model__max_depth": [8, 12, 20, None], "model__min_samples_split": [2, 5],
                 "model__min_samples_leaf": [1, 2, 4], "model__max_features": ["sqrt", 0.5, 0.85]},
    },
    "XGBoost": {
        "search": "halving",
        "fixed": {"tree_method": "hist", "eval_metric": "logloss"},
        "grid": {"selector__percentile": [90, 100], "model__n_estimators": [200, 400, 600],
                 "model__max_depth": [6, 8, 10], "model__learning_rate": [0.02, 0.05, 0.1],
                 "model__subsample": [0.7, 0.9], "model__colsample_bytree": [0.7, 0.9],
                 "model__min_child_weight": [1, 4]},
    },
    "SVM": {
        "search": "grid",
        "fixed": {"kernel": "rbf", "class_weight": "balanced", "probability": True, "cache_size": 1000},
        "grid": {"scaler": ["RobustScaler", "StandardScaler"], "model__C": [0.5, 1.0, 2.0, 8.0]},
    },
    "Naive Bayes": {
        "search": "grid",
        "fixed": {},
        "grid": {"selector__percentile": [50, 65, 80, 90, 100], "scaler": ["RobustScaler", "StandardScaler"],
                 "model__var_smoothing": [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]},
    },
    "MLP": {
        "search": "grid",
        "fixed": {"solver": "adam", "max_iter": 450, "early_stopping": True, "n_iter_no_change": 20,
                  "validation_fraction": 0.1},
        "grid": {"selector__percentile": [90, 100], "scaler": ["RobustScaler", "StandardScaler"],
                 "model__hidden_layer_sizes": [(128, 64, 32), (256, 128), (256, 128, 64)],
                 "model__alpha": [5e-4, 1e-3, 5e-3], "model__learning_rate_init": [1e-3, 2e-3, 3e-3]},
    },
    # Unsupervised: the anomalies are predicted illicit (estimators.py)
    "IsolationForest": {
        "search": "grid",
        "fixed": {},
        "grid": {"selector__percentile": [75, 100], "model__n_estimators": [200, 400],
                 "model__max_samples": ["auto", 0.5], "model__contamination": ["auto", 0.3, 0.4, 0.5]},
    },
}


# Parallel processes training the tabular models (one model and seed each,
# single-threaded, so the results do not depend on the number of processes)
EGO_CENTRIC_MODEL_WORKERS = CPU_COUNT

# The feature ablation and SHAP analyses explain the best model (highest mean test
# F1-score over the seeds, see wilcoxon_tests.best_model), trained with this seed
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
