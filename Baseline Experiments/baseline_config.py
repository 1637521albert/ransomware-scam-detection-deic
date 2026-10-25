# Configuration of the baseline experiments, used by baseline_experiments.py:
# representative BitcoinHeist-based methods of the literature, reimplemented and
# evaluated under the protocol of this work (same seed addresses, splits, training
# seeds, threshold and metrics as both methodologies).

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

# Stratified folds of the hyperparameter searches and of the cross-validation of every
# run, and decision threshold of the illicit class: as in Methodology II
CV_FOLDS = 5
THRESHOLD = 0.5

# Features of the seed addresses: the descriptive features of every seed address in
# Methodology I (expansion.py of the multi-instance methodology, before normalization),
# i.e. the node features of the GNNs without any graph. They are computed for the same
# seed addresses and splits as the executions of both methodologies, so the methods are
# trained and tested on exactly their data, and they describe every address as a whole,
# the classification unit of this work. The topological features of Methodology II are
# evaluated with conventional models by that methodology itself, and the attributes of the
# BitcoinHeist dataset describe an address on one day at a time, a temporally distributed
# unit that differs from the unique addresses classified here. The table must describe
# every seed address of every split (the experiment stops otherwise).
FEATURE_SETS = {
    "address": "address features of Methodology I, without the graph",
}

# Methods, as described by the original works (their model, base learners, meta-model and
# search space where the paper gives them), searched and evaluated with the protocol of
# this work: every single model is searched on the training split with stratified
# CV_FOLDS-fold cross-validation scored by the F1-score at THRESHOLD, as in Methodology II
# ("grid": every combination; "halving": successive halving; "random": RANDOM_SEARCH_ITERATIONS
# random candidates). A search space is written as a list of values, {"randint": [low, high]}
# (integers from low to high - 1, as scipy.stats.randint) or {"uniform": [low, high]}.
#
# A stacking ensemble combines the searched single models named in "base" (each with its
# own searched hyperparameters) through its "meta" model, trained on their out-of-fold
# probabilities and, with "passthrough", also on the features. The meta model is a
# searched single model (its name) or {"estimator": ..., "fixed": ...} with fixed settings.
#
# Every method is trained on the same data, with the same preprocessing (baseline_models.py)
# and without class balancing (every split is balanced), as the reviewers' requested
# comparison requires. The original works also filter, balance (ADASYN or SMOTE with Tomek
# links, or random undersampling and oversampling), remove outliers by z-score, scale the
# features in their own way (min-max, standardization or a power transform) and, in Dib et
# al. (2024), encode the address string as a feature; none of these is applied here (the
# address is an identifier, which cannot carry over to the unseen addresses of the
# inductive setting).

# Candidates of every random search (the original works do not give their number)
RANDOM_SEARCH_ITERATIONS = 50


def _without_selection(grid):
    # Methodology II also searches its feature-selection percentile and scaler; the
    # baselines keep every feature and scale them in the same way (baseline_models.py)
    return {k: v for k, v in grid.items() if not k.startswith("selector__") and k != "scaler"}


# Table 3 of Nayyer et al. (2023): random search over these ranges ("[1, 10]" for the
# maximum depth is read as the integers 1 to 10, which contains its selected value, 10)
NAYYER_TREE = {"model__max_depth": {"randint": [1, 11]}, "model__min_samples_split": {"randint": [2, 20]},
               "model__min_samples_leaf": {"randint": [1, 10]}, "model__max_features": ["sqrt", "log2", None]}

METHODS = {
    # Abu Al-Haija and Alsulami (2021): their "optimizable decision trees" are MATLAB's
    # optimizable ensemble of bootstrap-aggregated (bagged) decision trees with the Gini
    # index, tuned by Bayesian optimization; they select 30 learners with an unlimited
    # number of splits. Bagged trees are a random forest that samples every feature at
    # every split (max_features None); the number of learners, the maximum number of splits
    # (as leaf nodes) and the number of features sampled, which MATLAB also optimizes, are
    # searched over values that include their selection
    "Bagged decision trees": {
        "work": "Abu Al-Haija and Alsulami (2021)", "cite": "abu_al-haija_high_2021",
        "estimator": "RandomForest", "search": "halving", "fixed": {"criterion": "gini", "bootstrap": True},
        "grid": {"model__n_estimators": [10, 30, 100, 300], "model__max_leaf_nodes": [16, 256, 4096, None],
                 "model__max_features": [None, 0.5, "sqrt"]},
    },
    # Alsaif (2023) tunes logistic regression, random forest and XGBoost with scikit-learn's
    # random search, and Dib et al. (2024) their random forest and XGBoost with Optuna,
    # without publishing the search spaces: logistic regression over its regularization
    # strength, and random forest and XGBoost over the grids and fixed settings of
    # Methodology II
    "Logistic regression": {
        "work": "Alsaif (2023)", "cite": "alsaif_machine_2023",
        "estimator": "LogisticRegression", "search": "grid", "fixed": {"max_iter": 2000},
        "grid": {"model__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
    },
    "Random forest": {
        "work": "Alsaif (2023); Dib et al. (2024)", "cite": "alsaif_machine_2023,dib_machine_2024",
        "estimator": "RandomForest", "search": config.EGO_CENTRIC_MODELS["RandomForest"]["search"],
        "fixed": dict(config.EGO_CENTRIC_MODELS["RandomForest"]["fixed"]),
        "grid": _without_selection(config.EGO_CENTRIC_MODELS["RandomForest"]["grid"]),
    },
    "XGBoost": {
        "work": "Alsaif (2023); Dib et al. (2024)", "cite": "alsaif_machine_2023,dib_machine_2024",
        "estimator": "XGBoost", "search": config.EGO_CENTRIC_MODELS["XGBoost"]["search"],
        "fixed": dict(config.EGO_CENTRIC_MODELS["XGBoost"]["fixed"]),
        "grid": _without_selection(config.EGO_CENTRIC_MODELS["XGBoost"]["grid"]),
    },
    # Dib et al. (2024) tune the depth, minimum samples to split, minimum samples per leaf,
    # maximum leaf nodes and minimum weight fraction per leaf of their tree with Optuna; the
    # ranges are not published, so these are chosen to cover the usual values
    "Decision tree (Dib et al.)": {
        "work": "Dib et al. (2024)", "cite": "dib_machine_2024",
        "estimator": "DecisionTree", "search": "random", "fixed": {},
        "grid": {"model__max_depth": {"randint": [2, 31]}, "model__min_samples_split": {"randint": [2, 21]},
                 "model__min_samples_leaf": {"randint": [1, 11]}, "model__max_leaf_nodes": {"randint": [8, 1025]},
                 "model__min_weight_fraction_leaf": {"uniform": [0.0, 0.05]}},
    },
    # Dib et al. (2024): the predictions of DT, RF and XGBoost are the input of an MLP
    # meta-model (its settings are not published: scikit-learn's defaults, with enough
    # iterations to converge)
    "Stacking: DT, RF, XGBoost + MLP": {
        "work": "Dib et al. (2024)", "cite": "dib_machine_2024",
        "estimator": "Stacking", "base": ["Decision tree (Dib et al.)", "Random forest", "XGBoost"],
        "meta": {"estimator": "MLP", "fixed": {"max_iter": 1000}}, "passthrough": False,
    },
    # Nayyer et al. (2023), Table 3: every model and the RF meta-model of their stacking
    # (their "RDNK" model), searched over their ranges
    "Decision tree (Nayyer et al.)": {
        "work": "Nayyer et al. (2023)", "cite": "nayyer_new_2023",
        "estimator": "DecisionTree", "search": "random", "fixed": {}, "grid": dict(NAYYER_TREE),
    },
    "Random forest (Nayyer et al.)": {
        "work": "Nayyer et al. (2023)", "cite": "nayyer_new_2023",
        "estimator": "RandomForest", "search": "random", "fixed": {},
        "grid": dict(NAYYER_TREE, model__n_estimators=[25, 50, 100, 150]),
    },
    "K-nearest neighbors": {
        "work": "Nayyer et al. (2023)", "cite": "nayyer_new_2023",
        "estimator": "KNN", "search": "random", "fixed": {},
        "grid": {"model__n_neighbors": {"randint": [1, 50]}, "model__weights": ["uniform", "distance"],
                 "model__p": [1, 2]},
    },
    "Naive Bayes": {
        "work": "Nayyer et al. (2023)", "cite": "nayyer_new_2023",
        "estimator": "NaiveBayes", "search": "grid", "fixed": {},
        "grid": {"model__var_smoothing": [1e-12, 1e-9, 1e-6]},
    },
    # The predictions of DT, NB and KNN, added to the features, are the input of an RF
    # meta-model, with the hyperparameters of their searched RF (the values they select
    # for the meta-model are those of their RF)
    "Stacking: DT, NB, KNN + RF": {
        "work": "Nayyer et al. (2023)", "cite": "nayyer_new_2023",
        "estimator": "Stacking", "base": ["Decision tree (Nayyer et al.)", "Naive Bayes", "K-nearest neighbors"],
        "meta": "Random forest (Nayyer et al.)", "passthrough": True,
    },
}

# Not reimplemented, as they need data or software outside this protocol (by their
# bibliography keys): the GATv2 of kushnerov_development_2026, pre-trained
# on the Elliptic dataset; the Chainlet Orbits of azad_chainlet_2025, computed on daily
# Bitcoin graphs; the PART rule learner of talabani_bitcoin_2022 (Weka); the deep
# Q-network of shetty_unveiling_2024; and the LightGBM and CatBoost members of the
# stacking of teymourpour_stacking_2025, which are not among the libraries of this work
# (its tree-ensemble stacking is represented by dib_machine_2024).

# Inside the run directory, as the results depend on the splits of the configuration
OUTPUTS_DIR = config.RUN_DIR / "baseline_experiments"
FEATURES_DIR = OUTPUTS_DIR / "features"         # <feature set>.csv, <feature set>.json
TUNING_DIR = OUTPUTS_DIR / "tuning"             # searches.json, search_<set>_<method>_<fingerprint>.csv
RUNS_DIR = OUTPUTS_DIR / "runs"                 # <set>/<method>-seed<seed>_<fingerprint>.json
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"   # <set>/<method>-seed<seed>_<fingerprint>.csv
TABLES_DIR = OUTPUTS_DIR / "tables"
TESTS_DIR = OUTPUTS_DIR / "tests"
