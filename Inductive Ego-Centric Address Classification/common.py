# What the steps of the tabular models share: output paths, the dataset, the
# fingerprints, the expansion configuration of the data and the saved results.
#
# A fingerprint identifies what a result was computed from. The search of a model
# (tune_hyperparameters.py) depends on the features (the content of the feature
# files), the training features, its grid and fixed settings, DATA_SEED and the code
# (estimators.py, load_dataset and tune_model); a run (train_models.py) on the
# features, the model, its hyperparameters and fixed settings and the code
# (estimators.py, load_dataset, cross_validate_model and fit_run). The code counts by
# its syntax tree, so editing comments does not invalidate anything.
#
# Every search, run and interpretation is saved under its fingerprint and is never
# overwritten by another one: results of other features (another expansion of the
# egonets, other feature parameters, another version of the code...) are kept next to
# the current ones, each with the expansion configuration of its egonets, and are listed
# in an index per kind (tuning/index.csv, runs/index.csv, interpretation/index.csv) to
# be compared by their expansion configuration. Going back to earlier features reuses
# their results. The result tables (ego_runs.csv...) and the tests only use the runs of
# the current features and code.

from pathlib import Path

import pandas as pd

import config
from estimators import MODELS, THRESHOLD
from shared.experiment import code_fingerprint, digest, file_fingerprint, read_json

CURRENT_DIR = Path(__file__).resolve().parent

RESULTS_DIR = config.EGO_CENTRIC_RESULTS_DIR
RUNS_DIR = RESULTS_DIR / "runs"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
INTERPRETATION_DIR = RESULTS_DIR / "interpretation"
TUNING_DIR = RESULTS_DIR / "tuning"
TUNING_FILE = TUNING_DIR / "searches.json"

# Stratified cross-validation folds of the searches and of the evaluation of every run
CV_FOLDS = 5
TRAINING_FEATURES = config.EGO_CENTRIC_TRAINING_FEATURES

EVAL_METRICS = ["accuracy", "precision", "recall", "f1", "auc_roc"]
METRICS = EVAL_METRICS + ["fit_time_s"]
CV_METRICS = [f"cv_{m}" for m in EVAL_METRICS] + ["cv_fit_time_s"]

# Code of the searches and of the runs: file -> its functions, or None for the whole file
TUNING_CODE = {CURRENT_DIR / "estimators.py": None, CURRENT_DIR / "common.py": ["load_dataset"],
               CURRENT_DIR / "tune_hyperparameters.py": ["tune_model"]}
RUN_CODE = {CURRENT_DIR / "estimators.py": None, CURRENT_DIR / "common.py": ["load_dataset"],
            CURRENT_DIR / "train_models.py": ["cross_validate_model", "fit_run"]}


def file_name(model_name):
    return model_name.replace(" ", "_").replace("(", "").replace(")", "")


def run_id(model_name, seed):
    return f"{file_name(model_name)}-seed{seed}"


def feature_files():
    return [config.EGO_CENTRIC_FEATURES_DIR / f"{split}.csv" for split in config.SPLITS]


def load_dataset():
    """Training features, labels and addresses of every split (EGO_CENTRIC_FEATURES_DIR)."""
    missing = [str(path) for path in feature_files() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {missing}: run extract_topological_features.py first.")

    df = pd.concat([pd.read_csv(path) for path in feature_files()], ignore_index=True)
    # Deterministic row order, independent of the order of the files
    df = df.sort_values("addr_str").sample(frac=1, random_state=config.DATA_SEED).reset_index(drop=True)

    data = {}
    for split in config.SPLITS:
        part = df[df["split"] == split]
        data[split] = (part[TRAINING_FEATURES].copy(), part["label"].astype(int).values, part["addr_str"].values)
    return data


# --------------------------------------------------------------------------
# Fingerprints and expansion configuration of the data
# --------------------------------------------------------------------------

_CACHE = {}


def data_fingerprint():
    """Content of the feature files and the training features."""
    if "data" not in _CACHE:
        _CACHE["data"] = digest([file_fingerprint(feature_files()), TRAINING_FEATURES])
    return _CACHE["data"]


def data_description():
    """Where the features come from, as recorded by the previous steps in the description
    of every feature file (<split>.json): the expansion configuration of the egonets, the
    feature parameters and their fingerprints."""
    if "description" not in _CACHE:
        infos = {}
        for split in config.SPLITS:
            path = config.EGO_CENTRIC_FEATURES_DIR / f"{split}.json"
            infos[split] = read_json(path) if path.exists() else {}
        first = infos[config.SPLITS[0]]
        _CACHE["description"] = {
            "data_fingerprint": data_fingerprint(),
            "exp_alg": first.get("exp_alg"),
            "expansion": first.get("expansion"),
            "features": {k: (first.get("features") or {}).get(k) for k in ("radius", "betweenness_k")},
            "egonets_fingerprints": {split: info.get("egonets_fingerprint") for split, info in infos.items()},
            "features_fingerprints": {split: info.get("fingerprint") for split, info in infos.items()},
        }
    return _CACHE["description"]


def expansion_columns(description):
    """Flat columns of a data description, for the indexes."""
    description = description or {}
    expansion = description.get("expansion") or {}
    features = description.get("features") or {}
    return {
        "data_fingerprint": description.get("data_fingerprint"),
        "exp_alg": description.get("exp_alg"),
        "limit_mode": expansion.get("limit_mode"),
        "limit": expansion.get("limit"),
        "hops": expansion.get("hops"),
        "forward_hops": expansion.get("forward_hops"),
        "backward_hops": expansion.get("backward_hops"),
        "radius": features.get("radius"),
        "betweenness_k": features.get("betweenness_k"),
    }


def clear_cache():
    """Forgets the fingerprint and description of the features (if they change)."""
    _CACHE.clear()


def tuning_fingerprint(name):
    return digest({"data": data_fingerprint(), "model": name, "spec": MODELS[name], "folds": CV_FOLDS,
                   "data_seed": config.DATA_SEED, "threshold": THRESHOLD, "code": code_fingerprint(TUNING_CODE)})


def run_fingerprint(name, params):
    return digest({"data": data_fingerprint(), "model": name, "fixed": MODELS[name]["fixed"], "params": params,
                   "folds": CV_FOLDS, "threshold": THRESHOLD, "code": code_fingerprint(RUN_CODE)})


# --------------------------------------------------------------------------
# Saved results, by fingerprint
# --------------------------------------------------------------------------

def search_results_path(name, fingerprint):
    return TUNING_DIR / f"search_{file_name(name)}_{fingerprint}.csv"


def run_path(name, seed, fingerprint):
    return RUNS_DIR / f"{run_id(name, seed)}_{fingerprint}.json"


def predictions_path(name, seed, fingerprint):
    return PREDICTIONS_DIR / f"{run_id(name, seed)}_{fingerprint}.csv"


def interpretation_dir(name, seed, fingerprint):
    return INTERPRETATION_DIR / f"{run_id(name, seed)}_{fingerprint}"


def load_searches():
    """Every saved search: fingerprint -> model, params, cross-validated F1-score, data..."""
    return read_json(TUNING_FILE) if TUNING_FILE.exists() else {}


def current_hyperparameters():
    """Searched hyperparameters of every model with a search of the current features,
    grid and code (None otherwise)."""
    searches = load_searches()
    current = {}
    for name in MODELS:
        entry = searches.get(tuning_fingerprint(name))
        current[name] = entry["params"] if entry else None
    return current


def write_index(path, rows, sort_by):
    """CSV listing saved results, with the expansion configuration of their data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values([c for c in sort_by if c in df.columns])
    df.to_csv(path, index=False)
