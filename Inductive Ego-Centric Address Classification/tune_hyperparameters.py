# STEP 3 (Methodology II) - HYPERPARAMETER SEARCH OF THE TABULAR MODELS
#
# Searches the hyperparameters of the eight tabular models (estimators.py) on the
# topological features of the training split (EGO_CENTRIC_FEATURES_DIR/train.csv,
# written by extract_topological_features.py): every model is searched over its grid
# (config.EGO_CENTRIC_MODELS), trying every combination ("grid") or, for the costliest
# models, with successive halving ("halving": every combination on a subsample, then the
# best third on three times more addresses, until the whole training split). Every
# candidate is scored with stratified 5-fold cross-validation (seeded with DATA_SEED)
# by the F1-score of the illicit class at the 0.5 threshold. Only
# the training split is used: the validation split is left for the interpretation and
# the test split for the evaluation.
#
# The search of a model is saved under its fingerprint (common.py): the features, its
# grid and fixed settings, and the code. A search with the current fingerprint is reused
# (--retune searches again); the searches of other fingerprints (e.g. other features)
# are kept, so going back to them does not search again. train_models.py runs this
# search itself when a model has no search of the current fingerprint, so this script
# only needs to be run on its own to search without training:
#
#   python tune_hyperparameters.py           # the models whose search is missing or outdated
#   python tune_hyperparameters.py --retune  # every model again
#
# Output in EGO_CENTRIC_RESULTS_DIR/tuning:
#   searches.json                      every search by fingerprint: model, hyperparameters,
#                                      cross-validated F1-score and the data it used
#   index.csv                          one row per search, with the expansion configuration
#                                      of its data
#   search_<model>_<fingerprint>.csv   every candidate of a search with its score

import argparse
import json
import shutil
import sys
import time
import warnings
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
for _p in (str(CURRENT_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Single-threaded numerical libraries (before importing them): the search runs one
# single-threaded fit per core
from shared.threads import limit_threads

import joblib
import pandas as pd
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV, ParameterGrid, StratifiedKFold

import config
from common import (CV_FOLDS, RESULTS_DIR, TUNING_DIR, TUNING_FILE, data_description, expansion_columns, load_dataset,
                    load_searches, search_results_path, tuning_fingerprint, write_index)
from estimators import MODELS, f1_at_threshold, make_pipeline, scale_pos_weight, to_estimator_grid, to_json_params
from shared.experiment import timestamp, write_json

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def tune_model(name, data, workers, fingerprint):
    """Searches the hyperparameters of `name` over its grid with stratified k-fold
    cross-validation on the training split (F1-score at the 0.5 threshold), and saves
    every candidate. The pipelines of a search share their fitted preprocessing steps
    (cache)."""
    X, y, _ = data["train"]
    spec = MODELS[name]
    grid = to_estimator_grid(spec["grid"])
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=config.DATA_SEED)
    cache_dir = TUNING_DIR / "_cache"
    pipe = make_pipeline(name, config.DATA_SEED, pos_weight=scale_pos_weight(y),
                         memory=joblib.Memory(location=str(cache_dir), verbose=0))
    n_candidates = len(ParameterGrid(grid))
    if spec["search"] == "halving":
        search = HalvingGridSearchCV(pipe, grid, factor=3, resource="n_samples",
                                     min_resources=min(len(y), max(2000, len(y) // 9)), cv=cv,
                                     scoring=f1_at_threshold, random_state=config.DATA_SEED,
                                     n_jobs=workers, refit=False)
    elif spec["search"] == "grid":
        search = GridSearchCV(pipe, grid, cv=cv, scoring=f1_at_threshold, n_jobs=workers, refit=False)
    else:
        raise ValueError(f"Unknown search '{spec['search']}' of {name}: 'grid' or 'halving'")

    print(f"Searching the hyperparameters of {name}: {n_candidates} candidates ({spec['search']}, "
          f"{CV_FOLDS}-fold cross-validation)...", flush=True)
    start = time.perf_counter()
    try:
        search.fit(X, y)
    finally:
        shutil.rmtree(str(cache_dir), ignore_errors=True)
    elapsed = time.perf_counter() - start

    results = pd.DataFrame(search.cv_results_)
    results["params"] = [json.dumps(to_json_params(p)) for p in results["params"]]
    keep = [c for c in ("iter", "n_resources", "params", "mean_test_score", "std_test_score", "rank_test_score")
            if c in results.columns]
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    results[keep].to_csv(search_results_path(name, fingerprint), index=False)

    best = to_json_params(search.best_params_)
    print(f"  {name}: cross-validated F1 {search.best_score_:.4f} with {best} ({elapsed:.0f} s)", flush=True)
    return {"model": name, "params": best, "cv_f1": float(search.best_score_), "search": spec["search"],
            "candidates": n_candidates, "time_s": round(elapsed, 1), "finished": timestamp()}


def write_tuning_index(searches):
    rows = [{"fingerprint": fingerprint, "model": entry["model"], "cv_f1": entry["cv_f1"], "search": entry["search"],
             "candidates": entry["candidates"], "time_s": entry["time_s"], "finished": entry["finished"],
             "params": json.dumps(entry["params"]), **expansion_columns(entry.get("data"))}
            for fingerprint, entry in searches.items()]
    write_index(TUNING_DIR / "index.csv", rows, ["data_fingerprint", "model", "finished"])


def tuned_hyperparameters(data, workers, retune=False):
    """Hyperparameters of every model: those of its saved search of the current
    fingerprint, otherwise (or with retune) searched and saved, model by model."""
    searches = load_searches()
    for name in MODELS:
        fingerprint = tuning_fingerprint(name)
        if fingerprint in searches and not retune:
            print(f"{name}: hyperparameters already searched with these features, grid and code: "
                  f"{searches[fingerprint]['params']}")
            continue
        searches[fingerprint] = {**tune_model(name, data, workers, fingerprint), "data": data_description()}
        write_json(TUNING_FILE, searches)
        write_tuning_index(searches)
    write_tuning_index(searches)
    return {name: searches[tuning_fingerprint(name)]["params"] for name in MODELS}


def main():
    limit_threads()
    parser = argparse.ArgumentParser(description="Search the hyperparameters of the ego-centric tabular models.")
    parser.add_argument("--workers", type=int, default=config.EGO_CENTRIC_MODEL_WORKERS, help="parallel processes")
    parser.add_argument("--retune", action="store_true", help="search every model again, even if it is up to date")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tuned_hyperparameters(load_dataset(), args.workers, retune=args.retune)
    print(f"\nHyperparameters saved at: {TUNING_FILE}")


if __name__ == "__main__":
    main()
