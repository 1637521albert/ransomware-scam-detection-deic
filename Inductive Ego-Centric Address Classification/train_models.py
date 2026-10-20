# STEP 4 (Methodology II) - TRAINING AND EVALUATION OF THE TABULAR MODELS
#
# Asks how many random-seed runs to execute and trains the eight tabular models
# (estimators.py) with their searched hyperparameters and every seed, on the
# topological features (EGO_CENTRIC_FEATURES_DIR/<split>.csv). The models whose
# hyperparameter search is missing or outdated are searched first
# (tune_hyperparameters.py). Every run is evaluated with stratified 5-fold
# cross-validation on the training split, then fitted on the whole training split and
# evaluated on the test split, predicting illicit when the probability of the illicit
# class is at least 0.5, as the GNNs of Methodology I. Then it builds the result tables
# and runs the Wilcoxon tests (wilcoxon_tests.py).
#
# Every run is saved under its fingerprint (common.py: the features, hyperparameters and
# code) with the expansion configuration of its data, and is reused while its fingerprint
# is current. Runs of other fingerprints are kept (and listed in runs/index.csv with
# their expansion configuration), but the result tables and the tests only include the
# runs of the current features and code.
#
#   python train_models.py                    # asks the number of seeds
#   python train_models.py --n-seeds 10       # without asking
#   python train_models.py --aggregate-only   # only rebuild the result tables and tests
#
# Output in EGO_CENTRIC_RESULTS_DIR:
#   experiment_config.json                full configuration of the experiment (data, expansion,
#                                         features, models, searched hyperparameters, evaluation, seeds)
#   runs/<model>-seed<seed>_<fingerprint>.json   cross-validation and test metrics of every run,
#                                                with its hyperparameters and data
#   runs/index.csv                        every saved run, with the expansion configuration of its data
#   predictions/<model>-seed<seed>_<fingerprint>.csv   test addresses with label and predicted probability
#   ego_runs.csv                          one row per run (test and cross-validation metrics)
#   ego_cv_folds.csv                      one row per (model, seed, fold) of the cross-validation
#   ego_summary.csv, ego_cv_summary.csv   mean, standard deviation and median per model

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
for _p in (str(CURRENT_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Every run is a single-threaded process, one per core (before importing the libraries)
from shared.threads import limit_threads

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

import config
import wilcoxon_tests
from common import (CV_FOLDS, CV_METRICS, EVAL_METRICS, METRICS, PREDICTIONS_DIR, RESULTS_DIR, RUNS_DIR,
                    TRAINING_FEATURES, current_hyperparameters, data_description, expansion_columns, load_dataset,
                    load_searches, predictions_path, run_fingerprint, run_path, write_index)
from estimators import MODELS, THRESHOLD, evaluate, make_pipeline, scale_pos_weight
from shared.experiment import (ask_n_seeds, experiment_snapshot, git_commit, read_json, timestamp,
                               training_seeds, write_json)
from shared.stats import summarize_runs
from tune_hyperparameters import tuned_hyperparameters

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RESULT_TABLES = ["ego_runs.csv", "ego_cv_folds.csv", "ego_summary.csv", "ego_cv_summary.csv"]


def cross_validate_model(pipe, X, y, seed):
    """Stratified k-fold cross-validation of `pipe` on (X, y): the mean and standard
    deviation of every metric, and the metrics of every fold."""
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    fold_rows = []
    for fold, (train_idx, val_idx) in enumerate(folds.split(X, y), start=1):
        fold_pipe = clone(pipe)
        start = time.perf_counter()
        fold_pipe.fit(X.iloc[train_idx], y[train_idx])
        fit_time = time.perf_counter() - start
        metrics, _, _ = evaluate(fold_pipe, X.iloc[val_idx], y[val_idx])
        fold_rows.append({"fold": fold, **{k: float(v) for k, v in metrics.items()}, "fit_time_s": fit_time})

    folds_df = pd.DataFrame(fold_rows)
    summary = {}
    for m in EVAL_METRICS:
        summary[f"cv_{m}"] = float(folds_df[m].mean())
        summary[f"cv_{m}_std"] = float(folds_df[m].std(ddof=1)) if len(folds_df) > 1 else 0.0
    summary["cv_fit_time_s"] = float(folds_df["fit_time_s"].sum())
    return summary, fold_rows


def fit_run(name, seed, params, data, fingerprint, description, commit):
    """Trains and evaluates one model with one seed (cross-validation on the training
    split, then fitted on the whole training split and evaluated on the test split), and
    saves its result files."""
    X_train, y_train, _ = data["train"]
    X_test, y_test, addr_test = data["test"]
    np.random.seed(seed)
    pipe = make_pipeline(name, seed, params, scale_pos_weight(y_train))

    cv_metrics, cv_folds = cross_validate_model(pipe, X_train, y_train, seed)

    start = time.perf_counter()
    pipe.fit(X_train, y_train)
    fit_time = time.perf_counter() - start

    metrics, prob, pred = evaluate(pipe, X_test, y_test)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"address": addr_test, "label": y_test, "prob_illicit": prob, "prediction": pred}).to_csv(
        predictions_path(name, seed, fingerprint), index=False)

    # Written last: its presence marks the run of this fingerprint as completed
    write_json(run_path(name, seed, fingerprint), {
        "model": name,
        "seed": seed,
        "threshold": THRESHOLD,
        "cv_folds": CV_FOLDS,
        **{k: float(v) for k, v in metrics.items()},
        "fit_time_s": fit_time,
        **cv_metrics,
        "cv_fold_metrics": cv_folds,
        "hyperparameters": params,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "fingerprint": fingerprint,
        "data": description,
        "git_commit": commit,
        "finished": timestamp(),
    })
    return fingerprint


def run_is_current(name, seed, fingerprint):
    path = run_path(name, seed, fingerprint)
    try:
        return path.exists() and read_json(path).get("fingerprint") == fingerprint
    except (OSError, ValueError):
        return False


def train_missing_runs(data, seeds, workers, hyperparameters):
    """Trains the runs without a result of their current fingerprint."""
    commit = git_commit(config.PROJECT_ROOT)
    fingerprints = {name: run_fingerprint(name, hyperparameters[name]) for name in MODELS}
    runs = [(name, seed) for seed in seeds for name in MODELS]
    pending = [(name, seed) for name, seed in runs if not run_is_current(name, seed, fingerprints[name])]
    print(f"{len(runs) - len(pending)} of {len(runs)} runs already trained with the current features, "
          f"hyperparameters and code, {len(pending)} to train ({workers} parallel single-threaded processes, "
          f"with {CV_FOLDS}-fold cross-validation per run).")
    if pending:
        Parallel(n_jobs=workers, verbose=10)(
            delayed(fit_run)(name, seed, hyperparameters[name], data, fingerprints[name], data_description(), commit)
            for name, seed in pending)


def save_experiment_config(seeds, workers):
    expansion_path = config.EGO_CENTRIC_EGONETS_DIR / "expansion_config.json"
    features_paths = {split: config.EGO_CENTRIC_FEATURES_DIR / f"{split}.json" for split in config.SPLITS}
    write_json(RESULTS_DIR / "experiment_config.json", experiment_snapshot(
        extra={
            "methodology": "inductive ego-centric address classification",
            "data": {
                "expansion": read_json(expansion_path) if expansion_path.exists() else None,
                "features": {split: read_json(path) if path.exists() else None
                             for split, path in features_paths.items()},
                "description": data_description(),
            },
            "training_features": TRAINING_FEATURES,
            "models": MODELS,
            "preprocessing": "clipping of negative values and log1p, mutual-information feature selection "
                             "(SelectPercentile) and scaling, fitted on the training split",
            "hyperparameter_search": {
                "method": f"grid or successive-halving grid search, {CV_FOLDS}-fold stratified cross-validation "
                          "on the training split, F1-score at the 0.5 threshold",
                "searches": load_searches(),
            },
            "n_seeds": len(seeds),
            "seeds": seeds,
            "evaluation": {
                "threshold": THRESHOLD,
                "cross_validation": f"{CV_FOLDS}-fold stratified cross-validation on the training split (seeded per run)",
                "test_split": "seed addresses of the test split",
                "metrics": EVAL_METRICS,
                "statistical_tests": "wilcoxon_tests.py",
                "interpretation": "interpret_model.py",
            },
            "resources": {"parallel_processes": workers, "threads_per_process": 1},
        },
        modules=("sklearn", "xgboost", "numpy", "pandas"),
    ))


def aggregate():
    """Result tables and tests of the runs trained with the current features,
    hyperparameters and code (the others are left out)."""
    results = [read_json(p) for p in sorted(RUNS_DIR.glob("*.json"))]
    write_index(RUNS_DIR / "index.csv", [
        {"model": r.get("model"), "seed": r.get("seed"), "fingerprint": r.get("fingerprint"),
         **{m: r.get(m) for m in ("f1", "auc_roc", "accuracy", "precision", "recall", "cv_f1", "cv_auc_roc")},
         "hyperparameters": json.dumps(r.get("hyperparameters")), "finished": r.get("finished"),
         **expansion_columns(r.get("data"))} for r in results], ["data_fingerprint", "model", "seed"])

    hyperparameters = current_hyperparameters()
    current = {name: run_fingerprint(name, params) for name, params in hyperparameters.items() if params is not None}
    valid = [r for r in results if r.get("model") in current and r.get("fingerprint") == current[r["model"]]]
    if len(valid) < len(results):
        print(f"{len(results) - len(valid)} saved runs of other features, hyperparameters or code are kept "
              f"(listed in {RUNS_DIR / 'index.csv'}) but left out of the result tables.")
    if not valid:
        # Tables of other runs would be read by the Wilcoxon tests as if they were current
        for name in RESULT_TABLES:
            if (RESULTS_DIR / name).exists():
                (RESULTS_DIR / name).unlink()
        print("No completed runs to aggregate.")
        return

    fold_records, flat_results = [], []
    for r in valid:
        r = dict(r)
        for fold in r.pop("cv_fold_metrics", None) or []:
            fold_records.append({"model": r["model"], "seed": r["seed"], **fold})
        r["hyperparameters"] = json.dumps(r.get("hyperparameters"))
        r.update(expansion_columns(r.pop("data", None)))
        flat_results.append(r)

    runs = pd.DataFrame(flat_results).sort_values(["model", "seed"])
    runs.to_csv(RESULTS_DIR / "ego_runs.csv", index=False)
    if fold_records:
        pd.DataFrame(fold_records).sort_values(["model", "seed", "fold"]).to_csv(RESULTS_DIR / "ego_cv_folds.csv", index=False)

    summary = summarize_runs(runs, ["model"], METRICS + [m for m in CV_METRICS if m in runs.columns])
    summary = summary.sort_values("f1_mean", ascending=False)
    summary.to_csv(RESULTS_DIR / "ego_summary.csv", index=False)

    pd.set_option("display.width", 200)
    if all(f"cv_{m}_mean" in summary.columns for m in EVAL_METRICS):
        cv_summary = summary.sort_values("cv_f1_mean", ascending=False)
        cv_summary.to_csv(RESULTS_DIR / "ego_cv_summary.csv", index=False)
        print(f"\n=== {CV_FOLDS}-fold cross-validation on the training split ===")
        print(cv_summary[["model", "n_runs", "cv_accuracy_mean", "cv_accuracy_std", "cv_f1_mean", "cv_f1_std",
                          "cv_f1_median", "cv_auc_roc_mean", "cv_auc_roc_std"]].round(4).to_string(index=False))

    print("\n=== Test split ===")
    print(summary[["model", "n_runs", "accuracy_mean", "accuracy_std", "f1_mean", "f1_std", "f1_median",
                   "auc_roc_mean", "auc_roc_std"]].round(4).to_string(index=False))
    wilcoxon_tests.run_all()


def main():
    limit_threads()
    parser = argparse.ArgumentParser(description="Train and evaluate the ego-centric tabular models.")
    parser.add_argument("--n-seeds", type=int, help="random-seed runs per model (asked if not given)")
    parser.add_argument("--workers", type=int, default=config.EGO_CENTRIC_MODEL_WORKERS, help="parallel processes")
    parser.add_argument("--aggregate-only", action="store_true", help="only rebuild the result tables and tests")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not args.aggregate_only:
        data = load_dataset()
        for split in config.SPLITS:
            print(f"{split:5s}: {len(data[split][1])} seed addresses")
        # Asked before the search, which can take hours, so the rest runs unattended
        seeds = training_seeds(ask_n_seeds(args.n_seeds))
        hyperparameters = tuned_hyperparameters(data, args.workers)
        save_experiment_config(seeds, args.workers)
        train_missing_runs(data, seeds, args.workers, hyperparameters)
    aggregate()
    print(f"\nResults saved at: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
