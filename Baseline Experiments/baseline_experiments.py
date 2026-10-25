# BASELINE EXPERIMENTS - REPRESENTATIVE BITCOINHEIST METHODS UNDER THE PROTOCOL OF THIS WORK
#
# Reimplements representative methods of the BitcoinHeist literature as their papers
# describe them (decision trees, random forests, XGBoost, k-nearest neighbors, Naive Bayes
# and the stacking ensembles of Nayyer et al., 2023, and Dib et al., 2024, see
# baseline_config.METHODS) and evaluates them exactly as the methodologies of this work, run separately from the pipeline (like the expansion experiments):
#   - the same seed addresses and splits (config.SPLITS_FILE), so every method is tested
#     on the test addresses of both methodologies;
#   - the protocol of Methodology II: hyperparameters searched on the training split with
#     stratified 5-fold cross-validation (seeded with DATA_SEED) scored by the F1-score at
#     the 0.5 threshold, then every run cross-validated on the training split, fitted on
#     the whole training split and evaluated on the test split; the validation split is
#     not used, as in Methodology II;
#   - the training seeds of both methodologies (config.TRAINING_SEEDS), so the runs are
#     compared with theirs seed by seed;
#   - the same metrics (accuracy, precision, recall, F1-score and AUC-ROC of the illicit
#     class) and the same paired tests (wilcoxon_tests.py).
# Every method is trained on each feature set of baseline_config.FEATURE_SETS (the
# BitcoinHeist attributes, the address features of Methodology I and the topological
# features of Methodology II), built by baseline_features.py from files that already
# exist. A feature set whose input files do not exist yet is skipped.
#
# The experiment only reads the outputs of the pipeline and writes to OUTPUTS_DIR
# (baseline_config), so it changes nothing of the experiments of both methodologies, and
# it can run before, during or after them (the comparisons with them use the results
# present when it aggregates; --aggregate-only rebuilds them later).
#
#   python "Baseline Experiments/baseline_experiments.py"                      # every feature set available
#   python "Baseline Experiments/baseline_experiments.py" --feature-sets bitcoinheist,address
#   python "Baseline Experiments/baseline_experiments.py" --n-seeds 10 --workers 8
#   python "Baseline Experiments/baseline_experiments.py" --aggregate-only     # only the tables and tests
#
# Every search and run is saved under its fingerprint (the feature table, the method and
# its hyperparameters, and the code) and reused while it is current; --retune searches
# again. Output in OUTPUTS_DIR:
#   features/<set>.csv, <set>.json     feature table of every seed address and its description
#   tuning/searches.json               every search by fingerprint, with its best hyperparameters
#   tuning/search_<set>_<method>_<fingerprint>.csv   every candidate of a search
#   runs/<set>/<method>-seed<seed>_<fingerprint>.json   cross-validation and test metrics of a run
#   predictions/<set>/<method>-seed<seed>_<fingerprint>.csv   test addresses, label and probability
#   baseline_runs.csv, baseline_cv_folds.csv, baseline_summary.csv   one row per run, per fold,
#                                      and mean, standard deviation and median per method
#   tables/baselines.csv               F1-score and AUC-ROC of every method and feature set, with
#                                      the best configuration of both methodologies
#   tests/baseline_tests.csv           paired Wilcoxon tests against both methodologies and
#                                      between feature sets
#   experiments_config.json            full configuration of the experiment

import argparse
import json
import re
import sys
import time
import warnings
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
for _p in (str(CURRENT_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Every fit is a single-threaded process, one per core (before importing the libraries)
from shared.threads import limit_threads

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import (GridSearchCV, HalvingGridSearchCV, ParameterGrid, RandomizedSearchCV,
                                     StratifiedKFold)

import config
import wilcoxon_tests
from baseline_config import (CV_FOLDS, FEATURE_SETS, FEATURES_DIR, METHODS, OUTPUTS_DIR, PREDICTIONS_DIR,
                             RANDOM_SEARCH_ITERATIONS, RUNS_DIR, TABLES_DIR, TESTS_DIR, THRESHOLD, TUNING_DIR)
from baseline_features import ID_COLUMNS, feature_table, load_splits, missing_inputs
from baseline_models import (evaluate, f1_at_threshold, is_stacking, make_pipeline, scale_pos_weight,
                             search_space, to_json_params)
from shared.experiment import (ask_n_seeds, code_fingerprint, digest, experiment_snapshot, git_commit, read_json,
                               timestamp, training_seeds, write_json)
from shared.stats import summarize_runs

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

METRICS = ["accuracy", "precision", "recall", "f1", "auc_roc"]
CV_METRICS = [f"cv_{m}" for m in METRICS]
TUNING_FILE = TUNING_DIR / "searches.json"

# Code of the searches and of the runs: file -> its functions, or None for the whole file
SEARCH_CODE = {CURRENT_DIR / "baseline_models.py": None,
               Path(__file__).resolve(): ["load_data", "search_method"]}
RUN_CODE = {CURRENT_DIR / "baseline_models.py": None,
            Path(__file__).resolve(): ["load_data", "cross_validate", "fit_run"]}


def file_name(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def load_data(table):
    """Features, labels and addresses of every split, in a deterministic row order that
    does not depend on the order of the table."""
    table = table.sort_values("address").sample(frac=1, random_state=config.DATA_SEED).reset_index(drop=True)
    features = [c for c in table.columns if c not in ID_COLUMNS]
    data = {}
    for split in config.SPLITS:
        part = table[table["split"] == split]
        data[split] = (part[features].astype(float).reset_index(drop=True), part["label"].astype(int).values,
                       part["address"].values)
    return data


# --------------------------------------------------------------------------
# Fingerprints and saved results
# --------------------------------------------------------------------------

def search_fingerprint(name, data_fingerprint):
    return digest({"data": data_fingerprint, "method": name, "spec": METHODS[name], "folds": CV_FOLDS,
                   "iterations": RANDOM_SEARCH_ITERATIONS, "data_seed": config.DATA_SEED, "threshold": THRESHOLD,
                   "code": code_fingerprint(SEARCH_CODE)})


def run_fingerprint(name, params, data_fingerprint):
    return digest({"data": data_fingerprint, "method": name, "spec": METHODS[name], "params": params,
                   "folds": CV_FOLDS, "threshold": THRESHOLD, "code": code_fingerprint(RUN_CODE)})


def run_path(feature_set, name, seed, fingerprint):
    return RUNS_DIR / feature_set / f"{file_name(name)}-seed{seed}_{fingerprint}.json"


def predictions_path(feature_set, name, seed, fingerprint):
    return PREDICTIONS_DIR / feature_set / f"{file_name(name)}-seed{seed}_{fingerprint}.csv"


def load_searches():
    return read_json(TUNING_FILE) if TUNING_FILE.exists() else {}


def method_params(name, single_params):
    """Hyperparameters of a method: its own search, or those of the base models and of a
    searched meta model of a stacking ensemble (None if one of them has no search)."""
    if not is_stacking(name):
        return single_params.get(name)
    members = list(METHODS[name]["base"])
    if isinstance(METHODS[name]["meta"], str):
        members.append(METHODS[name]["meta"])
    params = {member: single_params.get(member) for member in members}
    return None if any(p is None for p in params.values()) else params


def n_candidates(spec):
    if spec["search"] == "random":
        return RANDOM_SEARCH_ITERATIONS
    return len(ParameterGrid(spec["grid"]))


# --------------------------------------------------------------------------
# Hyperparameter searches
# --------------------------------------------------------------------------

def search_method(name, data, workers):
    """Searches the hyperparameters of a single model over its grid with stratified
    k-fold cross-validation on the training split (F1-score at the threshold), as the
    searches of Methodology II. Returns the best hyperparameters, its score and every
    candidate."""
    X, y, _ = data["train"]
    spec = METHODS[name]
    pipe = make_pipeline(name, config.DATA_SEED, scale_pos_weight(y))
    grid = search_space(spec["grid"])
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=config.DATA_SEED)
    if spec["search"] == "random":
        search = RandomizedSearchCV(pipe, grid, n_iter=RANDOM_SEARCH_ITERATIONS, cv=cv, scoring=f1_at_threshold,
                                    random_state=config.DATA_SEED, n_jobs=workers, refit=False)
    elif spec["search"] == "halving":
        search = HalvingGridSearchCV(pipe, grid, factor=3, resource="n_samples",
                                     min_resources=min(len(y), max(2000, len(y) // 9)), cv=cv,
                                     scoring=f1_at_threshold, random_state=config.DATA_SEED,
                                     n_jobs=workers, refit=False)
    elif spec["search"] == "grid":
        search = GridSearchCV(pipe, grid, cv=cv, scoring=f1_at_threshold, n_jobs=workers, refit=False)
    else:
        raise ValueError(f"Unknown search '{spec['search']}' of {name}: 'grid', 'halving' or 'random'")
    search.fit(X, y)
    results = pd.DataFrame(search.cv_results_)
    results["params"] = [json.dumps(to_json_params(p)) for p in results["params"]]
    keep = [c for c in ("iter", "n_resources", "params", "mean_test_score", "std_test_score", "rank_test_score")
            if c in results.columns]
    return to_json_params(search.best_params_), float(search.best_score_), results[keep]


def searched_params(feature_set, data, data_fingerprint, workers, retune=False):
    """Hyperparameters of every single model on a feature set: those of its saved search
    of the current fingerprint, otherwise (or with retune) searched and saved."""
    searches = load_searches()
    params = {}
    for name, spec in METHODS.items():
        if is_stacking(name):
            continue
        fingerprint = search_fingerprint(name, data_fingerprint)
        if fingerprint in searches and not retune:
            params[name] = searches[fingerprint]["params"]
            continue
        candidates_count = n_candidates(spec)
        print(f"{feature_set} | {name}: searching {candidates_count} candidates ({spec['search']}, "
              f"{CV_FOLDS}-fold cross-validation)...", flush=True)
        start = time.perf_counter()
        best, score, candidates = search_method(name, data, workers)
        elapsed = time.perf_counter() - start
        TUNING_DIR.mkdir(parents=True, exist_ok=True)
        candidates.to_csv(TUNING_DIR / f"search_{feature_set}_{file_name(name)}_{fingerprint}.csv", index=False)
        print(f"  cross-validated F1 {score:.4f} with {best} ({elapsed:.0f} s)", flush=True)
        searches[fingerprint] = {"feature_set": feature_set, "method": name, "params": best, "cv_f1": score,
                                 "search": spec["search"], "candidates": candidates_count, "time_s": round(elapsed, 1),
                                 "data_fingerprint": data_fingerprint, "finished": timestamp()}
        write_json(TUNING_FILE, searches)
        params[name] = best
    return params


def current_params(data_fingerprint):
    """Hyperparameters of every method from the saved searches of the current fingerprint."""
    searches = load_searches()
    single = {}
    for name in METHODS:
        if not is_stacking(name):
            entry = searches.get(search_fingerprint(name, data_fingerprint))
            single[name] = entry["params"] if entry else None
    return {name: method_params(name, single) for name in METHODS}


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------

def cross_validate(pipe, X, y, seed):
    """Stratified k-fold cross-validation of `pipe` on the training split (seeded per run,
    as in Methodology II): the mean and standard deviation of every metric, and the
    metrics of every fold."""
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
    for m in METRICS:
        summary[f"cv_{m}"] = float(folds_df[m].mean())
        summary[f"cv_{m}_std"] = float(folds_df[m].std(ddof=1)) if len(folds_df) > 1 else 0.0
    return summary, fold_rows


def fit_run(feature_set, name, seed, params, data, fingerprint, description, commit):
    """Trains and evaluates one method with one seed on one feature set (cross-validation
    on the training split, then fitted on the whole training split and evaluated on the
    test split), and saves its result files."""
    X_train, y_train, _ = data["train"]
    X_test, y_test, addr_test = data["test"]
    np.random.seed(seed)
    pipe = make_pipeline(name, seed, scale_pos_weight(y_train), params)

    cv_metrics, cv_folds = cross_validate(pipe, X_train, y_train, seed)

    start = time.perf_counter()
    pipe.fit(X_train, y_train)
    fit_time = time.perf_counter() - start

    metrics, prob, pred = evaluate(pipe, X_test, y_test)
    path = predictions_path(feature_set, name, seed, fingerprint)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"address": addr_test, "label": y_test, "prob_illicit": prob, "prediction": pred}).to_csv(
        path, index=False)

    # Written last: its presence marks the run of this fingerprint as completed
    write_json(run_path(feature_set, name, seed, fingerprint), {
        "feature_set": feature_set,
        "method": name,
        "work": METHODS[name]["work"],
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


def run_is_current(feature_set, name, seed, fingerprint):
    path = run_path(feature_set, name, seed, fingerprint)
    try:
        return path.exists() and read_json(path).get("fingerprint") == fingerprint
    except (OSError, ValueError):
        return False


def train_missing_runs(feature_set, data, data_fingerprint, description, seeds, params, workers):
    """Trains the runs of a feature set without a result of their current fingerprint."""
    commit = git_commit(config.PROJECT_ROOT)
    fingerprints = {name: run_fingerprint(name, params[name], data_fingerprint) for name in METHODS}
    runs = [(name, seed) for seed in seeds for name in METHODS]
    pending = [(name, seed) for name, seed in runs if not run_is_current(feature_set, name, seed, fingerprints[name])]
    print(f"{feature_set}: {len(runs) - len(pending)} of {len(runs)} runs already trained with the current features, "
          f"hyperparameters and code, {len(pending)} to train ({workers} parallel single-threaded processes).",
          flush=True)
    if pending:
        Parallel(n_jobs=workers, verbose=10)(
            delayed(fit_run)(feature_set, name, seed, params[name], data, fingerprints[name], description, commit)
            for name, seed in pending)


# --------------------------------------------------------------------------
# Result tables and tests
# --------------------------------------------------------------------------

def current_runs(feature_sets):
    """The saved runs of every feature set trained with its current features,
    hyperparameters and code."""
    runs = []
    for feature_set in feature_sets:
        description_path = FEATURES_DIR / f"{feature_set}.json"
        if not description_path.exists():
            continue
        data_fingerprint = read_json(description_path)["fingerprint"]
        for name, params in current_params(data_fingerprint).items():
            if params is None:
                continue
            fingerprint = run_fingerprint(name, params, data_fingerprint)
            for path in sorted((RUNS_DIR / feature_set).glob(f"{file_name(name)}-seed*_{fingerprint}.json")):
                runs.append(read_json(path))
    return runs


def formatted(values):
    values = pd.Series(values, dtype=float)
    std = values.std(ddof=1) if len(values) > 1 else 0.0
    return f"{values.mean():.3f} ± {std:.3f} [{values.median():.3f}]"


def methodology_runs():
    """The runs of the best configuration of each methodology (highest mean F1-score),
    if their results exist: name -> runs (seed and metrics)."""
    best = {}
    gnn = wilcoxon_tests.load_gnn_runs(wilcoxon_tests.gnn_runs_file())
    if gnn is not None:
        name = wilcoxon_tests.best_model(gnn)
        best[f"Methodology I: {name}"] = gnn[gnn["configuration"] == name]
    ego = wilcoxon_tests.load_ego_runs()
    if ego is not None:
        name = wilcoxon_tests.best_model(ego)
        best[f"Methodology II: {name}"] = ego[ego["configuration"] == name]
    return best


def paper_table(runs, feature_sets, references):
    """F1-score and AUC-ROC of every method on every feature set, and of the best
    configuration of each methodology on the same test addresses."""
    rows = []
    for name, spec in METHODS.items():
        row = {"method": name, "work": spec["work"]}
        for feature_set in feature_sets:
            part = runs[(runs["method"] == name) & (runs["feature_set"] == feature_set)]
            for metric, label in (("f1", "F1"), ("auc_roc", "AUC-ROC")):
                row[f"{feature_set} {label}"] = formatted(part[metric]) if len(part) else ""
        rows.append(row)
    for name, part in references.items():
        rows.append({"method": name, "work": "this work",
                     "reference F1": formatted(part["f1"]), "reference AUC-ROC": formatted(part["auc_roc"])})
    return pd.DataFrame(rows)


def statistical_tests(runs, feature_sets, references):
    """Paired tests (wilcoxon_tests.compare): the best configuration of each methodology
    against every method and feature set, and, for every method, the features of this
    work against the BitcoinHeist attributes."""
    rows = []
    for reference, ref_runs in references.items():
        for (feature_set, name), part in runs.groupby(["feature_set", "method"], sort=False):
            rows += wilcoxon_tests.compare(ref_runs, part, reference, f"{name} ({feature_set})",
                                           f"{reference.split(':')[0]} vs baselines")
    if "bitcoinheist" in feature_sets:
        base = runs[runs["feature_set"] == "bitcoinheist"]
        for feature_set in [s for s in feature_sets if s != "bitcoinheist"]:
            other = runs[runs["feature_set"] == feature_set]
            for name in METHODS:
                a, b = other[other["method"] == name], base[base["method"] == name]
                if len(a) and len(b):
                    rows += wilcoxon_tests.compare(a, b, f"{name} ({feature_set})", f"{name} (bitcoinheist)",
                                                   f"{feature_set} vs bitcoinheist features")
    return rows


def aggregate(feature_sets):
    """Result tables and tests of the current runs (runs of other features,
    hyperparameters or code are kept but left out)."""
    results = current_runs(feature_sets)
    if not results:
        print("No completed runs to aggregate.")
        return
    fold_rows, flat = [], []
    for r in results:
        r = dict(r)
        for fold in r.pop("cv_fold_metrics", None) or []:
            fold_rows.append({"feature_set": r["feature_set"], "method": r["method"], "seed": r["seed"], **fold})
        r["hyperparameters"] = json.dumps(r.get("hyperparameters"))
        r["data_fingerprint"] = (r.pop("data", None) or {}).get("fingerprint")
        flat.append(r)
    runs = pd.DataFrame(flat).sort_values(["feature_set", "method", "seed"])
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    runs.to_csv(OUTPUTS_DIR / "baseline_runs.csv", index=False)
    pd.DataFrame(fold_rows).sort_values(["feature_set", "method", "seed", "fold"]).to_csv(
        OUTPUTS_DIR / "baseline_cv_folds.csv", index=False)
    summary = summarize_runs(runs, ["feature_set", "method"], METRICS + ["fit_time_s"] + CV_METRICS)
    summary.sort_values(["feature_set", "f1_mean"], ascending=[True, False]).to_csv(
        OUTPUTS_DIR / "baseline_summary.csv", index=False)

    present = [s for s in feature_sets if s in set(runs["feature_set"])]
    references = methodology_runs()
    table = paper_table(runs, present, references)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / "baselines.csv", index=False)

    tests = statistical_tests(runs, present, references)
    TESTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(tests).to_csv(TESTS_DIR / "baseline_tests.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print("\n=== Test split: F1-score and AUC-ROC, mean ± standard deviation [median] over the seeds ===")
    print(table.to_string(index=False))
    if not references:
        print("\nNo results of the methodologies yet: run --aggregate-only after them for the comparisons.")
    if tests:
        f1 = pd.DataFrame(tests)
        f1 = f1[f1["metric"] == "f1"][["group", "a", "b", "n", "mean_diff", "ci_low", "ci_high", "p_value"]]
        print("\n=== Paired tests (F1-score) ===")
        print(f1.round(4).to_string(index=False))


def save_experiment_config(descriptions, seeds, workers):
    """The configuration of the experiment, with the feature sets built so far."""
    write_json(OUTPUTS_DIR / "experiments_config.json", experiment_snapshot(
        extra={
            "experiment": "representative BitcoinHeist methods under the protocol of this work",
            "feature_sets": descriptions,
            "methods": METHODS,
            "preprocessing": "median imputation, signed log1p and standardization, fitted on the training split",
            "hyperparameter_search": {
                "method": f"grid, successive-halving grid or random search ({RANDOM_SEARCH_ITERATIONS} candidates), "
                          f"{CV_FOLDS}-fold stratified cross-validation on the training split (seeded with DATA_SEED), "
                          f"F1-score at the {THRESHOLD} threshold",
                "searches": load_searches(),
            },
            "stacking": "base models with their own searched hyperparameters, and the meta model of the original "
                        f"work trained on their {CV_FOLDS}-fold out-of-fold probabilities (and the features, with "
                        "passthrough)",
            "n_seeds": len(seeds),
            "seeds": seeds,
            "evaluation": {
                "threshold": THRESHOLD,
                "cross_validation": f"{CV_FOLDS}-fold stratified cross-validation on the training split (seeded per run)",
                "test_split": "seed addresses of the test split (config.SPLITS_FILE)",
                "metrics": METRICS,
                "statistical_tests": "wilcoxon_tests.compare: exact Wilcoxon signed-rank test and bootstrap CI",
            },
            "resources": {"parallel_processes": workers, "threads_per_process": 1},
        },
        modules=("sklearn", "xgboost", "numpy", "pandas", "scipy"),
    ))


def selected_feature_sets(answer):
    """The feature sets to run: the given ones (all their inputs must exist), or every
    feature set whose inputs exist."""
    if answer:
        selected = [s.strip() for s in answer.split(",") if s.strip()]
        unknown = [s for s in selected if s not in FEATURE_SETS]
        if unknown:
            raise SystemExit(f"Unknown feature sets {unknown}; available: {', '.join(FEATURE_SETS)}")
        for feature_set in selected:
            missing = missing_inputs(feature_set)
            if missing:
                raise SystemExit(f"{feature_set}: missing inputs {missing}")
        return selected
    selected = []
    for feature_set in FEATURE_SETS:
        missing = missing_inputs(feature_set)
        if missing:
            print(f"{feature_set}: skipped, missing inputs {missing}")
        else:
            selected.append(feature_set)
    return selected


def main():
    limit_threads()
    parser = argparse.ArgumentParser(description="Representative BitcoinHeist methods under the protocol of this work.")
    parser.add_argument("--feature-sets", help=f"comma-separated, any of {', '.join(FEATURE_SETS)} "
                                               "(default: every feature set whose inputs exist)")
    parser.add_argument("--n-seeds", type=int, help="random-seed runs per method (asked if not given)")
    parser.add_argument("--workers", type=int, default=config.EGO_CENTRIC_MODEL_WORKERS, help="parallel processes")
    parser.add_argument("--retune", action="store_true", help="search every model again, even if it is up to date")
    parser.add_argument("--aggregate-only", action="store_true", help="only rebuild the result tables and tests")
    args = parser.parse_args()

    if args.aggregate_only:
        sets = [s.strip() for s in args.feature_sets.split(",")] if args.feature_sets else list(FEATURE_SETS)
        aggregate(sets)
        return

    feature_sets = selected_feature_sets(args.feature_sets)
    if not feature_sets:
        raise SystemExit("No feature set has its inputs.")
    # Asked before the searches, so the rest runs unattended
    seeds = training_seeds(ask_n_seeds(args.n_seeds))
    splits = load_splits()
    descriptions = {}
    for feature_set in feature_sets:
        table, description = feature_table(feature_set, splits)
        descriptions[feature_set] = description
        data = load_data(table)
        data_fingerprint = description["fingerprint"]
        print(f"{feature_set}: {len(description['features'])} features; "
              + ", ".join(f"{split} {len(data[split][1])}" for split in config.SPLITS), flush=True)
        single = searched_params(feature_set, data, data_fingerprint, args.workers, retune=args.retune)
        params = {name: method_params(name, single) for name in METHODS}
        save_experiment_config(descriptions, seeds, args.workers)
        train_missing_runs(feature_set, data, data_fingerprint, description, seeds, params, args.workers)
    aggregate(feature_sets)
    print(f"\nResults saved at: {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
