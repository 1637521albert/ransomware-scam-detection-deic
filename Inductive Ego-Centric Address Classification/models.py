# STEP 3 (Methodology II) - MODEL TRAINING, EVALUATION AND INTERPRETABILITY
#
# Asks how many random-seed runs to execute, trains the eight tabular models on
# the topological features (EGO_CENTRIC_FEATURES_DIR/<split>.csv, written by
# extract_topological_features.py) with that many seeds, evaluates them on the
# test split, runs the Wilcoxon tests (wilcoxon_tests.py) and analyzes the best
# model, the one with the highest mean test F1-score over the seeds
# (wilcoxon_tests.best_model):
#
#   python models.py                         # asks the number of seeds
#   python models.py --n-seeds 10            # without asking
#   python models.py --skip-interpretation   # without the ablation and SHAP analyses
#   python models.py --interpret-only        # only the ablation and SHAP analyses (of the trained runs)
#   python models.py --aggregate-only        # only rebuild the result tables and tests
#
# The interpretation (feature ablation, then SHAP) explains the best model: the one with
# the highest mean test F1-score over the seeds. Each part is saved when it finishes and
# skipped when it is already computed for that model, so an interrupted interpretation
# resumes from the missing part (delete interpretation/shap_values.csv to compute the
# SHAP values again).
#
# All the cores are used: every (model, seed) run trains in its own
# single-threaded process (config.EGO_CENTRIC_MODEL_WORKERS at a time), so the
# results do not depend on the number of processes. Completed runs are skipped,
# so an interrupted execution can simply be relaunched.
#
# Output in EGO_CENTRIC_RESULTS_DIR:
#   experiment_config.json                full configuration of the experiment (data, expansion,
#                                         features, models, evaluation, seeds, resources)
#   runs/<model>-seed<seed>.json          metrics of every run
#   predictions/<model>-seed<seed>.csv    test addresses with label and predicted probability
#   ego_runs.csv                          one row per run
#   ego_summary.csv                       mean, standard deviation and median per model
#   interpretation/                       permutation importance, ablation curve and SHAP values of the
#                                         best model trained with EGO_CENTRIC_INTERPRETATION_SEED, and
#                                         interpretation.json (which model and why)

import argparse
import os
import sys
import time

# Every training run, ablation retraining and SHAP chunk runs in its own single-threaded
# process (one per core), so the numerical libraries are single-threaded too (set before
# importing them): the processes do not compete for the cores, the OpenMP runtime of
# XGBoost and scikit-learn is never started before forking the workers (which can hang
# them), and a model trained again for the interpretation is identical to its run.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
from functools import partial
from pathlib import Path

import joblib
from joblib import Parallel, delayed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import wilcoxon_tests
from shared.experiment import (ask_n_seeds, experiment_snapshot, git_commit, read_json, timestamp,
                               training_seeds, write_json)
from shared.parallel import parallel_map
from shared.stats import summarize_runs

# Validation addresses explained by each SHAP worker task. The chunks and their
# random seeds do not depend on the number of workers, so the SHAP values are
# the same with any number of cores.
SHAP_CHUNK_SIZE = 100

RESULTS_DIR = config.EGO_CENTRIC_RESULTS_DIR
RUNS_DIR = RESULTS_DIR / "runs"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
INTERPRETATION_DIR = RESULTS_DIR / "interpretation"

METRICS = ["accuracy", "precision", "recall", "f1", "auc_roc", "fit_time_s"]

# Features and hyperparameters of the models (config.py)
TRAINING_FEATURES = config.EGO_CENTRIC_TRAINING_FEATURES
HP = config.EGO_CENTRIC_HYPERPARAMETERS

# Feature names in the figures
FEATURE_NAMES_LATEX = {
    '2star_out': r'$\mathcal{B}_{2}$',
    'split_2+':  r'$\mathcal{B}_{3+}$',
    '2star_in':  r'$\mathcal{A}_{2}$',
    'merge_2+':  r'$\mathcal{A}_{3+}$',
    'peeling_2': r'$\mathcal{P}_{2}$',
    'peeling_3': r'$\mathcal{P}_{3+}$',
    'in_deg':       'In-Degree',
    'out_deg':      'Out-Degree',
    'pagerank':     'PageRank',
    'betweenness':  'Betweenness',
    'eccentricity': 'Eccentricity',
    'density':      'Density'
}

def get_label(feature_name):
    """Helper function to retrieve formatted LaTeX label for features."""
    return FEATURE_NAMES_LATEX.get(feature_name, feature_name)

# Isolation Forest as a classifier (anomaly score as the probability of the illicit class)
class IsolationForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=200, contamination='auto', random_state=None, n_jobs=None):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        self.model_.fit(X)
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        pred_iso = self.model_.predict(X)
        return (pred_iso == -1).astype(int)

    def predict_proba(self, X):
        scores = -self.model_.decision_function(X)
        if len(scores) == 0:
            return np.array([]).reshape(0, 2)
        if (scores.max() - scores.min()) == 0:
            scores_scaled = np.full_like(scores, 0.5)
        else:
            scores_scaled = (scores - scores.min()) / (scores.max() - scores.min())
        return np.vstack((1 - scores_scaled, scores_scaled)).T


def build_models(seed, scale_pos_weight):
    """The eight pipelines, with every random component seeded with `seed`. Every model
    is single-threaded; the runs are parallelized across processes instead."""
    # The mutual information estimate adds random noise, so it is seeded too
    mutual_info = partial(mutual_info_classif, random_state=seed)

    def pipeline(model):
        scaler = RobustScaler() if HP["scaler"] == "RobustScaler" else StandardScaler()
        return Pipeline([
            ('selector', SelectPercentile(mutual_info, percentile=HP["selector_percentile"])),
            ('log', FunctionTransformer(np.log1p, validate=True)),
            ('scaler', scaler),
            ('model', model),
        ])

    return {
        'L1 (Lasso)': pipeline(LogisticRegression(penalty=HP["logreg_penalty_l1"], solver='saga', max_iter=HP["logreg_max_iter"], C=HP["logreg_C"], class_weight='balanced', random_state=seed)),
        'ElasticNet': pipeline(LogisticRegression(penalty=HP["logreg_penalty_el"], solver='saga', l1_ratio=HP["logreg_l1_ratio"], max_iter=HP["logreg_max_iter"], C=HP["logreg_C"], class_weight='balanced', random_state=seed)),
        'RandomForest': pipeline(RandomForestClassifier(n_estimators=HP["rf_n_estimators"], max_depth=HP["rf_max_depth"], class_weight='balanced', random_state=seed, n_jobs=1)),
        'XGBoost': pipeline(XGBClassifier(n_estimators=HP["xgb_n_estimators"], max_depth=HP["xgb_max_depth"], learning_rate=HP["xgb_learning_rate"], subsample=HP["xgb_subsample"], colsample_bytree=HP["xgb_colsample_bytree"], scale_pos_weight=scale_pos_weight, eval_metric='logloss', random_state=seed, n_jobs=1)),
        'IsolationForest': pipeline(IsolationForestClassifier(n_estimators=HP["iso_n_estimators"], contamination=HP["iso_contamination"], random_state=seed, n_jobs=1)),
        'SVM': pipeline(SVC(C=HP["svm_C"], kernel=HP["svm_kernel"], class_weight='balanced', probability=True, random_state=seed)),
        'Naive Bayes': pipeline(GaussianNB()),
        'MLP': pipeline(MLPClassifier(hidden_layer_sizes=HP["mlp_hidden_layers"], alpha=HP["mlp_alpha"], max_iter=HP["mlp_max_iter"], solver=HP["mlp_solver"], learning_rate_init=HP["mlp_learning_rate_init"], random_state=seed, early_stopping=True, n_iter_no_change=20, validation_fraction=0.1)),
    }


def file_name(model_name):
    return model_name.replace(" ", "_").replace("(", "").replace(")", "")


def run_id(model_name, seed):
    return f"{file_name(model_name)}-seed{seed}"


def load_dataset():
    features_files = [config.EGO_CENTRIC_FEATURES_DIR / f"{split}.csv" for split in config.SPLITS]
    missing = [str(path) for path in features_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {missing}: run extract_topological_features.py first.")

    df = pd.concat([pd.read_csv(path) for path in features_files], ignore_index=True)
    # Deterministic row order, independent of the order of the files
    df = df.sort_values("addr_str").sample(frac=1, random_state=config.DATA_SEED).reset_index(drop=True)

    data = {}
    for split in config.SPLITS:
        part = df[df["split"] == split]
        data[split] = (part[TRAINING_FEATURES].copy(), part["label"].astype(int).values, part["addr_str"].values)
    return data


def evaluate(model, X, y):
    prob = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "auc_roc": roc_auc_score(y, prob),
    }, prob, pred


def fit_run(name, seed, data, scale_pos_weight, commit):
    """Trains and evaluates one model with one seed, and saves its result files."""
    X_train, y_train, _ = data["train"]
    X_test, y_test, addr_test = data["test"]
    np.random.seed(seed)
    pipe = build_models(seed, scale_pos_weight)[name]

    start = time.perf_counter()
    pipe.fit(X_train, y_train)
    fit_time = time.perf_counter() - start

    metrics, prob, pred = evaluate(pipe, X_test, y_test)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"address": addr_test, "label": y_test, "prob_illicit": prob, "prediction": pred}).to_csv(
        PREDICTIONS_DIR / f"{run_id(name, seed)}.csv", index=False)

    write_json(RUNS_DIR / f"{run_id(name, seed)}.json", {
        "model": name,
        "seed": seed,
        **{k: float(v) for k, v in metrics.items()},
        "fit_time_s": fit_time,
        "n_test": int(len(y_test)),
        "git_commit": commit,
        "finished": timestamp(),
    })
    return run_id(name, seed)


def train_missing_runs(data, seeds, workers):
    _, y_train, _ = data["train"]
    scale_pos_weight = float(np.sum(y_train == 0) / max(1, np.sum(y_train == 1)))
    commit = git_commit(config.PROJECT_ROOT)

    model_names = list(build_models(0, scale_pos_weight))
    runs = [(name, seed) for seed in seeds for name in model_names]
    pending = [(name, seed) for name, seed in runs if not (RUNS_DIR / f"{run_id(name, seed)}.json").exists()]
    print(f"{len(runs) - len(pending)} of {len(runs)} runs already completed, {len(pending)} to train "
          f"({workers} parallel single-threaded processes).")
    if pending:
        Parallel(n_jobs=workers, verbose=10)(
            delayed(fit_run)(name, seed, data, scale_pos_weight, commit) for name, seed in pending)


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
            },
            "training_features": TRAINING_FEATURES,
            "models": sorted(build_models(0, 1.0)),
            "hyperparameters": HP,
            "preprocessing": "mutual-information feature selection, log1p and scaling, fitted on the training split",
            "n_seeds": len(seeds),
            "seeds": seeds,
            "evaluation": {
                "test_split": "seed addresses of the test split",
                "metrics": ["accuracy", "precision", "recall", "f1", "auc_roc"],
                "statistical_tests": "wilcoxon_tests.py",
                "interpretation": "the model with the highest mean test F1-score over the seeds, trained with seed "
                                  f"{config.EGO_CENTRIC_INTERPRETATION_SEED}: permutation importance, feature ablation "
                                  "and SHAP on the validation split",
            },
            "resources": {"parallel_processes": workers, "threads_per_process": 1},
        },
        modules=("sklearn", "xgboost", "shap", "numpy", "pandas"),
    ))


def aggregate():
    results = [read_json(p) for p in sorted(RUNS_DIR.glob("*.json"))]
    if not results:
        print("No completed runs to aggregate.")
        return

    runs = pd.DataFrame(results).sort_values(["model", "seed"])
    runs.to_csv(RESULTS_DIR / "ego_runs.csv", index=False)

    summary = summarize_runs(runs, ["model"], METRICS).sort_values("f1_mean", ascending=False)
    summary.to_csv(RESULTS_DIR / "ego_summary.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n" + summary[["model", "n_runs", "accuracy_mean", "accuracy_std", "f1_mean", "f1_std", "f1_median",
                          "auc_roc_mean", "auc_roc_std"]].round(4).to_string(index=False))
    wilcoxon_tests.run_all()


def interpreted_model(data):
    """The best model (highest mean test F1-score over the seeds), trained with
    EGO_CENTRIC_INTERPRETATION_SEED: loaded if it was saved, otherwise trained again
    exactly as its run (same seed, single-threaded) and saved."""
    runs = pd.read_csv(RESULTS_DIR / "ego_runs.csv")
    name = wilcoxon_tests.best_model(runs, column="model")
    seed = config.EGO_CENTRIC_INTERPRETATION_SEED
    if seed not in set(runs.loc[runs["model"] == name, "seed"]):
        raise RuntimeError(f"{name} was not trained with the interpretation seed {seed}.")
    selection = {
        "model": name,
        "seed": seed,
        "criterion": "highest mean test F1-score over the seeds (ties: alphabetical order)",
        "f1_mean": runs.groupby("model")["f1"].mean().round(6).to_dict(),
        "n_seeds": int(runs.loc[runs["model"] == name, "seed"].nunique()),
    }

    model_path = INTERPRETATION_DIR / f"{run_id(name, seed)}.joblib"
    if model_path.exists():
        return name, joblib.load(model_path), selection
    X_train, y_train, _ = data["train"]
    scale_pos_weight = float(np.sum(y_train == 0) / max(1, np.sum(y_train == 1)))
    print(f"Training {name} with seed {seed} for the interpretation...")
    np.random.seed(seed)
    pipe = build_models(seed, scale_pos_weight)[name]
    pipe.fit(X_train, y_train)
    INTERPRETATION_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)
    return name, pipe, selection


def interpretation_inputs(data, pipe):
    """The steps of the interpreted pipeline and the selected features, raw and transformed."""
    X_train_raw = data["train"][0]
    X_val_raw = data["val"][0]
    selector = pipe.named_steps['selector']
    log = pipe.named_steps['log']
    scaler = pipe.named_steps['scaler']
    names = X_train_raw.columns[selector.get_support()].tolist()
    X_train_log = log.transform(selector.transform(X_train_raw))
    X_val_log = log.transform(selector.transform(X_val_raw))
    return {
        "model": pipe.named_steps['model'], "selector": selector, "names": names,
        # Inputs of the model: selected features, log1p and scaling
        "X_train": scaler.transform(X_train_log), "X_val": scaler.transform(X_val_log),
        # Values shown in the SHAP plots: selected features after log1p
        "X_val_log": X_val_log,
    }


def ablation(name, pipe, data, inputs, workers):
    """Ranks the selected features and retrains the model with the k most important ones
    for every k, evaluated with the F1-score (and AUC-ROC) on the validation split."""
    X_val_raw, y_val, _ = data["val"]
    y_train = data["train"][1]
    model_step, names = inputs["model"], inputs["names"]

    if hasattr(model_step, 'coef_'):
        importances = np.abs(model_step.coef_[0])
        importance_type = "|coefficient|"
    elif hasattr(model_step, 'feature_importances_'):
        importances = model_step.feature_importances_
        importance_type = "feature_importance (Gini)"
    else:
        # Every selected feature is permuted in turn on the validation split (the features
        # dropped by the selector have no effect)
        print(f"Computing permutation importance (F1-score) to rank the features of {name}...")
        r = permutation_importance(pipe, X_val_raw, y_val, n_repeats=10, random_state=config.DATA_SEED,
                                   n_jobs=workers, scoring='f1')
        importances = r.importances_mean[inputs["selector"].get_support()]
        importance_type = "permutation importance (mean decrease of the F1-score)"
    pd.DataFrame({"feature": names, "importance": importances}).sort_values(
        "importance", ascending=False).to_csv(INTERPRETATION_DIR / "feature_importance.csv", index=False)

    ordered_indices = np.argsort(importances)[::-1]
    ordered_names = [names[i] for i in ordered_indices]
    print(f"Feature ranking criterion: {importance_type}")
    print(f"Top 5 ranked features: {ordered_names[:5]}")

    X_train, X_val = inputs["X_train"], inputs["X_val"]

    def ablation_scores(k):
        # Retrains the model with the k most important features (each k in its own process)
        subset = ordered_indices[:k]
        model_abl = clone(model_step)
        model_abl.fit(X_train[:, subset], y_train)
        return (f1_score(y_val, model_abl.predict(X_val[:, subset]), zero_division=0),
                roc_auc_score(y_val, model_abl.predict_proba(X_val[:, subset])[:, 1]))

    scores = parallel_map(ablation_scores, list(range(1, len(ordered_indices) + 1)), workers,
                          desc=f"Retraining {name} incrementally", chunks_per_worker=1)
    results = pd.DataFrame({
        'k_features': range(1, len(scores) + 1),
        'f1': [f1 for f1, _ in scores],
        'auc_roc': [auc for _, auc in scores],
        'feature_added': ordered_names,
    })
    results.to_csv(INTERPRETATION_DIR / "ablation.csv", index=False)
    print("\n=== Feature Ablation Study Results ===")
    print(results.to_string(index=False))

    plt.figure(figsize=(7, 6))
    plt.plot(results['k_features'], results['f1'], marker='o', linestyle='-', color='#1f77b4', linewidth=1.5)
    plt.xticks(ticks=results['k_features'], labels=[get_label(f) for f in results['feature_added']],
               rotation=45, ha='right', fontsize=10)
    plt.xlabel('Feature Added (Cumulative)', fontsize=11)
    plt.ylabel("F1-score", fontsize=11)
    plt.title(f"Feature Ablation Study: {name} Performance", fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(INTERPRETATION_DIR / f"ablation_{file_name(name)}.pdf", bbox_inches='tight')
    plt.close()
    return {"importance_ranking": importance_type, "ablation_metric": "F1-score (and AUC-ROC) on the validation split"}


def kernel_shap_values(explainer, X, workers):
    """Kernel SHAP values of the illicit class, computed in fixed chunks of addresses in
    parallel (the chunks and their seeds do not depend on the number of workers)."""
    def chunk_values(start):
        # With few features every coalition is enumerated and the values are exact;
        # otherwise the coalitions are sampled, with a seed per chunk
        np.random.seed(config.DATA_SEED + start)
        values = explainer.shap_values(X[start:start + SHAP_CHUNK_SIZE], nsamples="auto", silent=True)
        # SHAP returns one array per class (a list), or a single array with the classes last
        return np.asarray(values[1] if isinstance(values, list) else np.asarray(values)[..., 1])

    chunks = parallel_map(chunk_values, list(range(0, len(X), SHAP_CHUNK_SIZE)), workers,
                          desc="SHAP values", chunks_per_worker=1)
    return np.concatenate(chunks, axis=0)


def shap_analysis(name, inputs, workers):
    """SHAP values of the interpreted model for every validation address, with the exact
    explainer of its family when there is one."""
    import shap
    model_step, names = inputs["model"], inputs["names"]
    X_train, X_val = inputs["X_train"], inputs["X_val"]
    labels = [FEATURE_NAMES_LATEX.get(n, n) for n in names]
    model_type = type(model_step).__name__
    n_features = X_val.shape[1]
    print(f"\nStarting SHAP analysis of {name} ({model_type}) on {len(X_val)} validation addresses...")

    # Background distribution of the features: a sample of the training split (fixed seed)
    sign = 1.0
    if model_type in ('RandomForestClassifier', 'XGBClassifier', 'IsolationForestClassifier'):
        # Exact tree SHAP, interventional on 100 training addresses (the cost grows with the
        # background size, and 100 addresses estimate the expected output well)
        background = shap.sample(X_train, 100, random_state=config.DATA_SEED)
        if model_type == 'IsolationForestClassifier':
            # Its probability is rescaled within every batch, so the anomaly score is
            # explained: the path length of the isolation forest, with the sign reversed
            # so that positive values push towards the anomalous (illicit) class
            explainer = shap.TreeExplainer(model_step.model_, background)
            sign = -1.0
            output = "anomaly score (negated isolation forest output)"
        else:
            explainer = shap.TreeExplainer(model_step, background)
            output = "log-odds of the illicit class" if model_type == 'XGBClassifier' else "probability of the illicit class"
        explanation = explainer(X_val, check_additivity=False)
        values = np.asarray(explanation.values)
        base = np.asarray(explanation.base_values)
        if values.ndim == 3:
            # One output per class: the illicit class
            values, base = values[..., 1], base[..., 1]
        method = "TreeExplainer (exact, interventional, 100 background addresses)"
    elif model_type == 'LogisticRegression':
        # Exact for a linear model: depends only on the mean of the training split
        explainer = shap.LinearExplainer(model_step, shap.maskers.Independent(X_train, max_samples=len(X_train)))
        explanation = explainer(X_val)
        values, base = np.asarray(explanation.values), np.asarray(explanation.base_values)
        output = "log-odds of the illicit class"
        method = "LinearExplainer (exact, independent features, training mean)"
    else:
        # MLP, SVM and Naive Bayes: model-agnostic kernel SHAP on the predicted probability,
        # with a background of 50 k-means centroids of the training split (deterministic)
        explainer = shap.KernelExplainer(model_step.predict_proba, shap.kmeans(X_train, 50))
        values = kernel_shap_values(explainer, X_val, workers)
        base = np.full(len(values), explainer.expected_value[1])
        output = "probability of the illicit class"
        exact = 2 * n_features + 2048 >= 2 ** n_features - 2
        method = ("KernelExplainer (50 k-means centroids, "
                  + ("every feature coalition enumerated: exact for this background)" if exact else "sampled coalitions)"))

    values, base = sign * values, sign * base
    explanation = shap.Explanation(values=values, base_values=base, data=inputs["X_val_log"], feature_names=labels)

    pd.DataFrame(values, columns=names).to_csv(INTERPRETATION_DIR / "shap_values.csv", index=False)
    pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(values).mean(axis=0)}).sort_values(
        "mean_abs_shap", ascending=False).to_csv(INTERPRETATION_DIR / "shap_importance.csv", index=False)

    for plot_type, title in [("dot", "Feature Impact (Beeswarm)"), ("bar", "Global Feature Importance")]:
        shap.summary_plot(explanation, plot_type=plot_type, show=False)
        plt.title(f'{title} - Model: {name}')
        plt.savefig(INTERPRETATION_DIR / f"shap_summary_{plot_type}_{file_name(name)}.pdf", bbox_inches='tight')
        plt.close()
    return {"shap_explainer": method, "shap_output": output, "shap_addresses": int(len(values))}


def interpret(data, workers):
    """Feature ablation and SHAP analysis of the best model, both on the validation split.
    Each part is saved when it finishes and is not computed again for the same model, so
    an interrupted interpretation resumes from the part that was missing."""
    name, pipe, selection = interpreted_model(data)
    info_path = INTERPRETATION_DIR / "interpretation.json"
    info = read_json(info_path) if info_path.exists() else {}
    if info.get("model") != name:
        # Results of another model (e.g. the best one with fewer seeds) are removed; the
        # trained models are kept
        for path in INTERPRETATION_DIR.iterdir():
            if path.suffix != ".joblib" and path.is_file():
                path.unlink()
        info = {**selection, "created": timestamp()}
        write_json(info_path, info)
    print(f"\nModel {name} selected for explainability and feature ablation analysis "
          f"(highest mean F1-score over {selection['n_seeds']} seeds: {selection['f1_mean'][name]:.4f}).")

    inputs = interpretation_inputs(data, pipe)
    if info.get("ablation") and (INTERPRETATION_DIR / "ablation.csv").exists():
        print(f"Feature ablation of {name} already computed.")
    else:
        print(f"\nStarting feature ablation study ({name}, on the validation split)...")
        info["ablation"] = ablation(name, pipe, data, inputs, workers)
        write_json(info_path, info)

    if info.get("shap") and (INTERPRETATION_DIR / "shap_values.csv").exists():
        print(f"SHAP analysis of {name} already computed.")
    else:
        info["shap"] = shap_analysis(name, inputs, workers)
        write_json(info_path, info)
    print(f"Interpretation saved at: {INTERPRETATION_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Train, evaluate and interpret the ego-centric models.")
    parser.add_argument("--n-seeds", type=int, help="random-seed runs per model (asked if not given)")
    parser.add_argument("--workers", type=int, default=config.EGO_CENTRIC_MODEL_WORKERS, help="parallel training processes")
    parser.add_argument("--aggregate-only", action="store_true", help="only rebuild the result tables and tests")
    parser.add_argument("--skip-interpretation", action="store_true", help="skip the ablation and SHAP analyses")
    parser.add_argument("--interpret-only", action="store_true",
                        help="only the ablation and SHAP analyses of the best model of the trained runs")
    args = parser.parse_args()
    if args.interpret_only and (args.aggregate_only or args.skip_interpretation):
        parser.error("--interpret-only cannot be combined with --aggregate-only or --skip-interpretation")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_dataset()
    for split in config.SPLITS:
        print(f"{split:5s}: {len(data[split][1])} seed addresses")

    if args.interpret_only:
        if not (RESULTS_DIR / "ego_runs.csv").exists():
            raise SystemExit(f"Missing {RESULTS_DIR / 'ego_runs.csv'}: train the models first.")
        interpret(data, args.workers)
        print(f"\nResults saved at: {RESULTS_DIR}")
        return

    if not args.aggregate_only:
        seeds = training_seeds(ask_n_seeds(args.n_seeds))
        save_experiment_config(seeds, args.workers)
        train_missing_runs(data, seeds, args.workers)
    aggregate()
    if not args.aggregate_only and not args.skip_interpretation:
        interpret(data, args.workers)
    print(f"\nResults saved at: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
