# STEP 3 (Methodology II) - MODEL TRAINING, 5-FOLD CROSS-VALIDATION, EVALUATION, HYPERPARAMETER SEARCH AND INTERPRETABILITY
#
# Asks how many random-seed runs to execute, trains the eight tabular models on
# the topological features (EGO_CENTRIC_FEATURES_DIR/<split>.csv, written by
# extract_topological_features.py) with that many seeds using the tuned
# hyperparameters, evaluates them with 5-fold Stratified Cross-Validation and on
# the test split, runs the Wilcoxon tests (wilcoxon_tests.py) and analyzes the
# best model, the one with the highest mean test F1-score over the seeds
# (wilcoxon_tests.best_model):
#
#   python models.py                         # asks the number of seeds
#   python models.py --n-seeds 10            # without asking
#   python models.py --skip-interpretation   # without the ablation and SHAP analyses
#   python models.py --interpret-only        # only the ablation and SHAP analyses (of the trained runs)
#   python models.py --aggregate-only        # only rebuild the result tables and tests
#   python models.py --tune-only             # run only the 5-fold CV hyperparameter search
#   python models.py --tune --n-seeds 10     # run hyperparameter search first, then multi-seed evaluation
#
# Output in EGO_CENTRIC_RESULTS_DIR:
#   experiment_config.json                full configuration of the experiment (data, expansion,
#                                         features, models, evaluation, seeds, resources)
#   runs/<model>-seed<seed>.json          5-fold CV and test metrics of every run
#   predictions/<model>-seed<seed>.csv    test addresses with label and predicted probability
#   ego_runs.csv                          one row per run (test + 5-fold CV metrics)
#   ego_cv_folds.csv                      one row per (model, seed, fold) from 5-fold CV
#   ego_summary.csv                       mean, standard deviation and median per model (test + CV)
#   ego_cv_summary.csv                    5-fold CV summary (mean, std, median per model)
#   tuning_results_pro/                   optimization_summary.csv, best_hyperparameters_pro.json, models/*.joblib
#   interpretation/                       permutation importance, ablation curve and SHAP values of the
#                                         best model trained with EGO_CENTRIC_INTERPRETATION_SEED, and
#                                         interpretation.json (which model and why)

import argparse
import contextlib
import copy
import json
import shutil
import sys
import time
import warnings

from functools import partial
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
for _p in (str(CURRENT_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Every training run, ablation retraining and SHAP chunk runs in its own process (one
# per core), so the numerical libraries are single-threaded (before importing them):
# no oversubscription, no OpenMP runtime started before forking (which can hang the
# workers), and a model trained again for the interpretation is identical to its run
from shared.threads import limit_threads

import joblib
from joblib import Parallel, delayed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    HalvingRandomSearchCV,
    ParameterGrid,
    PredefinedSplit,
    StratifiedKFold,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

import config
import wilcoxon_tests
from shared.experiment import (ask_n_seeds, experiment_snapshot, git_commit, read_json, timestamp,
                               training_seeds, write_json)
from shared.parallel import parallel_map
from shared.stats import summarize_runs

try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Validation addresses explained by each SHAP worker task. The chunks and their
# random seeds do not depend on the number of workers, so the SHAP values are
# the same with any number of cores.
SHAP_CHUNK_SIZE = 100

# Number of folds for Stratified Cross-Validation during model evaluation
CV_FOLDS = 5

RESULTS_DIR = config.EGO_CENTRIC_RESULTS_DIR
RUNS_DIR = RESULTS_DIR / "runs"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
INTERPRETATION_DIR = RESULTS_DIR / "interpretation"
DEFAULT_TUNING_DIR = RESULTS_DIR / "tuning_results_pro"

EVAL_METRICS = ["accuracy", "precision", "recall", "f1", "auc_roc"]
CV_METRICS = [f"cv_{m}" for m in EVAL_METRICS] + ["cv_fit_time_s"]
METRICS = EVAL_METRICS + ["fit_time_s"]

# Features and hyperparameters of the models (config.py)
TRAINING_FEATURES = config.EGO_CENTRIC_TRAINING_FEATURES
HP = config.EGO_CENTRIC_HYPERPARAMETERS

# Complete tuned hyperparameters per model (8 base tabular models, no ensembles or engineered features)
TUNED_HYPERPARAMETERS = {
    "RandomForest": {
        "search_type": "halving",
        "selector__percentile": 90,
        "scaler": "RobustScaler",
        "model__max_depth": int(HP.get("rf_max_depth", 12)),
        "model__max_features": float(HP.get("rf_max_features", 0.85)),
        "model__min_samples_leaf": int(HP.get("rf_min_samples_leaf", 2)),
        "model__min_samples_split": int(HP.get("rf_min_samples_split", 3)),
        "model__n_estimators": int(HP.get("rf_n_estimators", 417)),
        "class_weight": "balanced",
        "optimal_threshold": 0.4861,
    },
    "XGBoost": {
        "search_type": "halving",
        "selector__percentile": 100,
        "scaler": "RobustScaler",
        "model__colsample_bytree": float(HP.get("xgb_colsample_bytree", 0.9026053324077057)),
        "model__gamma": float(HP.get("xgb_gamma", 0.11799245987447787)),
        "model__learning_rate": float(HP.get("xgb_learning_rate", 0.018894879719012592)),
        "model__max_depth": int(HP.get("xgb_max_depth", 8)),
        "model__min_child_weight": int(HP.get("xgb_min_child_weight", 1)),
        "model__n_estimators": int(HP.get("xgb_n_estimators", 406)),
        "model__scale_pos_weight": float(HP.get("xgb_scale_pos_weight", 1.2)),
        "model__subsample": float(HP.get("xgb_subsample", 0.6888117872841409)),
        "tree_method": "hist",
        "eval_metric": "logloss",
        "optimal_threshold": 0.4648,
    },
    "MLP": {
        "search_type": "grid",
        "selector__percentile": 100,
        "scaler": "RobustScaler",
        "model__alpha": float(HP.get("mlp_alpha", 0.005)),
        "model__hidden_layer_sizes": tuple(HP.get("mlp_hidden_layers", (256, 128))),
        "model__learning_rate_init": float(HP.get("mlp_learning_rate_init", 0.003)),
        "solver": str(HP.get("mlp_solver", "adam")),
        "max_iter": int(HP.get("mlp_max_iter", 450)),
        "early_stopping": True,
        "n_iter_no_change": 20,
        "validation_fraction": 0.1,
        "optimal_threshold": 0.5016,
    },
    "SVM": {
        "search_type": "grid",
        "selector__percentile": 100,
        "scaler": "RobustScaler",
        "model__C": float(HP.get("svm_C", 8.0)),
        "kernel": str(HP.get("svm_kernel", "rbf")),
        "class_weight": "balanced",
        "probability": True,
        "cache_size": 1000,
        "optimal_threshold": 0.4188,
    },
    "L1 (Lasso)": {
        "search_type": "grid",
        "selector__percentile": 100,
        "scaler": "RobustScaler",
        "model__C": float(HP.get("logreg_C", 0.05)),
        "penalty": str(HP.get("logreg_penalty_l1", "l1")),
        "solver": "saga",
        "class_weight": "balanced",
        "max_iter": int(HP.get("logreg_max_iter", 2500)),
        "optimal_threshold": 0.4627,
    },
    "ElasticNet": {
        "search_type": "grid",
        "selector__percentile": 100,
        "scaler": "StandardScaler",
        "model__C": float(HP.get("logreg_C_el", 0.01)),
        "model__l1_ratio": float(HP.get("logreg_l1_ratio", 0.5)),
        "penalty": str(HP.get("logreg_penalty_el", "elasticnet")),
        "solver": "saga",
        "class_weight": "balanced",
        "max_iter": int(HP.get("logreg_max_iter", 2500)),
        "optimal_threshold": 0.471,
    },
    "Naive Bayes": {
        "search_type": "grid",
        "selector__percentile": 100,
        "scaler": "RobustScaler",
        "model__var_smoothing": float(HP.get("nb_var_smoothing", 1e-10)),
        "optimal_threshold": 0.1292,
    },
    "IsolationForest": {
        "search_type": "grid",
        "selector__percentile": 75,
        "scaler": "RobustScaler",
        "model__contamination": float(HP["iso_contamination"]) if HP.get("iso_contamination") != "auto" else 0.42,
        "model__flip_score": bool(HP.get("iso_flip_score", True)),
        "model__max_samples": HP.get("iso_max_samples", "auto"),
        "model__n_estimators": int(HP.get("iso_n_estimators", 400)),
        "optimal_threshold": 0.8626,
    },
}

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


class TunableIsolationForestClassifier(BaseEstimator, ClassifierMixin):
    """Isolation Forest as a classifier with tunable score polarity and parameters."""
    def __init__(self, n_estimators=400, contamination=0.42, max_samples='auto',
                 flip_score=True, random_state=None, n_jobs=1):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.flip_score = flip_score
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        ).fit(X)
        raw = self.model_.decision_function(X)
        scores = raw if self.flip_score else -raw
        self.score_min_ = float(np.min(scores)) if len(scores) > 0 else 0.0
        self.score_max_ = float(np.max(scores)) if len(scores) > 0 else 1.0
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        pred = self.model_.predict(X)
        return (pred == 1).astype(int) if self.flip_score else (pred == -1).astype(int)

    def predict_proba(self, X):
        raw = self.model_.decision_function(X)
        scores = raw if self.flip_score else -raw
        if len(scores) == 0:
            return np.array([]).reshape(0, 2)
        span = scores.max() - scores.min()
        if span == 0:
            train_span = getattr(self, "score_max_", 1.0) - getattr(self, "score_min_", 0.0)
            if train_span > 0:
                scaled = np.clip((scores - self.score_min_) / train_span, 0.0, 1.0)
            else:
                scaled = np.full_like(scores, 0.5)
        else:
            scaled = (scores - scores.min()) / span
        return np.vstack((1 - scaled, scaled)).T


# Backward-compatible alias for IsolationForestClassifier
IsolationForestClassifier = TunableIsolationForestClassifier


def get_model_threshold(model_name):
    """Returns the tuned decision threshold for `model_name` (defaults to 0.5)."""
    cfg = TUNED_HYPERPARAMETERS.get(model_name, {})
    return float(cfg.get("optimal_threshold", 0.5))


def load_hyperparameters_from_json(json_path):
    """Loads a `best_hyperparameters_pro.json` file and updates `TUNED_HYPERPARAMETERS` in-place
    for the 8 base models (ignoring any ensemble or feature-engineering keys)."""
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Hyperparameter file not found: {path}")
    with open(path, encoding="utf-8") as f:
        loaded = json.load(f)
    for name, params in loaded.items():
        if name not in TUNED_HYPERPARAMETERS:
            continue
        filtered = {k: v for k, v in params.items() if k in TUNED_HYPERPARAMETERS[name]}
        TUNED_HYPERPARAMETERS[name].update(filtered)
        if "model__hidden_layer_sizes" in TUNED_HYPERPARAMETERS[name]:
            val = TUNED_HYPERPARAMETERS[name]["model__hidden_layer_sizes"]
            if isinstance(val, list):
                TUNED_HYPERPARAMETERS[name]["model__hidden_layer_sizes"] = tuple(val)
    return TUNED_HYPERPARAMETERS


def _make_scaler(scaler_spec):
    if isinstance(scaler_spec, str):
        return RobustScaler() if scaler_spec == "RobustScaler" else StandardScaler()
    return clone(scaler_spec)


def build_models(seed, scale_pos_weight):
    """The eight pipelines configured with the tuned hyperparameters from
    `TUNED_HYPERPARAMETERS`, with every random component seeded with `seed`.
    Every model is single-threaded; the runs are parallelized across processes instead."""
    mutual_info = partial(mutual_info_classif, random_state=seed)

    def pipeline_for(model_name, estimator):
        cfg = TUNED_HYPERPARAMETERS[model_name]
        percentile = int(cfg.get("selector__percentile", HP.get("selector_percentile", 100)))
        scaler = _make_scaler(cfg.get("scaler", HP.get("scaler", "RobustScaler")))
        return Pipeline([
            ('selector', SelectPercentile(mutual_info, percentile=percentile)),
            ('log', FunctionTransformer(np.log1p, validate=True)),
            ('scaler', scaler),
            ('model', estimator),
        ])

    rf_cfg = TUNED_HYPERPARAMETERS["RandomForest"]
    xgb_cfg = TUNED_HYPERPARAMETERS["XGBoost"]
    mlp_cfg = TUNED_HYPERPARAMETERS["MLP"]
    svm_cfg = TUNED_HYPERPARAMETERS["SVM"]
    l1_cfg = TUNED_HYPERPARAMETERS["L1 (Lasso)"]
    el_cfg = TUNED_HYPERPARAMETERS["ElasticNet"]
    nb_cfg = TUNED_HYPERPARAMETERS["Naive Bayes"]
    iso_cfg = TUNED_HYPERPARAMETERS["IsolationForest"]

    xgb_spw = float(xgb_cfg.get("model__scale_pos_weight", scale_pos_weight))

    return {
        'L1 (Lasso)': pipeline_for(
            'L1 (Lasso)',
            LogisticRegression(
                penalty=l1_cfg.get("penalty", "l1"),
                solver=l1_cfg.get("solver", "saga"),
                max_iter=int(l1_cfg.get("max_iter", 2500)),
                C=float(l1_cfg.get("model__C", 0.05)),
                class_weight=l1_cfg.get("class_weight", "balanced"),
                random_state=seed,
            ),
        ),
        'ElasticNet': pipeline_for(
            'ElasticNet',
            LogisticRegression(
                penalty=el_cfg.get("penalty", "elasticnet"),
                solver=el_cfg.get("solver", "saga"),
                l1_ratio=float(el_cfg.get("model__l1_ratio", 0.5)),
                max_iter=int(el_cfg.get("max_iter", 2500)),
                C=float(el_cfg.get("model__C", 0.01)),
                class_weight=el_cfg.get("class_weight", "balanced"),
                random_state=seed,
            ),
        ),
        'RandomForest': pipeline_for(
            'RandomForest',
            RandomForestClassifier(
                n_estimators=int(rf_cfg.get("model__n_estimators", 417)),
                max_depth=rf_cfg.get("model__max_depth", 12),
                max_features=rf_cfg.get("model__max_features", 0.85),
                min_samples_split=int(rf_cfg.get("model__min_samples_split", 3)),
                min_samples_leaf=int(rf_cfg.get("model__min_samples_leaf", 2)),
                class_weight=rf_cfg.get("class_weight", "balanced"),
                random_state=seed,
                n_jobs=1,
            ),
        ),
        'XGBoost': pipeline_for(
            'XGBoost',
            XGBClassifier(
                n_estimators=int(xgb_cfg.get("model__n_estimators", 406)),
                max_depth=int(xgb_cfg.get("model__max_depth", 8)),
                learning_rate=float(xgb_cfg.get("model__learning_rate", 0.018894879719012592)),
                subsample=float(xgb_cfg.get("model__subsample", 0.6888117872841409)),
                colsample_bytree=float(xgb_cfg.get("model__colsample_bytree", 0.9026053324077057)),
                min_child_weight=int(xgb_cfg.get("model__min_child_weight", 1)),
                gamma=float(xgb_cfg.get("model__gamma", 0.11799245987447787)),
                scale_pos_weight=xgb_spw,
                tree_method=xgb_cfg.get("tree_method", "hist"),
                eval_metric=xgb_cfg.get("eval_metric", "logloss"),
                random_state=seed,
                n_jobs=1,
            ),
        ),
        'IsolationForest': pipeline_for(
            'IsolationForest',
            TunableIsolationForestClassifier(
                n_estimators=int(iso_cfg.get("model__n_estimators", 400)),
                contamination=iso_cfg.get("model__contamination", 0.42),
                max_samples=iso_cfg.get("model__max_samples", "auto"),
                flip_score=bool(iso_cfg.get("model__flip_score", True)),
                random_state=seed,
                n_jobs=1,
            ),
        ),
        'SVM': pipeline_for(
            'SVM',
            SVC(
                C=float(svm_cfg.get("model__C", 8.0)),
                kernel=svm_cfg.get("kernel", "rbf"),
                class_weight=svm_cfg.get("class_weight", "balanced"),
                probability=True,
                cache_size=int(svm_cfg.get("cache_size", 1000)),
                random_state=seed,
            ),
        ),
        'Naive Bayes': pipeline_for(
            'Naive Bayes',
            GaussianNB(
                var_smoothing=float(nb_cfg.get("model__var_smoothing", 1e-10)),
            ),
        ),
        'MLP': pipeline_for(
            'MLP',
            MLPClassifier(
                hidden_layer_sizes=tuple(mlp_cfg.get("model__hidden_layer_sizes", (256, 128))),
                alpha=float(mlp_cfg.get("model__alpha", 0.005)),
                max_iter=int(mlp_cfg.get("max_iter", 450)),
                solver=mlp_cfg.get("solver", "adam"),
                learning_rate_init=float(mlp_cfg.get("model__learning_rate_init", 0.003)),
                random_state=seed,
                early_stopping=bool(mlp_cfg.get("early_stopping", True)),
                n_iter_no_change=int(mlp_cfg.get("n_iter_no_change", 20)),
                validation_fraction=float(mlp_cfg.get("validation_fraction", 0.1)),
            ),
        ),
    }


# =====================================================================
# HYPERPARAMETER SEARCH MODE
# =====================================================================

def estimate_total_fits(mode, n_candidates, space, n_splits):
    if mode == "grid":
        return len(ParameterGrid(space)) * n_splits
    total = 0
    cand = n_candidates
    while cand >= 1:
        total += cand * n_splits
        if cand <= 3:
            break
        cand = cand // 3
    return total


@contextlib.contextmanager
def tqdm_joblib(desc, total):
    if not HAS_TQDM:
        yield None
        return

    pbar = tqdm(total=total, desc=desc, unit="fit", dynamic_ncols=True, leave=True)

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            pbar.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield pbar
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        pbar.close()


def find_optimal_threshold(probs_val, y_val):
    precisions, recalls, thresholds = precision_recall_curve(y_val, probs_val)
    if len(thresholds) == 0:
        return 0.5
    f1_scores = (2 * precisions[:-1] * recalls[:-1]) / np.maximum(precisions[:-1] + recalls[:-1], 1e-12)
    best_idx = int(np.argmax(f1_scores))
    return float(np.clip(thresholds[best_idx], 0.05, 0.95))


def optimal_f1_score(y_true, y_prob):
    """Maximum F1 along the Precision-Recall curve (invariant to 0.5 threshold miscalibration)."""
    probs = y_prob[:, 1] if getattr(y_prob, "ndim", 1) == 2 else y_prob
    thr = find_optimal_threshold(probs, y_true)
    return float(f1_score(y_true, (probs >= thr).astype(int), zero_division=0))


def build_scorer(scoring_mode="f1_05"):
    if scoring_mode == "f1_opt":
        return make_scorer(optimal_f1_score, response_method="predict_proba")
    if scoring_mode == "roc_auc":
        return "roc_auc"
    return make_scorer(f1_score, zero_division=0)


def compute_tuning_metrics(y_true, probs, threshold=0.5):
    pred = (probs >= threshold).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "auc_roc": round(float(roc_auc_score(y_true, probs)), 4),
    }


def clean_params(params):
    cleaned = {}
    for k, v in params.items():
        if k == "scaler":
            cleaned[k] = type(v).__name__
        elif isinstance(v, (np.floating, np.integer)):
            cleaned[k] = v.item()
        else:
            cleaned[k] = v
    return cleaned


def build_search_configs(seed, scale_pos_weight, illicit_ratio, cache_dir):
    mutual_info = partial(mutual_info_classif, random_state=seed)
    mem = joblib.Memory(location=str(cache_dir), verbose=0)

    def make_pipe(model):
        return Pipeline([
            ('selector', SelectPercentile(mutual_info, percentile=100)),
            ('log', FunctionTransformer(np.log1p, validate=True)),
            ('scaler', RobustScaler()),
            ('model', model),
        ], memory=mem)

    models = {
        'RandomForest': make_pipe(RandomForestClassifier(class_weight='balanced', random_state=seed, n_jobs=1)),
        'XGBoost': make_pipe(XGBClassifier(tree_method='hist', scale_pos_weight=scale_pos_weight, eval_metric='logloss', random_state=seed, n_jobs=1)),
        'MLP': make_pipe(MLPClassifier(solver='adam', random_state=seed, early_stopping=True, n_iter_no_change=20, validation_fraction=0.1, max_iter=450)),
        'SVM': make_pipe(SVC(kernel='rbf', class_weight='balanced', probability=True, cache_size=1000, random_state=seed)),
        'L1 (Lasso)': make_pipe(LogisticRegression(penalty='l1', solver='saga', class_weight='balanced', max_iter=2500, random_state=seed)),
        'ElasticNet': make_pipe(LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, class_weight='balanced', max_iter=2500, random_state=seed)),
        'Naive Bayes': make_pipe(GaussianNB()),
        'IsolationForest': make_pipe(TunableIsolationForestClassifier(random_state=seed, n_jobs=1)),
    }

    search_spaces = {
        'RandomForest': ("halving", 45, {
            'selector__percentile': [80, 90, 100],
            'model__n_estimators': randint(200, 650),
            'model__max_depth': [8, 12, 18, 26, None],
            'model__min_samples_split': randint(2, 12),
            'model__min_samples_leaf': randint(1, 5),
            'model__max_features': ['sqrt', 0.5, 0.7, 0.85],
        }),
        'XGBoost': ("halving", 60, {
            'selector__percentile': [70, 80, 90, 100],
            'model__n_estimators': randint(200, 650),
            'model__max_depth': randint(6, 14),
            'model__learning_rate': loguniform(0.01, 0.12),
            'model__subsample': uniform(0.65, 0.35),
            'model__colsample_bytree': uniform(0.65, 0.35),
            'model__min_child_weight': randint(1, 8),
            'model__gamma': uniform(0.0, 0.5),
            'model__scale_pos_weight': [1.0, scale_pos_weight, round(scale_pos_weight * 1.2, 4), round(scale_pos_weight * 1.4, 4)],
        }),
        'MLP': ("grid", None, {
            'selector__percentile': [90, 100],
            'scaler': [RobustScaler(), StandardScaler()],
            'model__hidden_layer_sizes': [(128, 64, 32), (256, 128), (256, 128, 64)],
            'model__alpha': [5e-4, 1e-3, 5e-3],
            'model__learning_rate_init': [1e-3, 2e-3, 3e-3],
        }),
        'SVM': ("grid", None, {
            'selector__percentile': [100],
            'scaler': [RobustScaler(), StandardScaler()],
            'model__C': [2.0, 8.0, 16.0, 32.0],
        }),
        'L1 (Lasso)': ("grid", None, {
            'selector__percentile': [70, 85, 100],
            'scaler': [RobustScaler(), StandardScaler()],
            'model__C': [0.01, 0.05, 0.2, 1.0, 5.0, 20.0],
        }),
        'ElasticNet': ("grid", None, {
            'selector__percentile': [75, 90, 100],
            'scaler': [RobustScaler(), StandardScaler()],
            'model__C': [0.01, 0.1, 0.5, 2.0, 10.0],
            'model__l1_ratio': [0.2, 0.5, 0.8],
        }),
        'Naive Bayes': ("grid", None, {
            'selector__percentile': [50, 65, 80, 90, 100],
            'scaler': [RobustScaler(), StandardScaler()],
            'model__var_smoothing': np.logspace(-10, -1, 10).tolist(),
        }),
        'IsolationForest': ("grid", None, {
            'selector__percentile': [75, 100],
            'model__n_estimators': [200, 400],
            'model__max_samples': ['auto', 0.5],
            'model__contamination': ['auto', round(illicit_ratio * 0.85, 2), round(illicit_ratio, 2)],
            'model__flip_score': [False, True],
        }),
    }
    return models, search_spaces


def run_hyperparameter_search(
    data,
    out_dir=DEFAULT_TUNING_DIR,
    cv_mode="kfold",
    folds=CV_FOLDS,
    scoring_mode="f1_05",
    models_filter=None,
    workers=config.EGO_CENTRIC_MODEL_WORKERS,
    seed=config.DATA_SEED,
):
    """Executes GridSearchCV / HalvingRandomSearchCV + threshold optimization and 5-fold CV
    evaluation for the eight tabular models, saves `optimization_summary.csv` and
    `best_hyperparameters_pro.json`, and updates `TUNED_HYPERPARAMETERS` in memory."""
    out_dir = Path(out_dir)
    cache_dir = out_dir / "_cache"
    models_dir = out_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, _ = data["train"]
    X_val, y_val, _ = data["val"]
    X_test, y_test, _ = data["test"]

    scale_pos_weight = float(np.sum(y_train == 0) / max(1, np.sum(y_train == 1)))
    illicit_ratio = float(np.mean(y_train == 1))
    print(f"\n[Hyperparameter Search] Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)} "
          f"(Illicit train: {illicit_ratio:.1%} | Workers: {workers} | CV: {cv_mode} ({folds} folds) | Scoring: {scoring_mode})")

    if cv_mode == "val":
        X_tune = pd.concat([X_train, X_val], ignore_index=True)
        y_tune = np.concatenate([y_train, y_val])
        cv_split = PredefinedSplit(np.concatenate([np.full(len(y_train), -1), np.zeros(len(y_val), dtype=int)]))
        n_splits = 1
    else:
        X_tune, y_tune = X_train, y_train
        cv_split = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        n_splits = folds

    base_models, search_spaces = build_search_configs(seed, scale_pos_weight, illicit_ratio, cache_dir)
    scorer = build_scorer(scoring_mode)
    selected_models = [m for m in (models_filter if models_filter else list(base_models.keys())) if m in base_models]

    summary_rows = []
    best_configs = {}
    global_t0 = time.perf_counter()

    try:
        for idx, name in enumerate(selected_models, start=1):
            pipe = base_models[name]
            mode, n_candidates, space = search_spaces[name]
            est_fits = estimate_total_fits(mode, n_candidates, space, n_splits)
            t0 = time.perf_counter()

            if mode == "halving":
                search = HalvingRandomSearchCV(
                    estimator=pipe,
                    param_distributions=space,
                    n_candidates=n_candidates,
                    factor=3,
                    resource="n_samples",
                    min_resources=max(2000, len(X_tune) // 9),
                    scoring=scorer,
                    cv=cv_split,
                    random_state=seed,
                    n_jobs=workers,
                    refit=False,
                )
            else:
                search = GridSearchCV(
                    estimator=pipe,
                    param_grid=space,
                    scoring=scorer,
                    cv=cv_split,
                    n_jobs=workers,
                    refit=False,
                )

            desc = f"[{idx}/{len(selected_models)}] {name:15s} ({mode})"
            with tqdm_joblib(desc=desc, total=est_fits):
                search.fit(X_tune, y_tune)

            best_pipe = clone(pipe).set_params(memory=None, **search.best_params_)
            model_step = best_pipe.named_steps["model"]
            if hasattr(model_step, "n_jobs") and not isinstance(model_step, LogisticRegression):
                best_pipe.set_params(model__n_jobs=workers)

            best_pipe.fit(X_train, y_train)
            elapsed = round(time.perf_counter() - t0, 1)

            probs_val = best_pipe.predict_proba(X_val)[:, 1]
            probs_test = best_pipe.predict_proba(X_test)[:, 1]

            opt_thr = find_optimal_threshold(probs_val, y_val)
            val_m_05 = compute_tuning_metrics(y_val, probs_val, 0.5)
            val_m_opt = compute_tuning_metrics(y_val, probs_val, opt_thr)
            test_m_05 = compute_tuning_metrics(y_test, probs_test, 0.5)
            test_m_opt = compute_tuning_metrics(y_test, probs_test, opt_thr)

            if hasattr(model_step, "n_jobs") and not isinstance(model_step, LogisticRegression):
                best_pipe.set_params(model__n_jobs=1)

            cv_summary, _ = cross_validate_model(
                best_pipe, X_train, y_train, seed=seed, threshold=opt_thr, n_splits=CV_FOLDS
            )

            clean_p = clean_params(search.best_params_)
            clean_p["optimal_threshold"] = round(opt_thr, 4)
            best_configs[name] = clean_p

            safe_name = file_name(name)
            joblib.dump({"pipeline": best_pipe, "threshold": opt_thr, "params": clean_p},
                        models_dir / f"{safe_name}.joblib")

            print(f"    -> {name:15s} finished in {elapsed:5.1f}s | "
                  f"{CV_FOLDS}-Fold CV F1: {cv_summary['cv_f1']:.4f} (±{cv_summary['cv_f1_std']:.4f}) | "
                  f"Test F1 (@0.5): {test_m_05['f1']:.4f} -> Test F1 (@{opt_thr:.2f}): {test_m_opt['f1']:.4f} | "
                  f"Test AUC: {test_m_opt['auc_roc']:.4f}")

            summary_rows.append({
                "model": name,
                "search_type": mode,
                "opt_threshold": round(opt_thr, 4),
                "cv_f1_mean": round(cv_summary["cv_f1"], 4),
                "cv_f1_std": round(cv_summary["cv_f1_std"], 4),
                "cv_auc_roc_mean": round(cv_summary["cv_auc_roc"], 4),
                "val_f1_05": val_m_05["f1"],
                "val_f1_opt": val_m_opt["f1"],
                "test_f1_05": test_m_05["f1"],
                "test_f1_opt": test_m_opt["f1"],
                "test_auc_roc": test_m_opt["auc_roc"],
                "test_precision_opt": test_m_opt["precision"],
                "test_recall_opt": test_m_opt["recall"],
                "test_accuracy_opt": test_m_opt["accuracy"],
                "time_s": elapsed,
                "best_params": json.dumps(clean_p),
            })

            pd.DataFrame(summary_rows).sort_values("val_f1_opt", ascending=False).to_csv(
                out_dir / "optimization_summary.csv", index=False
            )
            with open(out_dir / "best_hyperparameters_pro.json", "w", encoding="utf-8") as f:
                json.dump(best_configs, f, indent=2)

        df_final = pd.DataFrame(summary_rows).sort_values("val_f1_opt", ascending=False)
        df_final.to_csv(out_dir / "optimization_summary.csv", index=False)
        best_json_path = out_dir / "best_hyperparameters_pro.json"
        with open(best_json_path, "w", encoding="utf-8") as f:
            json.dump(best_configs, f, indent=2)
        load_hyperparameters_from_json(best_json_path)

    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)

    total_min = (time.perf_counter() - global_t0) / 60.0
    print("\n" + "=" * 98)
    print(f"FINAL OPTIMIZATION SUMMARY (Total time: {total_min:.1f} min | Sorted by Val F1)")
    print("=" * 98)
    cols = ["model", "opt_threshold", "cv_f1_mean", "cv_f1_std", "val_f1_opt", "test_f1_05", "test_f1_opt", "test_auc_roc", "time_s"]
    print(df_final[cols].to_string(index=False))
    print(f"\nTuning results and models saved at: {out_dir.resolve()}")
    return best_configs


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


def evaluate(model, X, y, threshold=None):
    prob = model.predict_proba(X)[:, 1]
    if threshold is None:
        pred = model.predict(X)
    else:
        pred = (prob >= float(threshold)).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "auc_roc": roc_auc_score(y, prob),
    }, prob, pred


def cross_validate_model(pipe, X, y, seed, threshold=None, n_splits=CV_FOLDS):
    """Evaluates `pipe` using `n_splits`-fold Stratified Cross-Validation on `(X, y)`.
    Returns aggregated CV mean/std metrics and the per-fold metric records."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_rows = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
        y_tr = y[train_idx]
        X_va = X.iloc[val_idx] if hasattr(X, "iloc") else X[val_idx]
        y_va = y[val_idx]

        fold_pipe = clone(pipe)
        t0 = time.perf_counter()
        fold_pipe.fit(X_tr, y_tr)
        fold_fit_time = time.perf_counter() - t0

        fold_metrics, _, _ = evaluate(fold_pipe, X_va, y_va, threshold=threshold)
        fold_rows.append({
            "fold": int(fold_idx),
            **{k: float(v) for k, v in fold_metrics.items()},
            "fit_time_s": float(fold_fit_time),
        })

    cv_df = pd.DataFrame(fold_rows)
    cv_summary = {}
    for m in EVAL_METRICS:
        vals = cv_df[m].astype(float)
        cv_summary[f"cv_{m}"] = float(vals.mean())
        cv_summary[f"cv_{m}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    cv_summary["cv_fit_time_s"] = float(cv_df["fit_time_s"].sum())
    return cv_summary, fold_rows


def fit_run(name, seed, data, scale_pos_weight, commit):
    """Trains and evaluates one model with one seed (using 5-fold Stratified Cross-Validation
    and evaluation on the test split), and saves its result files."""
    X_train, y_train, _ = data["train"]
    X_test, y_test, addr_test = data["test"]
    np.random.seed(seed)
    pipe = build_models(seed, scale_pos_weight)[name]
    opt_thr = get_model_threshold(name)

    # 1. 5-Fold Stratified Cross-Validation evaluation on the training split
    cv_metrics, cv_folds = cross_validate_model(
        pipe, X_train, y_train, seed=seed, threshold=opt_thr, n_splits=CV_FOLDS
    )

    # 2. Full training split fit and evaluation on the test split
    start = time.perf_counter()
    pipe.fit(X_train, y_train)
    fit_time = time.perf_counter() - start

    metrics, prob, pred = evaluate(pipe, X_test, y_test, threshold=opt_thr)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"address": addr_test, "label": y_test, "prob_illicit": prob, "prediction": pred}).to_csv(
        PREDICTIONS_DIR / f"{run_id(name, seed)}.csv", index=False)

    write_json(RUNS_DIR / f"{run_id(name, seed)}.json", {
        "model": name,
        "seed": seed,
        "threshold": opt_thr,
        "cv_folds": CV_FOLDS,
        **{k: float(v) for k, v in metrics.items()},
        "fit_time_s": fit_time,
        **cv_metrics,
        "cv_fold_metrics": cv_folds,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "git_commit": commit,
        "finished": timestamp(),
    })
    return run_id(name, seed)


def _run_complete(run_path):
    if not run_path.exists():
        return False
    try:
        info = read_json(run_path)
        return "cv_f1" in info and info.get("cv_folds") == CV_FOLDS
    except Exception:
        return False


def train_missing_runs(data, seeds, workers):
    _, y_train, _ = data["train"]
    scale_pos_weight = float(np.sum(y_train == 0) / max(1, np.sum(y_train == 1)))
    commit = git_commit(config.PROJECT_ROOT)

    model_names = list(build_models(0, scale_pos_weight))
    runs = [(name, seed) for seed in seeds for name in model_names]
    pending = [(name, seed) for name, seed in runs if not _run_complete(RUNS_DIR / f"{run_id(name, seed)}.json")]
    print(f"{len(runs) - len(pending)} of {len(runs)} runs already completed, {len(pending)} to train "
          f"({workers} parallel single-threaded processes, with {CV_FOLDS}-fold CV per run).")
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
            "tuned_hyperparameters": copy.deepcopy(TUNED_HYPERPARAMETERS),
            "preprocessing": (
                "mutual-information feature selection (SelectPercentile), log1p and "
                "RobustScaler/StandardScaler, fitted on the training split with per-model "
                "calibrated decision thresholds"
            ),
            "n_seeds": len(seeds),
            "seeds": seeds,
            "evaluation": {
                "cross_validation": f"{CV_FOLDS}-fold StratifiedKFold on the training split (seeded per run)",
                "test_split": "seed addresses of the test split",
                "metrics": EVAL_METRICS,
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
    valid_models = set(TUNED_HYPERPARAMETERS.keys())
    raw_results = [read_json(p) for p in sorted(RUNS_DIR.glob("*.json"))]
    raw_results = [r for r in raw_results if r.get("model") in valid_models]
    if not raw_results:
        print("No completed runs to aggregate.")
        return

    fold_records = []
    flat_results = []
    for r in raw_results:
        r_copy = dict(r)
        folds_list = r_copy.pop("cv_fold_metrics", None)
        flat_results.append(r_copy)
        if isinstance(folds_list, list):
            for f_item in folds_list:
                fold_records.append({
                    "model": r["model"],
                    "seed": r["seed"],
                    **f_item,
                })

    runs = pd.DataFrame(flat_results).sort_values(["model", "seed"])
    runs.to_csv(RESULTS_DIR / "ego_runs.csv", index=False)

    if fold_records:
        folds_df = pd.DataFrame(fold_records).sort_values(["model", "seed", "fold"])
        folds_df.to_csv(RESULTS_DIR / "ego_cv_folds.csv", index=False)

    summary_metrics = METRICS + [m for m in CV_METRICS if m in runs.columns]
    summary = summarize_runs(runs, ["model"], summary_metrics).sort_values("f1_mean", ascending=False)
    summary.to_csv(RESULTS_DIR / "ego_summary.csv", index=False)

    pd.set_option("display.width", 200)
    if all(f"cv_{m}_mean" in summary.columns for m in EVAL_METRICS):
        cv_summary = summary.sort_values("cv_f1_mean", ascending=False)
        cv_summary.to_csv(RESULTS_DIR / "ego_cv_summary.csv", index=False)
        print(f"\n=== {CV_FOLDS}-Fold Cross-Validation Evaluation Summary ===")
        print(cv_summary[["model", "n_runs", "cv_accuracy_mean", "cv_accuracy_std",
                          "cv_f1_mean", "cv_f1_std", "cv_f1_median",
                          "cv_auc_roc_mean", "cv_auc_roc_std"]].round(4).to_string(index=False))

    print("\n=== Test Split Evaluation Summary ===")
    print(summary[["model", "n_runs", "accuracy_mean", "accuracy_std", "f1_mean", "f1_std", "f1_median",
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
        pipe = joblib.load(model_path)
        if 'log' in pipe.named_steps:
            return name, pipe, selection
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
    _, y_val, _ = data["val"]
    y_train = data["train"][1]
    model_step, names = inputs["model"], inputs["names"]
    X_train, X_val = inputs["X_train"], inputs["X_val"]
    opt_thr = get_model_threshold(name)

    if hasattr(model_step, 'coef_'):
        importances = np.abs(model_step.coef_[0])
        importance_type = "|coefficient|"
    elif hasattr(model_step, 'feature_importances_'):
        importances = model_step.feature_importances_
        importance_type = "feature_importance (Gini)"
    else:
        # Every selected feature is permuted in turn on the validation split
        # using the calibrated threshold of `name`
        print(f"Computing permutation importance (F1-score) to rank the features of {name}...")

        def _thr_f1_scorer(est, X_eval, y_eval):
            probs = est.predict_proba(X_eval)[:, 1]
            return float(f1_score(y_eval, (probs >= opt_thr).astype(int), zero_division=0))

        r = permutation_importance(model_step, X_val, y_val, n_repeats=10, random_state=config.DATA_SEED,
                                   n_jobs=workers, scoring=_thr_f1_scorer)
        importances = r.importances_mean
        importance_type = "permutation importance (mean decrease of the F1-score)"
    pd.DataFrame({"feature": names, "importance": importances}).sort_values(
        "importance", ascending=False).to_csv(INTERPRETATION_DIR / "feature_importance.csv", index=False)

    ordered_indices = np.argsort(importances)[::-1]
    ordered_names = [names[i] for i in ordered_indices]
    print(f"Feature ranking criterion: {importance_type}")
    print(f"Top 5 ranked features: {ordered_names[:5]}")

    def ablation_scores(k):
        # Retrains the model with the k most important features (each k in its own process)
        subset = ordered_indices[:k]
        model_abl = clone(model_step)
        model_abl.fit(X_train[:, subset], y_train)
        probs_k = model_abl.predict_proba(X_val[:, subset])[:, 1]
        preds_k = (probs_k >= opt_thr).astype(int)
        return (f1_score(y_val, preds_k, zero_division=0),
                roc_auc_score(y_val, probs_k))

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
    if model_type in ('RandomForestClassifier', 'XGBClassifier', 'IsolationForestClassifier', 'TunableIsolationForestClassifier'):
        # Exact tree SHAP, interventional on 100 training addresses (the cost grows with the
        # background size, and 100 addresses estimate the expected output well)
        background = shap.sample(X_train, 100, random_state=config.DATA_SEED)
        if model_type in ('IsolationForestClassifier', 'TunableIsolationForestClassifier'):
            # Its probability is rescaled within every batch, so the anomaly score is
            # explained: the path length of the isolation forest, with the sign matching flip_score
            explainer = shap.TreeExplainer(model_step.model_, background)
            sign = 1.0 if getattr(model_step, "flip_score", False) else -1.0
            output = "anomaly score (isolation forest output aligned with illicit class)"
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
        if INTERPRETATION_DIR.exists():
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
    limit_threads()
    parser = argparse.ArgumentParser(description="Train, cross-validate (5 folds), evaluate, tune and interpret the ego-centric models.")
    parser.add_argument("--n-seeds", type=int, help="random-seed runs per model (asked if not given)")
    parser.add_argument("--workers", type=int, default=config.EGO_CENTRIC_MODEL_WORKERS, help="parallel training processes")
    parser.add_argument("--aggregate-only", action="store_true", help="only rebuild the result tables and tests")
    parser.add_argument("--skip-interpretation", action="store_true", help="skip the ablation and SHAP analyses")
    parser.add_argument("--interpret-only", action="store_true",
                        help="only the ablation and SHAP analyses of the best model of the trained runs")
    # Hyperparameter search mode options
    parser.add_argument("--tune", action="store_true",
                        help="run hyperparameter search (GridSearchCV / HalvingRandomSearchCV) before training")
    parser.add_argument("--tune-only", action="store_true",
                        help="run only the hyperparameter search and exit")
    parser.add_argument("--tune-out-dir", type=str, default=str(DEFAULT_TUNING_DIR),
                        help="directory to save hyperparameter tuning results and models")
    parser.add_argument("--tune-cv-mode", choices=["val", "kfold"], default="kfold",
                        help="validation split mode for hyperparameter tuning: StratifiedKFold 'kfold' (default) or fixed 'val'")
    parser.add_argument("--tune-folds", type=int, default=CV_FOLDS,
                        help="number of folds when --tune-cv-mode=kfold (default: 5)")
    parser.add_argument("--tune-scoring", choices=["f1_05", "f1_opt", "roc_auc"], default="f1_05",
                        help="scoring metric during search: 'f1_05' (@0.5), 'f1_opt' (optimal PR threshold), or 'roc_auc'")
    parser.add_argument("--tune-models", nargs="+", default=None,
                        help="optional subset of models to tune")
    parser.add_argument("--hyperparams-file", type=str, default=None,
                        help="optional path to best_hyperparameters_pro.json to override default tuned hyperparameters")
    args = parser.parse_args()
    if args.interpret_only and (args.aggregate_only or args.skip_interpretation or args.tune or args.tune_only):
        parser.error("--interpret-only cannot be combined with --aggregate-only, --skip-interpretation, --tune or --tune-only")

    if args.hyperparams_file:
        load_hyperparameters_from_json(args.hyperparams_file)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_dataset()
    for split in config.SPLITS:
        print(f"{split:5s}: {len(data[split][1])} seed addresses")

    if args.tune or args.tune_only:
        run_hyperparameter_search(
            data=data,
            out_dir=Path(args.tune_out_dir),
            cv_mode=args.tune_cv_mode,
            folds=args.tune_folds,
            scoring_mode=args.tune_scoring,
            models_filter=args.tune_models,
            workers=args.workers,
            seed=config.DATA_SEED,
        )
        if args.tune_only:
            return

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
