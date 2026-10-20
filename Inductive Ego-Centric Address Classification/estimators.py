# Tabular models of Methodology II: how every model is built and evaluated.
#
# Every model is a pipeline: clipping of negative values and log1p of the topological
# features (ClipLog1p), mutual-information feature selection, scaling and the model,
# with the fixed settings of config.EGO_CENTRIC_MODELS and the hyperparameters found by
# tune_hyperparameters.py. Every evaluation predicts illicit when the probability of
# the illicit class is at least THRESHOLD, the threshold of the GNNs of Methodology I.
#
# This module is imported by the scripts of the steps, so the pipelines they save with
# joblib can be loaded by any of them (a class defined in a script run as __main__ is
# only found by that script). Its whole code is part of the fingerprints of the
# hyperparameter searches and of the runs (common.py): changing it (not its comments)
# searches and trains again.

from functools import partial

from shared.threads import limit_threads  # noqa: F401 (single-threaded libraries, before numpy)

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

import config

# Predicted illicit when the probability of the illicit class is at least this value
THRESHOLD = 0.5

# Models, their fixed settings and their search grids
MODELS = config.EGO_CENTRIC_MODELS

SCALERS = {"RobustScaler": RobustScaler, "StandardScaler": StandardScaler}


class ClipLog1p(BaseEstimator, TransformerMixin):
    """log1p of the features, after setting negative values to 0 (log1p is not defined
    below -1). The topological features are counts and non-negative measures, so the
    clipping only guards against invalid values."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log1p(np.clip(np.asarray(X, dtype=float), 0.0, None))


class IsolationForestClassifier(BaseEstimator, ClassifierMixin):
    """Isolation forest as a classifier: the anomalies are predicted illicit.

    The anomaly score s = -decision_function(X) is positive for the addresses that the
    isolation forest predicts as anomalies (with its contamination). It is mapped to a
    probability of the illicit class with a fixed function, fitted on the training
    split: linear from the lowest training score (0) to 0 (0.5) and from 0 (0.5) to the
    highest training score (1), clipped to [0, 1]. The probability of an address only
    depends on its own score (not on the other addresses evaluated with it), and it is
    at least 0.5 exactly when the address is an anomaly, so THRESHOLD gives the decision
    of the isolation forest (a score of exactly 0, on the boundary, is predicted
    illicit, as predict does)."""

    def __init__(self, n_estimators=100, contamination="auto", max_samples="auto", random_state=None, n_jobs=1):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        self.model_ = IsolationForest(n_estimators=self.n_estimators, contamination=self.contamination,
                                      max_samples=self.max_samples, random_state=self.random_state,
                                      n_jobs=self.n_jobs).fit(X)
        scores = self.anomaly_score(X)
        # The bounds keep both linear pieces defined if every score has the same sign
        self.score_low_ = min(float(scores.min()), -1e-12)
        self.score_high_ = max(float(scores.max()), 1e-12)
        self.classes_ = np.array([0, 1])
        return self

    def anomaly_score(self, X):
        return -self.model_.decision_function(X)

    def predict_proba(self, X):
        scores = self.anomaly_score(X)
        prob = np.where(scores < 0, 0.5 * (1.0 - scores / self.score_low_), 0.5 + 0.5 * scores / self.score_high_)
        prob = np.clip(prob, 0.0, 1.0)
        return np.column_stack((1.0 - prob, prob))

    def predict(self, X):
        return (self.anomaly_score(X) >= 0).astype(int)


ESTIMATORS = {
    "L1 (Lasso)": LogisticRegression,
    "ElasticNet": LogisticRegression,
    "RandomForest": RandomForestClassifier,
    "XGBoost": XGBClassifier,
    "SVM": SVC,
    "Naive Bayes": GaussianNB,
    "MLP": MLPClassifier,
    "IsolationForest": IsolationForestClassifier,
}


def scale_pos_weight(y):
    return float(np.sum(y == 0) / max(1, np.sum(y == 1)))


def make_estimator(name, seed, pos_weight):
    """The model `name` with its fixed settings, seeded with `seed` and single-threaded
    (the runs are parallelized across processes instead)."""
    cls = ESTIMATORS[name]
    kwargs = dict(MODELS[name]["fixed"])
    accepted = cls().get_params()
    if "random_state" in accepted:
        kwargs["random_state"] = seed
    if "n_jobs" in accepted:
        kwargs["n_jobs"] = 1
    if "scale_pos_weight" in accepted and "scale_pos_weight" not in kwargs:
        kwargs["scale_pos_weight"] = pos_weight
    return cls(**kwargs)


def to_estimator_params(params):
    """Pipeline parameters from their JSON form: scaler names to scalers, lists to tuples."""
    params = dict(params)
    if isinstance(params.get("scaler"), str):
        params["scaler"] = SCALERS[params["scaler"]]()
    if isinstance(params.get("model__hidden_layer_sizes"), list):
        params["model__hidden_layer_sizes"] = tuple(params["model__hidden_layer_sizes"])
    return params


def to_json_params(params):
    """JSON form of pipeline parameters (scalers by name, plain Python numbers)."""
    clean = {}
    for key, value in params.items():
        if key == "scaler" and not isinstance(value, str):
            value = type(value).__name__
        elif isinstance(value, np.generic):
            value = value.item()
        elif isinstance(value, tuple):
            value = list(value)
        clean[key] = value
    return clean


def to_estimator_grid(grid):
    """Search grid of config.EGO_CENTRIC_MODELS with the scalers as objects."""
    return {key: [SCALERS[v]() for v in values] if key == "scaler" else list(values) for key, values in grid.items()}


def make_pipeline(name, seed, params=None, pos_weight=1.0, memory=None):
    """Clipping and log1p of the features, mutual-information selection (seeded), scaling
    and the model, with the given (searched) parameters. A step missing from the
    parameters keeps its default: percentile 100 and RobustScaler."""
    pipe = Pipeline([
        ('log', ClipLog1p()),
        ('selector', SelectPercentile(partial(mutual_info_classif, random_state=seed), percentile=100)),
        ('scaler', RobustScaler()),
        ('model', make_estimator(name, seed, pos_weight)),
    ], memory=memory)
    if params:
        pipe.set_params(**to_estimator_params(params))
    return pipe


def f1_at_threshold(estimator, X, y):
    """F1-score of the illicit class predicted with THRESHOLD (scorer of the searches)."""
    prob = estimator.predict_proba(X)[:, 1]
    return f1_score(y, (prob >= THRESHOLD).astype(int), zero_division=0)


def evaluate(model, X, y):
    """Metrics of the illicit class predicted with THRESHOLD, the probabilities and the
    predictions."""
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= THRESHOLD).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "auc_roc": roc_auc_score(y, prob),
    }, prob, pred
