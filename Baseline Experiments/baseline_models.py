# Models of the baseline experiments: how every method of baseline_config.METHODS is
# built and evaluated.
#
# Every method is a pipeline fitted on the training split only: median imputation of
# missing values, a signed logarithm (sign(x) log(1 + |x|)), standardization and the
# model. The logarithm and the standardization are monotone, so the tree models are
# unaffected, while the distance- and density-based models (k-nearest neighbors, Naive
# Bayes) are not dominated by the heavy-tailed amounts in satoshis. A stacking ensemble
# applies the same preprocessing once and combines its base models, each with its own
# searched hyperparameters, through its meta model (baseline_config.METHODS), trained on
# their out-of-fold probabilities and, with passthrough, on the features too. Every
# evaluation predicts illicit when the probability of the illicit class is at least
# THRESHOLD, as both methodologies.
#
# This module is imported by the experiment and by its worker processes, so its classes
# are found when the pipelines are sent to them. Its whole code is part of the
# fingerprints of the searches and of the runs: changing it (not its comments) searches
# and trains again.

from shared.threads import limit_threads  # noqa: F401 (single-threaded libraries, before numpy)

import numpy as np
from scipy.stats import randint, uniform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from baseline_config import CV_FOLDS, METHODS, THRESHOLD

ESTIMATORS = {
    "DecisionTree": DecisionTreeClassifier,
    "RandomForest": RandomForestClassifier,
    "XGBoost": XGBClassifier,
    "KNN": KNeighborsClassifier,
    "NaiveBayes": GaussianNB,
    "MLP": MLPClassifier,
    "LogisticRegression": LogisticRegression,
}


class SignedLog1p(BaseEstimator, TransformerMixin):
    """sign(x) log(1 + |x|) of every feature: monotone, defined for negative values (such
    as the -1 block of an address without transactions) and compressing heavy tails."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.sign(X) * np.log1p(np.abs(X))


def is_stacking(name):
    return METHODS[name]["estimator"] == "Stacking"


def scale_pos_weight(y):
    return float(np.sum(y == 0) / max(1, np.sum(y == 1)))


def search_space(grid):
    """The search space of a method as scikit-learn takes it: lists of values, and the
    distributions {"randint": [low, high]} and {"uniform": [low, high]}."""
    space = {}
    for key, values in grid.items():
        if isinstance(values, dict) and "randint" in values:
            space[key] = randint(*values["randint"])
        elif isinstance(values, dict) and "uniform" in values:
            low, high = values["uniform"]
            space[key] = uniform(low, high - low)
        else:
            space[key] = list(values)
    return space


def make_estimator(name, seed, pos_weight=1.0, params=None):
    """The single model `name` with its fixed settings and searched hyperparameters (in
    their pipeline form, model__*), seeded with `seed` and single-threaded (the runs are
    parallelized across processes instead)."""
    return estimator_from_spec(METHODS[name], seed, pos_weight, params)


def estimator_from_spec(spec, seed, pos_weight=1.0, params=None):
    """A model from its specification (estimator and fixed settings) and hyperparameters."""
    cls = ESTIMATORS[spec["estimator"]]
    kwargs = dict(spec.get("fixed", {}))
    accepted = cls().get_params()
    if "random_state" in accepted:
        kwargs["random_state"] = seed
    if "n_jobs" in accepted:
        kwargs["n_jobs"] = 1
    if "scale_pos_weight" in accepted and "scale_pos_weight" not in kwargs:
        kwargs["scale_pos_weight"] = pos_weight
    for key, value in (params or {}).items():
        kwargs[key.split("__", 1)[1]] = value
    return cls(**kwargs)


def preprocessing():
    return [("impute", SimpleImputer(strategy="median")), ("log", SignedLog1p()), ("scale", StandardScaler())]


def make_pipeline(name, seed, pos_weight=1.0, params=None):
    """The method `name` as a pipeline. params: for a single model, its searched
    hyperparameters (model__*); for a stacking ensemble, the searched hyperparameters of
    each of its base models and of a searched meta model, by name."""
    if not is_stacking(name):
        return Pipeline(preprocessing() + [("model", make_estimator(name, seed, pos_weight, params))])
    spec, params = METHODS[name], params or {}
    base = [(f"m{i}", make_estimator(member, seed, pos_weight, params.get(member)))
            for i, member in enumerate(spec["base"])]
    meta = spec["meta"]
    if isinstance(meta, str):
        final = make_estimator(meta, seed, pos_weight, params.get(meta))
    else:
        final = estimator_from_spec(meta, seed, pos_weight)
    stacking = StackingClassifier(
        estimators=base,
        final_estimator=final,
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed),
        stack_method="predict_proba",
        passthrough=spec.get("passthrough", False),
        n_jobs=1,
    )
    return Pipeline(preprocessing() + [("model", stacking)])


def to_json_params(params):
    """JSON form of searched hyperparameters (plain Python numbers)."""
    return {key: value.item() if isinstance(value, np.generic) else value for key, value in params.items()}


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
