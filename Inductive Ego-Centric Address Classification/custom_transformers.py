import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import IsolationForest

DEFAULT_BASE_FEATURES = [
    "2star_out", "split_2+", "2star_in", "merge_2+", "peeling_2", "peeling_3",
    "in_deg", "out_deg", "pagerank", "betweenness", "eccentricity", "density",
]

RATIO_FEATURE_NAMES = [
    "total_deg",
    "deg_ratio",
    "split_merge_ratio",
    "star_ratio",
    "peeling_intensity",
]

REQUIRED_RATIO_COLS = (
    "2star_out", "split_2+", "2star_in", "merge_2+",
    "peeling_2", "peeling_3", "in_deg", "out_deg",
)


# =====================================================================
# 1. ENGINYERIA DE CARACTERÍSTIQUES TOPOLÒGIQUES + LOG1P
# =====================================================================
class TopologicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Aplica log1p a les 12 variables base i opcionalment afegeix 5 ràtios topològics.
    Resol dinàmicament els índexs de les columnes si X és un DataFrame per evitar
    errors quan l'ordre de les columnes canvia entre scripts (p. ex. config.py).
    """
    def __init__(self, add_ratios=True):
        self.add_ratios = add_ratios

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
            col_map = {name: idx for idx, name in enumerate(self.feature_names_in_)}
            self.idx_map_ = {
                col: col_map.get(col, default_idx)
                for default_idx, col in enumerate(REQUIRED_RATIO_COLS)
            }
        else:
            n_cols = X.shape[1]
            self.feature_names_in_ = (
                list(DEFAULT_BASE_FEATURES) if n_cols == len(DEFAULT_BASE_FEATURES)
                else [f"f{i}" for i in range(n_cols)]
            )
            self.idx_map_ = {col: idx for idx, col in enumerate(REQUIRED_RATIO_COLS)}
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        X_arr = np.clip(X_arr, a_min=0.0, a_max=None)

        if not self.add_ratios:
            return np.log1p(X_arr)

        if hasattr(X, "columns"):
            col_map = {name: idx for idx, name in enumerate(X.columns)}
            idx = {col: col_map.get(col, i) for i, col in enumerate(REQUIRED_RATIO_COLS)}
        else:
            idx = getattr(
                self,
                "idx_map_",
                {col: i for i, col in enumerate(REQUIRED_RATIO_COLS)},
            )

        two_out = X_arr[:, idx["2star_out"]]
        split2 = X_arr[:, idx["split_2+"]]
        two_in = X_arr[:, idx["2star_in"]]
        merge2 = X_arr[:, idx["merge_2+"]]
        peel2 = X_arr[:, idx["peeling_2"]]
        peel3 = X_arr[:, idx["peeling_3"]]
        in_deg = X_arr[:, idx["in_deg"]]
        out_deg = X_arr[:, idx["out_deg"]]

        total_deg = in_deg + out_deg
        deg_ratio = (in_deg + 1.0) / (out_deg + 1.0)
        split_merge_ratio = (split2 + 1.0) / (merge2 + 1.0)
        star_ratio = (two_in + 1.0) / (two_out + 1.0)
        peeling_intensity = (peel2 + peel3) / (total_deg + 1.0)

        X_log = np.log1p(X_arr)
        extras = np.column_stack([
            np.log1p(total_deg),
            np.log(deg_ratio),
            np.log(split_merge_ratio),
            np.log(star_ratio),
            peeling_intensity,
        ])
        return np.hstack([X_log, extras])

    def get_feature_names_out(self, input_features=None):
        base = (
            list(input_features)
            if input_features is not None
            else list(getattr(self, "feature_names_in_", DEFAULT_BASE_FEATURES))
        )
        if self.add_ratios:
            return base + list(RATIO_FEATURE_NAMES)
        return base


# =====================================================================
# 2. ISOLATION FOREST AMB POLARITAT I LLINDAR AJUSTABLES
# =====================================================================
class TunableIsolationForestClassifier(BaseEstimator, ClassifierMixin):
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
            n_jobs=self.n_jobs
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

