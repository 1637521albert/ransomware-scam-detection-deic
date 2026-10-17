# STEP 3 (Methodology II) - MODEL TRAINING, EVALUATION AND INTERPRETABILITY
#
# Asks how many random-seed runs to execute, trains the eight tabular models on
# the topological features (EGO_CENTRIC_FEATURES_FILE, written by
# extract_topological_features.py) with that many seeds, evaluates them on the
# test split, runs the Wilcoxon tests (wilcoxon_tests.py) and analyzes the best
# model (config.EGO_CENTRIC_BEST_MODEL):
#
#   python models.py                         # asks the number of seeds
#   python models.py --n-seeds 10            # without asking
#   python models.py --skip-interpretation   # without the ablation and SHAP analyses
#   python models.py --aggregate-only        # only rebuild the result tables and tests
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
#                                         best model trained with EGO_CENTRIC_INTERPRETATION_SEED

import argparse
import sys
import time
from functools import partial
from pathlib import Path

import joblib
from joblib import Parallel, delayed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

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

# --- Training Features Definition ---
TRAINING_FEATURES = [
    'in_deg', 'out_deg',       # Degrees
    '2star_in', 'merge_2+',    # Aggregation
    '2star_out', 'split_2+',   # Branching
    'peeling_2', 'peeling_3',  # Peeling Chains
    'pagerank', 'betweenness', 'eccentricity', 'density' # Centrality & Graph Metrics
]

# --- LaTeX Feature Name Mapping ---
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

# --- Hyperparameters Dictionary ---
HP = {
    "selector_percentile": 90,
    "scaler": "StandardScaler",

    "logreg_C": 0.3,
    "logreg_penalty_l1": "l1",
    "logreg_penalty_el": "elasticnet",
    "logreg_l1_ratio": 0.5,
    "logreg_max_iter": 5000,

    "rf_n_estimators": 500,
    "rf_max_depth": 20,

    "xgb_n_estimators": 1000,
    "xgb_max_depth": 8,
    "xgb_learning_rate": 0.1,
    "xgb_subsample": 0.9,
    "xgb_colsample_bytree": 0.9,

    "iso_n_estimators": 500,
    "iso_contamination": "auto",
    "svm_C": 1.0,
    "svm_kernel": "rbf",

    "mlp_hidden_layers": (100, 50),
    "mlp_alpha": 0.0001,
    "mlp_max_iter": 1000,
    "mlp_solver": "adam",
    "mlp_learning_rate_init": 0.001
}

# --- Custom Isolation Forest Wrapper Class ---
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


# --- Dataset Loading ---
def load_dataset():
    features_file = config.EGO_CENTRIC_FEATURES_FILE
    if not features_file.exists():
        raise FileNotFoundError(f"Missing {features_file}: run extract_topological_features.py first.")

    df = pd.read_csv(features_file)
    # Deterministic row order, independent of the order of the file
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

    if name == config.EGO_CENTRIC_BEST_MODEL and seed == config.EGO_CENTRIC_INTERPRETATION_SEED:
        INTERPRETATION_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, INTERPRETATION_DIR / f"{run_id(name, seed)}.joblib")

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
    features_path = config.EGO_CENTRIC_FEATURES_FILE.with_suffix(".json")
    write_json(RESULTS_DIR / "experiment_config.json", experiment_snapshot(
        extra={
            "methodology": "inductive ego-centric address classification",
            "data": {
                "expansion": read_json(expansion_path) if expansion_path.exists() else None,
                "features": read_json(features_path) if features_path.exists() else None,
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
                "interpretation": f"{config.EGO_CENTRIC_BEST_MODEL} trained with seed {config.EGO_CENTRIC_INTERPRETATION_SEED}: "
                                  "permutation importance, feature ablation and SHAP on the validation split",
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


def interpret(data, workers):
    """Feature ablation and SHAP analysis of the best model, both on the validation set."""
    X_train_raw, y_train, _ = data["train"]
    X_val_raw, y_val, _ = data["val"]
    best_model_name = config.EGO_CENTRIC_BEST_MODEL
    model_path = INTERPRETATION_DIR / f"{run_id(best_model_name, config.EGO_CENTRIC_INTERPRETATION_SEED)}.joblib"
    if not model_path.exists():
        print(f"Missing {model_path}: train the runs first.")
        return
    if (INTERPRETATION_DIR / "shap_values.csv").exists() and (INTERPRETATION_DIR / "ablation.csv").exists():
        print(f"\nInterpretation already computed: {INTERPRETATION_DIR}")
        return

    best_model_pipe = joblib.load(model_path)
    best_model_safe_name = file_name(best_model_name)
    print(f"\nModel {best_model_name} selected for explainability and feature ablation analysis.")

    # --- Feature Ablation Study ---
    print(f"\nStarting Robust Feature Ablation Study ({best_model_name}, on validation set)...")
    model_step = best_model_pipe.named_steps['model']
    selector_step = best_model_pipe.named_steps['selector']
    scaler_step = best_model_pipe.named_steps['scaler']
    log_step = best_model_pipe.named_steps['log']

    if hasattr(model_step, 'coef_'):
        importances = np.abs(model_step.coef_[0])
        importance_type = "|coefficient|"
    elif hasattr(model_step, 'feature_importances_'):
        importances = model_step.feature_importances_
        importance_type = "feature_importance (Gini)"
    else:
        print(f"Computing Permutation Importance to rank features for {best_model_name}...")
        r = permutation_importance(
            best_model_pipe, X_val_raw, y_val,
            n_repeats=10, random_state=config.DATA_SEED, n_jobs=-1, scoring='roc_auc'
        )
        support_mask = selector_step.get_support()
        importances = r.importances_mean[support_mask]
        importance_type = "permutation_importance (Mean Decrease AUC)"

    sel_features_names = X_train_raw.columns[selector_step.get_support()].tolist()
    pd.DataFrame({"feature": sel_features_names, "importance": importances}).sort_values(
        "importance", ascending=False).to_csv(INTERPRETATION_DIR / "feature_importance.csv", index=False)

    # The ablation retrains on the same transformed features as the pipeline (log + scaling)
    X_train_scaled = scaler_step.transform(log_step.transform(selector_step.transform(X_train_raw)))
    X_val_scaled = scaler_step.transform(log_step.transform(selector_step.transform(X_val_raw)))

    ordered_indices = np.argsort(importances)[::-1]
    ordered_feats_names = [sel_features_names[i] for i in ordered_indices]
    print(f"Feature ranking criterion: {importance_type}")
    print(f"Top 5 ranked features: {ordered_feats_names[:5]}")

    def ablation_auc(k):
        # Retrains the model with the k most important features (each k in its own process)
        subset_indices = ordered_indices[:k]
        model_abl = clone(model_step)
        model_abl.fit(X_train_scaled[:, subset_indices], y_train)
        return roc_auc_score(y_val, model_abl.predict_proba(X_val_scaled[:, subset_indices])[:, 1])

    auc_list = parallel_map(ablation_auc, list(range(1, len(ordered_indices) + 1)), workers,
                            desc=f"Retraining {best_model_name} incrementally", chunks_per_worker=1)

    ablation_results_df = pd.DataFrame({
        'k_features': range(1, len(auc_list) + 1),
        'auc_roc': auc_list,
        'feature_added': ordered_feats_names
    })
    ablation_results_df.to_csv(INTERPRETATION_DIR / "ablation.csv", index=False)
    print("\n=== Feature Ablation Study Results ===")
    print(ablation_results_df.to_string(index=False))

    x_labels_latex = [get_label(f) for f in ablation_results_df['feature_added']]
    plt.figure(figsize=(7, 6))
    plt.plot(ablation_results_df['k_features'], ablation_results_df['auc_roc'],
             marker='o', linestyle='-', color='#1f77b4', linewidth=1.5)
    plt.xticks(ticks=ablation_results_df['k_features'], labels=x_labels_latex, rotation=45, ha='right', fontsize=10)
    plt.xlabel('Feature Added (Cumulative)', fontsize=11)
    plt.ylabel("AUC-ROC", fontsize=11)
    plt.title(f"Feature Ablation Study: {best_model_name} Performance", fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(INTERPRETATION_DIR / f"ablation_{best_model_safe_name}.pdf", bbox_inches='tight')
    plt.close()

    # --- Deep SHAP Analysis ---
    import shap
    print(f"\nStarting deep SHAP analysis for the best model: {best_model_name}")

    raw_feature_names = sel_features_names
    feature_names = [FEATURE_NAMES_LATEX.get(name, name) for name in raw_feature_names]
    X_train_data = selector_step.transform(X_train_raw)
    X_val_data = selector_step.transform(X_val_raw)
    X_val_plot = log_step.transform(X_val_data)
    X_val_plot_df = pd.DataFrame(X_val_plot, columns=feature_names)
    X_val_model = scaler_step.transform(X_val_plot)
    X_train_model = scaler_step.transform(log_step.transform(X_train_data))

    model_type_name = type(model_step).__name__
    print(f"Detected model type: {model_type_name}. Configuring optimized SHAP explainer...")

    if model_type_name in ['RandomForestClassifier', 'XGBClassifier', 'IsolationForestClassifier']:
        N_SAMPLES_SHAP = 500
        explainer = shap.TreeExplainer(model_step, X_train_model)
    elif model_type_name in ['LogisticRegression', 'GaussianNB']:
        N_SAMPLES_SHAP = 700
        explainer = shap.LinearExplainer(model_step, shap.kmeans(X_train_model, 75))
    else:
        N_SAMPLES_SHAP = len(X_val_model)
        explainer = shap.KernelExplainer(model_step.predict_proba, shap.kmeans(X_train_model, 50))

    print(f"Computing SHAP values for {best_model_name}...")
    if isinstance(explainer, shap.KernelExplainer):
        def kernel_shap_chunk(start):
            # The kernel explainer samples feature coalitions at random, so every chunk has its own seed
            np.random.seed(config.DATA_SEED + start)
            values = np.asarray(explainer.shap_values(X_val_model[start:min(start + SHAP_CHUNK_SIZE, N_SAMPLES_SHAP)],
                                                      nsamples="auto", silent=True))
            # Older SHAP versions return one array per class instead of a 3-D array
            return values[1] if values.ndim == 3 and values.shape[0] == 2 else values[:, :, 1]

        chunks = parallel_map(kernel_shap_chunk, list(range(0, N_SAMPLES_SHAP, SHAP_CHUNK_SIZE)),
                              workers, desc="SHAP values", chunks_per_worker=1)
        shap_values_class1_array = np.concatenate(chunks, axis=0)
        shap_values_class1 = shap.Explanation(
            values=shap_values_class1_array,
            base_values=explainer.expected_value[1],
            data=X_val_plot_df.iloc[:N_SAMPLES_SHAP].values,
            feature_names=feature_names
        )
    else:
        shap_values_obj = explainer(X_val_model[:N_SAMPLES_SHAP], check_additivity=False)
        shap_values_class1 = shap_values_obj[..., 1]
        shap_values_class1.data = X_val_plot_df.iloc[:N_SAMPLES_SHAP].values
        shap_values_class1.feature_names = feature_names

    pd.DataFrame(shap_values_class1.values, columns=raw_feature_names).to_csv(INTERPRETATION_DIR / "shap_values.csv", index=False)
    pd.DataFrame({"feature": raw_feature_names, "mean_abs_shap": np.abs(shap_values_class1.values).mean(axis=0)}).sort_values(
        "mean_abs_shap", ascending=False).to_csv(INTERPRETATION_DIR / "shap_importance.csv", index=False)

    shap.summary_plot(shap_values_class1, plot_type="dot", show=False)
    plt.title(f'Feature Impact (Beeswarm) - Model: {best_model_name}')
    plt.savefig(INTERPRETATION_DIR / f"shap_summary_dot_{best_model_safe_name}.pdf", bbox_inches='tight')
    plt.close()

    shap.summary_plot(shap_values_class1, plot_type="bar", show=False)
    plt.title(f'Global Feature Importance - Model: {best_model_name}')
    plt.savefig(INTERPRETATION_DIR / f"shap_summary_bar_{best_model_safe_name}.pdf", bbox_inches='tight')
    plt.close()
    print(f"Interpretation saved at: {INTERPRETATION_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Train, evaluate and interpret the ego-centric models.")
    parser.add_argument("--n-seeds", type=int, help="random-seed runs per model (asked if not given)")
    parser.add_argument("--workers", type=int, default=config.EGO_CENTRIC_MODEL_WORKERS, help="parallel training processes")
    parser.add_argument("--aggregate-only", action="store_true", help="only rebuild the result tables and tests")
    parser.add_argument("--skip-interpretation", action="store_true", help="skip the ablation and SHAP analyses")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_dataset()
    for split in config.SPLITS:
        print(f"{split:5s}: {len(data[split][1])} seed addresses")

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
