# STEP 5 (Methodology II) - INTERPRETATION OF THE BEST TABULAR MODEL
#
# Explains the best model of the runs trained by train_models.py, the one with the
# highest mean test F1-score over the seeds (wilcoxon_tests.best_model), trained with
# EGO_CENTRIC_INTERPRETATION_SEED exactly as its run. Both analyses use the validation
# split:
#   - Feature ablation: the selected features are ranked by the absolute coefficients
#     (linear models), the impurity importance (tree models) or the permutation
#     importance on the F1-score (the others), and the model is retrained with the k
#     most important ones for every k, measuring its F1-score and AUC-ROC.
#   - SHAP: the contribution of every feature to the prediction of every validation
#     address, with the exact explainer of the model family when there is one.
#
# Every interpretation is saved in its own directory, named after the model, the seed
# and the fingerprint of its run (common.py), and is never overwritten by another one:
# the interpretations of other models, seeds or fingerprints (e.g. of another expansion
# of the egonets) are kept, and interpretation/index.csv lists them all with the
# expansion configuration of their data. Every analysis is saved when it finishes and is
# not computed again in the same directory, so an interrupted interpretation resumes
# from the missing part.
#
#   python interpret_model.py
#
# Output in EGO_CENTRIC_RESULTS_DIR/interpretation:
#   index.csv                              every saved interpretation, with the expansion
#                                          configuration of its data
#   <model>-seed<seed>_<fingerprint>/      interpretation.json (which model and why, its data
#                                          and the explainers), model.joblib (the trained
#                                          model), feature_importance.csv, ablation.csv and its
#                                          figure, shap_values.csv, shap_importance.csv and
#                                          the SHAP summary figures

import argparse
import sys
import warnings
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
for _p in (str(CURRENT_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Every ablation retraining and SHAP chunk runs in its own process (one per core), so
# the numerical libraries are single-threaded (before importing them); a model trained
# again for the interpretation is identical to its run
from shared.threads import limit_threads

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, roc_auc_score

import config
import wilcoxon_tests
from common import (INTERPRETATION_DIR, RESULTS_DIR, current_hyperparameters, data_description, expansion_columns,
                    file_name, interpretation_dir, load_dataset, run_fingerprint, write_index)
from estimators import THRESHOLD, f1_at_threshold, make_pipeline, scale_pos_weight
from shared.experiment import read_json, timestamp, write_json
from shared.parallel import parallel_map

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Validation addresses explained by each SHAP worker task. The chunks and their
# random seeds do not depend on the number of workers, so the SHAP values are
# the same with any number of cores.
SHAP_CHUNK_SIZE = 100

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
    """Formatted (LaTeX) label of a feature in the figures."""
    return FEATURE_NAMES_LATEX.get(feature_name, feature_name)


def interpreted_model(data, hyperparameters):
    """The best model (highest mean test F1-score over the seeds), trained with
    EGO_CENTRIC_INTERPRETATION_SEED exactly as its run: loaded from its directory if it
    was saved there, otherwise trained again and saved. Returns its name, the pipeline,
    the selection and its directory."""
    runs = pd.read_csv(RESULTS_DIR / "ego_runs.csv")
    name = wilcoxon_tests.best_model(runs, column="model")
    seed = config.EGO_CENTRIC_INTERPRETATION_SEED
    if seed not in set(runs.loc[runs["model"] == name, "seed"]):
        raise RuntimeError(f"{name} was not trained with the interpretation seed {seed}.")
    params = hyperparameters[name]
    fingerprint = run_fingerprint(name, params)
    selection = {
        "model": name,
        "seed": seed,
        "hyperparameters": params,
        "fingerprint": fingerprint,
        "data": data_description(),
        "criterion": "highest mean test F1-score over the seeds (ties: alphabetical order)",
        "f1_mean": runs.groupby("model")["f1"].mean().round(6).to_dict(),
        "n_seeds": int(runs.loc[runs["model"] == name, "seed"].nunique()),
    }

    out_dir = interpretation_dir(name, seed, fingerprint)
    model_path = out_dir / "model.joblib"
    if model_path.exists():
        return name, joblib.load(model_path), selection, out_dir

    X_train, y_train, _ = data["train"]
    print(f"Training {name} with seed {seed} for the interpretation...")
    np.random.seed(seed)
    pipe = make_pipeline(name, seed, params, scale_pos_weight(y_train))
    pipe.fit(X_train, y_train)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Written to a temporary file first, so an interrupted save is never loaded
    tmp_path = out_dir / "model.joblib.tmp"
    joblib.dump(pipe, tmp_path)
    tmp_path.replace(model_path)
    return name, pipe, selection, out_dir


def interpretation_inputs(data, pipe):
    """The steps of the interpreted pipeline and the selected features, raw and transformed."""
    X_train_raw = data["train"][0]
    X_val_raw = data["val"][0]
    log = pipe.named_steps['log']
    selector = pipe.named_steps['selector']
    scaler = pipe.named_steps['scaler']
    names = X_train_raw.columns[selector.get_support()].tolist()
    X_train_log = selector.transform(log.transform(X_train_raw))
    X_val_log = selector.transform(log.transform(X_val_raw))
    return {
        "model": pipe.named_steps['model'], "selector": selector, "names": names,
        # Inputs of the model: log1p, selected features and scaling
        "X_train": scaler.transform(X_train_log), "X_val": scaler.transform(X_val_log),
        # Values shown in the SHAP plots: selected features after log1p
        "X_val_log": X_val_log,
    }


def ablation(name, data, inputs, workers, out_dir):
    """Ranks the selected features and retrains the model with the k most important ones
    for every k, evaluated with the F1-score (and AUC-ROC) on the validation split."""
    _, y_val, _ = data["val"]
    y_train = data["train"][1]
    model_step, names = inputs["model"], inputs["names"]
    X_train, X_val = inputs["X_train"], inputs["X_val"]

    if hasattr(model_step, 'coef_'):
        importances = np.abs(model_step.coef_[0])
        importance_type = "|coefficient|"
    elif hasattr(model_step, 'feature_importances_'):
        importances = model_step.feature_importances_
        importance_type = "feature_importance (Gini)"
    else:
        # Every selected feature is permuted in turn on the validation split
        print(f"Computing permutation importance (F1-score) to rank the features of {name}...")
        r = permutation_importance(model_step, X_val, y_val, n_repeats=10, random_state=config.DATA_SEED,
                                   n_jobs=workers, scoring=f1_at_threshold)
        importances = r.importances_mean
        importance_type = "permutation importance (mean decrease of the F1-score)"
    pd.DataFrame({"feature": names, "importance": importances}).sort_values(
        "importance", ascending=False).to_csv(out_dir / "feature_importance.csv", index=False)

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
        return (f1_score(y_val, (probs_k >= THRESHOLD).astype(int), zero_division=0),
                roc_auc_score(y_val, probs_k))

    scores = parallel_map(ablation_scores, list(range(1, len(ordered_indices) + 1)), workers,
                          desc=f"Retraining {name} incrementally", chunks_per_worker=1)
    results = pd.DataFrame({
        'k_features': range(1, len(scores) + 1),
        'f1': [f1 for f1, _ in scores],
        'auc_roc': [auc for _, auc in scores],
        'feature_added': ordered_names,
    })
    results.to_csv(out_dir / "ablation.csv", index=False)
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
    plt.savefig(out_dir / f"ablation_{file_name(name)}.pdf", bbox_inches='tight')
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


def shap_analysis(name, inputs, workers, out_dir):
    """SHAP values of the interpreted model for every validation address, with the exact
    explainer of its family when there is one."""
    import shap
    model_step, names = inputs["model"], inputs["names"]
    X_train, X_val = inputs["X_train"], inputs["X_val"]
    labels = [get_label(n) for n in names]
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
            # The path length of the isolation forest (longer: more normal) is explained, with
            # its sign reversed, so positive values point to an anomaly, predicted illicit
            explainer = shap.TreeExplainer(model_step.model_, background)
            sign = -1.0
            output = "anomaly score (isolation forest path length, sign reversed)"
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

    pd.DataFrame(values, columns=names).to_csv(out_dir / "shap_values.csv", index=False)
    pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(values).mean(axis=0)}).sort_values(
        "mean_abs_shap", ascending=False).to_csv(out_dir / "shap_importance.csv", index=False)

    for plot_type, title in [("dot", "Feature Impact (Beeswarm)"), ("bar", "Global Feature Importance")]:
        shap.summary_plot(explanation, plot_type=plot_type, show=False)
        plt.title(f'{title} - Model: {name}')
        plt.savefig(out_dir / f"shap_summary_{plot_type}_{file_name(name)}.pdf", bbox_inches='tight')
        plt.close()
    return {"shap_explainer": method, "shap_output": output, "shap_addresses": int(len(values))}


def write_interpretation_index():
    rows = []
    for info_path in sorted(INTERPRETATION_DIR.glob("*/interpretation.json")):
        info = read_json(info_path)
        rows.append({"directory": info_path.parent.name, "model": info.get("model"), "seed": info.get("seed"),
                     "fingerprint": info.get("fingerprint"), "n_seeds": info.get("n_seeds"),
                     "f1_mean": (info.get("f1_mean") or {}).get(info.get("model")),
                     "ablation": bool(info.get("ablation")), "shap": bool(info.get("shap")),
                     "updated": info.get("updated"), **expansion_columns(info.get("data"))})
    write_index(INTERPRETATION_DIR / "index.csv", rows, ["data_fingerprint", "model", "seed"])


def interpret(data, workers, hyperparameters):
    """Feature ablation and SHAP analysis of the best model, both on the validation split,
    in the directory of its model, seed and fingerprint. Each part is saved when it
    finishes and is not computed again there, so an interrupted interpretation resumes
    from the part that was missing. Other directories are never modified."""
    name, pipe, selection, out_dir = interpreted_model(data, hyperparameters)
    info_path = out_dir / "interpretation.json"
    info = read_json(info_path) if info_path.exists() else {"created": timestamp()}
    # The selection (e.g. the mean F1-scores, with more seeds) is updated; the analyses are kept
    info.update(selection, updated=timestamp())
    write_json(info_path, info)
    print(f"\nModel {name} selected for explainability and feature ablation analysis "
          f"(highest mean F1-score over {selection['n_seeds']} seeds: {selection['f1_mean'][name]:.4f}), "
          f"saved at {out_dir}.")

    inputs = interpretation_inputs(data, pipe)
    if info.get("ablation") and (out_dir / "ablation.csv").exists():
        print(f"Feature ablation of {name} already computed.")
    else:
        print(f"\nStarting feature ablation study ({name}, on the validation split)...")
        info["ablation"] = ablation(name, data, inputs, workers, out_dir)
        write_json(info_path, info)

    if info.get("shap") and (out_dir / "shap_values.csv").exists():
        print(f"SHAP analysis of {name} already computed.")
    else:
        info["shap"] = shap_analysis(name, inputs, workers, out_dir)
        write_json(info_path, info)
    write_interpretation_index()
    print(f"Interpretation saved at: {out_dir} (all interpretations: {INTERPRETATION_DIR / 'index.csv'})")


def main():
    limit_threads()
    parser = argparse.ArgumentParser(description="Feature ablation and SHAP analysis of the best ego-centric model.")
    parser.add_argument("--workers", type=int, default=config.EGO_CENTRIC_MODEL_WORKERS, help="parallel processes")
    args = parser.parse_args()

    hyperparameters = current_hyperparameters()
    outdated = [name for name, params in hyperparameters.items() if params is None]
    if outdated or not (RESULTS_DIR / "ego_runs.csv").exists():
        raise SystemExit("The models are not trained with the current features and code "
                         f"(hyperparameters missing or outdated: {outdated or 'none'}): run train_models.py first.")
    interpret(load_dataset(), args.workers, hyperparameters)


if __name__ == "__main__":
    main()
