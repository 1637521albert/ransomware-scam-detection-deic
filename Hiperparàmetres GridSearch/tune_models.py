# tune_models_pro.py
# Script autònom, 100% paral·lelitzat i amb barra de progrés tqdm (sense errors de pickling).

import argparse
import contextlib
import json
import os
import shutil
import time
import warnings
from functools import partial
from pathlib import Path

# Limitar fils de llibreries numèriques abans d'importar-les per evitar oversubscription
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
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
    cross_val_predict,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from custom_transformers import (
    TopologicalFeatureEngineer,
    TunableIsolationForestClassifier,
)

try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Les 12 variables topològiques originals del projecte
TRAINING_FEATURES = [
    '2star_out', 'split_2+', '2star_in', 'merge_2+', 'peeling_2', 'peeling_3',
    'in_deg', 'out_deg', 'pagerank', 'betweenness', 'eccentricity', 'density'
]


# =====================================================================
# 0. HOOK DE TQDM PER A JOBLIB (SENSE OVERHEAD DE CPU)
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


# =====================================================================
# 1 i 2. TRANSFORMADORS I CLASSIFICADORS PERSONALITZATS
# S'importen des de custom_transformers.py per garantir que la serialització
# (pickle / joblib / loky multiprocessing) funcioni sense errors de __main__.
# =====================================================================


# =====================================================================
# 3. CÀRREGA DE DADES I UTILITATS D'AVALUACIÓ
# =====================================================================
def load_data(data_dir, seed=42):
    data_dir = Path(data_dir)
    dfs = []
    for split in ("train", "val", "test"):
        path = data_dir / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"No s'ha trobat el fitxer requerit: {path}")
        part = pd.read_csv(path)
        if "split" not in part.columns:
            part["split"] = split
        dfs.append(part)

    df = pd.concat(dfs, ignore_index=True)
    if "addr_str" in df.columns:
        df = df.sort_values("addr_str").sample(frac=1, random_state=seed).reset_index(drop=True)

    return {
        s: (df[df["split"] == s][TRAINING_FEATURES].copy(),
            df[df["split"] == s]["label"].astype(int).values)
        for s in ("train", "val", "test")
    }


def find_optimal_threshold(probs_val, y_val):
    precisions, recalls, thresholds = precision_recall_curve(y_val, probs_val)
    if len(thresholds) == 0:
        return 0.5
    f1_scores = (2 * precisions[:-1] * recalls[:-1]) / np.maximum(precisions[:-1] + recalls[:-1], 1e-12)
    best_idx = int(np.argmax(f1_scores))
    return float(np.clip(thresholds[best_idx], 0.05, 0.95))


def optimal_f1_score(y_true, y_prob):
    """Calcula l'F1 màxim sobre la corba Precision-Recall (invariant al descalibratge del llindar 0.5)."""
    probs = y_prob[:, 1] if getattr(y_prob, "ndim", 1) == 2 else y_prob
    thr = find_optimal_threshold(probs, y_true)
    return float(f1_score(y_true, (probs >= thr).astype(int), zero_division=0))


def build_scorer(scoring_mode="f1_05"):
    if scoring_mode == "f1_opt":
        return make_scorer(optimal_f1_score, response_method="predict_proba")
    if scoring_mode == "roc_auc":
        return "roc_auc"
    return make_scorer(f1_score, zero_division=0)


def compute_metrics(y_true, probs, threshold=0.5):
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


# =====================================================================
# 4. PIPELINES I ESPAIS D'HIPERPARÀMETRES
# =====================================================================
def build_search_configs(seed, scale_pos_weight, illicit_ratio, cache_dir):
    mutual_info = partial(mutual_info_classif, random_state=seed)
    mem = joblib.Memory(location=str(cache_dir), verbose=0)

    def make_pipe(model):
        return Pipeline([
            ('fe', TopologicalFeatureEngineer(add_ratios=True)),
            ('selector', SelectPercentile(mutual_info, percentile=100)),
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
            'fe__add_ratios': [False, True],
            'selector__percentile': [80, 90, 100],
            'model__n_estimators': randint(200, 650),
            'model__max_depth': [8, 12, 18, 26, None],
            'model__min_samples_split': randint(2, 12),
            'model__min_samples_leaf': randint(1, 5),
            'model__max_features': ['sqrt', 0.5, 0.7, 0.85],
        }),
        'XGBoost': ("halving", 60, {
            'fe__add_ratios': [False, True],
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
            'fe__add_ratios': [False, True],
            'selector__percentile': [90, 100],
            'scaler': [RobustScaler(), StandardScaler()],
            'model__hidden_layer_sizes': [(128, 64, 32), (256, 128), (256, 128, 64)],
            'model__alpha': [5e-4, 1e-3, 5e-3],
            'model__learning_rate_init': [1e-3, 2e-3, 3e-3],
        }),
        'SVM': ("grid", None, {
            'fe__add_ratios': [False, True],
            'selector__percentile': [100],
            'scaler': [RobustScaler(), StandardScaler()],
            'model__C': [2.0, 8.0, 16.0, 32.0],
        }),
        'L1 (Lasso)': ("grid", None, {
            'fe__add_ratios': [False, True],
            'selector__percentile': [70, 85, 100],
            'scaler': [RobustScaler(), StandardScaler()],
            'model__C': [0.01, 0.05, 0.2, 1.0, 5.0, 20.0],
        }),
        'ElasticNet': ("grid", None, {
            'fe__add_ratios': [False, True],
            'selector__percentile': [75, 90, 100],
            'scaler': [RobustScaler(), StandardScaler()],
            'model__C': [0.01, 0.1, 0.5, 2.0, 10.0],
            'model__l1_ratio': [0.2, 0.5, 0.8],
        }),
        'Naive Bayes': ("grid", None, {
            'fe__add_ratios': [False, True],
            'selector__percentile': [50, 65, 80, 90, 100],
            'scaler': [RobustScaler(), StandardScaler()],
            'model__var_smoothing': np.logspace(-10, -1, 10).tolist(),
        }),
        'IsolationForest': ("grid", None, {
            'fe__add_ratios': [False, True],
            'selector__percentile': [75, 100],
            'model__n_estimators': [200, 400],
            'model__max_samples': ['auto', 0.5],
            'model__contamination': ['auto', round(illicit_ratio * 0.85, 2), round(illicit_ratio, 2)],
            'model__flip_score': [False, True],
        }),
    }
    return models, search_spaces


# =====================================================================
# 5. EXECUCIÓ PRINCIPAL
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Optimització completa i paral·lelitzada de tots els models.")
    parser.add_argument("--data-dir", type=str, default=".", help="Carpeta on es troben train.csv, val.csv i test.csv")
    parser.add_argument("--out-dir", type=str, default="./tuning_results_pro", help="Carpeta on desar els resultats")
    parser.add_argument("--cv-mode", choices=["val", "kfold"], default="val", help="Validació fixa ('val') o StratifiedKFold ('kfold')")
    parser.add_argument("--folds", type=int, default=3, help="Nombre de folds si --cv-mode=kfold")
    parser.add_argument("--scoring", choices=["f1_05", "f1_opt", "roc_auc"], default="f1_05",
                        help="Mètrica de selecció d'hiperparàmetres: 'f1_05' (llindar 0.5), 'f1_opt' (llindar PR òptim) o 'roc_auc'")
    parser.add_argument("--models", nargs="+", default=None, help="Filtrar models específics (opcional)")
    parser.add_argument("--workers", type=int, default=-1, help="Nuclis de CPU (-1 = tots els disponibles)")
    parser.add_argument("--seed", type=int, default=42, help="Llavor aleatòria")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "_cache"
    models_dir = out_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(args.data_dir, args.seed)
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]

    scale_pos_weight = float(np.sum(y_train == 0) / max(1, np.sum(y_train == 1)))
    illicit_ratio = float(np.mean(y_train == 1))
    n_cores = os.cpu_count() if args.workers == -1 else args.workers
    print(f"Dataset carregat -> Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)} "
          f"(Il·lícits train: {illicit_ratio:.1%} | Nuclis actius: {n_cores})")

    if args.cv_mode == "val":
        X_tune = pd.concat([X_train, X_val], ignore_index=True)
        y_tune = np.concatenate([y_train, y_val])
        cv_split = PredefinedSplit(np.concatenate([np.full(len(y_train), -1), np.zeros(len(y_val), dtype=int)]))
        n_splits = 1
    else:
        X_tune, y_tune = X_train, y_train
        cv_split = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        n_splits = args.folds

    base_models, search_spaces = build_search_configs(args.seed, scale_pos_weight, illicit_ratio, cache_dir)
    scorer = build_scorer(args.scoring)
    selected_models = [m for m in (args.models if args.models else list(base_models.keys())) if m in base_models]

    summary_rows = []
    best_configs = {}
    val_probs_dict = {}
    test_probs_dict = {}
    best_pipes_dict = {}
    global_t0 = time.perf_counter()

    try:
        # -----------------------------------------------------------------
        # FASE 1: CERCA D'HIPERPARÀMETRES + RE-ENTRENAMENT MULTINUCLI
        # -----------------------------------------------------------------
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
                    random_state=args.seed,
                    n_jobs=args.workers,
                    refit=False,
                )
            else:
                search = GridSearchCV(
                    estimator=pipe,
                    param_grid=space,
                    scoring=scorer,
                    cv=cv_split,
                    n_jobs=args.workers,
                    refit=False,
                )

            desc = f"[{idx}/{len(selected_models)}] {name:15s} ({mode})"
            with tqdm_joblib(desc=desc, total=est_fits):
                search.fit(X_tune, y_tune)

            # Re-entrenem el millor pipeline sobre X_train usant TOTS els nuclis disponibles
            best_pipe = clone(pipe).set_params(memory=None, **search.best_params_)
            model_step = best_pipe.named_steps["model"]
            if hasattr(model_step, "n_jobs") and not isinstance(model_step, LogisticRegression):
                best_pipe.set_params(model__n_jobs=args.workers)

            best_pipe.fit(X_train, y_train)
            elapsed = round(time.perf_counter() - t0, 1)

            probs_val = best_pipe.predict_proba(X_val)[:, 1]
            probs_test = best_pipe.predict_proba(X_test)[:, 1]
            val_probs_dict[name] = probs_val
            test_probs_dict[name] = probs_test

            # Optimització del llindar sobre el conjunt de validació
            opt_thr = find_optimal_threshold(probs_val, y_val)
            val_m_05 = compute_metrics(y_val, probs_val, 0.5)
            val_m_opt = compute_metrics(y_val, probs_val, opt_thr)
            test_m_05 = compute_metrics(y_test, probs_test, 0.5)
            test_m_opt = compute_metrics(y_test, probs_test, opt_thr)

            clean_p = clean_params(search.best_params_)
            clean_p["optimal_threshold"] = round(opt_thr, 4)
            best_configs[name] = clean_p

            if hasattr(model_step, "n_jobs") and not isinstance(model_step, LogisticRegression):
                best_pipe.set_params(model__n_jobs=1)

            best_pipes_dict[name] = clone(best_pipe)

            safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
            joblib.dump({"pipeline": best_pipe, "threshold": opt_thr, "params": clean_p},
                        models_dir / f"{safe_name}.joblib")

            print(f"    -> {name:15s} completat en {elapsed:5.1f}s | "
                  f"Test F1 (@0.5): {test_m_05['f1']:.4f} -> Test F1 (@{opt_thr:.2f}): {test_m_opt['f1']:.4f} | "
                  f"Test AUC: {test_m_opt['auc_roc']:.4f} | Ratios: {clean_p.get('fe__add_ratios')}")

            summary_rows.append({
                "model": name,
                "search_type": mode,
                "add_ratios": clean_p.get("fe__add_ratios", False),
                "opt_threshold": round(opt_thr, 4),
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

        # -----------------------------------------------------------------
        # FASE 2: ENSEMBLING OOF AUTOMÀTIC (AVALUACIÓ DINÀMICA DEL TOP-N)
        # -----------------------------------------------------------------
        if len(summary_rows) >= 2:
            df_base = pd.DataFrame(summary_rows)
            df_base = df_base[df_base["search_type"] != "ensemble"].sort_values("val_f1_opt", ascending=False)
            candidate_names = [m for m in df_base["model"].tolist() if m in best_pipes_dict]

            max_k = min(4, len(candidate_names))
            top_eval_models = candidate_names[:max_k]

            print(f"\n[Fase 2] Generant prediccions Out-Of-Fold (5-Fold CV) sobre Train ({len(X_train)} mostres) "
                  f"per als millors models: {top_eval_models}...")

            oof_train_dict = {}
            cv_oof = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
            for m in top_eval_models:
                t_oof = time.perf_counter()
                p = clone(best_pipes_dict[m])
                oof_train_dict[m] = cross_val_predict(
                    p, X_train, y_train, cv=cv_oof, method="predict_proba", n_jobs=args.workers
                )[:, 1]
                print(f"    -> OOF {m:15s} completat en {time.perf_counter() - t_oof:5.1f}s")

            best_stacking_k = None
            best_stacking_val_f1 = -1.0

            # Avaluem Stacking i Soft-Voting per a k = 2, 3, ... max_k
            for k in range(2, max_k + 1):
                names_k = candidate_names[:k]
                k_label = f"Top{k}"

                # A) Soft-Voting
                t0 = time.perf_counter()
                weights = np.array([df_base.loc[df_base["model"] == m, "val_f1_opt"].values[0] for m in names_k])
                weights = weights / np.maximum(weights.sum(), 1e-12)

                ens_val_probs = sum(w * val_probs_dict[m] for w, m in zip(weights, names_k))
                ens_test_probs = sum(w * test_probs_dict[m] for w, m in zip(weights, names_k))
                ens_thr = find_optimal_threshold(ens_val_probs, y_val)

                ens_val_05 = compute_metrics(y_val, ens_val_probs, 0.5)
                ens_val_opt = compute_metrics(y_val, ens_val_probs, ens_thr)
                ens_test_05 = compute_metrics(y_test, ens_test_probs, 0.5)
                ens_test_opt = compute_metrics(y_test, ens_test_probs, ens_thr)
                ens_time = round(time.perf_counter() - t0, 2)

                voting_name = f"Ensemble_SoftVoting_{k_label}"
                voting_info = {
                    "models": names_k,
                    "weights": [round(float(w), 4) for w in weights],
                    "optimal_threshold": round(ens_thr, 4),
                }
                best_configs[voting_name] = voting_info
                print(f"    -> {voting_name:25s} ({', '.join(names_k)}) | "
                      f"Val F1: {ens_val_opt['f1']:.4f} | Test F1 (@{ens_thr:.2f}): {ens_test_opt['f1']:.4f} | "
                      f"Test AUC: {ens_test_opt['auc_roc']:.4f}")

                summary_rows.append({
                    "model": voting_name,
                    "search_type": "ensemble",
                    "add_ratios": True,
                    "opt_threshold": round(ens_thr, 4),
                    "val_f1_05": ens_val_05["f1"],
                    "val_f1_opt": ens_val_opt["f1"],
                    "test_f1_05": ens_test_05["f1"],
                    "test_f1_opt": ens_test_opt["f1"],
                    "test_auc_roc": ens_test_opt["auc_roc"],
                    "test_precision_opt": ens_test_opt["precision"],
                    "test_recall_opt": ens_test_opt["recall"],
                    "test_accuracy_opt": ens_test_opt["accuracy"],
                    "time_s": ens_time,
                    "best_params": json.dumps(voting_info),
                })

                # B) Stacking OOF Classifier (Entrenat sobre 25.018 mostres OOF de train sense fuga!)
                t0 = time.perf_counter()
                X_meta_train = np.column_stack([oof_train_dict[m] for m in names_k])
                X_meta_val = np.column_stack([val_probs_dict[m] for m in names_k])
                X_meta_test = np.column_stack([test_probs_dict[m] for m in names_k])

                meta_clf = LogisticRegression(class_weight="balanced", random_state=args.seed)
                meta_clf.fit(X_meta_train, y_train)

                stack_val_probs = meta_clf.predict_proba(X_meta_val)[:, 1]
                stack_test_probs = meta_clf.predict_proba(X_meta_test)[:, 1]
                stack_thr = find_optimal_threshold(stack_val_probs, y_val)

                stack_val_05 = compute_metrics(y_val, stack_val_probs, 0.5)
                stack_val_opt = compute_metrics(y_val, stack_val_probs, stack_thr)
                stack_test_05 = compute_metrics(y_test, stack_test_probs, 0.5)
                stack_test_opt = compute_metrics(y_test, stack_test_probs, stack_thr)
                stack_time = round(time.perf_counter() - t0, 2)

                stack_name = f"Ensemble_Stacking_{k_label}"
                stack_info = {
                    "models": names_k,
                    "meta_coefs": [round(float(c), 4) for c in meta_clf.coef_[0]],
                    "optimal_threshold": round(stack_thr, 4),
                    "val_f1_opt": stack_val_opt["f1"],
                }
                best_configs[stack_name] = stack_info
                joblib.dump({"meta_model": meta_clf, "base_models": names_k, "threshold": stack_thr},
                            models_dir / f"{stack_name}.joblib")

                if stack_val_opt["f1"] > best_stacking_val_f1:
                    best_stacking_val_f1 = stack_val_opt["f1"]
                    best_stacking_k = k
                    best_configs["Ensemble_Stacking_Best"] = {**stack_info, "best_k": k}
                    joblib.dump({"meta_model": meta_clf, "base_models": names_k, "threshold": stack_thr},
                                models_dir / "Ensemble_Stacking_Best.joblib")

                print(f"    -> {stack_name:25s} ({', '.join(names_k)}) | "
                      f"Val F1: {stack_val_opt['f1']:.4f} | Test F1 (@{stack_thr:.2f}): {stack_test_opt['f1']:.4f} | "
                      f"Test AUC: {stack_test_opt['auc_roc']:.4f} | Coefs: {stack_info['meta_coefs']}")

                summary_rows.append({
                    "model": stack_name,
                    "search_type": "ensemble",
                    "add_ratios": True,
                    "opt_threshold": round(stack_thr, 4),
                    "val_f1_05": stack_val_05["f1"],
                    "val_f1_opt": stack_val_opt["f1"],
                    "test_f1_05": stack_test_05["f1"],
                    "test_f1_opt": stack_test_opt["f1"],
                    "test_auc_roc": stack_test_opt["auc_roc"],
                    "test_precision_opt": stack_test_opt["precision"],
                    "test_recall_opt": stack_test_opt["recall"],
                    "test_accuracy_opt": stack_test_opt["accuracy"],
                    "time_s": stack_time,
                    "best_params": json.dumps(stack_info),
                })

            if best_stacking_k is not None:
                print(f"\n>>> Millor Ensemble Trobat: Top-{best_stacking_k} Stacking amb Val F1 = {best_stacking_val_f1:.4f} (desat a Ensemble_Stacking_Best.joblib)")

        df_final = pd.DataFrame(summary_rows).sort_values("val_f1_opt", ascending=False)
        df_final.to_csv(out_dir / "optimization_summary.csv", index=False)
        with open(out_dir / "best_hyperparameters_pro.json", "w", encoding="utf-8") as f:
            json.dump(best_configs, f, indent=2)

    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)

    total_min = (time.perf_counter() - global_t0) / 60.0
    print("\n" + "=" * 98)
    print(f"RESUM FINAL D'OPTIMITZACIÓ (Temps total: {total_min:.1f} min | Ordenat per Val F1)")
    print("=" * 98)
    cols = ["model", "add_ratios", "opt_threshold", "val_f1_opt", "test_f1_05", "test_f1_opt", "test_auc_roc", "time_s"]
    print(df_final[cols].to_string(index=False))
    print(f"\nResultats i models desats a: {out_dir.resolve()}")


if __name__ == "__main__":
    main()