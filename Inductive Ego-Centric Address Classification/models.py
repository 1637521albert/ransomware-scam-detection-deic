#!/usr/bin/env python
# coding: utf-8

import os
import time
import shap
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Preprocessing & Selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, FunctionTransformer
from sklearn.feature_selection import mutual_info_classif, SelectPercentile
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin, clone

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Metrics & Evaluation
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve, 
    average_precision_score, precision_recall_curve, 
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score,
    recall_score, f1_score
)
from sklearn.inspection import permutation_importance

# Statistics
from scipy.stats import mannwhitneyu, iqr, ttest_rel

# Project path resolution & Environment configuration
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.paths import get_required_env_path

# Configuration setup
shap.initjs()
sns.set_theme(style="whitegrid")

SEEDS = [42, 185, 205, 10, 436]

RANDOM_STATE = SEEDS[0]

RADIUS = 6

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

RES_DIR = Path(__file__).resolve().parent

print(f"Target dataset directory resolved at: {RES_DIR}\n")

# --- Dataset Loading ---
print("Loading dataset...")
csv_path = RES_DIR / "full_dataset_6.csv"
if not csv_path.exists():
    raise FileNotFoundError(f"Dataset file not found at: {csv_path}")

df = pd.read_csv(csv_path)
df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
print("Dataset loaded and shuffled successfully!\n")

# --- Data Splitting (Train/Val/Test) ---
print("Splitting data according to predefined 'split' column...")

y = df['label'].astype(int).values
splits = df['split'].values
X = df[TRAINING_FEATURES].copy() 
feature_names = X.columns.tolist()

# Boolean masks for splits
mask_train = (splits == 'train')
mask_val   = (splits == 'val')
mask_test  = (splits == 'test')

# Raw feature separation
X_train_raw, y_train = X[mask_train], y[mask_train]
X_val_raw,   y_val   = X[mask_val],   y[mask_val]
X_test_raw,  y_test  = X[mask_test],  y[mask_test]

print(f"Total samples: {len(df)}")
print(f"Train: {len(y_train)} samples ({len(y_train)/len(df):.1%})")
print(f"Val:   {len(y_val)} samples ({len(y_val)/len(df):.1%})")
print(f"Test:  {len(y_test)} samples ({len(y_test)/len(df):.1%})\n")

try:
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
    print(f"Calculated 'scale_pos_weight' for XGBoost (class 0 / class 1): {scale_pos_weight:.2f}")
except ZeroDivisionError:
    print("Error: No class 1 samples found in y_train. Setting 'scale_pos_weight = 1'.")
    scale_pos_weight = 1

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
    "scale_pos_weight": scale_pos_weight,
    
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
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        self.model.fit(X)
        return self

    def predict(self, X):
        pred_iso = self.model.predict(X)
        return (pred_iso == -1).astype(int)

    def predict_proba(self, X):
        scores = -self.model.decision_function(X)
        if len(scores) == 0:
            return np.array([]).reshape(0, 2)
        if (scores.max() - scores.min()) == 0:
            scores_scaled = np.full_like(scores, 0.5)
        else:
            scores_scaled = (scores - scores.min()) / (scores.max() - scores.min())
        return np.vstack((1 - scores_scaled, scores_scaled)).T

# --- Pipeline Components ---
selector = SelectPercentile(mutual_info_classif, percentile=HP["selector_percentile"])
log_transformer = FunctionTransformer(np.log1p, validate=True)
scaler = RobustScaler() if HP["scaler"] == "RobustScaler" else StandardScaler()

# 1. Logistic Regression (L1 - Lasso)
pipe_l1 = Pipeline([
    ('selector', selector), ('log', log_transformer), ('scaler', scaler),
    ('model', LogisticRegression(penalty=HP["logreg_penalty_l1"], solver='saga', max_iter=HP["logreg_max_iter"], C=HP["logreg_C"], class_weight='balanced', random_state=RANDOM_STATE))
], verbose=False)

# 2. Logistic Regression (ElasticNet)
pipe_el = Pipeline([
    ('selector', selector), ('log', log_transformer), ('scaler', scaler),
    ('model', LogisticRegression(penalty=HP["logreg_penalty_el"], solver='saga', l1_ratio=HP["logreg_l1_ratio"], max_iter=HP["logreg_max_iter"], C=HP["logreg_C"], class_weight='balanced', random_state=RANDOM_STATE))
], verbose=False)

# 3. Random Forest
pipe_rf = Pipeline([
    ('selector', selector), ('log', log_transformer), ('scaler', scaler),
    ('model', RandomForestClassifier(n_estimators=HP["rf_n_estimators"], max_depth=HP["rf_max_depth"], class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1))
], verbose=False)

# 4. XGBoost
pipe_xgb = Pipeline([
    ('selector', selector), ('log', log_transformer), ('scaler', scaler),
    ('model', XGBClassifier(n_estimators=HP["xgb_n_estimators"], max_depth=HP["xgb_max_depth"], learning_rate=HP["xgb_learning_rate"], subsample=HP["xgb_subsample"], colsample_bytree=HP["xgb_colsample_bytree"], scale_pos_weight=scale_pos_weight, use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1))
], verbose=False)

# 5. Isolation Forest
pipe_iso = Pipeline([
    ('selector', selector), ('log', log_transformer), ('scaler', scaler),
    ('model', IsolationForestClassifier(n_estimators=HP["iso_n_estimators"], contamination=HP["iso_contamination"], random_state=RANDOM_STATE, n_jobs=-1))
], verbose=False)

# 6. Support Vector Machine (SVM)
pipe_svm = Pipeline([
    ('selector', selector), ('log', log_transformer), ('scaler', scaler),
    ('model', SVC(C=HP["svm_C"], kernel=HP["svm_kernel"], class_weight='balanced', probability=True, random_state=RANDOM_STATE))
], verbose=False)

# 7. Naive Bayes
pipe_nb = Pipeline([
    ('selector', selector), ('log', log_transformer), ('scaler', scaler),
    ('model', GaussianNB())
], verbose=False)

# 8. Multi-layer Perceptron (MLP)
pipe_mlp = Pipeline([
    ('selector', selector), ('log', log_transformer), ('scaler', scaler),
    ('model', MLPClassifier(hidden_layer_sizes=HP["mlp_hidden_layers"], alpha=HP["mlp_alpha"], max_iter=HP["mlp_max_iter"], solver=HP["mlp_solver"], learning_rate_init=HP["mlp_learning_rate_init"], random_state=RANDOM_STATE, early_stopping=True, n_iter_no_change=20, validation_fraction=0.1))
], verbose=False)

print("All ML classification pipelines successfully initialized.")

# --- Evaluation Function ---
def evaluate_models(models_dict, X_test, y_test):
    results = []
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_prob)
        else:
            roc_auc = None
        results.append({
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc
        })
    return pd.DataFrame(results).set_index("model")

models_base = {
    'L1 (Lasso)': pipe_l1,
    'ElasticNet': pipe_el,
    'RandomForest': pipe_rf,
    'XGBoost': pipe_xgb,              
    'IsolationForest': pipe_iso,
    'SVM': pipe_svm,           
    'Naive Bayes': pipe_nb,    
    'MLP': pipe_mlp            
}

metrics = {metric: {name: [] for name in models_base.keys()} for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']}

print("\n=== Starting Iterative Training across Multiple Seeds ===")
for current_seed in tqdm(SEEDS, desc="Iteration Seeds"):
    models_iter = {}
    for name, pipe in models_base.items():
        pipe_clone = clone(pipe)
        if 'model__random_state' in pipe_clone.get_params():
            pipe_clone.set_params(model__random_state=current_seed)
        pipe_clone.fit(X_train_raw, y_train)
        models_iter[name] = pipe_clone
        
    metrics_df = evaluate_models(models_iter, X_test_raw, y_test)
    for name in models_base.keys():
        for metric in metrics.keys():
            metrics[metric][name].append(metrics_df.loc[name, metric])

# --- Results Summary Table ---
print("\n=== MODEL PERFORMANCE SUMMARY (Mean ± Standard Deviation) ===")
print(f"{'Model':<18s} | {'Accuracy':<16s} | {'Precision':<16s} | {'Recall':<16s} | {'F1-score':<16s} | {'ROC AUC':<16s}")
print("-" * 110)

for name in models_base.keys():
    row_str = f"{name:<18s} | "
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
        mean_val = np.mean(metrics[metric][name])
        std_val = np.std(metrics[metric][name])
        if pd.isna(mean_val):
            row_str += f"{'N/A':<16s} | "
        else:
            row_str += f"{mean_val:.3f} ± {std_val:.3f} | "
    print(row_str)

# --- Statistical Significance Testing ---
print("\n=== STATISTICAL SIGNIFICANCE TESTING (Paired t-test: MLP vs others) ===")
for name in ['XGBoost', 'RandomForest', 'SVM']:
    t_stat, p_val = ttest_rel(metrics['f1_score']['MLP'], metrics['f1_score'][name])
    sig = "SIGNIFICANT" if p_val < 0.05 else "NOT significant"
    print(f"MLP vs {name:<14s} -> p-value: {p_val:.5e} ({sig})")

# Best Model Assignment for Interpretability & Ablation
best_model_name = 'MLP'
best_model_pipe = models_iter['MLP']
best_model_safe_name = "MLP"
print(f"\nModel {best_model_name} selected for explainability and feature ablation analysis.")

# --- Feature Ablation Study ---
print(f"\nStarting Robust Feature Ablation Study ({best_model_name}, on validation set)...")
model_step = best_model_pipe.named_steps['model']
selector_step = best_model_pipe.named_steps['selector']
scaler_step = best_model_pipe.named_steps['scaler']

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
        n_repeats=10, random_state=42, n_jobs=-1, scoring='roc_auc'
    )
    support_mask = selector_step.get_support()
    importances = r.importances_mean[support_mask]
    importance_type = "permutation_importance (Mean Decrease AUC)"

sel_features_names = X_train_raw.columns[selector_step.get_support()].tolist()
X_train_sel = selector_step.transform(X_train_raw)
X_val_sel = selector_step.transform(X_val_raw)
X_train_scaled = scaler_step.transform(X_train_sel)
X_val_scaled = scaler_step.transform(X_val_sel)

if len(importances) != len(sel_features_names):
    print(f"Error: Feature importances length ({len(importances)}) does not match features count ({len(sel_features_names)})")
else:
    ordered_indices = np.argsort(importances)[::-1]
    ordered_feats_names = [sel_features_names[i] for i in ordered_indices]
    
    print(f"Feature ranking criterion: {importance_type}")
    print(f"Top 5 ranked features: {ordered_feats_names[:5]}")

    auc_list = []
    model_abl = clone(model_step) 
    step_size = 1 
    
    print(f"Retraining {best_model_name} incrementally...")
    for k in tqdm(range(1, len(ordered_indices) + 1, step_size)):
        subset_indices = ordered_indices[:k]
        X_train_sub = X_train_scaled[:, subset_indices]
        X_val_sub   = X_val_scaled[:, subset_indices]
        
        model_abl.fit(X_train_sub, y_train)
        
        if hasattr(model_abl, "predict_proba"):
            y_val_prob_k = model_abl.predict_proba(X_val_sub)[:, 1]
            auc_k = roc_auc_score(y_val, y_val_prob_k)
            auc_list.append(auc_k)
        else:
            auc_list.append(0.5)

    ablation_results_df = pd.DataFrame({
        'k_features': range(1, len(auc_list) + 1, step_size),
        'roc_auc': auc_list,
        'feature_added': ordered_feats_names[::step_size]
    })

    print("\n=== Feature Ablation Study Results ===")
    print(ablation_results_df.head(15).to_string(index=False))
    
    # Plotting Ablation Curve
    x_labels_latex = [get_label(f) for f in ablation_results_df['feature_added']]
    plt.figure(figsize=(7, 6))
    plt.plot(ablation_results_df['k_features'], ablation_results_df['roc_auc'], 
             marker='o', linestyle='-', color='#1f77b4', linewidth=1.5)
    plt.xticks(
        ticks=ablation_results_df['k_features'],
        labels=x_labels_latex,
        rotation=45,
        ha='right',
        fontsize=10
    )
    plt.xlabel('Feature Added (Cumulative)', fontsize=11)
    plt.ylabel("AUC-ROC", fontsize=11)
    plt.title(f"Feature Ablation Study: {best_model_name} Performance", fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    figures_dir = PROJECT_ROOT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir / f"ablation_{best_model_safe_name}.pdf", bbox_inches='tight')
    plt.show()

# --- Deep SHAP Analysis ---
print(f"\nStarting deep SHAP analysis for the best model: {best_model_name}")

if 'selector' in best_model_pipe.named_steps:
    selector_step = best_model_pipe.named_steps['selector']
    X_train_data = selector_step.transform(X_train_raw)
    X_val_data = selector_step.transform(X_val_raw)
    raw_feature_names = X_train_raw.columns[selector_step.get_support()].tolist()
else:
    X_train_data = X_train_raw
    X_val_data = X_val_raw
    raw_feature_names = X_train_raw.columns.tolist()

feature_names = [FEATURE_NAMES_LATEX.get(name, name) for name in raw_feature_names]

log_step = best_model_pipe.named_steps['log']
scaler_step = best_model_pipe.named_steps['scaler']
model_step = best_model_pipe.named_steps['model']

X_val_real_df = pd.DataFrame(X_val_data, columns=feature_names)
X_val_plot = log_step.transform(X_val_data)
X_val_plot_df = pd.DataFrame(X_val_plot, columns=feature_names)
X_val_model = scaler_step.transform(X_val_plot)
X_train_log = log_step.transform(X_train_data)
X_train_model = scaler_step.transform(X_train_log)

model_type_name = type(model_step).__name__
print(f"Detected model type: {model_type_name}. Configuring optimized SHAP explainer...")

explainer = None
k_clusters = 'N/A'

if model_type_name in ['RandomForestClassifier', 'XGBClassifier', 'IsolationForestClassifier']:
    N_SAMPLES_SHAP = 500 
    explainer = shap.TreeExplainer(model_step, X_train_model)
elif model_type_name in ['LogisticRegression', 'GaussianNB']:
    k_clusters = 75
    N_SAMPLES_SHAP = 700
    background_summary = shap.kmeans(X_train_model, k_clusters)
    explainer = shap.LinearExplainer(model_step, background_summary)
else:
    k_clusters = 50
    N_SAMPLES_SHAP = 8340
    background_summary = shap.kmeans(X_train_model, k_clusters)
    explainer = shap.KernelExplainer(model_step.predict_proba, background_summary)

print(f"Computing SHAP values for {best_model_name}...")

if isinstance(explainer, shap.KernelExplainer):
    shap_values_raw = explainer.shap_values(X_val_model[:N_SAMPLES_SHAP], nsamples="auto")
    shap_values_class1_array = shap_values_raw[:, :, 1] 
    base_value = explainer.expected_value[1]
    
    shap_values_class1 = shap.Explanation(
        values=shap_values_class1_array,
        base_values=base_value,
        data=X_val_plot_df.iloc[:N_SAMPLES_SHAP].values,
        feature_names=feature_names
    )
else:
    shap_values_obj = explainer(X_val_model[:N_SAMPLES_SHAP], check_additivity=False)
    shap_values_class1 = shap_values_obj[..., 1]
    shap_values_class1.data = X_val_plot_df.iloc[:N_SAMPLES_SHAP].values
    shap_values_class1.feature_names = feature_names

print("SHAP calculation completed successfully.")

# Generate SHAP Plots
print("Generating SHAP Summary Plot (Beeswarm)...")
shap.summary_plot(shap_values_class1, plot_type="dot", show=False)
plt.title(f'Feature Impact (Beeswarm) - Model: {best_model_name}')
plt.savefig(figures_dir / f"shap_summary_dot_{best_model_safe_name}.pdf", bbox_inches='tight')
plt.show()

print("Generating SHAP Bar Plot (Global Importance)...")
shap.summary_plot(shap_values_class1, plot_type="bar", show=False)
plt.title(f'Global Feature Importance - Model: {best_model_name}')
plt.savefig(figures_dir / f"shap_summary_bar_{best_model_safe_name}.pdf", bbox_inches='tight')
plt.show()
