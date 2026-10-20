# STATISTICAL TESTS - PAIRED WILCOXON SIGNED-RANK TESTS
#
# Compares the configurations from the results of the training scripts. All
# configurations of both methodologies are trained on the same seeds and
# evaluated on the same test addresses, so they are compared seed by seed:
#
#   1. Methodology I: the GNN architectures at each depth, and 4 vs 2 layers
#      of each architecture (gnn_runs.csv, from run_experiments.py)
#   2. Methodology II: the best model (highest mean F1-score over the seeds)
#      against every other model
#      (ego_runs.csv, from train_models.py)
#   3. Between methodologies: the best GNN configuration against every model
#      of the ego-centric methodology
#
# For every comparison (a vs b) and metric, it reports the mean paired
# difference (a - b), its 95% percentile bootstrap confidence interval and the
# exact two-sided Wilcoxon signed-rank test. Only the seeds present in both
# configurations are used, and comparisons with fewer than MIN_SEEDS_FOR_TESTS
# common seeds are skipped.
#
#   python wilcoxon_tests.py                 # all available comparisons
#   python wilcoxon_tests.py --metric f1     # print another metric (all are saved)
#   python wilcoxon_tests.py --gnn-runs FILE # GNN results from another file
#
# Output in STATISTICAL_TESTS_DIR: gnn_tests.csv, ego_tests.csv,
# cross_methodology_tests.csv and tests_config.json.

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from shared.experiment import experiment_snapshot, write_json
from shared.stats import paired_comparison

METRICS = ["accuracy", "precision", "recall", "f1", "auc_roc"]
EGO_RUNS_FILE = config.EGO_CENTRIC_RESULTS_DIR / "ego_runs.csv"

# Architecture comparisons in the order and direction of the paper
ARCHITECTURE_PAIRS = [("HGT", "GAT"), ("HGT", "HAN"), ("GAT", "HAN")]


def gnn_runs_file(path=None):
    """The GNN results: the given file, or the runs trained with this configuration."""
    if path:
        return Path(path)
    return config.MULTI_INSTANCE_RESULTS_DIR / "gnn_runs.csv"


def load_gnn_runs(path):
    """GNN runs with one 'configuration' column (e.g. GAT-2) and the metric names of the ego runs."""
    if not path.exists():
        return None
    print(f"GNN results: {path}")
    runs = pd.read_csv(path)
    runs["configuration"] = runs["arch"] + "-" + runs["layers"].astype(str)
    return runs.rename(columns={"test_" + m: m for m in METRICS})


def load_ego_runs():
    if not EGO_RUNS_FILE.exists():
        return None
    return pd.read_csv(EGO_RUNS_FILE).rename(columns={"model": "configuration"})


def compare(runs_a, runs_b, name_a, name_b, group):
    """Paired tests of configuration a against b, for every metric, on their common seeds."""
    a = runs_a.set_index("seed")
    b = runs_b.set_index("seed")
    seeds = sorted(set(a.index) & set(b.index))
    if len(seeds) < config.MIN_SEEDS_FOR_TESTS:
        print(f"  skipped {name_a} vs {name_b}: {len(seeds)} common seeds (minimum {config.MIN_SEEDS_FOR_TESTS})")
        return []
    rows = []
    for metric in METRICS:
        test = paired_comparison(a.loc[seeds, metric], b.loc[seeds, metric],
                                 n_resamples=config.BOOTSTRAP_RESAMPLES, seed=config.DATA_SEED)
        rows.append({"group": group, "a": name_a, "b": name_b, "metric": metric, **test})
    return rows


def gnn_tests(runs):
    configs = set(runs["configuration"])
    layers = sorted(runs["layers"].unique(), reverse=True)
    rows = []
    for l in layers:
        for x, y in ARCHITECTURE_PAIRS:
            a, b = f"{x}-{l}", f"{y}-{l}"
            if a in configs and b in configs:
                rows += compare(runs[runs["configuration"] == a], runs[runs["configuration"] == b], a, b, f"{l} layers")
    if len(layers) > 1:
        high, low = max(layers), min(layers)
        for arch in ["HGT", "GAT", "HAN"]:
            a, b = f"{arch}-{high}", f"{arch}-{low}"
            if a in configs and b in configs:
                rows += compare(runs[runs["configuration"] == a], runs[runs["configuration"] == b], a, b, f"{high} vs {low} layers")
    return rows


def best_model(runs, column="configuration", metric="f1"):
    """The configuration with the highest mean metric over its seeds (ties: the first
    in alphabetical order). Also the model interpreted by interpret_model.py."""
    means = runs.groupby(column)[metric].mean()
    return max(sorted(means.index), key=lambda name: means[name])


def ego_tests(runs):
    best = best_model(runs)
    rows = []
    for name in sorted(set(runs["configuration"]) - {best}):
        rows += compare(runs[runs["configuration"] == best], runs[runs["configuration"] == name], best, name, f"{best} vs others")
    return rows


def cross_tests(gnn_runs, ego_runs):
    """The best GNN configuration (highest mean F1-score) against every ego-centric model."""
    best_gnn = best_model(gnn_runs)
    rows = []
    for name in sorted(set(ego_runs["configuration"])):
        rows += compare(gnn_runs[gnn_runs["configuration"] == best_gnn], ego_runs[ego_runs["configuration"] == name],
                        best_gnn, name, "multi-instance vs ego-centric")
    return rows


def show(rows, metric, title):
    if not rows:
        return
    table = pd.DataFrame(rows)
    table = table[table["metric"] == metric][["a", "b", "n", "mean_diff", "ci_low", "ci_high", "W", "p_value"]]
    print(f"\n=== {title} ({metric}) ===")
    print(table.round(4).to_string(index=False))


def run_all(metric="f1", gnn_runs_path=None):
    config.STATISTICAL_TESTS_DIR.mkdir(parents=True, exist_ok=True)
    gnn_path = gnn_runs_file(gnn_runs_path)
    gnn_runs, ego_runs = load_gnn_runs(gnn_path), load_ego_runs()
    if gnn_runs is None and ego_runs is None:
        print("No results found: run run_experiments.py and/or train_models.py first.")
        return

    outputs = {}
    if gnn_runs is not None:
        outputs["gnn_tests.csv"] = ("Methodology I: GNN configurations", gnn_tests(gnn_runs))
    if ego_runs is not None:
        outputs["ego_tests.csv"] = ("Methodology II: tabular models", ego_tests(ego_runs))
    if gnn_runs is not None and ego_runs is not None:
        outputs["cross_methodology_tests.csv"] = ("Multi-instance vs ego-centric", cross_tests(gnn_runs, ego_runs))

    for file_name, (title, rows) in outputs.items():
        pd.DataFrame(rows).to_csv(config.STATISTICAL_TESTS_DIR / file_name, index=False)
        show(rows, metric, title)

    write_json(config.STATISTICAL_TESTS_DIR / "tests_config.json", experiment_snapshot(
        extra={
            "test": "exact two-sided Wilcoxon signed-rank test on the paired per-seed differences (a - b)",
            "confidence_interval": "95% percentile bootstrap of the mean paired difference",
            "bootstrap_resamples": config.BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": config.DATA_SEED,
            "min_seeds": config.MIN_SEEDS_FOR_TESTS,
            "metrics": METRICS,
            "inputs": {"gnn_runs": str(gnn_path) if gnn_runs is not None else None,
                       "ego_runs": str(EGO_RUNS_FILE) if ego_runs is not None else None},
            "seeds": {
                "gnn": sorted(gnn_runs["seed"].unique().tolist()) if gnn_runs is not None else [],
                "ego": sorted(ego_runs["seed"].unique().tolist()) if ego_runs is not None else [],
            },
        },
        modules=("numpy", "pandas", "scipy"),
    ))
    print(f"\nStatistical tests saved at: {config.STATISTICAL_TESTS_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Paired Wilcoxon signed-rank tests between the trained configurations.")
    parser.add_argument("--metric", default="f1", choices=METRICS, help="metric printed on screen (all are saved)")
    parser.add_argument("--gnn-runs", help="CSV with the GNN runs (default: see gnn_runs_file)")
    args = parser.parse_args()
    run_all(args.metric, args.gnn_runs)


if __name__ == "__main__":
    main()
