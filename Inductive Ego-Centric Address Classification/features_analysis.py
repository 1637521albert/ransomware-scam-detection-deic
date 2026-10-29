# AUXILIARY ANALYSIS (Methodology II) - TOPOLOGICAL FEATURES OF THE SEED ADDRESSES BY CLASS
#
# How every feature the models are trained on (config.EGO_CENTRIC_TRAINING_FEATURES)
# is distributed among the illicit and the licit seed addresses: the count features
# (motifs, peeling chains, degrees, eccentricity) by the share of the seeds with none,
# 1 to 3 or more (BINS), and every feature by its mean, standard deviation, quartiles
# and share of zeros. For instance, peeling_2 and peeling_3 count the paths of 2 and 3
# consecutive transactions with one input and two outputs that start at the seed, and
# 2star_out (B2) the transactions whose only input is the seed and that have two
# outputs (see extract_topological_features.py). It also contrasts the two classes on
# every feature: the probability that an illicit seed has a higher value than a licit
# one (AUC of the feature alone, ties counted as half) and the two-sided Mann-Whitney
# U test. It is computed for every split and for all of them together, from the feature
# files alone, so it runs as soon as the features are extracted: as a step of the
# pipeline (features-ego) or on its own. It takes seconds and is computed again every
# time, with the fingerprints of the features it comes from.
#
#   python features_analysis.py
#
# Output in EGO_CENTRIC_RESULTS_DIR/features_analysis:
#   summary.csv            every split, class and feature: seeds, mean, standard deviation,
#                          minimum, quartiles, maximum, share of zeros and, for the count
#                          features, share of the seeds in each bin
#   distribution.csv       seeds of every split, class and count feature with each value,
#                          and their share of the class
#   contrast.csv           every split and feature: AUC of the feature alone and the
#                          Mann-Whitney U test between the illicit and the licit seeds
#   features_analysis.json the features, bins and classes, and the data description and
#                          fingerprints of the features
#   count_features.png     share of the seeds of each class in each bin (all splits)
#   continuous_features.png  distribution of the other features by class (all splits)

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
for _p in (str(CURRENT_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

import config
from common import RESULTS_DIR, data_description, feature_files
from shared.experiment import git_commit, timestamp, write_json

OUT_DIR = RESULTS_DIR / "features_analysis"

# Features the models are trained on (column -> name in the manuscript)
NAMES = {"2star_out": "B2", "split_2+": "B3+", "2star_in": "A2", "merge_2+": "A3+",
         "peeling_2": "P2", "peeling_3": "P3+", "in_deg": "In-Degree", "out_deg": "Out-Degree",
         "pagerank": "PageRank", "betweenness": "Betweenness", "eccentricity": "Eccentricity",
         "density": "Density"}
FEATURES = list(config.EGO_CENTRIC_TRAINING_FEATURES)
# Value of a count feature: (name, lowest, highest), highest None = no limit
BINS = [("0", 0, 0), ("1-3", 1, 3), ("4+", 4, None)]
CLASSES = {1: "illicit", 0: "licit"}
ALL_SPLITS = "all"
QUANTILES = {"q25": 0.25, "median": 0.5, "q75": 0.75}


def name(feature):
    return NAMES.get(feature, feature)


def load_features():
    """Training features, label and split of every seed address (EGO_CENTRIC_FEATURES_DIR)."""
    missing = [str(path) for path in feature_files() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {missing}: run extract_topological_features.py first.")
    columns = ["addr_str", "label", "split", *FEATURES]
    return pd.concat([pd.read_csv(path, usecols=columns) for path in feature_files()], ignore_index=True)


def count_features(df):
    """The features whose values are all whole numbers (counts), in the order of FEATURES."""
    return [f for f in FEATURES if np.allclose(df[f], np.round(df[f]))]


def parts(df):
    """(split, seeds of the split) for every split and for all of them."""
    for split in config.SPLITS + [ALL_SPLITS]:
        yield split, (df if split == ALL_SPLITS else df[df["split"] == split])


def in_bin(values, low, high):
    mask = values >= low
    if high is not None:
        mask &= values <= high
    return mask


def summary(df, counts):
    rows = []
    for split, part in parts(df):
        for label, cls in CLASSES.items():
            seeds = part[part["label"] == label]
            for feature in FEATURES:
                values = seeds[feature]
                row = {"split": split, "class": cls, "feature": feature, "name": name(feature), "seeds": len(values),
                       "mean": values.mean(), "std": values.std(), "min": values.min(),
                       **{k: values.quantile(q) for k, q in QUANTILES.items()},
                       "max": values.max(), "share_zero": (values == 0).mean() if len(values) else np.nan}
                for bin_name, low, high in BINS:
                    row[f"share_{bin_name}"] = (in_bin(values, low, high).mean() if feature in counts and len(values)
                                                else np.nan)
                rows.append(row)
    return pd.DataFrame(rows)


def distribution(df, counts):
    rows = []
    for split, part in parts(df):
        for label, cls in CLASSES.items():
            seeds = part[part["label"] == label]
            for feature in counts:
                values = seeds[feature].astype(int)
                for value, n in values.value_counts().sort_index().items():
                    rows.append({"split": split, "class": cls, "feature": feature, "value": int(value),
                                 "seeds": int(n), "share": n / len(values)})
    return pd.DataFrame(rows)


def contrast(df):
    """AUC of every feature alone (illicit vs licit) and the two-sided Mann-Whitney U test."""
    rows = []
    for split, part in parts(df):
        illicit, licit = part[part["label"] == 1], part[part["label"] == 0]
        for feature in FEATURES:
            if not len(illicit) or not len(licit):
                continue
            test = mannwhitneyu(illicit[feature], licit[feature], alternative="two-sided")
            rows.append({"split": split, "feature": feature, "name": name(feature),
                         "illicit_seeds": len(illicit), "licit_seeds": len(licit),
                         "auc": test.statistic / (len(illicit) * len(licit)), "U": test.statistic,
                         "p_value": test.pvalue})
    return pd.DataFrame(rows)


def plot_counts(summary_df, counts, path):
    """Share of the seeds of each class in each bin, one panel per count feature (all splits)."""
    data = summary_df[summary_df["split"] == ALL_SPLITS]
    bins = [b[0] for b in BINS]
    x = np.arange(len(bins))
    width = 0.8 / len(CLASSES)
    ncols = min(4, len(counts))
    nrows = int(np.ceil(len(counts) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), sharey=True, squeeze=False)
    for ax, feature in zip(axes.flat, counts):
        for i, cls in enumerate(CLASSES.values()):
            row = data[(data["feature"] == feature) & (data["class"] == cls)].iloc[0]
            bars = ax.bar(x + (i - (len(CLASSES) - 1) / 2) * width, [100 * row[f"share_{b}"] for b in bins], width,
                          label=cls)
            ax.bar_label(bars, fmt="%.1f", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(bins)
        ax.set_title(name(feature))
        ax.set_ylim(0, 115)
    for ax in axes.flat[len(counts):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Seed addresses (%)")
    axes.flat[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_continuous(df, features, path):
    """Box plots of every other feature by class (all splits; whiskers at 1.5 IQR, no outliers)."""
    if not features:
        return
    fig, axes = plt.subplots(1, len(features), figsize=(3.2 * len(features), 3.2), squeeze=False)
    for ax, feature in zip(axes.flat, features):
        ax.boxplot([df.loc[df["label"] == label, feature] for label in CLASSES], labels=list(CLASSES.values()),
                   showfliers=False)
        ax.set_title(name(feature))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    df = load_features()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts = count_features(df)

    summary_df = summary(df, counts)
    contrast_df = contrast(df)
    summary_df.to_csv(OUT_DIR / "summary.csv", index=False)
    distribution(df, counts).to_csv(OUT_DIR / "distribution.csv", index=False)
    contrast_df.to_csv(OUT_DIR / "contrast.csv", index=False)
    plot_counts(summary_df, counts, OUT_DIR / "count_features.png")
    plot_continuous(df, [f for f in FEATURES if f not in counts], OUT_DIR / "continuous_features.png")
    write_json(OUT_DIR / "features_analysis.json", {
        "created": timestamp(),
        "git_commit": git_commit(config.PROJECT_ROOT),
        "features": {f: name(f) for f in FEATURES},
        "count_features": counts,
        "bins": [{"name": n, "lowest": low, "highest": high} for n, low, high in BINS],
        "classes": {str(label): cls for label, cls in CLASSES.items()},
        "contrast": "AUC of the feature alone (P[illicit > licit] + P[tie]/2) and two-sided Mann-Whitney U test",
        "seeds": summary_df[summary_df["feature"] == FEATURES[0]]
                 .pivot(index="split", columns="class", values="seeds").astype(int).to_dict(orient="index"),
        "data": data_description(),
    })

    shown = summary_df[summary_df["split"] == ALL_SPLITS]
    aucs = contrast_df[contrast_df["split"] == ALL_SPLITS].set_index("feature")["auc"]
    print("Topological features of the seed addresses by class (all splits):")
    for feature in FEATURES:
        print(f"\n  {name(feature)} ({feature}), AUC of the feature alone {aucs[feature]:.3f}")
        for _, row in shown[shown["feature"] == feature].iterrows():
            bins = (", ".join(f"{b[0]}: {100 * row[f'share_{b[0]}']:.1f}%" for b in BINS) + "; "
                    if feature in counts else "")
            print(f"    {row['class']:8s} ({row['seeds']} seeds)  {bins}mean {row['mean']:.4g}, "
                  f"median {row['median']:.4g}, zero {100 * row['share_zero']:.1f}%")
    print(f"\nSaved in {OUT_DIR}")


if __name__ == "__main__":
    main()
