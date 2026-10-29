# AUXILIARY ANALYSIS (Methodology II) - PEELING CHAINS AND BINARY BRANCHES OF THE SEED ADDRESSES
#
# How many peeling chains every seed address starts, and how many binary branches it
# makes, by class: the share of the illicit and of the licit seeds with none, 1 to 3 or
# more (BINS). They are features of extract_topological_features.py: peeling_2 and
# peeling_3 count the paths of 2 and 3 consecutive transactions with one input and two
# outputs that start at the seed, so a seed with peeling_2 = 0 starts no chain, and
# 2star_out (B2) counts the transactions whose only input is the seed and that have two
# outputs. It is computed for every split and for all of them together, from the feature
# files alone, so it runs as soon as the features are extracted: as a step of the
# pipeline (peeling-ego) or on its own. It takes seconds and is computed again every
# time, with the fingerprints of the features it comes from.
#
#   python peeling_analysis.py
#
# Output in EGO_CENTRIC_RESULTS_DIR/peeling_chains:
#   distribution.csv     seeds of every split, class and feature with each count, and
#                        their share of the class
#   summary.csv          share of the seeds of every split, class and feature in each bin,
#                        and the mean and median count
#   peeling_chains.json  the features, bins and classes, and the data description and
#                        fingerprints of the features
#   peeling_chains.png   share of the seeds of each class in each bin (all splits)

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

import config
from common import RESULTS_DIR, data_description, feature_files
from shared.experiment import git_commit, timestamp, write_json

OUT_DIR = RESULTS_DIR / "peeling_chains"

# Features (column -> name in the manuscript) and what each one counts
FEATURES = {"peeling_2": "P2", "peeling_3": "P3", "2star_out": "B2"}
COUNTED = {"peeling_2": "Peeling chains started by the seed", "peeling_3": "Peeling chains started by the seed",
           "2star_out": "Binary branches of the seed"}
# Count of a seed: (name, lowest, highest), highest None = no limit
BINS = [("0", 0, 0), ("1-3", 1, 3), ("4+", 4, None)]
CLASSES = {1: "illicit", 0: "licit"}
ALL_SPLITS = "all"


def load_features():
    """Features, label and split of every seed address (EGO_CENTRIC_FEATURES_DIR)."""
    missing = [str(path) for path in feature_files() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {missing}: run extract_topological_features.py first.")
    columns = ["addr_str", "label", "split", *FEATURES]
    return pd.concat([pd.read_csv(path, usecols=columns) for path in feature_files()], ignore_index=True)


def groups(df):
    """(split, class, feature, count of its seeds) for every split and all of them."""
    for split in config.SPLITS + [ALL_SPLITS]:
        part = df if split == ALL_SPLITS else df[df["split"] == split]
        for label, name in CLASSES.items():
            seeds = part[part["label"] == label]
            for feature in FEATURES:
                yield split, name, feature, seeds[feature].astype(int)


def in_bin(values, low, high):
    mask = values >= low
    if high is not None:
        mask &= values <= high
    return mask


def distribution(df):
    rows = []
    for split, name, feature, values in groups(df):
        for count, seeds in values.value_counts().sort_index().items():
            rows.append({"split": split, "class": name, "feature": feature, "count": int(count),
                         "seeds": int(seeds), "share": seeds / len(values)})
    return pd.DataFrame(rows)


def summary(df):
    rows = []
    for split, name, feature, values in groups(df):
        row = {"split": split, "class": name, "feature": feature, "seeds": len(values)}
        for bin_name, low, high in BINS:
            row[f"share_{bin_name}"] = in_bin(values, low, high).mean() if len(values) else np.nan
        row["mean"] = values.mean()
        row["median"] = values.median()
        rows.append(row)
    return pd.DataFrame(rows)


def plot(summary_df, path):
    """Share of the seeds of each class in each bin, one panel per feature (all splits)."""
    data = summary_df[summary_df["split"] == ALL_SPLITS]
    bins = [b[0] for b in BINS]
    x = np.arange(len(bins))
    width = 0.8 / len(CLASSES)
    fig, axes = plt.subplots(1, len(FEATURES), figsize=(4.5 * len(FEATURES), 3.5), sharey=True)
    for ax, (feature, label) in zip(np.atleast_1d(axes), FEATURES.items()):
        for i, name in enumerate(CLASSES.values()):
            row = data[(data["feature"] == feature) & (data["class"] == name)].iloc[0]
            shares = [100 * row[f"share_{b}"] for b in bins]
            bars = ax.bar(x + (i - (len(CLASSES) - 1) / 2) * width, shares, width, label=name)
            ax.bar_label(bars, fmt="%.1f", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(bins)
        ax.set_xlabel(f"{COUNTED[feature]} ({label})")
        ax.set_ylim(0, 110)
    np.atleast_1d(axes)[0].set_ylabel("Seed addresses (%)")
    np.atleast_1d(axes)[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    df = load_features()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    distribution_df = distribution(df)
    summary_df = summary(df)
    distribution_df.to_csv(OUT_DIR / "distribution.csv", index=False)
    summary_df.to_csv(OUT_DIR / "summary.csv", index=False)
    plot(summary_df, OUT_DIR / "peeling_chains.png")
    write_json(OUT_DIR / "peeling_chains.json", {
        "created": timestamp(),
        "git_commit": git_commit(config.PROJECT_ROOT),
        "features": FEATURES,
        "bins": [{"name": name, "lowest": low, "highest": high} for name, low, high in BINS],
        "classes": {str(label): name for label, name in CLASSES.items()},
        "seeds": summary_df[summary_df["feature"] == next(iter(FEATURES))]
                 .pivot(index="split", columns="class", values="seeds").astype(int).to_dict(orient="index"),
        "data": data_description(),
    })

    shown = summary_df[summary_df["split"] == ALL_SPLITS]
    print("Peeling chains started and binary branches made by the seed addresses (all splits):")
    for _, row in shown.iterrows():
        shares = ", ".join(f"{b[0]}: {100 * row[f'share_{b[0]}']:.1f}%" for b in BINS)
        print(f"  {FEATURES[row['feature']]} {row['class']:8s} ({row['seeds']} seeds)  {shares}  "
              f"mean {row['mean']:.2f}, median {row['median']:g}")
    print(f"\nSaved in {OUT_DIR}")


if __name__ == "__main__":
    main()
