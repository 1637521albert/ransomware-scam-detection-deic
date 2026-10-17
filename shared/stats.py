# Paired statistical comparisons between configurations trained on the same seeds.

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def _exact_wilcoxon(differences):
    # The keyword selecting the exact distribution changed name across SciPy versions
    try:
        return wilcoxon(differences, method="exact")
    except TypeError:
        return wilcoxon(differences, mode="exact")


def paired_comparison(a, b, n_resamples=10000, seed=0):
    """Compares two configurations evaluated on the same seeds (a - b).

    Returns the mean paired difference, its 95% percentile bootstrap confidence
    interval and the exact two-sided Wilcoxon signed-rank test.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b

    rng = np.random.RandomState(seed)
    boot = rng.choice(diff, size=(n_resamples, len(diff)), replace=True).mean(axis=1)
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

    if np.allclose(diff, 0):
        statistic, p_value = float("nan"), 1.0
    else:
        statistic, p_value = _exact_wilcoxon(diff)

    return {
        "mean_diff": float(diff.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "W": float(statistic),
        "p_value": float(p_value),
        "n": int(len(diff)),
    }


def summarize_runs(runs, group_cols, metrics):
    """Mean, standard deviation and median of every metric for each configuration."""
    rows = []
    for key, group in runs.groupby(group_cols, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key))
        row["n_runs"] = len(group)
        for metric in metrics:
            values = group[metric].astype(float)
            row[metric + "_mean"] = values.mean()
            row[metric + "_std"] = values.std(ddof=1) if len(values) > 1 else 0.0
            row[metric + "_median"] = values.median()
        rows.append(row)
    return pd.DataFrame(rows)
