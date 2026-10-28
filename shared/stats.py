# Paired statistical comparisons between configurations trained on the same seeds.

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

# Largest number of nonzero differences whose 2^n sign assignments are enumerated
EXACT_MAX_N = 20


def _exact_wilcoxon(differences):
    """Two-sided Wilcoxon signed-rank test: the statistic (the smaller of the rank sums of
    the positive and the negative differences) and its exact p-value, the share of the 2^n
    equally likely sign assignments whose statistic is at most the observed one. Tied
    absolute differences take their average rank, so the p-value stays exact with ties,
    where SciPy switches to the normal approximation. Zero differences are discarded, as
    in SciPy's default."""
    d = np.asarray(differences, dtype=float)
    d = d[d != 0]
    if len(d) > EXACT_MAX_N:
        return wilcoxon(d)
    ranks = rankdata(np.abs(d))
    statistic = min(ranks[d > 0].sum(), ranks[d < 0].sum())
    signs = (np.arange(2 ** len(d))[:, None] >> np.arange(len(d))) & 1
    positive = signs @ ranks
    extreme = np.minimum(positive, ranks.sum() - positive) <= statistic + 1e-9
    return float(statistic), float(extreme.mean())


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
