# Verification of a splits file (address,label,family,split).
#
# Used by create_splits.py, for the splits of the pipeline and the seed addresses of
# the expansion experiments. It checks that the file is well formed and, if the
# BitcoinHeist CSV is given, that every seed address has the label and family it
# has in BitcoinHeist.

import pandas as pd

COLUMNS = ["address", "label", "family", "split"]


def read_bitcoinheist(path):
    """Illicit addresses with their family, and the licit (white only) addresses."""
    df = pd.read_csv(path, usecols=["address", "label"]).dropna(subset=["address", "label"])
    df["address"] = df["address"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    white = df["label"].str.lower() == "white"
    families = df[~white].groupby("address")["label"].unique()
    family = {address: values[0] for address, values in families.items() if len(values) == 1}
    multiple = sorted(address for address, values in families.items() if len(values) > 1)
    licit = set(df.loc[white, "address"]) - set(families.index)
    return family, licit, multiple


def verify_splits(splits, splits_list, val_per_class=None, test_per_class=None, bitcoinheist_csv=None):
    """Checks of a splits DataFrame. Returns a dict with the checks and counts, and raises
    ValueError listing every failed check."""
    errors = []
    missing = [c for c in COLUMNS if c not in splits.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if splits[COLUMNS].isna().any().any():
        errors.append("empty values")
    duplicated = int(splits["address"].duplicated().sum())
    if duplicated:
        errors.append(f"{duplicated} addresses appear more than once (in the same or different splits)")
    unknown_splits = sorted(set(splits["split"]) - set(splits_list))
    if unknown_splits:
        errors.append(f"unknown splits {unknown_splits}")
    unknown_labels = sorted(set(splits["label"]) - {0, 1})
    if unknown_labels:
        errors.append(f"labels other than 0 and 1: {unknown_labels}")

    counts = splits.groupby(["split", "label"]).size()
    per_split = {s: {"illicit": int(counts.get((s, 1), 0)), "licit": int(counts.get((s, 0), 0))} for s in splits_list}
    for s, n in per_split.items():
        if n["illicit"] != n["licit"]:
            errors.append(f"{s}: {n['illicit']} illicit and {n['licit']} licit addresses (classes not balanced)")
    for s, expected in [("val", val_per_class), ("test", test_per_class)]:
        if expected is not None and per_split.get(s, {}).get("illicit") != expected:
            errors.append(f"{s}: {per_split.get(s)} addresses per class, {expected} expected")

    checks = {"n_addresses": int(len(splits)), "per_split": per_split}

    if bitcoinheist_csv is not None:
        family, licit, multiple = read_bitcoinheist(bitcoinheist_csv)
        illicit_seeds = splits[splits["label"] == 1]
        licit_seeds = splits[splits["label"] == 0]
        not_illicit = int((~illicit_seeds["address"].isin(set(family) | set(multiple))).sum())
        not_licit = int((~licit_seeds["address"].isin(licit)).sum())
        single = illicit_seeds[illicit_seeds["address"].isin(set(family))]
        wrong_family = int((single["address"].map(family) != single["family"]).sum())
        wrong_white = int((licit_seeds["family"] != "white").sum())
        if not_illicit:
            errors.append(f"{not_illicit} illicit seeds are not illicit in BitcoinHeist")
        if not_licit:
            errors.append(f"{not_licit} licit seeds are not white-only addresses in BitcoinHeist")
        if wrong_family:
            errors.append(f"{wrong_family} illicit seeds have a family different from BitcoinHeist")
        if wrong_white:
            errors.append(f"{wrong_white} licit seeds do not have the family 'white'")
        checks["bitcoinheist"] = {
            "illicit_addresses": len(family) + len(multiple),
            "illicit_addresses_with_several_families": len(multiple),
            "illicit_addresses_used": int(illicit_seeds["address"].isin(set(family) | set(multiple)).sum()),
            "licit_addresses": len(licit),
        }

    if errors:
        raise ValueError("Invalid splits:\n  " + "\n  ".join(errors))
    return checks
