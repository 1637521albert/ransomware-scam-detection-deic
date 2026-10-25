# Feature tables of the baseline experiments: one row per seed address of the splits
# (address, label, split and the features of baseline_config.FEATURE_SETS).
#
# Every table is built from files that the pipeline or the BitcoinHeist dataset already
# provide, and only read: nothing of the methodologies is changed or computed again.
#   bitcoinheist  RSD_BITCOINHEIST_CSV: the daily records of every seed address, summarized
#   address       MULTI_INSTANCE_GRAPHS_DIR/<split>/addr_feats.csv (expansion.py of
#                 Methodology I): the row of every seed address, before normalization
#   topological   EGO_CENTRIC_FEATURES_DIR/<split>.csv (Methodology II)
# A table must describe every seed address of every split, with its label: otherwise
# the experiment stops, so every feature set is evaluated on the same addresses as both
# methodologies.
#
# A table is saved in FEATURES_DIR with its fingerprint (the content of its input files
# and of the splits file, and the code that builds it) and reused while the fingerprint
# is current.

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
for _p in (str(CURRENT_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
from baseline_config import BITCOINHEIST_ATTRIBUTES, BITCOINHEIST_STATISTICS, FEATURE_SETS, FEATURES_DIR
from shared.experiment import code_fingerprint, digest, file_fingerprint, read_json, timestamp, write_json
from shared.paths import get_required_env_path
from shared.splits import verify_splits

ID_COLUMNS = ["address", "label", "split"]
# Columns of addr_feats.csv that are not features of the GNNs (serialize_graph.build_graph);
# full_type is encoded as in serialize_graph.py
ADDRESS_NON_FEATURES = ["addr_str", "class", "full_type", "incoming_counterparties", "outgoing_counterparties"]
CHUNK_ROWS = 1000000


def load_splits():
    """The seed addresses of every split (config.SPLITS_FILE), checked."""
    if not config.SPLITS_FILE.exists():
        raise FileNotFoundError(f"Missing {config.SPLITS_FILE}: run create_splits.py first.")
    splits = pd.read_csv(config.SPLITS_FILE)
    verify_splits(splits, config.SPLITS)
    return splits[ID_COLUMNS].copy()


def input_files(feature_set):
    """The files a feature table is built from (besides the splits file)."""
    if feature_set == "bitcoinheist":
        return [get_required_env_path("RSD_BITCOINHEIST_CSV")]
    if feature_set == "address":
        return [config.MULTI_INSTANCE_GRAPHS_DIR / split / "addr_feats.csv" for split in config.SPLITS]
    if feature_set == "topological":
        return [config.EGO_CENTRIC_FEATURES_DIR / f"{split}.csv" for split in config.SPLITS]
    raise ValueError(f"Unknown feature set '{feature_set}'; available: {', '.join(FEATURE_SETS)}")


def bitcoinheist_table(splits):
    """The daily BitcoinHeist records of every seed address, summarized per address: the
    statistics of every attribute over its records, its number of records and of active
    days, the share of its records that are looped, and its first and last active day
    (days since 2009-01-01)."""
    path = get_required_env_path("RSD_BITCOINHEIST_CSV")
    seeds = set(splits["address"])
    columns = ["address", "year", "day"] + BITCOINHEIST_ATTRIBUTES
    parts = []
    for chunk in pd.read_csv(path, usecols=columns, chunksize=CHUNK_ROWS):
        chunk["address"] = chunk["address"].astype(str).str.strip()
        parts.append(chunk[chunk["address"].isin(seeds)])
    records = pd.concat(parts, ignore_index=True)
    dates = pd.to_datetime(records["year"].astype(int).astype(str), format="%Y") + \
        pd.to_timedelta(records["day"].astype(int) - 1, unit="D")
    records["day_index"] = (dates - pd.Timestamp("2009-01-01")).dt.days

    grouped = records.groupby("address")
    table = grouped[BITCOINHEIST_ATTRIBUTES].agg(BITCOINHEIST_STATISTICS)
    table.columns = [f"{statistic}_{attribute}" for attribute, statistic in table.columns]
    table["n_records"] = grouped.size()
    table["n_active_days"] = grouped["day_index"].nunique()
    table["looped_ratio"] = (records["looped"] > 0).groupby(records["address"]).mean()
    table["first_day"] = grouped["day_index"].min()
    table["last_day"] = grouped["day_index"].max()
    table = table.reset_index()
    return splits.merge(table, on="address", how="left", validate="one_to_one")


def address_table(splits):
    """The row of every seed address in the address features of Methodology I (the file of
    its own split), with its full type encoded as in serialize_graph.py."""
    sys.path.insert(0, str(PROJECT_ROOT / "Inductive Multi-Instance Address Classification"))
    from serialize_graph import FULL_TYPE_COLUMNS, encode_full_type

    parts = []
    for split in config.SPLITS:
        seeds = splits[splits["split"] == split]
        wanted = set(seeds["address"])
        rows = [chunk[chunk["addr_str"].astype(str).isin(wanted)]
                for chunk in pd.read_csv(config.MULTI_INSTANCE_GRAPHS_DIR / split / "addr_feats.csv",
                                         chunksize=CHUNK_ROWS)]
        rows = pd.concat(rows, ignore_index=True)
        rows["addr_str"] = rows["addr_str"].astype(str)
        part = seeds.merge(rows, left_on="address", right_on="addr_str", how="left", validate="one_to_one")
        # The label written by the expansion must be the label of the split
        found = part["class"].notna()
        mismatched = int((part.loc[found, "class"].astype(int) != part.loc[found, "label"]).sum())
        if mismatched:
            raise ValueError(f"{split}: {mismatched} seed addresses have another label in addr_feats.csv")
        full_type = pd.DataFrame(np.full((len(part), len(FULL_TYPE_COLUMNS)), np.nan), columns=FULL_TYPE_COLUMNS)
        full_type.loc[found.values, :] = encode_full_type(part.loc[found, "full_type"].to_numpy())
        features = [c for c in rows.columns if c not in ADDRESS_NON_FEATURES]
        parts.append(pd.concat([part[ID_COLUMNS].reset_index(drop=True), full_type,
                                part[features].reset_index(drop=True)], axis=1))
    return pd.concat(parts, ignore_index=True)


def topological_table(splits):
    """The topological features of Methodology II of every seed address."""
    features = list(config.EGO_CENTRIC_TRAINING_FEATURES)
    parts = []
    for split in config.SPLITS:
        rows = pd.read_csv(config.EGO_CENTRIC_FEATURES_DIR / f"{split}.csv")
        rows["addr_str"] = rows["addr_str"].astype(str)
        seeds = splits[splits["split"] == split]
        part = seeds.merge(rows[["addr_str", "label"] + features].rename(columns={"label": "feature_label"}),
                           left_on="address", right_on="addr_str", how="left", validate="one_to_one")
        found = part["feature_label"].notna()
        mismatched = int((part.loc[found, "feature_label"].astype(int) != part.loc[found, "label"]).sum())
        if mismatched:
            raise ValueError(f"{split}: {mismatched} seed addresses have another label in {split}.csv")
        parts.append(part[ID_COLUMNS + features])
    return pd.concat(parts, ignore_index=True)


BUILDERS = {"bitcoinheist": bitcoinheist_table, "address": address_table, "topological": topological_table}
CODE = {Path(__file__).resolve(): ["load_splits", "bitcoinheist_table", "address_table", "topological_table",
                                   "check_table"]}


def check_table(feature_set, table, splits):
    """Every seed address of every split once, with its label and at least one known
    feature; infinite values become missing (imputed by the models on the training
    split). Returns the number of missing values per feature."""
    if len(table) != len(splits) or table["address"].duplicated().any():
        raise ValueError(f"{feature_set}: {len(table)} rows for {len(splits)} seed addresses")
    features = [c for c in table.columns if c not in ID_COLUMNS]
    values = table[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    table[features] = values
    absent = values.isna().all(axis=1)
    if absent.any():
        missing = table.loc[absent].groupby("split").size().to_dict()
        raise ValueError(f"{feature_set}: {int(absent.sum())} seed addresses have no features ({missing}); "
                         f"every seed address must be described, so all the feature sets are evaluated on the "
                         f"same addresses")
    return {c: int(n) for c, n in values.isna().sum().items() if n}


def table_fingerprint(feature_set):
    return digest({"feature_set": feature_set, "inputs": file_fingerprint(input_files(feature_set)),
                   "splits": file_fingerprint([config.SPLITS_FILE]), "code": code_fingerprint(CODE),
                   "attributes": BITCOINHEIST_ATTRIBUTES, "statistics": BITCOINHEIST_STATISTICS,
                   "topological": list(config.EGO_CENTRIC_TRAINING_FEATURES)})


def missing_inputs(feature_set):
    """The input files of a feature set that do not exist (BitcoinHeist: if the variable
    is not set)."""
    try:
        return [str(p) for p in input_files(feature_set) if not Path(p).exists()]
    except RuntimeError as error:
        return [str(error)]


def feature_table(feature_set, splits):
    """The feature table of a feature set: the saved one if its fingerprint is current,
    otherwise built, checked and saved. Returns the table and its description."""
    fingerprint = table_fingerprint(feature_set)
    csv_path, json_path = FEATURES_DIR / f"{feature_set}.csv", FEATURES_DIR / f"{feature_set}.json"
    if csv_path.exists() and json_path.exists() and read_json(json_path).get("fingerprint") == fingerprint:
        print(f"{feature_set}: features already built from the current inputs ({csv_path})")
        return pd.read_csv(csv_path), read_json(json_path)

    print(f"{feature_set}: building the features of {len(splits)} seed addresses...", flush=True)
    table = BUILDERS[feature_set](splits)
    missing = check_table(feature_set, table, splits)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    description = {
        "feature_set": feature_set,
        "description": FEATURE_SETS[feature_set],
        "fingerprint": fingerprint,
        "created": timestamp(),
        "inputs": [str(p) for p in input_files(feature_set)],
        "n_addresses": int(len(table)),
        "features": [c for c in table.columns if c not in ID_COLUMNS],
        "missing_values": missing,
    }
    # Written last: with its fingerprint, it marks the table as complete
    write_json(json_path, description)
    if missing:
        print(f"{feature_set}: missing values (imputed with the training median): {missing}")
    return table, description
