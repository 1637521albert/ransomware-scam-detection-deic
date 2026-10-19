# STEP 0 - SEED ADDRESSES AND SPLITS
#
# Provides the labeled seed addresses and their training, validation and test
# splits. Both methodologies read the same file, so they are trained and
# evaluated on exactly the same addresses. Depending on config.SPLITS_MODE:
#
# "paper" (default): the splits of the paper, provided in paper/splits.csv, are
# verified (shared/splits.py; also against BitcoinHeist if RSD_BITCOINHEIST_CSV
# is set) and summarized. If BlockSci is available (RSD_BLOCKSCI_CONFIG), every
# seed address is also checked to exist in the chain, so that no seed is
# silently skipped by the expansions. No random sampling is used.
#
# "new": new splits are selected from BitcoinHeist and the chain:
#   - Illicit seeds: every address with a ransomware label that exists in the
#     BlockSci chain (addresses that also appear as white are illicit).
#   - Licit seeds: the same number of white addresses, sampled at random.
#   - Each class is split with SPLIT_FRACTIONS (60/20/20): validation and test
#     get the rounded fraction of the class and training the rest, so every split
#     is balanced and the fractions are exact up to one address per class.
#
# The candidates are sorted before every random step, so the selection only
# depends on DATA_SEED and is identical in every execution. The addresses are
# looked up in the chain by parallel workers (one per core), and the results are
# taken in the original order, so the selection does not depend on them.
#
# Output: SPLITS_FILE (address,label,family,split; only in "new" mode, the
# paper one is part of the repository) and SPLITS_SUMMARY_FILE.

import os
import random

import pandas as pd

import config
from shared.experiment import git_commit, timestamp, write_json
from shared.parallel import blocksci_workers, parallel_map
from shared.paths import get_required_env_path
from shared.splits import verify_splits


def in_chain(chain, addr_str):
    try:
        return chain.address_from_string(addr_str) is not None
    except Exception:
        return False


def summarize(checks, extra):
    summary = {
        "created": timestamp(),
        "git_commit": git_commit(config.PROJECT_ROOT),
        "mode": config.SPLITS_MODE,
        "splits_file": str(config.SPLITS_FILE),
        **extra,
        "checks": checks,
        "per_split": checks["per_split"],
    }
    write_json(config.SPLITS_SUMMARY_FILE, summary)
    print("\nSeed addresses per split (illicit + licit):")
    for split in config.SPLITS:
        n = summary["per_split"][split]
        print(f"  {split:5s}: {n['illicit']} + {n['licit']}")


def split_sizes(n):
    """Addresses per class of every split, for n addresses per class."""
    val = int(n * config.SPLIT_FRACTIONS["val"] + 0.5)
    test = int(n * config.SPLIT_FRACTIONS["test"] + 0.5)
    return {"train": n - val - test, "val": val, "test": test}


def check_existing_splits():
    """Fails if the existing new splits do not follow SPLIT_FRACTIONS (e.g. they were
    created by an earlier version with other sizes)."""
    splits = pd.read_csv(config.SPLITS_FILE)
    counts = splits.groupby(["split", "label"]).size()
    expected = split_sizes(int((splits["label"] == 1).sum()))
    found = {s: (int(counts.get((s, 1), 0)), int(counts.get((s, 0), 0))) for s in config.SPLITS}
    if any(found[s] != (expected[s], expected[s]) for s in config.SPLITS):
        raise RuntimeError(f"{config.SPLITS_FILE} has {found} (illicit, licit) addresses per split, but "
                           f"SPLIT_FRACTIONS gives {expected} per class. Delete {config.RUN_DIR} (its data were "
                           "built from these splits) and run the pipeline again.")


def check_in_chain(addresses):
    """Number of seed addresses found in the BlockSci chain, or None if BlockSci is not
    available. Fails if any address is missing."""
    if not os.getenv("RSD_BLOCKSCI_CONFIG"):
        print("RSD_BLOCKSCI_CONFIG is not set: the addresses were not checked against the chain.")
        return None
    try:
        import blocksci
    except ImportError:
        print("BlockSci is not installed: the addresses were not checked against the chain.")
        return None
    chain = blocksci.Blockchain(str(get_required_env_path("RSD_BLOCKSCI_CONFIG")))
    found = parallel_map(lambda addr_str: in_chain(chain, addr_str), addresses, blocksci_workers(),
                         desc="Checking the seed addresses in the chain")
    missing = [a for a, ok in zip(addresses, found) if not ok]
    if missing:
        raise ValueError(f"{len(missing)} seed addresses are not in the chain, e.g. {missing[:5]}")
    print(f"All {len(addresses)} seed addresses are in the chain.")
    return len(addresses)


def paper_splits():
    """Verifies the splits of the paper and writes their summary."""
    if not config.PAPER_SPLITS_FILE.exists():
        raise FileNotFoundError(f"Missing {config.PAPER_SPLITS_FILE}: it is built with paper/build_splits.py "
                                "(see paper/README.md).")
    bitcoinheist_csv = os.getenv("RSD_BITCOINHEIST_CSV")
    splits = pd.read_csv(config.PAPER_SPLITS_FILE)
    checks = verify_splits(splits, config.SPLITS, config.PAPER_VAL_PER_CLASS, config.PAPER_TEST_PER_CLASS,
                           bitcoinheist_csv)
    if not bitcoinheist_csv:
        print("RSD_BITCOINHEIST_CSV is not set: the labels were not checked against BitcoinHeist.")
    checks["in_chain"] = check_in_chain(list(splits["address"]))
    summarize(checks, {"source": "paper", "bitcoinheist_csv": bitcoinheist_csv})
    print(f"\nPaper splits verified: {config.PAPER_SPLITS_FILE}")


def first_in_chain(chain, candidates, n, workers, name):
    """The first n candidates (all of them if n is None) that exist in the chain, in
    their order. They are checked in parallel batches and the first ones found are
    kept, as a sequential scan would do."""
    check = lambda addr_str: in_chain(chain, addr_str)
    batch_size = len(candidates) if n is None else max(10000, 4 * n)
    found_addresses = []
    for start in range(0, len(candidates), max(1, batch_size)):
        batch = candidates[start:start + batch_size]
        progress = f" ({len(found_addresses)}/{n} found)" if n is not None else ""
        found = parallel_map(check, batch, workers, desc=f"Checking {name} addresses{progress}")
        found_addresses += [a for a, ok in zip(batch, found) if ok]
        if n is not None and len(found_addresses) >= n:
            break
    return found_addresses if n is None else found_addresses[:n]


def select_splits(chain, bitcoinheist_csv, n_per_class=None):
    """New seed addresses and splits: all the illicit BitcoinHeist addresses in the chain
    (or n_per_class of them, sampled at random) and the same number of randomly sampled
    licit ones, each class split with SPLIT_FRACTIONS. Returns the splits, their checks
    and a description of the selection."""
    rng = random.Random(config.DATA_SEED)

    df = pd.read_csv(bitcoinheist_csv, usecols=["address", "label"]).dropna(subset=["address", "label"])
    df["address"] = df["address"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()

    illicit_rows = df[df["label"].str.lower() != "white"]
    family = illicit_rows.drop_duplicates("address").set_index("address")["label"].to_dict()
    illicit_candidates = sorted(family)
    licit_candidates = sorted(set(df.loc[df["label"].str.lower() == "white", "address"]) - set(family))
    rng.shuffle(licit_candidates)
    if n_per_class is not None:
        rng.shuffle(illicit_candidates)

    workers = blocksci_workers()
    print(f"{workers} workers ({config.CPU_COUNT} cores)")
    illicit = first_in_chain(chain, illicit_candidates, n_per_class, workers, "illicit")
    if n_per_class is not None and len(illicit) < n_per_class:
        raise RuntimeError(f"Only {len(illicit)} illicit addresses found in the chain for {n_per_class} requested.")
    licit = first_in_chain(chain, licit_candidates, len(illicit), workers, "licit")
    if len(licit) < len(illicit):
        raise RuntimeError(f"Only {len(licit)} licit addresses found for {len(illicit)} illicit ones.")

    sizes = split_sizes(len(illicit))
    if min(sizes.values()) < 1:
        raise RuntimeError(f"{len(illicit)} illicit addresses are not enough for the splits {sizes}.")

    rows = []
    for label, addresses in [(1, illicit), (0, licit)]:
        addresses = list(addresses)
        rng.shuffle(addresses)
        test = addresses[:sizes["test"]]
        val = addresses[sizes["test"]:sizes["test"] + sizes["val"]]
        train = addresses[sizes["test"] + sizes["val"]:]
        for split, split_addresses in [("train", train), ("val", val), ("test", test)]:
            for addr_str in split_addresses:
                rows.append({
                    "address": addr_str,
                    "label": label,
                    "family": family.get(addr_str, "white"),
                    "split": split,
                })

    splits = pd.DataFrame(rows)
    checks = verify_splits(splits, config.SPLITS, sizes["val"], sizes["test"], bitcoinheist_csv)
    info = {
        "data_seed": config.DATA_SEED,
        "bitcoinheist_csv": str(bitcoinheist_csv),
        "illicit_candidates": len(illicit_candidates),
        "illicit_per_class": "all in the chain" if n_per_class is None else n_per_class,
        "illicit_used": len(illicit),
        "split_fractions": config.SPLIT_FRACTIONS,
        "per_class": sizes,
        "fraction_of_all_addresses": {s: round(n / len(illicit), 6) for s, n in sizes.items()},
    }
    return splits, checks, info


def main():
    if config.SPLITS_MODE == "paper":
        paper_splits()
        return

    if config.SPLITS_FILE.exists():
        check_existing_splits()
        print(f"Splits already exist, nothing to do: {config.SPLITS_FILE}")
        print("Delete the file to create them again.")
        return

    import blocksci
    bitcoinheist_csv = get_required_env_path("RSD_BITCOINHEIST_CSV")
    chain = blocksci.Blockchain(str(get_required_env_path("RSD_BLOCKSCI_CONFIG")))
    splits, checks, info = select_splits(chain, bitcoinheist_csv)
    config.SPLITS_FILE.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(config.SPLITS_FILE, index=False)
    summarize(checks, {"source": "create_splits.py", **info})
    print(f"\nSaved: {config.SPLITS_FILE}")


if __name__ == "__main__":
    main()
