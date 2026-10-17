# STEP 0 - SEED ADDRESSES AND SPLITS (BlockSci environment)
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
#   - Per class, VAL_PER_CLASS and TEST_PER_CLASS addresses go to validation and
#     test, and the remaining ones to training.
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
    checks = verify_splits(splits, config.SPLITS, config.VAL_PER_CLASS, config.TEST_PER_CLASS, bitcoinheist_csv)
    if not bitcoinheist_csv:
        print("RSD_BITCOINHEIST_CSV is not set: the labels were not checked against BitcoinHeist.")
    checks["in_chain"] = check_in_chain(list(splits["address"]))
    summarize(checks, {"source": "paper", "bitcoinheist_csv": bitcoinheist_csv})
    print(f"\nPaper splits verified: {config.PAPER_SPLITS_FILE}")


def main():
    if config.SPLITS_MODE == "paper":
        paper_splits()
        return

    if config.SPLITS_FILE.exists():
        print(f"Splits already exist, nothing to do: {config.SPLITS_FILE}")
        print("Delete the file to create them again.")
        return

    import blocksci
    blocksci_config = get_required_env_path("RSD_BLOCKSCI_CONFIG")
    bitcoinheist_csv = get_required_env_path("RSD_BITCOINHEIST_CSV")
    chain = blocksci.Blockchain(str(blocksci_config))
    rng = random.Random(config.DATA_SEED)

    df = pd.read_csv(bitcoinheist_csv, usecols=["address", "label"]).dropna(subset=["address", "label"])
    df["address"] = df["address"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()

    illicit_rows = df[df["label"].str.lower() != "white"]
    family = illicit_rows.drop_duplicates("address").set_index("address")["label"].to_dict()
    illicit_candidates = sorted(family)
    licit_candidates = sorted(set(df.loc[df["label"].str.lower() == "white", "address"]) - set(family))
    rng.shuffle(licit_candidates)

    workers = blocksci_workers()
    print(f"{workers} workers ({config.CPU_COUNT} cores)")
    check = lambda addr_str: in_chain(chain, addr_str)

    found = parallel_map(check, illicit_candidates, workers, desc="Checking illicit addresses")
    illicit = [a for a, ok in zip(illicit_candidates, found) if ok]

    # The shuffled licit candidates are checked in parallel batches, and the first
    # ones found in the chain are kept, as a sequential scan would do
    licit = []
    batch_size = max(10000, 4 * len(illicit))
    for start in range(0, len(licit_candidates), batch_size):
        batch = licit_candidates[start:start + batch_size]
        found = parallel_map(check, batch, workers, desc=f"Checking licit addresses ({len(licit)}/{len(illicit)} found)")
        licit += [a for a, ok in zip(batch, found) if ok]
        if len(licit) >= len(illicit):
            break
    licit = licit[:len(illicit)]

    if len(licit) < len(illicit):
        raise RuntimeError(f"Only {len(licit)} licit addresses found for {len(illicit)} illicit ones.")

    n_eval = config.VAL_PER_CLASS + config.TEST_PER_CLASS
    if len(illicit) <= n_eval:
        raise RuntimeError(f"{len(illicit)} illicit addresses are not enough for {n_eval} validation and test seeds per class.")

    rows = []
    for label, addresses in [(1, illicit), (0, licit)]:
        addresses = list(addresses)
        rng.shuffle(addresses)
        test = addresses[:config.TEST_PER_CLASS]
        val = addresses[config.TEST_PER_CLASS:n_eval]
        train = addresses[n_eval:]
        for split, split_addresses in [("train", train), ("val", val), ("test", test)]:
            for addr_str in split_addresses:
                rows.append({
                    "address": addr_str,
                    "label": label,
                    "family": family.get(addr_str, "white"),
                    "split": split,
                })

    splits = pd.DataFrame(rows)
    checks = verify_splits(splits, config.SPLITS, config.VAL_PER_CLASS, config.TEST_PER_CLASS, bitcoinheist_csv)
    config.SPLITS_FILE.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(config.SPLITS_FILE, index=False)

    summarize(checks, {
        "source": "create_splits.py",
        "data_seed": config.DATA_SEED,
        "bitcoinheist_csv": str(bitcoinheist_csv),
        "illicit_candidates": len(illicit_candidates),
        "illicit_in_chain": len(illicit),
    })
    print(f"\nSaved: {config.SPLITS_FILE}")


if __name__ == "__main__":
    main()
