# STEP 0 - SEED ADDRESSES AND SPLITS (BlockSci environment)
#
# Selects the labeled seed addresses from BitcoinHeist and splits them into
# training, validation and test sets. Both methodologies read the resulting
# file, so they are trained and evaluated on exactly the same addresses.
#
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
# Output: SPLITS_FILE (address,label,family,split) and SPLITS_SUMMARY_FILE.

import random

import pandas as pd
import blocksci

import config
from shared.experiment import git_commit, timestamp, write_json
from shared.parallel import parallel_map
from shared.paths import get_required_env_path

BLOCKSCI_CONFIG = get_required_env_path("RSD_BLOCKSCI_CONFIG")
BITCOINHEIST_CSV = get_required_env_path("RSD_BITCOINHEIST_CSV")


def in_chain(chain, addr_str):
    try:
        return chain.address_from_string(addr_str) is not None
    except Exception:
        return False


def main():
    if config.SPLITS_FILE.exists():
        print(f"Splits already exist, nothing to do: {config.SPLITS_FILE}")
        print("Delete the file to create them again.")
        return

    chain = blocksci.Blockchain(str(BLOCKSCI_CONFIG))
    rng = random.Random(config.DATA_SEED)

    df = pd.read_csv(BITCOINHEIST_CSV, usecols=["address", "label"]).dropna(subset=["address", "label"])
    df["address"] = df["address"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()

    illicit_rows = df[df["label"].str.lower() != "white"]
    family = illicit_rows.drop_duplicates("address").set_index("address")["label"].to_dict()
    illicit_candidates = sorted(family)
    licit_candidates = sorted(set(df.loc[df["label"].str.lower() == "white", "address"]) - set(family))
    rng.shuffle(licit_candidates)

    workers = config.CPU_COUNT
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
    config.SPLITS_FILE.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(config.SPLITS_FILE, index=False)

    counts = splits.groupby(["split", "label"]).size()
    summary = {
        "created": timestamp(),
        "git_commit": git_commit(config.PROJECT_ROOT),
        "data_seed": config.DATA_SEED,
        "bitcoinheist_csv": str(BITCOINHEIST_CSV),
        "illicit_candidates": len(illicit_candidates),
        "illicit_in_chain": len(illicit),
        "per_split": {
            split: {"illicit": int(counts.get((split, 1), 0)), "licit": int(counts.get((split, 0), 0))}
            for split in config.SPLITS
        },
    }
    write_json(config.SPLITS_SUMMARY_FILE, summary)

    print("\nSeed addresses per split (illicit + licit):")
    for split in config.SPLITS:
        n = summary["per_split"][split]
        print(f"  {split:5s}: {n['illicit']} + {n['licit']}")
    print(f"\nSaved: {config.SPLITS_FILE}")


if __name__ == "__main__":
    main()
