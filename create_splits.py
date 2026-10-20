# STEP 0 - SEED ADDRESSES AND SPLITS (requires BlockSci)
#
# Selects the labeled seed addresses and their training, validation and test
# splits. Both methodologies read the same file, so they are trained and
# evaluated on exactly the same addresses:
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
# Output: SPLITS_FILE (address,label,family,split) and SPLITS_SUMMARY_FILE.
#
# The selection is deterministic: with the same inputs it always gives the same seed
# addresses. The splits are therefore reused while their inputs do not change: the
# functions that select the addresses (SELECTION_CODE, by their syntax tree, so other
# edits of this file do not count), the BitcoinHeist file (its content), the BlockSci
# chain (its configuration file), DATA_SEED and the split fractions. They form the
# fingerprint saved in SPLITS_SUMMARY_FILE; with another fingerprint the splits are
# selected again. The expansions of both methodologies include the seed addresses of
# every split in their own fingerprints, so only new seed addresses rebuild the data.

import random
from pathlib import Path

import pandas as pd

import config
from shared.experiment import code_fingerprint, digest, file_fingerprint, git_commit, read_json, timestamp, write_json
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


# Functions that decide which addresses are selected, by kind of selection (the checks
# of shared/splits.py only validate the result)
SELECTION_CODE = {
    "splits": ["in_chain", "first_in_chain", "select_addresses", "split_sizes", "select_splits"],
    "seeds": ["in_chain", "first_in_chain", "select_addresses", "select_seeds"],
}


def selection_fingerprint(bitcoinheist_csv, kind, **parameters):
    """What a selection of seed addresses depends on: kind ("splits" or "seeds") and its
    parameters, the functions that select them, the content of the BitcoinHeist file, the
    BlockSci chain and DATA_SEED."""
    code = code_fingerprint({Path(__file__).resolve(): SELECTION_CODE[kind]})
    return digest({"kind": kind, "parameters": parameters, "code": code, "data_seed": config.DATA_SEED,
                   "bitcoinheist": file_fingerprint([bitcoinheist_csv]),
                   "blocksci_config": str(get_required_env_path("RSD_BLOCKSCI_CONFIG"))})


def saved_fingerprint(data_file, summary_file):
    """Fingerprint of a saved selection (in its summary), or None."""
    if not data_file.exists() or not summary_file.exists():
        return None
    return read_json(summary_file).get("fingerprint")


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


def select_addresses(chain, bitcoinheist_csv, rng, n_per_class=None):
    """All the illicit BitcoinHeist addresses in the chain (or n_per_class of them, sampled
    at random) and the same number of randomly sampled licit ones. Returns the illicit
    and licit addresses, the family of every illicit address and the number of illicit
    candidates."""
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
    return illicit, licit, family, len(illicit_candidates)


def select_seeds(chain, bitcoinheist_csv, n_per_class):
    """n_per_class illicit BitcoinHeist addresses in the chain, sampled at random, and as
    many licit ones, without splits (address,label,family). Returns the seeds, their
    checks and a description of the selection."""
    rng = random.Random(config.DATA_SEED)
    illicit, licit, family, n_candidates = select_addresses(chain, bitcoinheist_csv, rng, n_per_class)
    seeds = pd.DataFrame([{"address": a, "label": label, "family": family.get(a, "white")}
                          for label, addresses in [(1, illicit), (0, licit)] for a in addresses])
    checks = verify_splits(seeds.assign(split="all"), ["all"], bitcoinheist_csv=bitcoinheist_csv)
    info = {
        "data_seed": config.DATA_SEED,
        "bitcoinheist_csv": str(bitcoinheist_csv),
        "illicit_candidates": n_candidates,
        "per_class": n_per_class,
    }
    return seeds, checks, info


def select_splits(chain, bitcoinheist_csv, n_per_class=None):
    """New seed addresses and splits: all the illicit BitcoinHeist addresses in the chain
    (or n_per_class of them, sampled at random) and the same number of randomly sampled
    licit ones, each class split with SPLIT_FRACTIONS. Returns the splits, their checks
    and a description of the selection."""
    rng = random.Random(config.DATA_SEED)
    illicit, licit, family, n_candidates = select_addresses(chain, bitcoinheist_csv, rng, n_per_class)

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
        "illicit_candidates": n_candidates,
        "illicit_per_class": "all in the chain" if n_per_class is None else n_per_class,
        "illicit_used": len(illicit),
        "split_fractions": config.SPLIT_FRACTIONS,
        "per_class": sizes,
        "fraction_of_all_addresses": {s: round(n / len(illicit), 6) for s, n in sizes.items()},
    }
    return splits, checks, info


def main():
    bitcoinheist_csv = get_required_env_path("RSD_BITCOINHEIST_CSV")
    fingerprint = selection_fingerprint(bitcoinheist_csv, "splits", split_fractions=config.SPLIT_FRACTIONS,
                                        splits=config.SPLITS)
    saved = saved_fingerprint(config.SPLITS_FILE, config.SPLITS_SUMMARY_FILE)
    if saved == fingerprint:
        print(f"Splits already selected with this code, data and parameters, nothing to do: {config.SPLITS_FILE}")
        return
    if config.SPLITS_FILE.exists():
        print(f"The existing splits were selected by other code, data or parameters ({saved or 'unknown'} "
              f"instead of {fingerprint}): selecting them again; the data built from them will be rebuilt")

    import blocksci
    chain = blocksci.Blockchain(str(get_required_env_path("RSD_BLOCKSCI_CONFIG")))
    splits, checks, info = select_splits(chain, bitcoinheist_csv)
    config.SPLITS_FILE.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(config.SPLITS_FILE, index=False)
    # Written last: with its fingerprint, it marks the splits as complete
    summarize(checks, {"source": "create_splits.py", "fingerprint": fingerprint, **info})
    print(f"\nSaved: {config.SPLITS_FILE}")


if __name__ == "__main__":
    main()
