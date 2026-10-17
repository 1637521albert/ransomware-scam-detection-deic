# Builds paper/splits.csv: the seed addresses and splits used in the paper.
#
# The seed addresses are read from the graphs on which the GNNs of the paper
# were trained (the labeled address nodes of each split), and/or from CSV files
# with the labeled addresses of each split. When both are given, they must
# contain exactly the same addresses, labels and splits. The family of every
# address is taken from BitcoinHeist, and the result is verified with
# shared/splits.py.
#
#   python paper/build_splits.py --graphs <dir>        # <dir>/<split>/graph.pth and addr_mapping.json
#   python paper/build_splits.py --labeled-csvs <dir>  # <dir>/<split>_labeled_addrs.csv (addr_str, class)
#   python paper/build_splits.py --graphs <dir> --labeled-csvs <dir>   # both, cross-checked
#
# The BitcoinHeist CSV is given with --bitcoinheist or RSD_BITCOINHEIST_CSV.
# Reading the graphs needs PyTorch; the CSVs only need pandas.
#
# Output: paper/splits.csv (address,label,family,split) and
# paper/splits_provenance.json (names and SHA-256 of the source files, checks
# and SHA-256 of splits.csv; no local paths are stored).

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from shared.experiment import git_commit, timestamp, write_json
from shared.splits import read_bitcoinheist, verify_splits


def from_graphs(directory):
    import torch
    rows, sizes = [], {}
    for split in config.SPLITS:
        graph = torch.load(directory / split / "graph.pth")
        with open(directory / split / "addr_mapping.json") as f:
            index_to_address = {int(index): address for address, index in json.load(f).items()}
        y = graph["addr"].y.numpy()
        for index in (y != -1).nonzero()[0]:
            rows.append({"address": index_to_address[int(index)], "label": int(y[index]), "split": split})
        sizes[split] = {
            "addr_nodes": int(graph["addr"].num_nodes),
            "tx_nodes": int(graph["tx"].num_nodes),
            "edges": {"-".join(edge_type): int(index.size(1)) for edge_type, index in graph.edge_index_dict.items()},
        }
    return pd.DataFrame(rows), sizes


def from_labeled_csvs(directory):
    parts = []
    for split in config.SPLITS:
        df = pd.read_csv(directory / f"{split}_labeled_addrs.csv", usecols=["addr_str", "class"])
        parts.append(pd.DataFrame({"address": df["addr_str"].astype(str).str.strip(),
                                   "label": df["class"].astype(int), "split": split}))
    return pd.concat(parts, ignore_index=True)


def as_dict(splits):
    return dict(zip(splits["address"], zip(splits["label"], splits["split"])))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Build paper/splits.csv from the graphs and/or labeled CSVs of the paper.")
    parser.add_argument("--graphs", type=Path, help="directory with <split>/graph.pth and <split>/addr_mapping.json")
    parser.add_argument("--labeled-csvs", type=Path, help="directory with <split>_labeled_addrs.csv")
    parser.add_argument("--bitcoinheist", type=Path, default=os.getenv("RSD_BITCOINHEIST_CSV"), help="BitcoinHeist CSV")
    parser.add_argument("--output", type=Path, default=config.PAPER_SPLITS_FILE)
    args = parser.parse_args()

    if not args.graphs and not args.labeled_csvs:
        parser.error("give --graphs and/or --labeled-csvs")
    if not args.bitcoinheist:
        parser.error("give --bitcoinheist or set RSD_BITCOINHEIST_CSV")

    sources, graph_sizes, candidates = {}, None, []
    if args.graphs:
        splits, graph_sizes = from_graphs(args.graphs.resolve())
        candidates.append(("graphs", splits))
        sources["graphs"] = {f"{split}/{name}": sha256(args.graphs / split / name)
                             for split in config.SPLITS for name in ("graph.pth", "addr_mapping.json")}
    if args.labeled_csvs:
        candidates.append(("labeled_csvs", from_labeled_csvs(args.labeled_csvs.resolve())))
        sources["labeled_csvs"] = {f"{split}_labeled_addrs.csv": sha256(args.labeled_csvs / f"{split}_labeled_addrs.csv")
                                   for split in config.SPLITS}

    if len(candidates) == 2:
        a, b = as_dict(candidates[0][1]), as_dict(candidates[1][1])
        if a != b:
            only_a, only_b = set(a) - set(b), set(b) - set(a)
            different = sum(1 for k in set(a) & set(b) if a[k] != b[k])
            raise SystemExit(f"The graphs and the labeled CSVs differ: {len(only_a)} addresses only in the graphs, "
                             f"{len(only_b)} only in the CSVs, {different} with another label or split.")
        print("The graphs and the labeled CSVs contain the same addresses, labels and splits.")
    splits = candidates[0][1]

    family, _, _ = read_bitcoinheist(args.bitcoinheist)
    splits["family"] = [family.get(a, "white") if l == 1 else "white" for a, l in zip(splits["address"], splits["label"])]
    order = {s: i for i, s in enumerate(config.SPLITS)}
    splits = splits.assign(_order=splits["split"].map(order)) \
                   .sort_values(["_order", "label", "address"], ascending=[True, False, True])[["address", "label", "family", "split"]]

    checks = verify_splits(splits, config.SPLITS, config.VAL_PER_CLASS, config.TEST_PER_CLASS, args.bitcoinheist)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(args.output, index=False)
    write_json(args.output.with_name("splits_provenance.json"), {
        "created": timestamp(),
        "git_commit": git_commit(config.PROJECT_ROOT),
        "sources": sources,
        "bitcoinheist_csv": {Path(args.bitcoinheist).name: sha256(args.bitcoinheist)},
        "graph_sizes": graph_sizes,
        "checks": checks,
        "sha256": sha256(args.output),
    })

    print("Seed addresses per split (illicit + licit):")
    for split, n in checks["per_split"].items():
        print(f"  {split:5s}: {n['illicit']} + {n['licit']}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
