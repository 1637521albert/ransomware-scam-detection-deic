# Data and original results of the paper

This folder contains what is needed to reproduce the experiments of the paper on exactly the same seed addresses, and the original results of the multi-instance methodology. It is used by the default `paper` mode of the pipeline (see the main [README](../README.md#two-modes-paper-splits-and-new-splits)).

The full experimental datasets built from these seed addresses (expanded graphs, serialized graphs and extracted features) are not included due to their size and are provided by the authors upon request.

| File | Content |
|---|---|
| `splits.csv` | Seed addresses of the paper and their split, shared by both methodologies: `address,label,family,split` (label 1 = illicit, 0 = licit; family = BitcoinHeist label). 20,849 illicit and 20,849 licit addresses: 12,849 per class for training and 4,000 per class for validation and for test. |
| `splits_provenance.json` | Sources from which `splits.csv` was built, the checks it passed and its SHA-256. |
| `gnn_runs.csv` | Original per-run results of the GNN experiments of the paper (one row per architecture, number of layers and random-seed run). |
| `build_splits.py` | Script that built `splits.csv`. |

## `splits.csv`

The seed addresses of the paper: every illicit BitcoinHeist address (20,849) and the same number of licit (white-only) BitcoinHeist addresses, with their split. It was built with `build_splits.py` from the labeled seed files of each split (`<split>_labeled_addrs.csv`, columns `addr_str` and `class`), whose names and SHA-256 hashes are recorded in `splits_provenance.json`:

```bash
python paper/build_splits.py --labeled-csvs <dir with <split>_labeled_addrs.csv> --bitcoinheist <BitcoinHeist CSV>
```

It can also be built from (or cross-checked against) the labeled address nodes of serialized graphs with `--graphs <dir with <split>/graph.pth and addr_mapping.json>`.

`build_splits.py`, and `create_splits.py` in `paper` mode, check that:

- every address appears once, in a single split (no overlap between training, validation and test);
- each split is balanced, with 4,000 addresses per class in validation and test;
- every illicit seed is illicit in BitcoinHeist, with its family, and every licit seed is a white-only BitcoinHeist address (if the BitcoinHeist CSV is given);
- every address exists in the BlockSci chain (`create_splits.py`, if BlockSci is available), so that no seed is skipped by the expansions.

## `gnn_runs.csv`

One row per run, with the columns of the `gnn_runs.csv` written by `run_experiments.py`:

`arch, layers, seed, test_accuracy, test_precision, test_recall, test_f1, test_auc_roc, epochs, best_epoch`

`wilcoxon_tests.py` reads this file in `paper` mode when the GNNs have not been trained again, so the statistical comparisons of the paper are rebuilt without retraining. The runs were trained with the protocol of the paper (`config.GNN_TRAINING`: best epoch by validation accuracy, early stopping after 10 epochs without improvement, at most 1,000 epochs).
