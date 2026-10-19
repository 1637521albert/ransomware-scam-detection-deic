# TRACE: Transparent Ransomware Attribution via Cryptocurrency Exploration

This repository contains the code to reproduce the experiments of *TRACE: Transparent Ransomware Attribution via Cryptocurrency Exploration*. Ransomware-related Bitcoin addresses are classified with two methodologies that start from the same labeled seed addresses:

- **Methodology I – Inductive Multi-Instance Address Classification:** the seed addresses of each split are expanded into one merged heterogeneous address–transaction graph, on which Graph Neural Networks (GAT, HGT and HAN) classify the seeds.
- **Methodology II – Inductive Ego-Centric Address Classification:** an independent egonet is expanded around every seed address, topological motifs and graph metrics are extracted from it, and tabular machine learning models classify the seeds.

Both methodologies use the same seed addresses: the 20,849 illicit BitcoinHeist addresses and 20,849 licit ones, with 12,849 addresses per class for training and 4,000 per class for validation and for test. The seed addresses and splits of the paper are included in [`paper/`](paper/), together with the original per-run results of the GNN experiments.

**Data availability.** The full experimental datasets (the expanded graphs, the serialized PyTorch Geometric graphs and the extracted features) are not distributed with the repository due to their size; they are provided by the authors upon request. They can also be reconstructed from the seed addresses in `paper/` and the public Bitcoin blockchain with the pipeline below.

The parameters of the paper are in [`config.py`](config.py). `run_pipeline.py --original` runs every step with them and only asks how many random-seed runs to train; without `--original`, it asks the parameters of the selected methodologies (see [Configurations](#configurations-original-and-custom)). Every step uses all the CPU cores, saves its outputs together with the full configuration that produced them, and skips the work that is already done, so an interrupted step can simply be launched again.

## Repository structure

```
requirements.txt                  Pinned dependencies of the Python environment (BlockSci is installed separately)
config.py                         Parameters of every experiment (paths, seeds, splits, expansion, training)
run_pipeline.py                   Every step of one or both methodologies with a single command
create_splits.py                  Step 0: seed addresses and train/validation/test splits (both methodologies)
wilcoxon_tests.py                 Paired Wilcoxon tests from the results of both methodologies
paper/
  splits.csv                      Seed addresses and splits of the paper (both methodologies)
  gnn_runs.csv                    Original per-run results of the GNN experiments of the paper
  build_splits.py                 Builds splits.csv from the graphs of the paper (see paper/README.md)
shared/
  paths.py                        Environment variable handling
  splits.py                       Verification of a splits file
  configuration.py                Configurable parameters, questions and configuration files
  experiment.py                   Training seeds, configuration snapshots, run metadata and result files
  stats.py                        Paired Wilcoxon tests, bootstrap confidence intervals and summaries
Inductive Multi-Instance Address Classification/
  expansion.py                    Step 1: graph expansion and feature extraction
  serialize_graph.py              Step 2: PyTorch Geometric graphs
  normalize_graphs.py             Step 3: feature normalization (fitted on the training graph)
  model.py                        Step 4: training and evaluation of one GNN configuration with one seed
  run_experiments.py              Step 5: all GNN runs in parallel, result tables and Wilcoxon tests
Inductive Ego-Centric Address Classification/
  expansion.py                    Step 1: egonet expansion
  extract_topological_features.py Step 2: motif counts and graph metrics
  models.py                       Step 3: training, evaluation, statistical tests, ablation and SHAP
Expansion Experiments/
  expansion_experiments.py        Expansion experiments of Table 2 of the paper (run separately)
  experiments_config.py           Their seed addresses, variants, depths and widths
expansion.md, VISUALIZATIONS/     Illustrations of the expansion variants
```

## Environment

The whole pipeline runs in a single Python 3.7 environment. The experiments of the paper were run with Python 3.7.6, PyTorch 1.13.1 and PyTorch Geometric 2.3.1, on CPU.

1. **Python libraries.** In a Python 3.7 environment (created with conda, venv or any other tool), install the pinned dependencies of [`requirements.txt`](requirements.txt):

   ```bash
   python -m pip install --upgrade "pip<24.1"    # pip 24.0, the last release for Python 3.7
   pip install -r requirements.txt
   ```

   pip is upgraded first because older versions fail to build PyTorch Geometric (*Package 'setuptools' requires a different Python*).

2. **BlockSci**, used by `create_splits.py`, both `expansion.py` and the expansion experiments. It is not on PyPI: it is built from source and its Python package (`blockscipy`) is installed into the same environment, following the [BlockSci setup instructions](https://citp.github.io/BlockSci/setup.html). BlockSci declares `pycrypto` as a dependency, which is unmaintained and fails on Python 3 (*NameError: name 'xrange' is not defined* when importing BlockSci); `pycryptodome` provides the same module and replaces it:

   ```bash
   pip uninstall -y pycrypto && pip install pycryptodome
   ```

3. **Data paths**, set as environment variables (see [Environment variables](#environment-variables)):

   ```bash
   export RSD_BLOCKSCI_CONFIG=<path to the BlockSci config file of the parsed chain>
   export RSD_BITCOINHEIST_CSV=<path to the BitcoinHeist CSV>
   ```

4. **Check:** prints the height of the last block of the chain.

   ```bash
   python -c "import os, blocksci; print(blocksci.Blockchain(os.environ['RSD_BLOCKSCI_CONFIG']).blocks[-1].height)"
   ```

Without BlockSci, the steps that do not read the chain (`create_splits.py` in `paper` mode and every step after the expansions) run with step 1 only.

## Environment variables

| Variable | Required by | Description |
|---|---|---|
| `RSD_BLOCKSCI_CONFIG` | `create_splits.py`, both `expansion.py`, `Expansion Experiments/expansion_experiments.py` | Path to the BlockSci config file of the parsed chain |
| `RSD_BITCOINHEIST_CSV` | `create_splits.py`, `Expansion Experiments/expansion_experiments.py` | Path to the [BitcoinHeist](https://www.kaggle.com/datasets/sapere0/bitcoinheist-ransomware-dataset) CSV |
| `RSD_OUTPUT_DIR` | all scripts (optional) | Directory for all generated data and results (default: `outputs/` in the repository) |
| `RSD_SPLITS_MODE` | all scripts (optional) | `paper` (default) or `new`, see [Two modes](#two-modes-paper-splits-and-new-splits) |
| `RSD_CONFIG_FILE` | all scripts (optional) | JSON file of a custom configuration (default: the original configuration), see [Configurations](#configurations-original-and-custom); set by `run_pipeline.py` |

## Running the experiments

### Two modes: paper splits and new splits

| Mode | Seed addresses and splits | Outputs |
|---|---|---|
| `paper` (default) | Those of the paper, in `paper/splits.csv`. `create_splits.py` only verifies them. | `outputs/paper/` |
| `new` | New ones, selected by `create_splits.py` with `DATA_SEED`: every illicit BitcoinHeist address in the chain and the same number of randomly sampled licit ones, each class split 60/20/20 into training, validation and test. | `outputs/new/` (the splits in `outputs/new/splits.csv`) |

The mode is chosen with `run_pipeline.py --mode paper|new` or, for the scripts run one by one, with `RSD_SPLITS_MODE`. Everything else (expansion, features, training, evaluation, tests) works the same in both modes. Each mode writes to its own directory, so the data and results of the paper splits and of new splits are never mixed. Both modes are deterministic: repeating a run with the same configuration and seeds gives the same result, and other training seeds can be chosen with the number of random-seed runs (see [Reproducibility](#reproducibility)).

In `paper` mode with the original configuration, the statistical tests use the original GNN runs of the paper (`paper/gnn_runs.csv`) until the GNNs are trained again, so the comparisons of the paper can be rebuilt without retraining the 60 GNN runs.

### Configurations: original and custom

| Configuration | Command | Parameters | Outputs |
|---|---|---|---|
| Original | `run_pipeline.py --original` | Those of the paper, in `config.py`. Only the number of random-seed runs is asked (or given with `--n-seeds`). | `outputs/<mode>/` |
| Custom | `run_pipeline.py` | Asked group by group for the selected methodologies, with the paper values as defaults. | `outputs/<mode>/custom-<id>/` |
| Saved custom | `run_pipeline.py --config <file>` | Those of a configuration file, e.g. the `configuration.json` saved by a custom run. | `outputs/<mode>/custom-<id>/` |

Without `--original` or `--config`, `run_pipeline.py` asks, for every group of parameters, whether to keep those of the paper or to choose new ones (Enter keeps the value in brackets):

- **Data generation** (always): `DATA_SEED`.
- **Methodology I:** the expansion (variant, width limit and *w*, forward and backward hops or *k*; the variants are illustrated in [`expansion.md`](expansion.md)), and the GNN architectures, numbers of layers and training parameters.
- **Methodology II:** the expansion (width limit and *w*, *k*), the radius and betweenness of the topological features, and the training features and hyperparameters of the tabular models and the model that is interpreted.
- **Statistical tests** (always): bootstrap resamples and minimum common seeds.

Only the parameters that change the data or the results are asked; the resources (workers, memory) stay in `config.py`. The chosen parameters are saved in `outputs/<mode>/custom-<id>/configuration.json`, where `<id>` is a hash of all the parameters, and every step reads them through `RSD_CONFIG_FILE`. All the data and results of a custom configuration are kept in its own directory, so different configurations are never mixed, and choosing the same parameters again resumes the same directory. A configuration with the same values as the paper is the original one. To relaunch or resume a custom configuration without answering again, or to run it in the other mode, use `--config <file>`; to run a single script with it, set `RSD_CONFIG_FILE=<file>`. Choosing the parameters needs an interactive terminal, so unattended runs use `--original` or `--config`.

### Reproducing the paper

The default `paper` mode uses the seed addresses and splits of the paper (`paper/splits.csv`). After setting the environment variables:

```bash
python run_pipeline.py ego-centric --original --n-seeds 10      # Methodology II: splits check, egonets, features, models, tests
python run_pipeline.py multi-instance --original --n-seeds 10   # Methodology I: graphs and the 60 GNN runs (CPU-intensive)
python wilcoxon_tests.py                             # tests of both methodologies
```

Until the GNNs are trained again, `wilcoxon_tests.py` uses the original GNN runs of the paper (`paper/gnn_runs.csv`). The first step (`splits`) verifies `paper/splits.csv` against BitcoinHeist and the chain before anything is expanded.

### Everything with one command

```bash
python run_pipeline.py --original                  # both methodologies, parameters of the paper
python run_pipeline.py                             # both methodologies, parameters chosen by the user
python run_pipeline.py multi-instance --original   # Methodology I only
python run_pipeline.py ego-centric --original      # Methodology II only
python run_pipeline.py --original --n-seeds 10     # without any question
python run_pipeline.py --config <file>             # a saved custom configuration
python run_pipeline.py --original --from train-gnn # start from a given step
python run_pipeline.py --list                      # show the steps and their commands
python run_pipeline.py --original --mode new       # new seed addresses and splits instead of the paper ones
```

`run_pipeline.py` checks that the environment has the required libraries, sets the configuration (original, chosen by the user or from a file), asks the number of random-seed runs once, and runs every step below in order with the current interpreter. With both methodologies, the Wilcoxon tests are run again at the end to compare them. It stops at the first failing step; as every step keeps its completed work, it can be relaunched with `--from <step>` and the same `--original` or `--config <file>` (the message of the failure gives the command). Every execution is recorded in `pipeline_runs/<timestamp>.json` of its output directory (configuration, commands, interpreter, status and duration of each step).

### Step by step

The steps can also be run one by one. The scripts can be launched from any directory, and use the original configuration unless `RSD_CONFIG_FILE` is set. The descriptions below give the parameters of the paper.

#### Step 0 – Seed addresses and splits

```bash
python create_splits.py
```

In `paper` mode, it verifies the splits of the paper (`paper/splits.csv`): every address appears in a single split, the classes are balanced with 4,000 addresses per class in validation and test, and, if `RSD_BITCOINHEIST_CSV` is set, every label and family agrees with BitcoinHeist. It does not need BlockSci.

In `new` mode, every illicit BitcoinHeist address present in the chain is used as a seed, together with the same number of randomly sampled licit (*white*) addresses. The addresses are looked up in the chain in parallel. Each class is split 60/20/20: validation and test get 20% of the class each (rounded to the nearest address) and training the rest, so the three splits are balanced and their fractions of all the seed addresses are exact up to one address per class. The selection depends only on `DATA_SEED` (the candidates are sorted before every random step), and the result is verified as above. Splits created with other sizes, e.g. by an earlier version, are rejected.

Both methodologies read the same splits file, so they are trained and evaluated on exactly the same addresses.

#### Methodology I – Inductive Multi-Instance Address Classification

```bash
cd "Inductive Multi-Instance Address Classification"
python expansion.py          # requires BlockSci
python serialize_graph.py
python normalize_graphs.py
python run_experiments.py    # all architectures x layers x seeds (asks the number of seeds)
```

1. **Expansion** (`expansion.py`): expands the seeds of each split with a forward–backward, transaction-centered, whole-scope expansion of *k* = 2 hops and a *random node* width limit *w* = 5, and exports the address, transaction, input, output and spent-output features of each graph. Each split uses all the cores: every BlockSci access (listing the transactions of the seeds, scanning the frontier of every hop, exploring the selected transactions and computing the features) runs in parallel workers, while the main process performs the order-dependent steps (random sampling, de-duplication and insertion into the graph) in the original order, so the graphs are identical to a single-process run. `tx_feats.csv` also records the hop where every transaction was added to the graph: `hop` (1 for the transactions of the seeds, then 2, 3… of the forward and backward hops) and `hop_direction` (`forward`, `backward`, or `both` for a hop-1 transaction where a seed is both an input and an output); they are not used as GNN features. The expanded graph and the address features (in blocks of 100,000 addresses) are saved as they are computed, so an interrupted expansion resumes where it stopped. Address histories are read as whole BlockSci arrays when this gives exactly the same values as reading them item by item, which is checked on a sample of addresses of every split.
2. **Serialization** (`serialize_graph.py`): builds a PyTorch Geometric `HeteroData` graph per split, with the address and transaction index mappings, serializing the three splits in parallel.
3. **Normalization** (`normalize_graphs.py`): clips the transaction locktime, applies logarithmic compression to heavy-tailed features and min–max/quantile scaling, all fitted on the training graph only, and saves the normalized graphs that every training run reads.
4. **Training** (`model.py`): trains one configuration with one seed. The best epoch is the one with the highest validation accuracy, and training stops after 10 epochs without improvement (at most 1,000 epochs), as in the paper. A single run can be launched with `python model.py --arch GAT --layers 2 --seed 42`.
5. **All experiments** (`run_experiments.py`): asks how many random-seed runs to execute (or `--n-seeds N`), trains GAT, HGT and HAN with 2 and 4 layers on that many seeds, builds the result tables and runs the Wilcoxon tests. As many runs as fit in the available memory train at the same time, sharing all the cores (`--workers N` to fix their number); `--aggregate-only` only rebuilds the tables and tests. The run being trained is printed when it starts, and its epochs (losses, validation accuracy, best epoch so far and duration) are written as they finish to `outputs/<mode>/multi_instance/results/runs/<arch>-<layers>l-seed<seed>/train.log`, e.g. `tail -f outputs/paper/multi_instance/results/runs/*/train.log`.

#### Methodology II – Inductive Ego-Centric Address Classification

```bash
cd "Inductive Ego-Centric Address Classification"
python expansion.py                       # requires BlockSci
python extract_topological_features.py
python models.py                          # asks the number of seeds
```

1. **Expansion** (`expansion.py`): expands an independent egonet around every seed with an all-over, transaction-centered expansion of *k* = 3 hops and a *random node* width limit *w* = 3, in parallel worker processes. Every transaction of an egonet records the hop where it was added (`hop`: 1 for the transactions of the seed, then 2, 3…).
2. **Feature extraction** (`extract_topological_features.py`): computes, in parallel workers, the aggregation, branching and peeling-chain motifs of each seed, its degrees, PageRank, betweenness and eccentricity, and the density of its egonet. The features of each split are saved to their own file as soon as the split is complete, and a split whose file already exists is skipped.
3. **Models** (`models.py`): asks how many random-seed runs to execute (or `--n-seeds N`), trains Logistic Regression (L1 and ElasticNet), Random Forest, XGBoost, SVM, Naive Bayes, MLP and Isolation Forest on that many seeds, one single-threaded process per run on all the cores, and evaluates them on the test split. It then runs the Wilcoxon tests and analyzes the best model (MLP) with a permutation-importance feature ablation and SHAP, both on the validation split and in parallel (the SHAP values are computed in fixed chunks of addresses with their own seeds, so they are reproducible with any number of cores). `--skip-interpretation` skips the last analyses and `--aggregate-only` only rebuilds the tables and tests.

#### Statistical tests

```bash
python wilcoxon_tests.py     # from the repository root, after training
```

The training scripts already run it, but it can be executed again at any time from the saved results. It compares, seed by seed, (1) the GNN architectures at each depth and 4 vs 2 layers of each architecture, (2) the best tabular model against every other model, and (3) the best GNN configuration against every tabular model, which is possible because both methodologies use the same seeds and test addresses. For every comparison and metric it reports the mean paired difference, its 95% bootstrap confidence interval (10,000 resamples) and the exact two-sided Wilcoxon signed-rank test. Comparisons with fewer than 6 common seeds are skipped. Only the results of the same mode and configuration are compared.

### Expansion experiments (Table 2)

```bash
python "Expansion Experiments/expansion_experiments.py"                    # asks which experiments to run
python "Expansion Experiments/expansion_experiments.py" --experiments 1,2,3
```

The expansion experiments of Table 2 of the paper measure the structure and cost of the expanded graphs. They are run separately from the pipeline, always with the original configuration, need BlockSci and `RSD_BITCOINHEIST_CSV`, and write to `outputs/expansion_experiments/`, independently of the mode:

1. **Configuration:** the expansion variants `all over tx`, `fw bw tx whole`, `fw bw addr dedicated` and `fw bw addr whole same` with *k* = 2 hops and *w* = 5.
2. **Hops:** `fw bw tx whole` with *w* = 5 and *k* = 1, 2, 3.
3. **Width:** `fw bw tx whole` with *k* = 2 and *w* = 1, 2, 5, 10.

All of them use a *random node* width limit and the same expansion algorithms as the multi-instance `expansion.py`, keeping only the graph structure. The seed addresses are 1,500 illicit BitcoinHeist addresses found in the chain, sampled at random, and 1,500 licit ones (3,000 in total), selected with `DATA_SEED` and saved in `seeds.csv`, which is reused. Every run expands all of them into a single graph, without training, validation and test splits, and starts from the same random state (`DATA_SEED`), so the graphs and their metrics are reproducible (only the runtime depends on the machine). For every run (experiment, variant, *k* and *w*) they compute the address, transaction and total nodes, total edges (input, output and spent-output, as in `serialize_graph.py`), density, average degree, largest strongly connected component (size and percentage of the nodes), runtime, width-limit activation and candidate retention rates and the width-limit counters (expansion steps, activated steps, candidates and retained candidates). Every run is saved in `runs/` and reused if it exists (delete its file to compute it again), every metric of an experiment in `<experiment>.csv`, and the tables of the paper (relative growth along *k* and *w*, as in Table 2) in `tables/<experiment>.csv`. The seed addresses and the variants, depths and widths are set in `Expansion Experiments/experiments_config.py`.

## Outputs

```
outputs/<mode>/                              paper/ or new/, one directory per mode, with the original configuration
  custom-<id>/                               a custom configuration: configuration.json and the same structure as below
  splits_summary.json                        verified seed addresses per split (paper mode: of paper/splits.csv)
  splits.csv                                 seed addresses, labels, families and splits (new mode only)
  pipeline_runs/<timestamp>.json             steps, commands, status and duration of every run_pipeline.py execution
  multi_instance/
    graphs/expansion_config.json, normalization.json
    graphs/<split>/                          feature CSVs, graph.pth, graph_normalized.pth, mappings, labeled_nodes.csv
    results/experiment_config.json           full configuration: data, expansion, normalization, training, evaluation, seeds, resources
    results/runs/<arch>-<layers>l-seed<seed>/
      result.json                            full configuration, training history, best epoch and test metrics
      predictions.csv                        test seeds with label and predicted probability
      model.pth, train.log                   best weights and training log
    results/gnn_runs.csv                     one row per run
    results/gnn_summary.csv                  mean, standard deviation and median per configuration
  ego_centric/
    egonets/<split>.jsonl, expansion_config.json
    features/<split>.csv, <split>.json       features of the seed addresses of each split
    results/experiment_config.json           full configuration: data, expansion, features, models, evaluation, seeds, resources
    results/runs/, predictions/              metrics and test predictions of every model and seed
    results/ego_runs.csv, ego_summary.csv
    results/interpretation/                  feature importance, ablation and SHAP values and figures
  statistical_tests/
    gnn_tests.csv, ego_tests.csv, cross_methodology_tests.csv, tests_config.json

outputs/expansion_experiments/               expansion experiments of Table 2
  seeds.csv, seeds_summary.json, experiments_config.json
  runs/<variant> <k> hops <w> random node limit.json   metrics of every run
  configuration.csv, hops.csv, width.csv     every metric of every run of each experiment
  tables/<experiment>.csv                    tables of the paper
```

Every result records the git commit of the code that produced it, and every expansion, feature extraction and experiment saves its configuration and the library versions used. The test predictions are saved for every run, so other metrics or per-address comparisons between configurations and methodologies can be computed without training again.

## Reproducibility

- **What gives the same results:** the same mode, configuration, number of seeds and input data (BitcoinHeist CSV and BlockSci chain) give the same splits, graphs, features and tabular-model results in every execution and on any number of cores. Three things can still change the results: another BlockSci chain (a parse up to another block height has other transactions, so the expansions, and in `new` mode the licit candidates found in the chain, differ); another number of threads per GNN run (each run uses `CPU_COUNT // parallel runs` threads, and with `GNN_PARALLEL_RUNS = "auto"` the number of parallel runs depends on the memory available when training starts, so the GNN metrics can differ slightly between machines or executions; it is recorded in every `result.json`, and setting `GNN_PARALLEL_RUNS` to a number in `config.py` fixes it on a given machine); and other library versions or hardware, which can change floating-point results (`requirements.txt` pins the versions).
- **Data:** all data-generation steps use `DATA_SEED`. The seed candidates are sorted before every random step, so the selection does not depend on the Python process, and every split is expanded from its own fixed random state.
- **Expansion in parallel:** every multi-instance split and every ego-centric egonet is expanded from its own fixed random state, so the result does not depend on the number of parallel processes or their order.
- **Training:** each GNN run fixes the Python, NumPy and PyTorch seeds and uses deterministic algorithms with a fixed number of threads (recorded in `result.json`; floating-point sums may differ slightly with another number of threads). Every random component of the tabular models is seeded, including the mutual-information feature selection, and each run is single-threaded, so the results do not depend on the number of processes. The same configuration and seed therefore always give the same result.
- **Seeds:** the first seeds are those of `config.TRAINING_SEEDS`, and more seeds are drawn deterministically from `DATA_SEED`, so an experiment with more seeds always contains the runs of an experiment with fewer.
- **Paired comparisons:** all configurations of both methodologies use the same training seeds and test addresses, so their results are compared seed by seed with `wilcoxon_tests.py`. Only the seeds present in both configurations of a comparison are used.

## Future work

The following improvements of the multi-instance training are supported by the code but not used in the paper, whose protocol is the default of `config.py`:

- **Longer early-stopping protocol.** A minimum number of epochs before early stopping and a longer patience (e.g. `min_epochs: 50, patience: 20` in `config.GNN_TRAINING`) would make the runs that stop after very few epochs less frequent, reducing the variability between random seeds. It increases the training time of every run.
- **Scheduling by configuration.** The number of parallel runs is chosen from the largest memory peak of all configurations (GAT with 4 layers, the most demanding one, needs about 65 GiB on the full training graph). Choosing it per configuration would train the smaller configurations (HAN and HGT) several at a time.
- **GPU training and mini-batching.** Every epoch is a full-graph pass on CPU. Training on a GPU with enough memory for the full graph, or with neighbor-sampled mini-batches, would reduce the training time considerably, at the cost of exact determinism (GPU) or of a different training procedure (mini-batching).
- **Retraining on the released splits.** Training the GNNs again in `paper` mode with the seeds of `config.TRAINING_SEEDS` would pair every GNN run with the ego-centric run of the same seed, allowing seed-level comparisons between the two methodologies.
