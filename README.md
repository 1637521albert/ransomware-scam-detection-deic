# TRACE: Transparent Ransomware Attribution via Cryptocurrency Exploration

This repository contains the code to reproduce the experiments of *TRACE: Transparent Ransomware Attribution via Cryptocurrency Exploration*. Ransomware-related Bitcoin addresses are classified with two methodologies that start from the same labeled seed addresses:

- **Methodology I – Inductive Multi-Instance Address Classification:** the seed addresses of each split are expanded into one merged heterogeneous address–transaction graph, on which Graph Neural Networks (GAT, HGT and HAN) classify the seeds.
- **Methodology II – Inductive Ego-Centric Address Classification:** an independent egonet is expanded around every seed address, topological motifs and graph metrics are extracted from it, and tabular machine learning models classify the seeds.

Both methodologies use the same seed addresses, selected by `create_splits.py` with `DATA_SEED`: every illicit BitcoinHeist address found in the chain and the same number of randomly sampled licit ones, each class split 60/20/20 into training, validation and test.

**Data availability.** The full experimental datasets (the expanded graphs, the serialized PyTorch Geometric graphs and the extracted features) are not distributed with the repository due to their size; they are provided by the authors upon request. They can also be reconstructed from BitcoinHeist and the public Bitcoin blockchain with the pipeline below.

The parameters of the paper are in [`config.py`](config.py). `run_pipeline.py --original` runs every step with them and only asks how many random-seed runs to train; without `--original`, it asks the parameters of the selected methodologies (see [Configurations](#configurations-original-and-custom)). Every step uses all the CPU cores (the steps with one worker process per core run their numerical libraries single-threaded, `shared/threads.py`, so the processes do not compete for the cores), saves its outputs together with the full configuration that produced them, and skips the work that is already done, so an interrupted step can simply be launched again.

## Repository structure

```
requirements.txt                  Pinned dependencies of the Python environment (BlockSci is installed separately)
config.py                         Parameters of every experiment (paths, seeds, splits, expansion, training)
run_pipeline.py                   Every step of one or both methodologies with a single command
create_splits.py                  Step 0: seed addresses and train/validation/test splits (both methodologies)
wilcoxon_tests.py                 Paired Wilcoxon tests from the results of both methodologies
shared/
  paths.py                        Environment variable handling
  splits.py                       Verification of a splits file
  configuration.py                Configurable parameters, questions and configuration files
  expansion.py                    Expansion variants, shared by both methodologies
  threads.py                      Single-threaded numerical libraries in the per-core worker processes
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
  models.py                       Steps 3-4: training, evaluation, statistical tests, and ablation and SHAP of the best model
Expansion Experiments/
  expansion_experiments.py        Expansion experiments of Table 2 of the paper (run separately)
  experiments_config.py           Their seed addresses, variants, depths and widths
expansion.md, VISUALIZATIONS/     Illustrations of the expansion variants
```

## Environment

The whole pipeline runs in a single Python 3.7 environment that contains both the pinned libraries of [`requirements.txt`](requirements.txt) and [BlockSci](https://citp.github.io/BlockSci/setup.html). The experiments of the paper were run with Python 3.7.6, PyTorch 1.13.1 and PyTorch Geometric 2.3.1, on CPU.

BlockSci is not on PyPI: it is built from source following its setup instructions, which compile its Python package (`blockscipy`, in the `blockscipy/` directory of the BlockSci sources) for a given Python version and install it in that Python environment. Installing the pinned libraries in that environment would replace the versions it already has, so a new environment is created for the project and the existing BlockSci build is made available in it:

```bash
# 1. New environment with the same Python version as the BlockSci build (3.7.6 in the paper)
conda create -n trace python=3.7.6
conda activate trace

# 2. Pinned libraries (pip 24.0, the last release for Python 3.7: older versions fail
#    to build PyTorch Geometric with "Package 'setuptools' requires a different Python")
python -m pip install --upgrade "pip<24.1"
pip install -r requirements.txt

# 3. BlockSci: add its Python package to the environment, without compiling it again
BLOCKSCI_DIR=<directory of the BlockSci sources>
echo "$BLOCKSCI_DIR/blockscipy" > "$(python -c 'import site; print(site.getsitepackages()[0])')/blockscipy.pth"

# 4. Dependencies of the BlockSci package (pycryptodome instead of pycrypto, see below)
pip install "multiprocess>=0.70.5" "dateparser>=0.6.0" pycryptodome

# 5. Data paths (see Environment variables)
export RSD_BLOCKSCI_CONFIG=<BlockSci config file of the parsed chain>
export RSD_BITCOINHEIST_CSV=<BitcoinHeist CSV>

# 6. Check: prints the height of the last block of the chain
python -c "import os, blocksci; print(blocksci.Blockchain(os.environ['RSD_BLOCKSCI_CONFIG']).blocks[-1].height)"
```

- **Step 3** writes a `.pth` file that adds the BlockSci package directory to the Python path of the new environment, as the editable installation of BlockSci does in its own environment. The compiled extension only works with the Python version it was built for, so both environments must use the same Python version (3.7).
- **Step 4:** BlockSci declares `pycrypto` as a dependency, but `pycrypto` is unmaintained and installs Python 2 code with a recent pip (*NameError: name 'xrange' is not defined* when importing BlockSci). `pycryptodome` provides the same `Crypto` module; `pip check` still reports the missing `pycrypto`, which can be ignored.
- **Step 5:** `export` only sets the variables in the current shell; with conda they can be saved in the environment with `conda env config vars set RSD_BLOCKSCI_CONFIG=... RSD_BITCOINHEIST_CSV=...`.
- **Without an existing BlockSci build,** it is built following its setup instructions with the new environment active, which installs `blockscipy` directly in it (steps 3 and 4 are then not needed).

Without BlockSci, the steps that do not read the chain (every step after the expansions) only need steps 1 and 2.

## Environment variables

| Variable | Required by | Description |
|---|---|---|
| `RSD_BLOCKSCI_CONFIG` | `create_splits.py`, both `expansion.py`, `Expansion Experiments/expansion_experiments.py` | Path to the BlockSci config file of the parsed chain |
| `RSD_BITCOINHEIST_CSV` | `create_splits.py`, `Expansion Experiments/expansion_experiments.py` | Path to the [BitcoinHeist](https://www.kaggle.com/datasets/sapere0/bitcoinheist-ransomware-dataset) CSV |
| `RSD_OUTPUT_DIR` | all scripts (optional) | Directory for all generated data and results (default: `outputs/` in the repository) |
| `RSD_CONFIG_FILE` | all scripts (optional) | JSON file of a custom configuration (default: the original configuration), see [Configurations](#configurations-original-and-custom); set by `run_pipeline.py` |

## Running the experiments

### Configurations: original and custom

| Configuration | Command | Parameters | Outputs |
|---|---|---|---|
| Original | `run_pipeline.py --original` | Those of the paper, in `config.py`. Only the number of random-seed runs is asked (or given with `--n-seeds`). | `outputs/` |
| Custom | `run_pipeline.py` | Asked group by group for the selected methodologies, with the paper values as defaults. | `outputs/custom-<id>/` |
| Saved custom | `run_pipeline.py --config <file>` | Those of a configuration file, e.g. the `configuration.json` saved by a custom run. | `outputs/custom-<id>/` |

Without `--original` or `--config`, `run_pipeline.py` asks, for every group of parameters, whether to keep those of the paper or to choose new ones (Enter keeps the value in brackets):

- **Data generation** (always): `DATA_SEED`.
- **Methodology I:** the expansion (variant, width limit and *w*, forward and backward hops or *k*; the variants are illustrated in [`expansion.md`](expansion.md)), and the GNN architectures, numbers of layers and training parameters.
- **Methodology II:** the expansion (variant, width limit and *w*, forward and backward hops or *k*), the radius and betweenness of the topological features, and the training features and hyperparameters of the tabular models and the model that is interpreted.
- **Statistical tests** (always): bootstrap resamples and minimum common seeds.

Only the parameters that change the data or the results are asked; the resources (workers, memory) stay in `config.py`. The chosen parameters are saved in `outputs/custom-<id>/configuration.json`, where `<id>` is a hash of all the parameters, and every step reads them through `RSD_CONFIG_FILE`. All the data and results of a custom configuration are kept in its own directory, so different configurations are never mixed, and choosing the same parameters again resumes the same directory. A configuration with the same values as the paper is the original one. To relaunch or resume a custom configuration without answering again, use `--config <file>`; to run a single script with it, set `RSD_CONFIG_FILE=<file>`. Choosing the parameters needs an interactive terminal, so unattended runs use `--original` or `--config`.

### Running the configuration of the paper

After setting the environment variables:

```bash
python run_pipeline.py ego-centric --original --n-seeds 10      # Methodology II: splits, egonets, features, models, tests
python run_pipeline.py multi-instance --original --n-seeds 10   # Methodology I: graphs and the 60 GNN runs (CPU-intensive)
python wilcoxon_tests.py                                         # tests of both methodologies
```

The first step (`splits`) selects the seed addresses and splits, which both methodologies then share.

### Everything with one command

```bash
python run_pipeline.py --original                  # both methodologies, parameters of the paper
python run_pipeline.py                             # both methodologies, parameters chosen by the user
python run_pipeline.py multi-instance --original   # Methodology I only
python run_pipeline.py ego-centric --original      # Methodology II only
python run_pipeline.py --original --n-seeds 10     # without any question
python run_pipeline.py --config <file>             # a saved custom configuration
python run_pipeline.py --original --from train-gnn # start from a given step
python run_pipeline.py ego-centric --original --from interpret-ego   # only the ablation and SHAP of Methodology II
python run_pipeline.py --list                      # show the steps and their commands
```

`run_pipeline.py` checks that the environment has the required libraries, sets the configuration (original, chosen by the user or from a file), asks the number of random-seed runs once, and runs every step below in order with the current interpreter. With both methodologies, the Wilcoxon tests are run again at the end to compare them. It stops at the first failing step; as every step keeps its completed work, it can be relaunched with `--from <step>` and the same `--original` or `--config <file>` (the message of the failure gives the command). Every execution is recorded in `pipeline_runs/<timestamp>.json` of its output directory (configuration, commands, interpreter, status and duration of each step).

### Step by step

The steps can also be run one by one. The scripts can be launched from any directory, and use the original configuration unless `RSD_CONFIG_FILE` is set. The descriptions below give the parameters of the paper.

#### Step 0 – Seed addresses and splits

```bash
python create_splits.py
```

Every illicit BitcoinHeist address present in the chain is used as a seed, together with the same number of randomly sampled licit (*white*) addresses. The addresses are looked up in the chain in parallel. Each class is split 60/20/20: validation and test get 20% of the class each (rounded to the nearest address) and training the rest, so the three splits are balanced and their fractions of all the seed addresses are exact up to one address per class. The selection depends only on `DATA_SEED` (the candidates are sorted before every random step), and the splits are verified: every address appears in a single split, the classes are balanced, and every label and family agrees with BitcoinHeist. Splits created with other sizes, e.g. by an earlier version, are rejected.

Both methodologies read the same splits file, so they are trained and evaluated on exactly the same addresses.

#### Methodology I – Inductive Multi-Instance Address Classification

```bash
cd "Inductive Multi-Instance Address Classification"
python expansion.py          # requires BlockSci
python serialize_graph.py
python normalize_graphs.py
python run_experiments.py    # all architectures x layers x seeds (asks the number of seeds)
```

1. **Expansion** (`expansion.py`): expands the seeds of each split with a forward–backward, transaction-centered, whole-scope expansion of *k* = 2 hops and a *random node* width limit *w* = 5, and exports the address, transaction, input, output and spent-output features of each graph. Each split uses all the cores, with any expansion variant: every BlockSci access (listing the candidate transactions of the frontier of every hop, exploring the selected transactions and computing the features) runs in parallel workers, while the main process performs the order-dependent steps (random sampling, de-duplication and insertion into the graph) in the order of the sequential algorithms of `shared/expansion.py`, which the ego-centric methodology also uses, so the graphs are identical to a single-process run. The candidates are handled as transaction indices, and only the transactions that are selected are hashed (the width limit discards most candidates); transactions are identified by their hash in the exported files, which only differs from their index for the two pairs of duplicated 2010 coinbase transactions (BIP30). The workers rebuild transactions from their index (`blocksci.Tx(index, chain)` or `chain.tx_with_index(index)`, checked at the start), and the main process keeps plain values only, with every address string and transaction hash stored once. `tx_feats.csv` also records the hop where every transaction was added to the graph: `hop` (1 for the transactions of the seeds, then 2, 3… of the forward and backward hops) and `hop_direction` (`forward`, `backward`, or `both` for a hop-1 transaction where a seed is both an input and an output; empty in the all-over variants); they are not used as GNN features. The features of an address only depend on its history in the chain (except its class), and the graphs of the splits share many addresses, mostly the busiest ones, whose features are the costliest: they are computed once and reused by the next splits (identified by their string and full type), giving exactly the same files; within every block, the addresses with most inputs and outputs in the graph (the busiest, with the longest histories) are given first to the workers, in small chunks, so that a few busy addresses do not keep one worker busy at the end of the block. The expanded graph and the address features (in blocks of 100,000 addresses) are saved as they are computed, so an interrupted expansion resumes where it stopped. Address histories are read as whole BlockSci arrays when this gives exactly the same values as reading them item by item, which is checked on a sample of addresses of every split.
2. **Serialization** (`serialize_graph.py`): builds a PyTorch Geometric `HeteroData` graph per split, with the address and transaction index mappings, serializing the three splits in parallel.
3. **Normalization** (`normalize_graphs.py`): clips the transaction locktime, applies logarithmic compression to heavy-tailed features and min–max/quantile scaling, all fitted on the training graph only, and saves the normalized graphs that every training run reads.
4. **Training** (`model.py`): trains one configuration with one seed. The best epoch is the one with the highest validation accuracy, and training stops after 10 epochs without improvement (at most 1,000 epochs), as in the paper. A single run can be launched with `python model.py --arch GAT --layers 2 --seed 42`.
5. **All experiments** (`run_experiments.py`): asks how many random-seed runs to execute (or `--n-seeds N`), trains GAT, HGT and HAN with 2 and 4 layers on that many seeds, builds the result tables and runs the Wilcoxon tests. The runs use all the cores and the memory: the graphs are loaded once and every run is a process forked from `run_experiments.py`, so all the runs share a single copy of the graphs in memory; the memory of every configuration is measured first with one epoch (`memory_probe.json`), and a run of that configuration reserves only its own memory (its peak minus the graphs, times `GNN_MEMORY_MARGIN`); the runs start in order of cost, the configurations with the longest epoch first, whenever their memory and threads are available; and every configuration has a fixed number of threads per run (the cores divided by the runs of that configuration that fit in memory), recorded in `result.json`, so the result of a run does not depend on the scheduling. `--workers N` trains *N* runs at a time with `CPU_COUNT // N` threads each instead; `--aggregate-only` only rebuilds the tables and tests. The run being trained is printed when it starts, and its epochs (losses, validation accuracy, best epoch so far and duration) are written as they finish to `outputs/multi_instance/results/runs/<arch>-<layers>l-seed<seed>/train.log`, e.g. `tail -f outputs/multi_instance/results/runs/*/train.log`.

#### Methodology II – Inductive Ego-Centric Address Classification

```bash
cd "Inductive Ego-Centric Address Classification"
python expansion.py                       # requires BlockSci
python extract_topological_features.py
python models.py                          # asks the number of seeds
python models.py --interpret-only         # only the ablation and SHAP of the trained models
```

1. **Expansion** (`expansion.py`): expands an independent egonet around every seed with an all-over, transaction-centered expansion of *k* = 3 hops and a *random node* width limit *w* = 3, in parallel worker processes. Any other expansion variant can be chosen (custom configuration): both methodologies use the same expansion algorithms (`shared/expansion.py`), so an egonet is the graph that the multi-instance expansion builds from its seed alone. Every transaction of an egonet records the hop where it was added (`hop`: 1 for the transactions of the seed, then 2, 3…) and, in the forward–backward variants, its direction (`hop_direction`). Every egonet is one task of the workers, so a seed with a large egonet does not hold back others, and its record is built directly from what the expansion stored: every address string and transaction hash is encoded once (memoized during the expansion), and BlockSci is only read for the values not read yet.
2. **Feature extraction** (`extract_topological_features.py`): computes, in parallel workers, the aggregation, branching and peeling-chain motifs of each seed, its degrees, PageRank, betweenness and eccentricity, and the density of its egonet. Every egonet is a task, read by its worker from the egonet file; the egonets of all the pending splits are queued in one stream, split after split and the largest first, so no worker waits at the end of a split. Only the centralities of the seed are computed, and the exact betweenness with igraph (the same values as networkx up to floating-point rounding). The graphs are built directly from the egonet records, without PyTorch, and every worker runs single-threaded numerical libraries, so the workers (one per core) do not compete for the cores. The features of each split are saved to their own file as soon as the split is complete, and a split whose file already exists is skipped.
3. **Models** (`models.py`): asks how many random-seed runs to execute (or `--n-seeds N`), trains Logistic Regression (L1 and ElasticNet), Random Forest, XGBoost, SVM, Naive Bayes, MLP and Isolation Forest on that many seeds, one single-threaded process per run on all the cores, evaluates them on the test split and runs the Wilcoxon tests (the best model against every other one).
4. **Interpretation** (`models.py`, step `interpret-ego` of the pipeline): explains the best model, the one with the highest mean test F1-score over the seeds (ties: alphabetical order; the same model as in the Wilcoxon tests), trained with `EGO_CENTRIC_INTERPRETATION_SEED` (loaded, or trained again exactly as its run). Both analyses use the validation split:
   - *Feature ablation:* the selected features are ranked by the absolute coefficients (linear models), the impurity importance (tree models) or the permutation importance on the F1-score (the others), and the model is retrained with the *k* most important ones for every *k*, measuring the F1-score (and AUC-ROC) of each, in parallel.
   - *SHAP:* the contribution of every feature to the prediction of every validation address, with the exact explainer of the model family when there is one: tree SHAP for Random Forest, XGBoost and Isolation Forest (interventional, 100 background training addresses; the isolation forest explains its anomaly score), linear SHAP for logistic regression, and kernel SHAP on the predicted probability for MLP, SVM and Naive Bayes (50 k-means centroids of the training split as background; with up to 11 selected features every feature coalition is enumerated, so the values are exact for that background). Kernel SHAP is computed in parallel, in fixed chunks of addresses with their own seeds, so it is reproducible with any number of cores.

   Each part is saved when it finishes (`interpretation/`, with `interpretation.json` describing the model, the selection and the explainers), and is not computed again for the same model, so an interrupted interpretation resumes from the missing part; a different best model (e.g. after adding seeds) replaces the previous results. `--interpret-only` runs only this step; `--skip-interpretation` skips it and `--aggregate-only` only rebuilds the tables and tests.

#### Statistical tests

```bash
python wilcoxon_tests.py     # from the repository root, after training
```

The training scripts already run it, but it can be executed again at any time from the saved results. It compares, seed by seed, (1) the GNN architectures at each depth and 4 vs 2 layers of each architecture, (2) the best tabular model against every other model, and (3) the best GNN configuration against every tabular model, which is possible because both methodologies use the same seeds and test addresses. For every comparison and metric it reports the mean paired difference, its 95% bootstrap confidence interval (10,000 resamples) and the exact two-sided Wilcoxon signed-rank test. Comparisons with fewer than 6 common seeds are skipped. Only the results of the same configuration are compared.

### Expansion experiments (Table 2)

```bash
python "Expansion Experiments/expansion_experiments.py"                    # asks which experiments to run
python "Expansion Experiments/expansion_experiments.py" --experiments 1,2,3
python "Expansion Experiments/expansion_experiments.py" --experiments 1,2,3 --workers 16  # workers of every expansion
```

The expansion experiments of Table 2 of the paper measure the structure and cost of the expanded graphs. They are run separately from the pipeline, always with the original configuration, need BlockSci and `RSD_BITCOINHEIST_CSV`, and write to `outputs/expansion_experiments/`, independently of the mode:

1. **Configuration:** the expansion variants `all over tx`, `fw bw tx whole`, `fw bw addr dedicated` and `fw bw addr whole same` with *k* = 2 hops and *w* = 5.
2. **Hops:** `fw bw tx whole` with *w* = 5 and *k* = 1, 2, 3.
3. **Width:** `fw bw tx whole` with *k* = 2 and *w* = 1, 2, 5, 10.

All of them use a *random node* width limit and the same expansion algorithms as the multi-instance `expansion.py`, keeping only the graph structure. The seed addresses are 1,500 illicit BitcoinHeist addresses found in the chain, sampled at random, and 1,500 licit ones (3,000 in total), selected with `DATA_SEED` and saved in `seeds.csv`, which is reused. Every run expands all of them into a single graph, without training, validation and test splits, and starts from the same random state (`DATA_SEED`), so the graphs and their metrics are reproducible. The runs are executed one after another, and every expansion is parallelized over all the cores (`--workers`, default: one per core, limited by the available memory), as are the lookups of the seed addresses in the chain. Every hop is computed in two parallel passes: the workers list the candidate transactions of every node of the frontier, the main process applies the width limit in the original order, the workers read the inputs and outputs of the selected transactions, and the main process inserts them in the graph in the original order. The graphs are therefore identical to a sequential expansion with any number of workers, and only the runtime depends on it (saved with every run). To save memory, the main process keeps only plain identifiers (transaction indices and address strings, shared between the edges) instead of BlockSci objects, the workers rebuild the BlockSci objects they need, and the graph of a run is freed before the next one. For every run (experiment, variant, *k* and *w*) they compute the address, transaction and total nodes, total edges (input, output and spent-output, as in `serialize_graph.py`), density, average degree, largest strongly connected component (size and percentage of the nodes), runtime, width-limit activation and candidate retention rates and the width-limit counters (expansion steps, activated steps, candidates and retained candidates). Every run is saved in `runs/` and reused if it exists (delete its file to compute it again), every metric of an experiment in `<experiment>.csv`, and the tables of the paper (relative growth along *k* and *w*, as in Table 2) in `tables/<experiment>.csv`. The seed addresses and the variants, depths and widths are set in `Expansion Experiments/experiments_config.py`.

## Command-line options

| Script | Option | Description |
|---|---|---|
| `run_pipeline.py` | `both` (default), `multi-instance`, `ego-centric` | Methodologies to run (positional) |
| | `--original` | Parameters of the paper; only the number of seeds is asked |
| | `--config <file>` | Saved custom configuration (`configuration.json`); without `--original` or `--config` the parameters are asked |
| | `--n-seeds N` | Random-seed runs per model or configuration (asked if not given) |
| | `--from <step>` | First step: `splits`, `expand-mi`, `expand-ego`, `serialize`, `normalize`, `train-gnn`, `extract-features`, `train-ego`, `interpret-ego`, `wilcoxon` (only those of the selected methodologies) |
| | `--list` | Show the steps and their commands without running them |
| | `--skip-check` | Do not check that the environment has the required libraries |
| `create_splits.py`, both `expansion.py`, `serialize_graph.py`, `normalize_graphs.py`, `extract_topological_features.py` | none | Configured by `config.py` and the environment variables; completed work is skipped |
| `run_experiments.py` (Methodology I) | `--n-seeds N` | Random-seed runs per configuration (asked if not given) |
| | `--workers N` / `auto` | `auto` (default, `GNN_PARALLEL_RUNS`): runs scheduled by the measured memory of every configuration, the most expensive first; *N*: *N* runs at a time with `CPU_COUNT // N` threads each |
| | `--archs ...`, `--layers ...` | Subset of architectures (`GAT HGT HAN`) and numbers of layers |
| | `--aggregate-only` | Only rebuild the result tables and tests |
| `model.py` (Methodology I) | `--arch`, `--layers`, `--seed` | One GNN run (required) |
| | `--threads N` | PyTorch threads of the run (default: all the cores) |
| | `--force` | Train again even if the run exists |
| | `--verbose` | Print every epoch |
| | `--probe` | Only measure the peak memory of one epoch |
| `models.py` (Methodology II) | `--n-seeds N` | Random-seed runs per model (asked if not given) |
| | `--workers N` | Parallel processes (default: one per core) |
| | `--skip-interpretation` | Train, evaluate and test, without the ablation and SHAP |
| | `--interpret-only` | Only the ablation and SHAP of the best model of the trained runs |
| | `--aggregate-only` | Only rebuild the result tables and tests |
| `wilcoxon_tests.py` | `--metric` | Metric printed on screen (`accuracy`, `precision`, `recall`, `f1`, `auc_roc`; all are saved) |
| | `--gnn-runs <csv>` | GNN runs to compare (default: those of the configuration) |
| `Expansion Experiments/expansion_experiments.py` | `--experiments 1,2,3` | Experiments to run (1 configuration, 2 hops, 3 width; asked if not given) |
| | `--workers N` | Parallel workers of every expansion (default: one per core, limited by the memory) |

## Outputs

```
outputs/                                     the original configuration
  custom-<id>/                               a custom configuration: configuration.json and the same structure as below
  splits.csv                                 seed addresses, labels, families and splits
  splits_summary.json                        verified seed addresses per split
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
    results/interpretation/                  interpretation.json, feature importance, ablation and SHAP values and figures of the best model
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

- **What gives the same results:** the same configuration, number of seeds and input data (BitcoinHeist CSV and BlockSci chain) give the same splits, graphs, features and tabular-model results in every execution and on any number of cores. Three things can still change the results: another BlockSci chain (a parse up to another block height has other transactions, so the seed addresses found in the chain and the expansions differ); another number of threads per GNN run (with `GNN_PARALLEL_RUNS = "auto"` the threads of a configuration are the cores divided by its runs that fit in the memory available when training starts, so the GNN metrics can differ slightly between machines or executions; it is recorded in every `result.json`, and setting `GNN_PARALLEL_RUNS` to a number in `config.py` fixes it on a given machine); and other library versions or hardware, which can change floating-point results (`requirements.txt` pins the versions).
- **Data:** all data-generation steps use `DATA_SEED`. The seed candidates are sorted before every random step, so the selection does not depend on the Python process, and every split is expanded from its own fixed random state.
- **Expansion in parallel:** every multi-instance split and every ego-centric egonet is expanded from its own fixed random state, so the result does not depend on the number of parallel processes or their order.
- **Training:** each GNN run fixes the Python, NumPy and PyTorch seeds and uses deterministic algorithms with a fixed number of threads (recorded in `result.json`; floating-point sums may differ slightly with another number of threads). Every random component of the tabular models is seeded, including the mutual-information feature selection, and each run is single-threaded, so the results do not depend on the number of processes. The same configuration and seed therefore always give the same result.
- **Seeds:** the first seeds are those of `config.TRAINING_SEEDS`, and more seeds are drawn deterministically from `DATA_SEED`, so an experiment with more seeds always contains the runs of an experiment with fewer.
- **Paired comparisons:** all configurations of both methodologies use the same training seeds and test addresses, so their results are compared seed by seed with `wilcoxon_tests.py`. Only the seeds present in both configurations of a comparison are used.

## Future work

The following improvements of the multi-instance training are supported by the code but not used in the paper, whose protocol is the default of `config.py`:

- **Longer early-stopping protocol.** A minimum number of epochs before early stopping and a longer patience (e.g. `min_epochs: 50, patience: 20` in `config.GNN_TRAINING`) would make the runs that stop after very few epochs less frequent, reducing the variability between random seeds. It increases the training time of every run.
- **GPU training and mini-batching.** Every epoch is a full-graph pass on CPU. Training on a GPU with enough memory for the full graph, or with neighbor-sampled mini-batches, would reduce the training time considerably, at the cost of exact determinism (GPU) or of a different training procedure (mini-batching).
