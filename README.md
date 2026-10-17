# TRACE: Transparent Ransomware Attribution via Cryptocurrency Exploration

This repository contains the code to reproduce the experiments of *TRACE: Transparent Ransomware Attribution via Cryptocurrency Exploration*. Ransomware-related Bitcoin addresses are classified with two methodologies that start from the same labeled seed addresses:

- **Methodology I – Inductive Multi-Instance Address Classification:** the seed addresses of each split are expanded into one merged heterogeneous address–transaction graph, on which Graph Neural Networks (GAT, HGT and HAN) classify the seeds.
- **Methodology II – Inductive Ego-Centric Address Classification:** an independent egonet is expanded around every seed address, topological motifs and graph metrics are extracted from it, and tabular machine learning models classify the seeds.

Every experiment is configured in [`config.py`](config.py). The only question asked is how many random-seed runs to train. Every step uses all the CPU cores, saves its outputs together with the full configuration that produced them, and skips the work that is already done, so an interrupted step can simply be launched again.

## Repository structure

```
config.py                         Parameters of every experiment (paths, seeds, splits, expansion, training)
run_pipeline.py                   Every step of one or both methodologies with a single command
create_splits.py                  Step 0: seed addresses and train/validation/test splits (both methodologies)
wilcoxon_tests.py                 Paired Wilcoxon tests from the results of both methodologies
shared/
  paths.py                        Environment variable handling
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
expansion.md, VISUALIZATIONS/     Illustrations of the expansion variants
```

## Environments

Two Python environments are used, so that BlockSci and its dependencies do not interfere with PyTorch:

| Environment | Used by | Main dependencies |
|---|---|---|
| **BlockSci** | `create_splits.py`, both `expansion.py` | [BlockSci](https://citp.github.io/BlockSci/setup.html), pandas, numpy, tqdm |
| **PyTorch** | every other script | PyTorch, [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html), scikit-learn, scipy, pandas, networkx, xgboost, shap, matplotlib, joblib, tqdm |

The experiments of the paper were run with Python 3.7.6, PyTorch 1.13.1 and PyTorch Geometric 2.3.1, on CPU.

## Environment variables

| Variable | Required by | Description |
|---|---|---|
| `RSD_BLOCKSCI_CONFIG` | `create_splits.py`, both `expansion.py` | Path to the BlockSci `config.blocksci` file |
| `RSD_BITCOINHEIST_CSV` | `create_splits.py` | Path to the [BitcoinHeist](https://www.kaggle.com/datasets/sapere0/bitcoinheist-ransomware-dataset) CSV |
| `RSD_OUTPUT_DIR` | all scripts (optional) | Directory for all generated data and results (default: `outputs/` in the repository) |
| `RSD_BLOCKSCI_PYTHON` | `run_pipeline.py` (optional) | Python interpreter of the BlockSci environment (default: the current one) |
| `RSD_TORCH_PYTHON` | `run_pipeline.py` (optional) | Python interpreter of the PyTorch environment (default: the current one) |

## Running the experiments

### Everything with one command

```bash
export RSD_BLOCKSCI_PYTHON=/path/to/blocksci-env/bin/python
export RSD_TORCH_PYTHON=/path/to/torch-env/bin/python

python run_pipeline.py                     # both methodologies
python run_pipeline.py multi-instance      # Methodology I only
python run_pipeline.py ego-centric         # Methodology II only
python run_pipeline.py --n-seeds 10        # without asking the number of seeds
python run_pipeline.py --from train-gnn    # start from a given step
python run_pipeline.py --list              # show the steps and their commands
```

`run_pipeline.py` checks that both environments have the required libraries, asks the number of random-seed runs once, and runs every step below in order, each with the interpreter of its environment. With both methodologies, the Wilcoxon tests are run again at the end to compare them. It stops at the first failing step; as every step keeps its completed work, it can be relaunched with `--from <step>`. Every execution is recorded in `outputs/pipeline_runs/<timestamp>.json` (commands, interpreters, status and duration of each step).

### Step by step

The steps can also be run one by one. The scripts can be launched from any directory.

#### Step 0 – Seed addresses and splits (BlockSci environment)

```bash
python create_splits.py
```

Every illicit BitcoinHeist address present in the chain is used as a seed, together with the same number of randomly sampled licit (*white*) addresses. The addresses are looked up in the chain in parallel. Per class, 4,000 addresses go to validation, 4,000 to test and the rest to training. Both methodologies read the resulting `splits.csv`, so they are trained and evaluated on exactly the same addresses.

#### Methodology I – Inductive Multi-Instance Address Classification

```bash
cd "Inductive Multi-Instance Address Classification"
python expansion.py          # BlockSci environment
python serialize_graph.py    # PyTorch environment, from here on
python normalize_graphs.py
python run_experiments.py    # all architectures x layers x seeds (asks the number of seeds)
```

1. **Expansion** (`expansion.py`): expands the seeds of each split with a forward–backward, transaction-centered, whole-scope expansion of *k* = 2 hops and a *random node* width limit *w* = 5, and exports the address, transaction, input, output and spent-output features of each graph. Each split uses all the cores: every BlockSci access (listing the transactions of the seeds, scanning the frontier of every hop, exploring the selected transactions and computing the features) runs in parallel workers, while the main process performs the order-dependent steps (random sampling, de-duplication and insertion into the graph) in the original order, so the graphs are identical to a single-process run.
2. **Serialization** (`serialize_graph.py`): builds a PyTorch Geometric `HeteroData` graph per split, with the address and transaction index mappings, serializing the three splits in parallel.
3. **Normalization** (`normalize_graphs.py`): clips the transaction locktime, applies logarithmic compression to heavy-tailed features and min–max/quantile scaling, all fitted on the training graph only, and saves the normalized graphs that every training run reads.
4. **Training** (`model.py`): trains one configuration with one seed. The best epoch is the one with the highest validation accuracy; training runs at least 50 epochs and stops after 20 epochs without improvement. A single run can be launched with `python model.py --arch GAT --layers 2 --seed 42`.
5. **All experiments** (`run_experiments.py`): asks how many random-seed runs to execute (or `--n-seeds N`), trains GAT, HGT and HAN with 2 and 4 layers on that many seeds, builds the result tables and runs the Wilcoxon tests. as many runs as fit in the available memory train at the same time, sharing all the cores (`--workers N` to fix their number); `--aggregate-only` only rebuilds the tables and tests.

#### Methodology II – Inductive Ego-Centric Address Classification

```bash
cd "Inductive Ego-Centric Address Classification"
python expansion.py                       # BlockSci environment
python extract_topological_features.py    # PyTorch environment, from here on
python models.py                          # asks the number of seeds
```

1. **Expansion** (`expansion.py`): expands an independent egonet around every seed with an all-over, transaction-centered expansion of *k* = 3 hops and a *random node* width limit *w* = 3, in parallel worker processes.
2. **Feature extraction** (`extract_topological_features.py`): computes, in parallel workers, the aggregation, branching and peeling-chain motifs of each seed, its degrees, PageRank, betweenness and eccentricity, and the density of its egonet.
3. **Models** (`models.py`): asks how many random-seed runs to execute (or `--n-seeds N`), trains Logistic Regression (L1 and ElasticNet), Random Forest, XGBoost, SVM, Naive Bayes, MLP and Isolation Forest on that many seeds, one single-threaded process per run on all the cores, and evaluates them on the test split. It then runs the Wilcoxon tests and analyzes the best model (MLP) with a permutation-importance feature ablation and SHAP, both on the validation split and in parallel (the SHAP values are computed in fixed chunks of addresses with their own seeds, so they are reproducible with any number of cores). `--skip-interpretation` skips the last analyses and `--aggregate-only` only rebuilds the tables and tests.

#### Statistical tests

```bash
python wilcoxon_tests.py     # from the repository root, after training
```

The training scripts already run it, but it can be executed again at any time from the saved results. It compares, seed by seed, (1) the GNN architectures at each depth and 4 vs 2 layers of each architecture, (2) the best tabular model against every other model, and (3) the best GNN configuration against every tabular model, which is possible because both methodologies use the same seeds and test addresses. For every comparison and metric it reports the mean paired difference, its 95% bootstrap confidence interval (10,000 resamples) and the exact two-sided Wilcoxon signed-rank test. Comparisons with fewer than 6 common seeds are skipped.

## Outputs

```
outputs/
  splits.csv, splits_summary.json            seed addresses, labels, families and splits
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
    features.csv, features.json
    results/experiment_config.json           full configuration: data, expansion, features, models, evaluation, seeds, resources
    results/runs/, predictions/              metrics and test predictions of every model and seed
    results/ego_runs.csv, ego_summary.csv
    results/interpretation/                  feature importance, ablation and SHAP values and figures
  statistical_tests/
    gnn_tests.csv, ego_tests.csv, cross_methodology_tests.csv, tests_config.json
```

Every result records the git commit of the code that produced it, and every expansion, feature extraction and experiment saves its configuration and the library versions used. The test predictions are saved for every run, so other metrics or per-address comparisons between configurations and methodologies can be computed without training again.

## Reproducibility

- **Data:** all data-generation steps use `DATA_SEED`. The seed candidates are sorted before every random step, so the selection does not depend on the Python process, and every split is expanded from its own fixed random state.
- **Expansion in parallel:** every multi-instance split and every ego-centric egonet is expanded from its own fixed random state, so the result does not depend on the number of parallel processes or their order.
- **Training:** each GNN run fixes the Python, NumPy and PyTorch seeds and uses deterministic algorithms with a fixed number of threads (recorded in `result.json`; floating-point sums may differ slightly with another number of threads). Every random component of the tabular models is seeded, including the mutual-information feature selection, and each run is single-threaded, so the results do not depend on the number of processes. The same configuration and seed therefore always give the same result.
- **Seeds:** the first seeds are those of `config.TRAINING_SEEDS`, and more seeds are drawn deterministically from `DATA_SEED`, so an experiment with more seeds always contains the runs of an experiment with fewer.
- **Paired comparisons:** all configurations of both methodologies use the same training seeds and test addresses, so their results are compared seed by seed with `wilcoxon_tests.py`.
