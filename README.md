# TRACE: Transparent Ransomware Attribution via Cryptocurrency Exploration

This repository contains the code of the experiments of *TRACE: Transparent Ransomware Attribution via Cryptocurrency Exploration*. Ransomware-related Bitcoin addresses are classified with two methodologies that start from the same labeled seed addresses:

- **Methodology I – Inductive Multi-Instance Address Classification:** the seed addresses of each split are expanded into one merged heterogeneous address–transaction graph, on which Graph Neural Networks (GAT, HGT and HAN) classify the seeds.
- **Methodology II – Inductive Ego-Centric Address Classification:** an independent egonet is expanded around every seed address, topological motifs and graph metrics are extracted from it, and tabular machine learning models classify the seeds.

Both methodologies use the same seed addresses and splits (every illicit BitcoinHeist address found in the chain and as many licit ones, each class split 60/20/20 into training, validation and test), the same training seeds and the same decision threshold, so their results are compared seed by seed with paired Wilcoxon tests.

**Data availability.** The full experimental datasets (the expanded graphs, the serialized PyTorch Geometric graphs and the extracted features) are not distributed with the repository due to their size; they are provided by the authors upon request. They can also be reconstructed from BitcoinHeist and the public Bitcoin blockchain with the pipeline below.

The parameters of every experiment are in [`config.py`](config.py) (the *original configuration*). `run_pipeline.py --original` runs every step with them and only asks how many random-seed runs to train; without `--original`, it asks the parameters of the selected methodologies (see [Configurations](#configurations-original-and-custom)). Every step uses all the CPU cores, saves its outputs with the configuration that produced them, and reuses its completed work only while it was computed by the same code, parameters and inputs (see [Reuse of completed work](#reuse-of-completed-work)), so an interrupted step can simply be launched again.

## Repository structure

```
requirements.txt                  Pinned dependencies of the Python environment (BlockSci is installed separately)
config.py                         Parameters of every experiment (paths, seeds, splits, expansions, models, training)
run_pipeline.py                   Every step of one or both methodologies with a single command
create_splits.py                  Step 0: seed addresses and train/validation/test splits (both methodologies)
wilcoxon_tests.py                 Paired Wilcoxon tests from the results of both methodologies
shared/
  expansion.py                    Expansion variants: sequential reference algorithms, one graph per process (ego-centric egonets)
  parallel_expansion.py           The same variants in parallel, one graph of many seeds (multi-instance, expansion experiments)
  parallel.py                     Ordered parallel map over forked worker processes
  configuration.py                Configurable parameters, questions and configuration files
  experiment.py                   Training seeds, fingerprints, configuration snapshots and result files
  splits.py                       Verification of a splits file
  stats.py                        Paired Wilcoxon tests, bootstrap confidence intervals and summaries
  threads.py                      Single-threaded numerical libraries in the per-core worker processes
  paths.py                        Environment variable handling
Inductive Multi-Instance Address Classification/
  expansion.py                    Step 1: graph expansion and feature extraction
  serialize_graph.py              Step 2: PyTorch Geometric graphs
  normalize_graphs.py             Step 3: feature normalization (fitted on the training graph)
  model.py                        Step 4: training and evaluation of one GNN configuration with one seed
  run_experiments.py              Step 5: all GNN runs in parallel, result tables and Wilcoxon tests
Inductive Ego-Centric Address Classification/
  expansion.py                    Step 1: egonet expansion
  extract_topological_features.py Step 2: motif counts and graph metrics
  tune_hyperparameters.py         Step 3: hyperparameter search of the tabular models
  train_models.py                 Step 4: training, evaluation and statistical tests of the tabular models
  interpret_model.py              Step 5: feature ablation and SHAP of the best model
  estimators.py                   The tabular model pipelines and their evaluation (imported by steps 3-5)
  common.py                       Paths, dataset, fingerprints and saved results (imported by steps 3-5)
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

# 6. Check: prints the height of the last block of the chain (676078 for the chain of the datasets)
python -c "import os, blocksci; print(blocksci.Blockchain(os.environ['RSD_BLOCKSCI_CONFIG']).blocks[-1].height)"
```

- **Step 3** writes a `.pth` file that adds the BlockSci package directory to the Python path of the new environment, as the editable installation of BlockSci does in its own environment. The compiled extension only works with the Python version it was built for, so both environments must use the same Python version (3.7).
- **Step 4:** BlockSci declares `pycrypto` as a dependency, but `pycrypto` is unmaintained and installs Python 2 code with a recent pip (*NameError: name 'xrange' is not defined* when importing BlockSci). `pycryptodome` provides the same `Crypto` module; `pip check` still reports the missing `pycrypto`, which can be ignored.
- **Step 5:** `export` only sets the variables in the current shell; with conda they can be saved in the environment with `conda env config vars set RSD_BLOCKSCI_CONFIG=... RSD_BITCOINHEIST_CSV=...`.
- **Without an existing BlockSci build,** it is built following its setup instructions with the new environment active, which installs `blockscipy` directly in it (steps 3 and 4 are then not needed).

**Blockchain of the datasets.** Every dataset of this project (the seed addresses, splits and graphs of both methodologies, and the expansion experiments) was built from the same BlockSci parse of the Bitcoin chain with 676,079 blocks (heights 0 to 676,078), so step 6 prints 676078. A parse up to another height gives other seed addresses and graphs (see [Reproducibility](#reproducibility)), and the fingerprints only record the path of the BlockSci config file, not the height of the chain it points to.

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
| Original | `run_pipeline.py --original` | Those of `config.py`. Only the number of random-seed runs is asked (or given with `--n-seeds`). | `outputs/` |
| Custom | `run_pipeline.py` | Asked group by group for the selected methodologies, with the original values as defaults. | `outputs/custom-<id>/` |
| Saved custom | `run_pipeline.py --config <file>` | Those of a configuration file, e.g. the `configuration.json` saved by a custom run. | `outputs/custom-<id>/` |

Without `--original` or `--config`, `run_pipeline.py` asks, for every group of parameters, whether to keep the original ones or to choose new ones (Enter keeps the value in brackets):

- **Data generation** (always): `DATA_SEED`.
- **Methodology I:** the expansion (variant, width limit and *w*, forward and backward hops or *k*; the variants are illustrated in [`expansion.md`](expansion.md)), and the GNN architectures, numbers of layers and training parameters.
- **Methodology II:** the expansion (variant, width limit and *w*, forward and backward hops or *k*), the radius and betweenness of the topological features, and the training features of the tabular models. Their hyperparameters are not asked: they are searched over the grids of `config.EGO_CENTRIC_MODELS`, always with 5-fold cross-validation.
- **Statistical tests** (always): bootstrap resamples and minimum common seeds.

Only the parameters that change the data or the results are asked; the resources (workers, memory) stay in `config.py`. The chosen parameters are saved in `outputs/custom-<id>/configuration.json`, where `<id>` is a hash of all the parameters, and every step reads them through `RSD_CONFIG_FILE`. All the data and results of a custom configuration are kept in its own directory, so different configurations are never mixed, and choosing the same parameters again resumes the same directory. A configuration with the same values as the original one is the original one. To relaunch or resume a custom configuration without answering again, use `--config <file>`; to run a single script with it, set `RSD_CONFIG_FILE=<file>`. Choosing the parameters needs an interactive terminal, so unattended runs use `--original` or `--config`.

### Everything with one command

```bash
python run_pipeline.py --original                  # both methodologies, original configuration
python run_pipeline.py                             # both methodologies, parameters chosen by the user
python run_pipeline.py multi-instance --original   # Methodology I only
python run_pipeline.py ego-centric --original      # Methodology II only
python run_pipeline.py --original --n-seeds 10     # without any question
python run_pipeline.py --config <file>             # a saved custom configuration
python run_pipeline.py --original --from train-gnn # start from a given step
python run_pipeline.py ego-centric --original --from interpret-ego   # only the ablation and SHAP of Methodology II
python run_pipeline.py --list                      # show the steps and their commands
```

`run_pipeline.py` checks that the environment has the required libraries, sets the configuration (original, chosen by the user or from a file), asks the number of random-seed runs once, and runs the steps below in order with the current interpreter:

| Step | Script | Methodology |
|---|---|---|
| `splits` | `create_splits.py` | both |
| `expand-mi`, `serialize`, `normalize`, `train-gnn` | `expansion.py`, `serialize_graph.py`, `normalize_graphs.py`, `run_experiments.py` | I |
| `expand-ego`, `extract-features`, `tune-ego`, `train-ego`, `interpret-ego` | `expansion.py`, `extract_topological_features.py`, `tune_hyperparameters.py`, `train_models.py`, `interpret_model.py` | II |
| `wilcoxon` | `wilcoxon_tests.py` | both (only when both methodologies are run, to compare them) |

It stops at the first failing step; as every step keeps its completed work, it can be relaunched with `--from <step>` and the same `--original` or `--config <file>` (the message of the failure gives the command). Every execution is recorded in `pipeline_runs/<timestamp>.json` of its output directory (configuration, commands, interpreter, status and duration of each step).

### Step by step

The steps can also be run one by one. The scripts can be launched from any directory, and use the original configuration unless `RSD_CONFIG_FILE` is set. The descriptions below give the parameters of the original configuration.

#### Step 0 – Seed addresses and splits

```bash
python create_splits.py
```

Every illicit BitcoinHeist address present in the chain is used as a seed, together with the same number of randomly sampled licit (*white*) addresses; addresses that appear with both labels are illicit. Each class is split 60/20/20: validation and test get 20% of the class each (rounded to the nearest address) and training the rest, so the three splits are balanced and their fractions are exact up to one address per class. The addresses are looked up in the chain in parallel, and the results are taken in their original order.

The selection is deterministic: the candidates are sorted before every random step and the random draws use `DATA_SEED`, so the same inputs always give the same seed addresses. The splits are verified (every address appears in a single split, the classes are balanced, and every label and family agrees with BitcoinHeist), saved in `splits.csv`, and reused while their inputs do not change (see [Reuse of completed work](#reuse-of-completed-work)). Both methodologies read this file, so they are trained and evaluated on exactly the same addresses.

#### Methodology I – Inductive Multi-Instance Address Classification

```bash
cd "Inductive Multi-Instance Address Classification"
python expansion.py          # requires BlockSci
python serialize_graph.py
python normalize_graphs.py
python run_experiments.py    # all architectures x layers x seeds (asks the number of seeds)
```

1. **Expansion** (`expansion.py`): expands the seeds of each split into one graph, with a forward–backward, transaction-centered, whole-scope expansion of *k* = 2 hops and a *random node* width limit *w* = 5 (any variant can be chosen), and exports the address, transaction, input, output and spent-output features of each graph to `graphs/<split>/`.
   - *Parallel expansion* (`shared/parallel_expansion.py`): every hop has two parallel passes. The workers list the candidate transactions of every node of the frontier (as transaction indices), the main process applies the width limit and removes duplicates in the order of the sequential algorithms of `shared/expansion.py`, the workers read the selected transactions (their hash and the inputs and outputs with an address), and the main process inserts them in the graph in that order. The graph is therefore identical to a sequential expansion with any number of workers. Only the selected transactions are hashed (the width limit discards most candidates); transactions are identified by their index during the expansion and by their hash in the exported files, which only differ for the two pairs of duplicated 2010 coinbase transactions (BIP30). Addresses are identified by their address string, except pay-to-pubkey (P2PK) addresses: BlockSci gives them the string of the pay-to-pubkey-hash address of the same key, so they are identified by `pubkey:` + that string (the `addr_str` of the exported files), and the two addresses, each with its own history and features, are two nodes instead of one.
   - *Width limit*: with the *random node* limit, every node of the frontier samples *w* transactions uniformly among its own distinct candidates, independently of the other nodes, and a transaction chosen by several nodes is added once. With the *random hop* limit, *w* transactions are sampled among the distinct candidates of the whole hop.
   - *Hops*: `tx_feats.csv` records the hop where every transaction was added (`hop`: 1 for the transactions of the seeds, then 2, 3…) and its direction (`hop_direction`: `forward`, `backward`, or `both` for a hop-1 transaction where a seed is both an input and an output; empty in the all-over variants). They are not used as GNN features.
   - *Address features*: computed from the whole history of every address in the chain. `n_incoming_txs` counts the transactions where the address receives and `n_outgoing_txs` those where it spends; `n_total_txs` counts each distinct transaction once; the `in_tx_*` and `out_tx_*` statistics follow BlockSci's naming (transactions where the address is an input or an output), and the waiting times are the blocks between consecutive transactions (at least two transactions). The features of an address do not depend on the split (except its class), so they are computed once and reused by the next splits; within every block of 100,000 addresses the busiest are given first to the workers, in small chunks. Address histories are read as whole BlockSci arrays when this gives exactly the same values as reading them item by item, which is checked on a sample of addresses of every split.
   - *Resuming*: the expanded graph and the address features (block by block) are saved as they are computed, so an interrupted expansion resumes where it stopped.
2. **Serialization** (`serialize_graph.py`): builds a PyTorch Geometric `HeteroData` graph per split, with the address and transaction index mappings, serializing the three splits in parallel. The categorical features are encoded as 0/1 columns with fixed rules, the same in every split, so they are already in [0, 1]: the address `full_type` as a one-hot script type (P2PK, P2PKH, P2SH, P2WPKH, P2WSH, other) plus two flags (the script involves multisig; it involves SegWit, native or nested in P2SH), the input `sequence_num` as a one-hot (final `0xFFFFFFFF`, `0xFFFFFFFE`, other: signals replace-by-fee), and the transaction `version` as a flag (version 2 or higher). Every graph has 60 address features, 6 transaction features, 5 input-edge features and 2 output-edge features (their names and order are in `normalize_graphs.py`). A graph is serialized again when its export or this code change, and the normalization and the training runs are redone when the graphs they were built from change.
3. **Normalization** (`normalize_graphs.py`): clips the transaction locktime, applies logarithmic compression to heavy-tailed features and min–max/quantile scaling, all fitted on the training graph only, and saves the normalized graphs that every training run reads.
4. **Training** (`model.py`): trains one configuration with one seed. Every epoch is one full-graph Adam step on the cross-entropy with label smoothing (`label_smoothing`, 0.1). The model that is validated, selected, saved and tested is an exponential moving average of the trained weights (`ema_decay`, 0.9: about the last 10 epochs), which removes the epoch-to-epoch noise of the full-graph steps; both cost nothing per epoch and 0 disables them. The best epoch is the one with the lowest validation loss (plain cross-entropy of the averaged weights), and training stops after `patience` epochs without improving it (see `config.GNN_TRAINING`). A completed run is only reused if it was trained on the current graphs with the current training hyperparameters and `model.py` code (`training_fingerprint` in its `result.json`); otherwise it is trained again. A test address is predicted illicit when its probability is at least 0.5. A single run can be launched with `python model.py --arch GAT --layers 2 --seed 42`.
5. **All experiments** (`run_experiments.py`): asks how many random-seed runs to execute (or `--n-seeds N`), trains GAT, HGT and HAN with 2 and 4 layers on that many seeds, builds the result tables and runs the Wilcoxon tests. The graphs are loaded once and every run is a process forked from `run_experiments.py`, so all the runs share a single copy of the graphs in memory. The memory of every configuration is measured first with one epoch (`memory_probe.json`), and a run of that configuration reserves only its own memory (its peak minus the graphs, times `GNN_MEMORY_MARGIN`); the runs start in order of cost, the configurations with the longest epoch first, whenever their memory and threads are available; and every configuration has a fixed number of threads per run (the cores divided by the runs of that configuration that fit in memory), recorded in `result.json`, so the result of a run does not depend on the scheduling. `--workers N` trains *N* runs at a time with `CPU_COUNT // N` threads each instead; `--aggregate-only` only rebuilds the tables and tests. The epochs of every run are written as they finish to `results/runs/<arch>-<layers>l-seed<seed>/train.log`, e.g. `tail -f outputs/multi_instance/results/runs/*/train.log`.

#### Methodology II – Inductive Ego-Centric Address Classification

```bash
cd "Inductive Ego-Centric Address Classification"
python expansion.py                       # requires BlockSci
python extract_topological_features.py
python tune_hyperparameters.py            # optional: train_models.py searches what is missing
python train_models.py                    # asks the number of seeds
python interpret_model.py                 # ablation and SHAP of the best model
```

1. **Expansion** (`expansion.py`): expands an independent egonet around every seed with an all-over, transaction-centered expansion of *k* = 3 hops and a *random node* width limit *w* = 3 (any variant can be chosen). The egonets use the sequential algorithms of `shared/expansion.py`, which give the same graph as the parallel expansion of Methodology I, so an egonet is the graph that the multi-instance expansion builds from its seed alone. Every egonet is one task of the worker processes, expanded from its own random state (from `DATA_SEED` and its seed address), so a seed with a large egonet does not hold back others and the egonets do not depend on the number or order of the workers. Every transaction records its `hop` and, in the forward–backward variants, its `hop_direction`. The egonets of a split are saved in `egonets/<split>.jsonl`, with their expansion configuration in `egonets/<split>.meta.json`.
2. **Feature extraction** (`extract_topological_features.py`): computes, in parallel workers, the aggregation, branching and peeling-chain motifs of each seed, its degrees, PageRank, betweenness and eccentricity, and the density of its egonet. The egonets of all the pending splits are queued in one stream, split after split and the largest first, so no worker waits at the end of a split. Only the centralities of the seed are computed, and the exact betweenness with igraph (the same values as networkx up to floating-point rounding). The features of each split are saved in `features/<split>.csv` as soon as the split is complete, with `features/<split>.json`, which records the expansion configuration of its egonets; every later result carries it.
3. **Hyperparameter search** (`tune_hyperparameters.py`): Logistic Regression (L1 and ElasticNet), Random Forest, XGBoost, SVM, Naive Bayes, MLP and Isolation Forest (`estimators.py`), each a pipeline of clipping of negative values and `log1p` of the 12 topological features, mutual-information feature selection, scaling and the model. Their hyperparameters are not fixed: every model is searched over its grid (`config.EGO_CENTRIC_MODELS`; successive halving for Random Forest and XGBoost) with stratified 5-fold cross-validation on the training split only, scored with the F1-score at the 0.5 threshold. The search is the longest part of this methodology (hours on 16 cores the first time, mainly SVM, MLP and the halving searches); it is saved model by model, so an interrupted search resumes from the next model, and it is never repeated for the same features, grids and code (`--retune` searches every model again).
4. **Training** (`train_models.py`): asks how many random-seed runs to execute (or `--n-seeds N`), searches the models whose search is missing or outdated, and trains every model with its searched hyperparameters on that many seeds, one single-threaded process per run on all the cores. Each run is evaluated with 5-fold cross-validation on the training split and, fitted on the whole training split, on the test split; every evaluation predicts illicit when the probability of the illicit class is at least 0.5, as the GNNs do. Isolation Forest is unsupervised and predicts the anomalies as illicit; its probability is a fixed function of the anomaly score of each address (fitted on the training split, at least 0.5 exactly for the anomalies), so it does not depend on the other addresses evaluated. It then builds the result tables (`ego_runs.csv`, `ego_summary.csv`, `ego_cv_folds.csv`, `ego_cv_summary.csv`) and runs the Wilcoxon tests (the best model against every other one). `--aggregate-only` only rebuilds the tables and tests.
5. **Interpretation** (`interpret_model.py`): explains the best model, the one with the highest mean test F1-score over the seeds (ties: alphabetical order; the same model as in the Wilcoxon tests), trained with `EGO_CENTRIC_INTERPRETATION_SEED` exactly as its run. Both analyses use the validation split:
   - *Feature ablation:* the selected features are ranked by the absolute coefficients (linear models), the impurity importance (tree models) or the permutation importance on the F1-score (the others), and the model is retrained with the *k* most important ones for every *k*, measuring the F1-score (and AUC-ROC) of each, in parallel.
   - *SHAP:* the contribution of every feature to the prediction of every validation address, with the exact explainer of the model family when there is one: tree SHAP for Random Forest, XGBoost and Isolation Forest (interventional, 100 background training addresses; the isolation forest explains its anomaly score), linear SHAP for logistic regression, and kernel SHAP on the predicted probability for MLP, SVM and Naive Bayes (50 k-means centroids of the training split as background; with up to 11 selected features every feature coalition is enumerated, so the values are exact for that background). Kernel SHAP is computed in parallel, in fixed chunks of addresses with their own seeds, so it is reproducible with any number of cores.

   Every interpretation is saved in its own directory, `interpretation/<model>-seed<seed>_<fingerprint>/`, with `interpretation.json` (the model, why it was selected, its data and the explainers) and the trained model. Each part is saved when it finishes, so an interrupted interpretation resumes from the missing part.

The searches, runs and interpretations of other features (e.g. another expansion of the egonets) are never overwritten: each one is saved under its fingerprint, and `tuning/index.csv`, `runs/index.csv` and `interpretation/index.csv` list all of them with the expansion configuration of their data (variant, width limit, hops, radius…), so they can be compared. The result tables and the tests only include the runs of the current features and code.

#### Statistical tests

```bash
python wilcoxon_tests.py     # from the repository root, after training
```

The training scripts already run it, but it can be executed again at any time from the saved results. It compares, seed by seed, (1) the GNN architectures at each depth and 4 vs 2 layers of each architecture, (2) the best tabular model against every other model, and (3) the best GNN configuration against every tabular model, which is possible because both methodologies use the same seeds, test addresses and decision threshold (0.5). For every comparison and metric it reports the mean paired difference, its 95% bootstrap confidence interval (10,000 resamples) and the exact two-sided Wilcoxon signed-rank test. Comparisons with fewer than 6 common seeds are skipped. Only the results of the same configuration, trained on its current data, are compared: the result tables of a methodology are removed when none of its runs is current.

### Expansion experiments (Table 2)

```bash
python "Expansion Experiments/expansion_experiments.py"                    # asks which experiments to run
python "Expansion Experiments/expansion_experiments.py" --experiments 1,2,3
python "Expansion Experiments/expansion_experiments.py" --experiments 1,2,3 --workers 16  # workers of every expansion
```

The expansion experiments of Table 2 of the paper measure the structure and cost of the expanded graphs. They are run separately from the pipeline, always with the original configuration, need BlockSci and `RSD_BITCOINHEIST_CSV`, and write to `outputs/expansion_experiments/`:

1. **Configuration:** the expansion variants `all over tx`, `fw bw tx whole`, `fw bw addr dedicated` and `fw bw addr whole same` with *k* = 2 hops and *w* = 5.
2. **Hops:** `fw bw tx whole` with *w* = 5 and *k* = 1, 2, 3.
3. **Width:** `fw bw tx whole` with *k* = 2 and *w* = 1, 2, 5, 10.

The seed addresses are 1,500 illicit BitcoinHeist addresses found in the chain, sampled at random, and 1,500 licit ones (3,000 in total), selected with `DATA_SEED` (as in Step 0) and saved in `seeds.csv`. Every run expands all of them into a single graph, without splits, with the parallel expansion of Methodology I (`shared/parallel_expansion.py`), so any expansion variant and width-limit mode can be used (`experiments_config.py`; the paper uses a *random node* width limit). Every run starts from the same random state (`DATA_SEED`), so the graphs and their metrics are reproducible and do not depend on the number of workers; only the runtime does (saved with every run). The runs are executed one after another, each parallelized over all the cores (`--workers`, default: one per core, limited by the available memory), and the graph of a run is freed before the next one.

**Comparable runtimes.** The tables compare the runtimes of the runs with each other, so all of them are measured in the same context (`experiments_config.py`):

- **One session.** All the runs of the selected experiments are timed in one execution of the script. The saved runs are reused only if all of them were timed in the same session, on the same machine, with the same workers and timing settings, and none was contended; otherwise all of them are run again. Run the three experiments together (`--experiments 1,2,3`), so the shared run (`fw bw tx whole`, *k* = 2, *w* = 5) belongs to one session with all the others.
- **Warm cache.** Every run is first expanded without timing (`WARMUP_RUNS`, default 1), then timed (`TIMED_RUNS`, default 1; the runtime is the median). The expansions of a run are identical, so every timed expansion finds the blockchain data it reads in the page cache, whatever ran before it, and no run gains from following a similar one. Both the warm-up and the timed runtimes are saved.
- **Idle machine.** A session only starts if, over 30 s, the other processes use at most 1 core on average and at least 75% of the memory is available; otherwise the script lists the busiest processes and stops (`--allow-busy` times the runs anyway). The cores used by the other processes during every timed expansion are saved with the run, and a run where they used more than 1 core on average is marked *contended*, reported at the end, and timed again by the next session. The load is read from `/proc` (Linux).

A session takes about twice the time of the expansions themselves (about 6 hours with 16 workers at the runtimes measured so far, mostly `fw bw addr whole same`). Do not run it while other experiments (such as the GNN training) use the machine.

For every run (experiment, variant, *k* and *w*) they compute the address, transaction and total nodes, total edges (input, output and spent-output, as in `serialize_graph.py`), density, average degree, largest strongly connected component (size and percentage of the nodes), runtime (with its session, the warm-up and timed runtimes and the load of the machine), width-limit activation and candidate retention rates and the width-limit counters (sampling steps, activated steps, candidates and retained candidates). Every run is saved in `runs/`, every metric of an experiment in `<experiment>.csv`, and the tables of the paper (relative growth along *k* and *w*, as in Table 2, and the runtimes relative to the first variant, depth or width) in `tables/<experiment>.csv`. The seed addresses and the variants, depths and widths are set in `Expansion Experiments/experiments_config.py`.

## Reuse of completed work

Every step saves, with its outputs, a *fingerprint* of what they were computed from: the code (by its syntax tree, so editing comments does not count), the parameters and the inputs. A step reuses its completed work only while the fingerprint is unchanged, and computes it again otherwise, so a change anywhere (code, configuration, BitcoinHeist data, chain) is never silently ignored, while an interrupted or repeated execution does not redo finished work.

| Step | Fingerprint | Saved in |
|---|---|---|
| Splits (`create_splits.py`) and experiment seeds | the selection functions, the BitcoinHeist file (content), the chain (configuration file), `DATA_SEED` and the split fractions (or seeds per class) | `splits_summary.json`, `seeds_summary.json` |
| Multi-instance expansion | `expansion.py`, `shared/expansion.py`, `shared/parallel_expansion.py`, the expansion configuration, `DATA_SEED` and the seed addresses of the split | `graphs/<split>/expansion_meta.json` |
| Serialization, normalization, GNN runs, memory probes | the expansion fingerprints of the graphs they were built from | `graph_meta.json`, `normalization.json`, `result.json`, `memory_probe.json` |
| Ego-centric egonets | `expansion.py`, `shared/expansion.py`, the expansion configuration, `DATA_SEED` and the seed addresses of the split | `egonets/<split>.meta.json` |
| Topological features | `extract_topological_features.py`, the feature parameters and the egonet fingerprint | `features/<split>.json` |
| Hyperparameter searches | the content of the feature files, the grid and fixed settings of the model and the training code | `tuning/searches.json` |
| Tabular runs and interpretations | the content of the feature files, the hyperparameters and fixed settings of the model and the training code | `runs/*_<fingerprint>.json`, `interpretation/*_<fingerprint>/` |
| Expansion experiments | the expansion and metrics code, the variant, hops, width limit and the seed addresses; the runtimes are reused only if all the runs of the selected experiments were timed in one session, with the same workers and timing settings | `runs/<run>.json` |

The selection of the seed addresses is deterministic, so selecting them again with unchanged inputs gives the same addresses; the expansions include the seed addresses themselves in their fingerprints, so only new seed addresses rebuild the data built from them. The searches, runs and interpretations of Methodology II are kept for every fingerprint (see above); the other steps keep only the outputs of the current fingerprint.

## Command-line options

| Script | Option | Description |
|---|---|---|
| `run_pipeline.py` | `both` (default), `multi-instance`, `ego-centric` | Methodologies to run (positional) |
| | `--original` | Original configuration of `config.py`; only the number of seeds is asked |
| | `--config <file>` | Saved custom configuration (`configuration.json`); without `--original` or `--config` the parameters are asked |
| | `--n-seeds N` | Random-seed runs per model or configuration (asked if not given) |
| | `--from <step>` | First step: `splits`, `expand-mi`, `expand-ego`, `serialize`, `normalize`, `train-gnn`, `extract-features`, `tune-ego`, `train-ego`, `interpret-ego`, `wilcoxon` (only those of the selected methodologies) |
| | `--list` | Show the steps and their commands without running them |
| | `--skip-check` | Do not check that the environment has the required libraries |
| `create_splits.py`, both `expansion.py`, `serialize_graph.py`, `normalize_graphs.py`, `extract_topological_features.py` | none | Configured by `config.py` and the environment variables (see [Reuse of completed work](#reuse-of-completed-work)) |
| `run_experiments.py` (Methodology I) | `--n-seeds N` | Random-seed runs per configuration (asked if not given) |
| | `--workers N` / `auto` | `auto` (default, `GNN_PARALLEL_RUNS`): runs scheduled by the measured memory of every configuration, the most expensive first; *N*: *N* runs at a time with `CPU_COUNT // N` threads each |
| | `--archs ...`, `--layers ...` | Subset of architectures (`GAT HGT HAN`) and numbers of layers |
| | `--aggregate-only` | Only rebuild the result tables and tests |
| `model.py` (Methodology I) | `--arch`, `--layers`, `--seed` | One GNN run (required) |
| | `--threads N` | PyTorch threads of the run (default: all the cores) |
| | `--force` | Train again even if the run exists |
| | `--verbose` | Print every epoch |
| | `--probe` | Only measure the peak memory of one epoch |
| `tune_hyperparameters.py` (Methodology II) | `--retune` | Search every model again, even if its search is up to date |
| | `--workers N` | Parallel processes (default: one per core) |
| `train_models.py` (Methodology II) | `--n-seeds N` | Random-seed runs per model (asked if not given) |
| | `--workers N` | Parallel processes (default: one per core) |
| | `--aggregate-only` | Only rebuild the result tables and tests |
| `interpret_model.py` (Methodology II) | `--workers N` | Parallel processes (default: one per core) |
| `wilcoxon_tests.py` | `--metric` | Metric printed on screen (`accuracy`, `precision`, `recall`, `f1`, `auc_roc`; all are saved) |
| | `--gnn-runs <csv>` | GNN runs to compare (default: those of the configuration) |
| `Expansion Experiments/expansion_experiments.py` | `--experiments 1,2,3` | Experiments to run (1 configuration, 2 hops, 3 width; asked if not given) |
| | `--workers N` | Parallel workers of every expansion (default: one per core, limited by the memory) |
| | `--allow-busy` | Time the runs even if other processes use the machine (the runtimes are then not comparable) |

## Outputs

```
outputs/                                     the original configuration
  custom-<id>/                               a custom configuration: configuration.json and the same structure as below
  splits.csv                                 seed addresses, labels, families and splits
  splits_summary.json                        verified seed addresses per split and their fingerprint
  pipeline_runs/<timestamp>.json             steps, commands, status and duration of every run_pipeline.py execution
  multi_instance/
    graphs/expansion_config.json             expansion configuration and fingerprint of every split
    graphs/normalization.json                fitted normalization and the expansion fingerprints of its graphs
    graphs/<split>/                          feature CSVs, expansion_meta.json, graph.pth, graph_meta.json,
                                             graph_normalized.pth, mappings, labeled_nodes.csv
    results/experiment_config.json           full configuration: data, expansion, normalization, training, evaluation, seeds, resources
    results/memory_probe.json                peak memory and epoch time of every GNN configuration
    results/runs/<arch>-<layers>l-seed<seed>/
      result.json                            full configuration, training history, best epoch and test metrics
      predictions.csv                        test seeds with label and predicted probability
      model.pth, train.log                   best weights and training log
    results/gnn_runs.csv                     one row per run trained on the current graphs
    results/gnn_summary.csv                  mean, standard deviation and median per configuration
  ego_centric/
    egonets/<split>.jsonl, <split>.meta.json egonets and their expansion configuration and fingerprint
    egonets/expansion_config.json
    features/<split>.csv, <split>.json       features of the seed addresses, with the expansion configuration and fingerprints
    results/experiment_config.json           full configuration: data, models, searches, evaluation, seeds, resources
    results/tuning/searches.json, index.csv  every hyperparameter search by fingerprint, with the expansion of its data
    results/tuning/search_<model>_<fingerprint>.csv   every candidate of a search
    results/runs/<model>-seed<seed>_<fingerprint>.json, index.csv   metrics of every run, with its data
    results/predictions/<model>-seed<seed>_<fingerprint>.csv        test addresses with label and predicted probability
    results/ego_runs.csv, ego_summary.csv, ego_cv_folds.csv, ego_cv_summary.csv   runs of the current features
    results/interpretation/index.csv         every interpretation, with the expansion of its data
    results/interpretation/<model>-seed<seed>_<fingerprint>/   interpretation.json, model.joblib, feature importance,
                                             ablation and SHAP values and figures
  statistical_tests/
    gnn_tests.csv, ego_tests.csv, cross_methodology_tests.csv, tests_config.json

outputs/expansion_experiments/               expansion experiments of Table 2
  seeds.csv, seeds_summary.json, experiments_config.json
  runs/<variant> <k> hops <w> random node limit.json   metrics and fingerprint of every run
  configuration.csv, hops.csv, width.csv     every metric of every run of each experiment
  tables/<experiment>.csv                    tables of the paper
```

Every result records the git commit of the code that produced it, and every expansion, feature extraction and experiment saves its configuration and the library versions used. The test predictions are saved for every run, so other metrics or per-address comparisons between configurations and methodologies can be computed without training again.

## Reproducibility

- **Same inputs, same results:** the same configuration, number of seeds and input data (BitcoinHeist CSV and BlockSci chain) give the same splits, graphs, features and tabular-model results in every execution and on any number of cores. Three things can still change the results: another BlockSci chain (the datasets were built from a chain of 676,079 blocks, heights 0 to 676,078; a parse up to another block height has other transactions, so the seed addresses found in the chain and the expansions differ); another number of threads per GNN run (with `GNN_PARALLEL_RUNS = "auto"` the threads of a configuration depend on the memory available when training starts, so the GNN metrics can differ slightly between machines or executions; it is recorded in every `result.json`, and setting `GNN_PARALLEL_RUNS` to a number in `config.py` fixes it on a given machine); and other library versions or hardware, which can change floating-point results (`requirements.txt` pins the versions).
- **Data:** all data-generation steps use `DATA_SEED`. The seed candidates are sorted before every random step, every multi-instance split is expanded from its own fixed random state, and every ego-centric egonet from its own random state, so the results do not depend on the number of parallel processes or their order.
- **Training:** each GNN run fixes the Python, NumPy and PyTorch seeds and uses deterministic algorithms with a fixed number of threads (recorded in `result.json`). Every random component of the tabular models is seeded, including the mutual-information feature selection and the cross-validation folds, and each run is single-threaded, so the results do not depend on the number of processes. The hyperparameter search uses only the training split and `DATA_SEED`.
- **Seeds:** the first training seeds are those of `config.TRAINING_SEEDS`, and more seeds are drawn deterministically from `DATA_SEED`, so an experiment with more seeds always contains the runs of an experiment with fewer.
- **Paired comparisons:** all configurations of both methodologies use the same training seeds, test addresses and 0.5 decision threshold, so their results are compared seed by seed with `wilcoxon_tests.py`. Only the seeds present in both configurations of a comparison are used.

## Future work

A possible improvement of the multi-instance training, not implemented:

- **GPU training and mini-batching.** Every epoch is a full-graph pass on CPU (one Adam step per epoch). Training on a GPU with enough memory for the full graph, or with neighbor-sampled mini-batches, would reduce the training time considerably, at the cost of exact determinism (GPU) or of a different training procedure (mini-batching).
