# TRACE: Transparent Ransomware Attribution via Cryptocurrency Examination

This is the repository for executing the experiments of [*TRACE: Transparent Ransomware Attribution via Cryptocurrency Examination*](https://www.discrypt.cat/). The code is divided in two different methodologies: the *Inductive Multi-Instance Address Classification* and the *Inductive Ego-Centric Address Classification*, which will have different data types and steps to be implemented.

## Dependencies and environments

In order to execute all parts of the experiments, two different Python environments must be set up so that the BlockSci library and its data structure can be executed with no interference with PyTorch and other newer libraries.

- Expansion environment compatible with BlockSci library and all its dependencies [(more info)](https://citp.github.io/BlockSci/setup.html).
- Model training environment compatible with PyTorch and PyTorch Geometric library and all its dependencies [(more info)](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

## Environment Variables

External paths and runtime options are read from environment variables:

| Variable | Used by | Description |
|---|---|---|
| `RSD_BLOCKSCI_CONFIG` | Both `expansion.py` scripts, `model.py` | Path to the BlockSci `config.blocksci` file |
| `RSD_BITCOINHEIST_CSV` | Both `expansion.py` scripts, `model.py` | Path to the BitcoinHeist CSV (seed addresses and labels) |
| `RSD_HETEROGENEOUS_EGONETS` | `extract_topological_features.py` | Directory containing the ego-network expansion runs |
| `RSD_USE_WANDB` | `model.py` (optional) | Enable Weights & Biases logging (`1`/`true`/`yes`); prompted interactively if unset |
| `RSD_DEBUG_MODE` | `model.py` (optional) | Enable debugging logs (`1`/`true`/`yes`); prompted interactively if unset |

## Parameter Configuration

Before running the pipeline, some initial configuration is required. The scripts prompt the user interactively for:

- the expansion algorithm's parameters: direction (forward-backward or all-over), approach (transaction-based or address-based), transaction address scope (whole or dedicated), side-address treatment (none, same or opposite), limit mode (random node, random hop or none) with its limit value, and the number of hops
- the number of seed addresses for each data split (**train**, **validation** and **test**)

These choices directly influence graph construction, model behavior, and the reproducibility of the experiments. Each run is stored in `outputs/<run name>/` inside the methodology folder, where the run name encodes the chosen parameters, together with a `graph_config.json` file recording them. The serialization and training scripts ask for the same parameters in order to locate the corresponding run. Illustrations of each expansion strategy are available in [`expansion.md`](expansion.md).

## Methodology I: Inductive Multi-Instance Address Classification

All the necessary files to construct the graph, extract node and edge features, and train the address-classification model are located in [`Inductive Multi-Instance Address Classification`](Inductive%20Multi-Instance%20Address%20Classification/). The preprocessing and training workflow is divided into three main stages:

### 1. Graph Expansion and Feature Extraction

The script [`expansion.py`](Inductive%20Multi-Instance%20Address%20Classification/expansion.py) implements the expansion procedure starting from the seed address set.

Seed addresses are taken from the BitcoinHeist CSV: addresses with a ransomware label are used as illicit seeds and addresses labelled `white` as licit seeds (addresses appearing with both labels are treated as illicit). Only addresses found in the BlockSci chain are kept, and the licit set is capped to the size of the illicit one. Each split then receives the requested number of illicit seeds plus the same number of licit seeds, so a split of `N` samples contains `2N` seed addresses.

For each of the **train**, **validation**, and **test** splits, it generates the following feature files:

- **`addr_feats.csv`** — aggregated and descriptive features for address nodes
- **`tx_feats.csv`** — descriptive features for transaction nodes
- **`input_feats.csv`** — feature set for all input edges
- **`output_feats.csv`** — feature set for all output edges
- **`spent_pairs.csv`** — pairs of spent/spending transactions, used to link transactions in the graph

These files include both raw blockchain attributes and structural/aggregated metrics derived during the expansion process.

---

### 2. Graph Serialization

The script [`serialize_graph.py`](Inductive%20Multi-Instance%20Address%20Classification/serialize_graph.py) converts the tabular CSV files into PyTorch Geometric **HeteroData** graph objects.
In addition to a serialized `graph.pth` for each split, this step also creates:

- **address-to-index** mappings (`addr_mapping.json`)
- **transaction-to-index** mappings (`tx_mapping.json`)

These mappings compress address strings and transaction hashes into integer identifiers, enabling efficient storage and training.

---

### 3. Model Training and Evaluation

The script [`model.py`](Inductive%20Multi-Instance%20Address%20Classification/model.py) defines and trains the Graph Neural Network model.
It provides user-selectable architectures (**GAT**, **HGT** or **HAN**) with a configurable number of layers and performs the following tasks:

- Loads the serialized heterogeneous graphs
- Normalizes both address and transaction features
- Trains the model on the designated training split
- Tracks validation performance to select the best checkpoint (saved as `best_model.pth`)
- Evaluates the final model on the test split

Debugging logs can be enabled via `RSD_DEBUG_MODE`.
For long-term, interactive experiment tracking, Weights & Biases (WandB) logging can also be activated via `RSD_USE_WANDB`.

## Methodology II: Inductive Ego-Centric Address Classification

All files required to execute the ego-centric topological feature extraction and classical machine learning pipeline are located in [`Inductive Ego-Centric Address Classification`](Inductive%20Ego-Centric%20Address%20Classification/) and are structured to handle tabular structural descriptors. The preprocessing, extraction, evaluation, and interpretability workflow is divided into three main stages:

### 1. Graph Expansion
The script [`expansion.py`](Inductive%20Ego-Centric%20Address%20Classification/expansion.py) executes the selected expansion strategy starting from the selected seed addresses. For each data split (**train**, **validation**, and **test**), it generates serialized subgraph files in line-delimited JSON format (`<split>_p<illicit %>.jsonl`, e.g. `train_p50.jsonl`), capturing the local ego-networks surrounding target addresses according to configurable parameters such as hop depth and breadth limits.

---

### 2. Topological Feature Extraction
The script [`extract_topological_features.py`](Inductive%20Ego-Centric%20Address%20Classification/extract_topological_features.py) asks the user to select one of the runs located in `RSD_HETEROGENEOUS_EGONETS` and processes its `.jsonl` files using parallel multiprocessing workers. It parses each ego-network into a NetworkX directed graph to compute structural graph metrics and behavioral transaction motifs:
- **Structural Centralities & Metrics**: PageRank, Betweenness Centrality, Clustering Coefficient, Eccentricity, Graph Density, In-Degree, and Out-Degree.
- **Aggregation Patterns**: Fan-In dynamics distinguishing binary aggregation from multi-aggregation/consolidation.
- **Branching Patterns**: Fan-Out dynamics distinguishing binary branching from multi-branching.
- **Recursive Patterns**: Multi-hop peeling chains capturing layering and obfuscation behavior.

The extracted descriptors are consolidated into a single tabular dataset `full_dataset_<radius>.csv` (`full_dataset_6.csv` with the default radius of 6), written to the selected run directory, containing all computed topological features alongside ground-truth labels and split assignments.

---

### 3. Model Training, Evaluation, and Interpretability
The script [`models.py`](Inductive%20Ego-Centric%20Address%20Classification/models.py) implements the complete machine learning experimental pipeline for tabular data. It expects `full_dataset_6.csv` to be placed in the same folder as the script. It performs the following tasks:
- **Data Pipeline**: Loads the consolidated dataset, maps predefined data splits, applies mutual information feature selection (top 90th percentile), logarithmic compression for heavy-tailed distributions, and feature scaling.
- **Supervised Training & Evaluation**: Trains and evaluates multiple supervised machine learning models (Random Forest, XGBoost, Multi-Layer Perceptron [MLP], Support Vector Machine [SVM], Logistic Regression with Lasso/ElasticNet, and Naive Bayes) alongside an unsupervised Isolation Forest baseline.
- **Multi-Seed Robustness**: Executes training across multiple random seeds and reports performance metrics (Accuracy, Precision, Recall, F1-score, and ROC-AUC) as mean $\pm$ standard deviation, together with paired t-tests comparing the MLP against the other models.
- **Feature Ablation Study**: Conducts a cumulative ablation analysis based on permutation importance ranking to measure performance degradation and evaluate feature efficiency.
- **Deep SHAP Interpretability**: Computes SHAP values using optimized explainers (Tree, Linear, or Kernel) to generate academic summary beeswarm (dot) plots and global feature importance bar charts.

Ablation and SHAP figures are saved as PDF files in the `figures/` directory at the repository root.

---
