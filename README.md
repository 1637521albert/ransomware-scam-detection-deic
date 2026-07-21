# TRACE: Transparent Ransomware Attribution via Cryptocurrency Examination

This is the repository for executing the experiments of [*TRACE: Transparent Ransomware Attribution via Cryptocurrency Examination*](https://www.discrypt.cat/). The code is divided in two different mothodologies: the *Inductive Multi-Instance Address Classification* and the *Inductive Ego-Centric Address Classification*, which will have different data types and steps to be implemented.

## Dependencies and environments

In order to execute al parts of the experiments, two different Python environments must be set up so that Blcksci library and its data structure can be executed with no iterference with all PyTorch and other new libraries.

- Expansion environment compatible with Blocksci library and all its dependencies [(more info)](https://citp.github.io/BlockSci/setup.html).
- Model training environment compatible with PyTorch and PyTorch Geometric library and all its dependencies [(more info)](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

## Parameter Configuration

Before running the pipeline, some initial configuration is required.  
Users must specify:

- the expansion algorithm’s parameters  
- the seed address set for each data split  

These choices directly influence graph construction, model behavior, and the reproducibility of the experiments.

## Methodolgy I: Inductive Multi-Instance Address Classification

All the necessary files to construct the graph, extract node and edge features, and train the address-classification model are located in [`Inductive Multi-Instance Address Classification`](Inductive%20Multi-Instance%20Address%20Classification/). The preprocessing and training workflow is divided into three main stages:

### 1. Graph Expansion and Feature Extraction

The script [`expansion.py`](Inductive%20Multi-Instance%20Address%20Classification/expansion.py) implements the expansion procedure starting from the seed address set.  
For each of the **train**, **validation**, and **test** splits, it generates the following feature files:

- **`addr_feats.csv`** — aggregated and descriptive features for address nodes  
- **`tx_feats.csv`** — descriptive features for transaction nodes  
- **`input_feats.csv`** — feature set for all input edges  
- **`output_feats.csv`** — feature set for all output edges  

These files include both raw blockchain attributes and structural/aggregated metrics derived during the expansion process.

Before running the scripts, set the required external paths:

- `RSD_BLOCKSCI_CONFIG` for the BlockSci `config.blocksci` file
- `RSD_BITCOINHEIST_CSV` for the BitcoinHeist CSV

---

### 2. Graph Serialization

The script [`serialize_graph.py`](Inductive%20Multi-Instance%20Address%20Classification/serialize_graph.py) converts the tabular CSV files into PyTorch Geometric **HeteroData** graph objects.  
In addition to serialized `.pth` graphs for each split, this step also creates:

- **address-to-index** mappings  
- **transaction-to-index** mappings  

These mappings compress address strings and transaction hashes into integer identifiers, enabling efficient storage and training.

---

### 3. Model Training and Evaluation

The script [`model.py`](Inductive%20Multi-Instance%20Address%20Classification/model.py) defines and trains the Graph Neural Network model.  
It provides user-selectable architectures and performs the following tasks:

- Loads the serialized heterogeneous graphs  
- Lormalizes both address and transaction features
- Trains the model on the designated training split  
- Tracks validation performance to select the best checkpoint  
- Evaluates the final model on the test split  

Debugging logs can be enabled via the corresponding configuration parameter.  
For long-term, interactive experiment tracking, Weights & Biases (WandB) logging can also be activated.

## Methodology II: Inductive Ego-Centric Address Classification

All files required to execute the ego-centric topological feature extraction and classical machine learning pipeline are structured to handle tabular structural descriptors. The preprocessing, extraction, evaluation, and interpretability workflow is divided into three main stages:

### 1. Graph Expansion
The script [`expansion.py`](expansion.py) executes the selected expansion strategy starting from the selected seed addresses. For each data split (**train**, **validation**, and **test**), it generates serialized subgraph files in line-delimited JSON format (`.jsonl`), capturing the local ego-networks surrounding target addresses according to configurable parameters such as hop depth and breadth limits.

---

### 2. Topological Feature Extraction
The script [`extract_topological_features.py`](extract_topological_features.py) processes the generated `.jsonl` files using parallel multiprocessing workers. It parses each ego-network into a NetworkX directed graph to compute structural graph metrics and behavioral transaction motifs:
- **Structural Centralities & Metrics**: PageRank, Betweenness Centrality, Eccentricity, Graph Density, In-Degree, and Out-Degree.
- **Aggregation Patterns ($\mathcal{A}$)**: Fan-In dynamics distinguishing binary aggregation ($\mathcal{A}_2$) from multi-aggregation/consolidation ($\mathcal{A}_{3+}$).
- **Branching Patterns ($\mathcal{B}$)**: Fan-Out dynamics distinguishing binary branching ($\mathcal{B}_2$) from multi-branching ($\mathcal{B}_{3+}$).
- **Recursive Patterns ($\mathcal{P}$)**: Multi-hop peeling chains ($\mathcal{P}_2, \mathcal{P}_3$) capturing layering and obfuscation behavior.

The extracted descriptors are consolidated into a single tabular dataset (e.g., `full_dataset_6.csv`) containing all computed topological features alongside ground-truth labels and split assignments.

---

### 3. Model Training, Evaluation, and Interpretability
The script [`models.py`](models.py) implements the complete machine learning experimental pipeline for tabular data. It performs the following tasks:
- **Data Pipeline**: Loads the consolidated dataset, maps predefined data splits, applies mutual information feature selection (e.g., top 90th percentile), logarithmic compression for heavy-tailed distributions, and feature scaling.
- **Supervised Training & Evaluation**: Trains and evaluates multiple supervised machine learning models (Random Forest, XGBoost, Multi-Layer Perceptron [MLP], Support Vector Machine [SVM], Logistic Regression with Lasso/ElasticNet, and Naive Bayes) alongside an unsupervised Isolation Forest baseline.
- **Multi-Seed Robustness**: Executes training across multiple random seeds and reports performance metrics (Accuracy, Precision, Recall, F1-score, and ROC-AUC) as mean $\pm$ standard deviation.
- **Statistical Significance**: Performs paired t-tests comparing the F1-score performance distributions of the best-performing model against competing baselines.
- **Feature Ablation Study**: Conducts a cumulative ablation analysis based on permutation importance ranking to measure performance degradation and evaluate feature efficiency.
- **Deep SHAP Interpretability**: Computes SHAP values using optimized explainers (Tree, Linear, or Kernel) to generate academic summary beeswarm (dot) plots and global feature importance bar charts.

---
