# TRACE: Transparent Ransomware Attribution via Cryptocurrency Examination

This is the repository for executing the experiments of [*TRACE: Transparent Ransomware Attribution via Cryptocurrency Examination*](https://www.discrypt.cat/). The code is divided in two different mothodologies: the *Inductive Multi-Instance Address Classification* and the *Inductive Ego-Centric Address Classification*, which will have different data types and steps to be implemented.

## Dependencies and environments

In order to execute al parts of the experiments, two different Python environments must be set up so that Blcksci library and its data structure can be executed with no iterference with all PyTorch and other new libraries.

- Expansion environment compatible with Blocksci library and all its dependencies [(more info)](https://citp.github.io/BlockSci/setup.html).
- Model training environment compatible with PyTorch and PyTorch Geometric library and all its dependencies [(more info)](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

## Methodolgy I: Inductive Ego-Centric Address Classification

## Methodolgy II: Inductive Multi-Instance Address Classification

All the necessary files to construct the graph, extract node and edge features, and train the address-classification model are located in [`Inductive Multi-Instance Address Classification`](Inductive%20Multi-Instance%20Address%20Classification/). The preprocessing and training workflow is divided into three main stages:

### 1. Graph Expansion and Feature Extraction

The script [`expansion.py`](Inductive%20Multi-Instance%20Address%20Classification/expansion.py) implements the ego-centric expansion procedure starting from the seed address set.  
For each of the **train**, **validation**, and **test** splits, it generates the following feature files:

- **`addr_feats.csv`** — aggregated and descriptive features for address nodes  
- **`tx_feats.csv`** — descriptive features for transaction nodes  
- **`input_feats.csv`** — feature set for all input edges  
- **`output_feats.csv`** — feature set for all output edges  

These files include both raw blockchain attributes and structural/aggregated metrics derived during the expansion process.

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

---

### Parameter Configuration

Before running the pipeline, some initial configuration is required.  
Users must specify:

- the expansion algorithm’s parameters  
- the seed address set for each data split  

These choices directly influence graph construction, model behavior, and the reproducibility of the experiments.

---

