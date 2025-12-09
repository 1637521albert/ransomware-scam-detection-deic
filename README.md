# TRACE: Transparent Ransomware Attribution via Cryptocurrency Examination

This is the repository for executing the experiments of [*TRACE: Transparent Ransomware Attribution via Cryptocurrency Examination*](https://www.discrypt.cat/). The code is divided in two different mothodologies: the *Inductive Multi-Instance Address Classification* and the *Inductive Ego-Centric Address Classification*, which will have different data types and steps to be implemented.

## Dependencies and environments

In order to execute al parts of the experiments, two different Python environments must be set up so that Blcksci library and its data structure can be executed with no iterference with all PyTorch and other new libraries.

- Expansion environment (Blocksci): 

## Expansion Algorithm

The results of expanding the address `1C9KA8hWUuASCdDq1EPB7PmcnFNqhb1so2` over 2 hops are shown. The green nodes represent addresses that have been explored during the graph expansion, the largest red node represents the illicit seed node, and the gray nodes are transactions between addresses. The direction of the edges indicates the flow of funds (`(addr -> tx)` is an input, `(tx -> addr)` is an output), and the layout of the graph shows past transactions at the top moving to future transactions at the bottom. 

### Transaction-Based Forward-Backward Whole-Scope Expansion
![Transaction-Based Forward-Backward Expansion](VISUALIZATIONS/tx-based%20fw-bw.png)

### Transaction-Based Forward-Backward Dedicated-Scope Expansion



#### Transaction-based All-Over Expansion
![Transaction-Based All-Over Expansion](VISUALIZATIONS/all%20over%20tx.png)

### Address-Based Forward-Backward Whole-Scope None-Side Addresses Treatment Expansion
![Address-Based Whole Tx None](VISUALIZATIONS/addr-based%20whole%20tx%20none.png)

### Address-Based Forward-Backward Whole-Scope Same-Side Addresses Treatment Expansion
![Address-Based Whole Tx Same](VISUALIZATIONS/addr-based%20whole%20tx%20same.png)

### Address-Based Forward-Backward Whole-Scope Opposite-Side Addresses Treatment Expansion



### Address-Based Forward-Backward Dedicated-Scope Expansion
![Address-Based Dedicated In:Out Expansion](VISUALIZATIONS/addr-based%20dedicated%20in:out.png)

#### Address-Based All-Over Expansion
![Address-Based All-Over Expansion](VISUALIZATIONS/addr-based%20all%20over.png)

## Methodology I: Inductive Ego-Centric Address Classification

All the necessary files to construct the graph, extract node and edge features, and train the address-classification model are located in the folder:

[`Inductive Multi-Instance Address Classification`](Inductive%20Multi-Instance%20Address%20Classification/)

The preprocessing and training workflow is divided into three main stages:

---

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

- loads the serialized heterogeneous graphs  
- trains the model on the designated training split  
- tracks validation performance to select the best checkpoint  
- evaluates the final model on the test split  

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

## Methodolgy II: Inductive Multi-Instance Address Classification

