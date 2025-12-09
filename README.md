# TRACE: Transparent Ransomware Attribution via Cryptocurrency Examination

This is the repository for executing the experiments of [*TRACE: Transparent Ransomware Attribution via Cryptocurrency Examination*](https://www.discrypt.cat/). The code is divided in two different mothodologies: the *Inductive Multi-Instance Address Classification* and the *Inductive Ego-Centric Address Classification*, which will have different data types and steps to be implemented.

## Dependencies and environments

In order to execute al parts of the experiments, two different Python environments must be set up so that Blcksci library and its data structure can be executed with no iterference with all PyTorch and other new libraries.

- Expansion environment (Blocksci): 

## Expansion Algorithm

The results of expanding the address `1C9KA8hWUuASCdDq1EPB7PmcnFNqhb1so2` over 2 hops are shown also. The green nodes represent addresses that have been explored during the graph expansion, the largest red node represents the illicit node, and the gray nodes are transactions between addresses. The direction of the edges indicates the flow of funds (`(addr -> tx)` is an input, `(tx -> addr)` is an output), and the layout of the graph shows past transactions at the top moving to future transactions at the bottom. 

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

## Graph Type

The directed graph generated from the `.pth` file has the following characteristics:

- **Graph Type:** Heterogeneous, bipartite directed graph with parallel edges but no loops.
- **Heterogeneous Graph Model:**
  - **Nodes:** Are the two main elements in the network, represented in two types of nodes:
    - **Transactions:** Represent transactions.
    - **Addresses:** Represent blockchain addresses.
  - **Edges:** Represent the flow of coins:
    - **Inputs:** Link address nodes to transaction nodes by representing the inputs of each transaction.
    - **Outputs:** Link transaction nodes to address nodes by representing the outputs of each transaction.

### Edge Properties
- **Direction:** Edges are directed to capture the flow of bitcoins:
  - From an address/es to a transaction/s (input).
  - From a transaction/s to an address/es (output).
- **Weight:** Edges can be weighted, representing the value transferred in bitcoins between nodes.

### Advantages of the Heterogeneous Model
1. **Separation of Node Types and Relationships:**
   - Differentiating transaction nodes from address nodes helps clarify the role of each type within Bitcoin transactions.
   - Differentiated edges between inputs and outputs accurately capture the flow of value.

2. **Coin Flow Modeling:**
   - Directed edges allow tracking the movement of Bitcoins from a source address to one or more destination addresses.

3. **Temporal Context Exploration:**
   - Temporal attributes like `block_height` and `age` provide a temporal perspective of operations, revealing patterns more comprehensively.

## Generated Files

The generated file is in a .pth format and represent an instance of a HeteroData object in which every type of node and edge, and all its features and labels, if needed, are encoded. The way to import this dataset consists in a simple command in torch: `torch.load('graph.pth')`. In this dataset, we can find all the information needed to apply the precomputation and the model to classify addresses of the network, whether are licits or illicits. Since all the information needed to use the dataset is clearly explained in the PyTorch geometric documentation, consult [this link](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.HeteroData.html) in case of any implementation doubt. The structure is the following:

### 1. Addresses
Contains information about the analyzed addresses.

- **Columns:**
  - `addr_str` (`string`): String representing the blockchain address.
  - `full_type` (`string`): Full type of the address, e.g., P2PKH, P2SH.
  - `class` (`int`): Binary label:
    - `1`: Address classified as malicious.
    - `0`: Address classified as legitimate.

### 2. Transactions
Includes information about transactions related to the addresses.

- **Columns:**
  - `hash` (`string`): Unique hash of the transaction.
  - `block_height` (`int`): Block number where the transaction is included.
  - `fee` (`float`): Fee paid for the transaction (in cryptocurrency units).
  - `is_coinbase` (`int`): Indicates if it is a coinbase transaction (`1`) or not (`0`).
  - `locktime` (`int`): Locktime value of the transaction.
  - `total_size` (`int`): Total size of the transaction (in bytes).
  - `version` (`int`): Transaction format version.

### 3. Inputs
Provides detailed information about transaction inputs.

- **Columns:**
  - `addr_str` (`string` or `None`): Address associated with the input (or `None` if absent).
  - `tx_hash` (`string`): Hash of the originating transaction for the input.
  - `age` (`int`): Age of the coin before being spent (in blocks).
  - `block` (`int`): Block number where the original input was included.
  - `index` (`int`): Index within the set of transaction inputs.
  - `sequence_num` (`int`): Input sequence number.
  - `value` (`float`): Value in Satoshis (sats).

### 4. Outputs
Describes transaction outputs, including information about fund distribution.

- **Columns:**
  - `tx_hash` (`string`): Hash of the transaction associated with this output.
  - `addr_str` (`string` or `None`): Destination address of the output (or `None` if absent).
  - `block` (`int`): Block number where the output is included.
  - `index` (`int`): Index within the set of transaction outputs.
  - `is_spent` (`int`): Indicates if the output has been spent (`1`) or not (`0`).
  - `value` (`float`): Value in Satoshis (sats).

