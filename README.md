# Estructura del Dataset

## Fitxers Generats

### 1. `addr_feats.csv`
Conté informació sobre les adreces analitzades.

- **Columnes:**
  - `addr` (`string`): Cadena que representa l'adreça al blockchain.
  - `full_type` (`string`): Tipus complet de l'adreça, per exemple: P2PKH, P2SH.
  - `class` (`int`): Etiqueta binària:
    - `1`: Adreça classificada com a maliciosa.
    - `0`: Adreça classificada com lícita.


### 2. `tx_feats.csv`
Inclou informació sobre les transaccions relacionades amb les adreces.

- **Columnes:**
  - `hash` (`string`): Hash únic de la transacció.
  - `block_height` (`int`): Número del bloc on es troba la transacció.
  - `fee` (`float`): Comissió pagada per la transacció (en unitats de la criptomoneda).
  - `is_coinbase` (`int`): Indica si és una transacció coinbase (`1`) o no (`0`).
  - `locktime` (`int`): Valor de "locktime" de la transacció.
  - `total_size` (`int`): Mida total de la transacció (en bytes).
  - `version` (`int`): Versió del format de la transacció.


### 3. `input_feats.csv`
Proporciona informació detallada sobre els inputs de les transaccions.

- **Columnes:**
  - `addr` (`string` o `None`): Adreça associada amb l'input (o `None` si no n'hi ha).
  - `tx` (`string`): Hash de la transacció d'origen de l'input.
  - `age` (`int`): Edat del coin abans de ser gastat (en blocs).
  - `block` (`int`): Número del bloc on es va incloure l'input original.
  - `index` (`int`): Índex dins del conjunt d'inputs de la transacció.
  - `sequence_num` (`int`): Número de seqüència de l'input.
  - `value` (`float`): Valor en Satoshis (sats).


### 4. `output_feats.csv`
Descriu els outputs de les transaccions, incloent informació sobre la distribució dels fons.

- **Columnes:**
  - `tx` (`string`): Hash de la transacció associada amb aquest output.
  - `addr` (`string` o `None`): Adreça de destinació de l'output (o `None` si no n'hi ha).
  - `block` (`int`): Número del bloc on es troba l'output.
  - `index` (`int`): Índex dins del conjunt d'outputs de la transacció.
  - `is_spent` (`int`): Indica si l'output ha estat gastat (`1`) o no (`0`).
  - `value` (`float`): Valor en en Satoshis (sats).


## Relació entre els Fitxers
Els fitxers CSV estan interconnectats per estructurar les dades del blockchain:
- Les **adreces** (`addr_feats.csv`) estan associades a **transaccions** (`tx_feats.csv`).
- Les **transaccions** tenen **inputs** (`input_feats.csv`) i **outputs** (`output_feats.csv`), connectant adreces en un graf dirigit.


## Tipus de Graf

El digraf generat a partir dels fitxers `.csv` té les següents característiques:

- **Tipus de graf:** Dígraf heterogeni, bipartit, amb arestes paral·leles però sense llaços.
- **Model de graf heterogeni:** 
  - Els **nodes** són de dos tipus: 
    - **Nodes de transacció**: Representen una transacció.
    - **Nodes d’adreça**: Representen adreces al blockchain.
  - Les **arestes** representen el flux de monedes:
    - **Inputs**: Connecten nodes d’adreça amb nodes de transacció.
    - **Outputs**: Connecten nodes de transacció amb nodes d’adreça.

### Propietats de les Arestes
- **Direcció:** Les arestes són dirigides per capturar el flux de bitcoins:
  - Des d'una adreça (input) cap a una transacció.
  - Des d'una transacció cap a una o més adreces (output).
- **Pes:** Les arestes poden estar ponderades, representant el valor transferit en bitcoins entre nodes.

### Avantatges del Model Heterogeni
1. **Separació de tipus de nodes i relacions:**
   - La diferenciació entre nodes de transacció i nodes d’adreça permet entendre millor el paper de cada tipus dins de les transaccions de Bitcoin.
   - Les arestes diferenciades entre inputs i outputs capturen el flux de valor amb precisió.

2. **Modelització de flux de monedes:**
   - La direccionalitat de les arestes permet rastrejar el moviment de bitcoins des d'una adreça d'origen fins a una o més adreces de destinació.

3. **Capacitat d'explorar context temporal:**
   - Atributs temporals com `block_height` i `age` proporcionen una visió temporal de les operacions, revelant patrons de manera més completa.

---

# Expansion Algorithm Taxonomy

In this part we will provide an overview of different algorithms used to expand address or transaction networks in blockchain analysis. The explanations focus on how each algorithm operates, explaining its specific approach and the relationships it explores.

## Transaction-Based Expansion
This approach focuses on exploring the network of transactions by tracking the flow of bitcoins through inputs and outputs. It identifies related transactions, emphasizing the movement of funds rather than the specific addresses involved.

### Forward-Backward Expansion
The forward-backward algorithm separates the exploration into two distinct phases:

- **Forward Expansion**: Captures the flow of bitcoins by identifying future transactions that spend the outputs of the current transactions, following the "spending path."
- **Backward Expansion**: Searches for the origins of bitcoins by exploring transactions that created the inputs of the current transactions, reconstructing the "funding path."

This approach provides flexibility, allowing to focus on specific aspects such as origins, destinations, or complete spending paths.

### All-Over Expansion
Starting with an initial set of addresses and their transactions, this approach explores connections between nodes through both inputs and outputs. Each hop identifies new transactions linked to the current set, both by their inputs (funding transactions) and outputs (spending transactions). 

### Comparison: Forward-Backward vs. All-Over
In contrast to the all-over method, the forward-backward approach separates exploration into forward and backward phases. This enables especifying the number of hops in each direction, facilitating targeted analyses. The all-over algorithm, by contrast, combines forward and backward exploration within each hop, simultaneously exploring funding and spending relationships of every address in the current set.

## Address-Based Expansion
This method centers on the network of addresses, rather than transactions. It explores relationships between addresses by tracing the transactions they participate in. Unlike transaction-based expansion, this approach emphasizes the connections and roles of individual addresses in the network.

### Dedicated Inputs/Outputs
A forward-backward expansion algorithm centered on addresses, this approach:

- **Forward Phase**: Explores outgoing relationships, identifying transactions funded by the current addresses and new addresses from transaction outputs.
- **Backward Phase**: Explores incoming relationships, identifying transactions that fund the current addresses and tracing inputs to their original addresses.

This method avoids rapid graph expansion by ignoring "side" addresses (inputs in the forward phase and outputs in the backward phase that do not correspond to the expanding addresses).

### Whole Transaction Implied Addresses
This approach also performs iterative forward and backward phases but includes all inputs and outputs associated with each transaction explored. Variants of this method handle "side" addresses differently:

- **None**: Side addresses remain as leaf nodes and are not included in further expansions.
- **Same**: Side addresses are expanded in the direction corresponding to their role (e.g., future for forward, past for backward).
- **Opposite**: Side addresses are expanded in the reverse direction of their role (e.g., past for forward, future for backward).

### All-Over 
In each hop, it examines both inputs and outputs of transactions involving the current set of addresses, identifying new related addresses and adding them to the expanding graph.  This results in a broader and more interconnected graph, as every discovered address has the potential to contribute to further all-over expansions.

---
# Estadístiques del Graf Resultant
