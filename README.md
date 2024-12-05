# ransomware-scam-detection-deic

# Estructura del dataset

## Fitxers Generats

### 1. `addr_feats.csv`
Conté informació sobre les adreces analitzades.

- **Columnes:**
  - `addr`: Cadena que representa l'adreça al blockchain (e.g., adreça Bitcoin).
  - `full_type`: Tipus complet de l'adreça per exemple: P2PKH, P2SH.
  - `class`: Etiqueta binària:
    - `1`: Adreça clasificada com a maliciosa.
    - `0`: Adreça classificada com lícita.

---

### 2. `tx_feats.csv`
Inclou informació sobre les transaccions relacionades amb les adreces.

- **Columnes:**
  - `hash`: Hash únic de la transacció.
  - `block_height`: Número del bloc on es troba la transacció.
  - `fee`: Comissió pagada per la transacció (en unitats de la criptomoneda).
  - `is_coinbase`: Indica si és una transacció coinbase (`1`) o no (`0`).
  - `locktime`: Valor de "locktime" de la transacció.
  - `total_size`: Mida total de la transacció (en bytes).
  - `version`: Versió del format de la transacció.

---

### 3. `input_feats.csv`
Proporciona informació detallada sobre els inputs de les transaccions.

- **Columnes:**
  - `addr`: Adreça associada amb l'input (o `None` si no n'hi ha).
  - `tx`: Hash de la transacció d'origen de l'input.
  - `age`: Edat del coin abans de ser gastat (en blocs).
  - `block`: Número del bloc on es va incloure l'input original.
  - `index`: Índex dins del conjunt d'inputs de la transacció.
  - `sequence_num`: Número de seqüència de l'input.
  - `value`: Valor en unitats de la criptomoneda.

---

### 4. `output_feats.csv`
Descriu els outputs de les transaccions, incloent informació sobre la distribució dels fons.

- **Columnes:**
  - `tx`: Hash de la transacció associada amb aquest output.
  - `addr`: Adreça de destinació de l'output (o `None` si no n'hi ha).
  - `block`: Número del bloc on es troba l'output.
  - `index`: Índex dins del conjunt d'outputs de la transacció.
  - `is_spent`: Indica si l'output ha estat gastat (`1`) o no (`0`).
  - `value`: Valor en unitats de la criptomoneda.

---

## Relació entre els Fitxers
Els fitxers CSV estan interconnectats per estructurar les dades del blockchain:
- Les **adreces** (`addr_feats.csv`) estan associades a **transaccions** (`tx_feats.csv`).
- Les **transaccions** tenen **inputs** (`input_feats.csv`) i **outputs** (`output_feats.csv`), connectant adreces en un gràfic direccional.

---

# Algorismes d'expansió

# Estadístiques del graf resultant
