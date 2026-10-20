# Expansion variants

An expansion grows a graph of addresses and transactions around the seed addresses, hop by hop. Every hop adds one layer of transactions and the addresses of their inputs and outputs. Methodology I expands the seeds of a split into one graph (`shared/parallel_expansion.py`), and Methodology II one egonet per seed (`shared/expansion.py`). Both implement the same eight variants, and give the same graph for the same seeds.

A variant is chosen in `config.py` (`MULTI_INSTANCE_EXPANSION`, `EGO_CENTRIC_EXPANSION`) by four fields, whose concatenation is its name:

| Field | Values | Meaning |
|---|---|---|
| `direction` | `"fw bw"`, `"all over"` | Forward (following the funds) and backward (towards their origin) separately, or both directions at once |
| `graph` | `" tx"`, `" addr"` | The frontier of every hop: transactions or addresses |
| `address` | `" whole"`, `" dedicated"`, `""` | A selected transaction brings all its inputs and outputs (whole), or only those linked to the frontier plus the other side (dedicated) |
| `side_direction` | `" none"`, `" same"`, `" opposite"`, `""` | Address-based whole variants: what happens to the *side addresses*, found on the same side of a transaction as the expanded address |

The number of hops is `forward_hops` and `backward_hops` in the forward–backward variants, and `hops` in the all-over variants.

**Width limit** (`limit_mode`, `limit` = *w*): without a limit (`""`) every candidate transaction is added. With `"random node"`, every node of the frontier samples *w* transactions uniformly among its own distinct candidates, independently of the other nodes (a transaction chosen by several nodes is added once). With `"random hop"`, *w* transactions are sampled among the distinct candidates of the whole hop. The random draws use `DATA_SEED`, so an expansion is reproducible.

**Hop records:** every transaction records the hop where it was added (`hop`, from 1) and, in the forward–backward variants, its direction (`hop_direction`: `forward`, `backward`, or `both` for a first-hop transaction found in both directions).

The figures show the expansion of the address `1C9KA8hWUuASCdDq1EPB7PmcnFNqhb1so2` over 2 hops. The green nodes are addresses explored during the expansion, the largest red node is the illicit seed, and the gray nodes are transactions. The direction of the edges is the flow of funds (`addr -> tx` is an input, `tx -> addr` an output), and the layout shows past transactions at the top and future transactions at the bottom.

## Transaction-based variants

The frontier of every hop is a set of transactions. The first hop takes the transactions of the seed addresses, and every next hop the transactions linked to the frontier.

### `fw bw tx whole` (forward–backward, whole scope)

The first hop takes the transactions where a seed is an input (forward) and those where it is an output (backward). Every next forward hop takes the transactions that spend the outputs of the forward frontier, and every next backward hop the transactions whose outputs the backward frontier spends. Every selected transaction is added whole, with all its inputs and outputs. This is the variant of Methodology I in the original configuration (*k* = 2, *w* = 5).

![Transaction-based forward-backward expansion](VISUALIZATIONS/tx-based%20fw-bw.png)

### `fw bw tx dedicated` (forward–backward, dedicated scope)

The same transactions as `fw bw tx whole`, but a forward transaction only brings the inputs that spend the frontier (the inputs of the seed in the first hop) and all its outputs, and a backward transaction only the outputs spent by the frontier and all its inputs.

### `all over tx` (all over)

The first hop takes every transaction of the seed addresses, and every next hop every transaction that spends from or is spent by the frontier, in both directions at once. Every transaction is added whole. This is the variant of Methodology II in the original configuration (*k* = 3, *w* = 3).

![Transaction-based all-over expansion](VISUALIZATIONS/all%20over%20tx.png)

## Address-based variants

The frontier of every hop is a set of addresses: the seeds, then the addresses found in the previous hop.

### `fw bw addr whole none`, `same` and `opposite` (forward–backward, whole scope)

A forward hop takes the transactions where a frontier address is an input, and a backward hop those where it is an output; every transaction is added whole. The addresses on the other side (outputs of a forward transaction, inputs of a backward one) form the next frontier in the same direction. The side addresses (the other inputs of a forward transaction, the other outputs of a backward one) are added to the graph, and the last field decides whether they are expanded:

- `none`: they are not expanded.
- `same`: they are expanded in the same direction as the hop that found them.
- `opposite`: they are expanded in the opposite direction (the side addresses of a forward hop join the next backward frontier, and vice versa).

![Address-based whole scope, side addresses not expanded](VISUALIZATIONS/addr-based%20whole%20tx%20none.png)

![Address-based whole scope, side addresses expanded in the same direction](VISUALIZATIONS/addr-based%20whole%20tx%20same.png)

### `fw bw addr dedicated` (forward–backward, dedicated scope)

A forward hop takes the transactions where a frontier address is an input, with only the inputs of the frontier addresses and all the outputs; a backward hop the transactions where it is an output, with only its outputs and all the inputs. The addresses on the other side form the next frontier.

![Address-based dedicated expansion](VISUALIZATIONS/addr-based%20dedicated%20in:out.png)

### `all over addr` (all over)

Every hop takes every transaction of the frontier addresses, in both directions at once, added whole, and every new address found becomes the next frontier.

![Address-based all-over expansion](VISUALIZATIONS/addr-based%20all%20over.png)
