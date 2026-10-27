# Parallel expansion of the address-transaction graph, shared by every script that
# expands many seed addresses into one graph (the multi-instance methodology and the
# expansion experiments).
#
# It runs every expansion variant of shared/expansion.py (the reference sequential
# algorithms, used by the ego-centric methodology to expand one egonet per process) in
# parallel, giving the same graph with any number of workers. Every hop has two parallel
# passes over forked workers (parallel_map keeps the order of the items), with the
# order-dependent steps in the main process:
#   1. Scan: the workers list the candidate transactions of every node of the frontier,
#      as transaction indices (compact arrays; their hashes are not computed).
#   2. Selection (main process): the candidates are de-duplicated and the width limit is
#      applied node by node, in the order of the sequential algorithm (random.sample
#      only depends on the number of candidates, so the same ones are selected).
#   3. Explore: the workers read the selected transactions: their hash and the inputs and
#      outputs with an address (all of them, or those of the frontier in the dedicated
#      variants), with their values.
#   4. Insertion (main process): the records are inserted in the graph in the order of
#      the sequential algorithm.
# The workers rebuild transactions from their index (TX_BY_INDEX, detected with
# detect_tx_by_index) and inherit the chain and the graph when they are forked.
# Transactions are identified by their index during the expansion and by their hash in
# the graph; both only differ for the two pairs of duplicated 2010 coinbase transactions
# (BIP30), which share a hash. The main process keeps plain values, with every address
# string and tx hash interned (stored once). expand() fills, in place:
#   addresses: address string -> (reference, label); the reference rebuilds the address:
#              ("seed", address string) or ("in"/"out", tx index, position) of the
#              input/output where it was found (resolve_address); found addresses get -1
#   txs:       tx hash -> (hop, direction, tx index) of the hop where the tx was added
#   inputs:    (spent tx hash, spent output index) -> (address string, tx hash, age,
#              sequence number, value)
#   outputs:   (tx hash, output index) -> (address string, is spent, value)
# Inputs and outputs without an address are not kept.
#
# Usage: configure(chain, workers), detect_tx_by_index(a tx of the chain), then
# expand(variant, addresses, txs, inputs, outputs, ...) for every graph. The random draws
# use the global random module, which the caller seeds.

import math
import sys
from array import array

from shared import expansion as shared_expansion
from shared.expansion import address_string, limit_expansion, spent_tx_hash, tx_hash
from shared.parallel import forked_pool, parallel_map

# Rebuilds a transaction from its index (set by detect_tx_by_index) and the indices of
# the txs in the graph, read by the scans (set before every scan)
TX_BY_INDEX = {"build": None, "name": None}
_SCAN_STATE = {"txs": set()}

# Chain, workers and verbosity of the expansions (configure), and an optional function
# called with (number of candidates, number kept) at every width-limit sampling
_CTX = {"chain": None, "workers": 1, "verbose": True, "on_sample": None}


def configure(chain, workers, verbose=True, on_sample=None):
    """Sets the chain and the number of worker processes of the next expansions. The
    workers are forked at every hop, so they inherit the chain and the graph."""
    _CTX.update(chain=chain, workers=max(1, int(workers)), verbose=verbose, on_sample=on_sample)
    shared_expansion.set_chain(chain)


def detect_tx_by_index(sample_tx):
    """The BlockSci call that builds a tx from its index, checked on a known tx."""
    import blocksci
    chain = _CTX["chain"]
    options = [
        ("blocksci.Tx(index, chain)", lambda i: blocksci.Tx(i, chain)),
        ("chain.tx_with_index(index)", lambda i: chain.tx_with_index(i)),
    ]
    for name, build in options:
        try:
            tx = build(sample_tx.index)
            if tx.index == sample_tx.index and str(tx.hash) == str(sample_tx.hash):
                TX_BY_INDEX.update(build=build, name=name)
                return name
        except Exception:
            continue
    raise RuntimeError("BlockSci does not build a transaction from its index with "
                       + " or ".join(name for name, _ in options))

def tx_of(i):
    return TX_BY_INDEX["build"](i)

def resolve_address(ref):
    # Rebuilds a BlockSci address from its reference
    if ref[0] == "seed":
        return _CTX["chain"].address_from_string(ref[1])
    tx = tx_of(ref[1])
    return tx.inputs[ref[2]].address if ref[0] == "in" else tx.outputs[ref[2]].address

def _spending_index_attr(out):
    return out.spending_tx_index if out.is_spent else None

def _spending_index_tx(out):
    tx = out.spending_tx
    return None if tx is None else tx.index

_spending_index = _spending_index_attr

def spending_tx_index(out):
    # Index of the tx spending an output (None if unspent), without building the tx
    # when BlockSci allows it
    global _spending_index
    try:
        return _spending_index(out)
    except AttributeError:
        _spending_index = _spending_index_tx
        return _spending_index(out)

# --------------------------------------------------------------------------
# Workers: scans
# --------------------------------------------------------------------------

def _indices(txes):
    return array("q", (tx.index for tx in txes))

def _new(indices):
    # Distinct indices not in the graph, in order
    txs, seen, new = _SCAN_STATE["txs"], set(), array("q")
    for t in indices:
        if t is not None and t not in txs and t not in seen:
            seen.add(t)
            new.append(t)
    return new

def _groups(pairs, skip_graph=True):
    # (tx index, selector) pairs -> [(tx index, [selectors])], by first appearance, for
    # the txs not in the graph (all of them if skip_graph is False)
    txs, groups = _SCAN_STATE["txs"], {}
    for t, selector in pairs:
        if not skip_graph or t not in txs:
            groups.setdefault(t, []).append(selector)
    return list(groups.items())

def scan_input_output_txes(ref):
    addr = resolve_address(ref)
    return _indices(addr.input_txes), _indices(addr.output_txes)

def scan_txes(ref):
    return _indices(resolve_address(ref).txes)

# Address-based whole and all over addr: the txs not in the graph, which are the
# candidates of the width limit (as in the tx-based variants)

def scan_new_txes(ref):
    return _new(tx.index for tx in resolve_address(ref).txes)

def scan_new_input_txes(ref):
    return _new(tx.index for tx in resolve_address(ref).input_txes)

def scan_new_output_txes(ref):
    return _new(tx.index for tx in resolve_address(ref).output_txes)

def scan_spending(i):
    return _new(spending_tx_index(out) for out in tx_of(i).outputs)

def scan_spent(i):
    return _new(inp.spent_tx_index for inp in tx_of(i).inputs)

def scan_spent_and_spending(i):
    tx = tx_of(i)
    return _new([inp.spent_tx_index for inp in tx.inputs] + [spending_tx_index(out) for out in tx.outputs])

# Dedicated variants: the inputs are selected by their spent output, the outputs by index.
# Every tx of a node is listed, also those in the graph (they get the links, see
# dedicated_candidates)

def scan_seed_dedicated(ref):
    addr = resolve_address(ref)
    return (_groups(((inp.tx_index, (inp.spent_tx_index, inp.spent_output.index)) for inp in addr.inputs), False),
            _groups(((out.tx_index, out.index) for out in addr.outputs), False))

def scan_dedicated_spending(i):
    return _groups(((t, (i, out.index)) for out in tx_of(i).outputs for t in [spending_tx_index(out)] if t is not None),
                   False)

def scan_dedicated_spent(i):
    return _groups(((inp.spent_tx_index, inp.spent_output.index) for inp in tx_of(i).inputs), False)

def scan_address_inputs(ref):
    return _groups(((inp.tx_index, (inp.spent_tx_index, inp.spent_output.index)) for inp in resolve_address(ref).inputs),
                   False)

def scan_address_outputs(ref):
    return _groups(((out.tx_index, out.index) for out in resolve_address(ref).outputs), False)

# --------------------------------------------------------------------------
# Workers: exploration
# --------------------------------------------------------------------------

def explore_record(task):
    """Hash, inputs and outputs (with an address) of a tx: all of them, or only the
    selected inputs ('input' mode, in the order of the selectors) or outputs ('output'),
    with all the outputs or inputs; 'link input' and 'link output' read only the selected
    inputs or outputs (the links of the dedicated variants to a tx of the graph).
    Inputs: (position, spent tx hash, spent output index, address string, age, sequence
    number, value); outputs: (position, index, address string, is spent, value)."""
    i, mode, selectors = task
    tx = tx_of(i)
    if mode in ("input", "link input"):
        by_outpoint = {(inp.spent_tx_index, inp.spent_output.index): (k, inp) for k, inp in enumerate(tx.inputs)}
        in_items = [by_outpoint[s] for s in selectors]
    else:
        in_items = [] if mode == "link output" else list(enumerate(tx.inputs))
    if mode in ("output", "link output"):
        by_index = {out.index: (j, out) for j, out in enumerate(tx.outputs)}
        out_items = [by_index[s] for s in selectors]
    else:
        out_items = [] if mode == "link input" else list(enumerate(tx.outputs))
    # keys: (address key, string) of the addresses read, so the main process can pass the
    # strings to the workers of the next passes (Graph.explored)
    ins, keys = [], []
    for k, inp in in_items:
        addr = inp.address
        addr_str = address_string(addr)
        if addr_str is not None:
            ins.append((k, spent_tx_hash(inp), inp.spent_output.index, addr_str, inp.age, inp.sequence_num, inp.value))
            keys.append((shared_expansion.address_key(addr), addr_str))
    outs = []
    for j, out in out_items:
        addr = out.address
        addr_str = address_string(addr)
        if addr_str is not None:
            outs.append((j, out.index, addr_str, out.is_spent, out.value))
            keys.append((shared_expansion.address_key(addr), addr_str))
    return tx_hash(tx), ins, outs, keys

# --------------------------------------------------------------------------
# Main process: selection and insertion
# --------------------------------------------------------------------------

intern = sys.intern

# Every hop forks one pool at its scan (the workers inherit the txs of the graph and the
# caches filled by the previous hops) and keeps it for its exploration; the pool is
# closed when the exploration ends, before the graph is modified (so the memory pages
# of the graph are not duplicated while the workers are alive). Small chunks, as a few
# txs (with thousands of inputs and outputs) are much costlier than the others.
_POOL = {"context": None, "pool": None}
CHUNKS = {"chunks_per_worker": 64, "max_chunk": 64}

def _close_pool():
    context = _POOL["context"]
    _POOL.update(context=None, pool=None)
    if context is not None:
        context.__exit__(None, None, None)

def scan(func, items, desc, txs_idx):
    _close_pool()
    _SCAN_STATE["txs"] = txs_idx
    workers = _CTX["workers"]
    if workers > 1:
        _POOL["context"] = forked_pool(workers)
        _POOL["pool"] = _POOL["context"].__enter__()
    try:
        return parallel_map(func, list(items), workers, desc=_desc(f"Scanning {desc}"), pool=_POOL["pool"], **CHUNKS)
    except BaseException:
        _close_pool()
        raise

def explore(tasks, desc):
    try:
        return parallel_map(explore_record, list(tasks), _CTX["workers"], desc=_desc(f"Exploring {desc}"),
                            pool=_POOL["pool"], **CHUNKS)
    finally:
        _close_pool()

def _desc(text):
    # Progress bars only when verbose
    return text if _CTX["verbose"] else None

def sample(candidates, limit):
    candidates = list(candidates)
    kept = limit_expansion(candidates, limit)
    if _CTX["on_sample"] is not None:
        _CTX["on_sample"](len(candidates), len(kept))
    return kept

_OPPOSITE = {"forward": "backward", "backward": "forward"}

class Graph:
    """The graph being expanded (see the top of this section)."""

    def __init__(self, addresses, txs, inputs, outputs):
        self.addresses, self.txs, self.inputs, self.outputs = addresses, txs, inputs, outputs
        self.idx = {value[2] for value in txs.values()}
        # The hashes (tx index -> hash) and address strings read by the workers are kept
        # in the caches of shared/expansion.py, which the workers of the next hops inherit
        shared_expansion.clear_tx_hash_cache()
        self.hash_of = shared_expansion.tx_hash_cache()

    def add_tx(self, i, hop, direction=None):
        # Same hop records as shared/expansion.py: a tx of the first hop found in both
        # directions is recorded as "both"
        h = self.hash_of[i]
        current = self.txs.get(h)
        if current is None:
            self.txs[h] = (hop, direction, i)
        elif direction in _OPPOSITE and current[:2] == (hop, _OPPOSITE[direction]):
            self.txs[h] = (hop, "both", current[2])
        self.idx.add(i)

    def explored(self, records, indices):
        """The records of explore_record as (hash, inputs, outputs), keeping their hash
        and address strings for the workers of the next hops."""
        result = []
        for i, (h, ins, outs, keys) in zip(indices, records):
            h = self.hash_of[i] = intern(h)
            for key, addr_str in keys:
                shared_expansion.remember_address_string(key, intern(addr_str))
            result.append((h, ins, outs))
        return result

    def put_input(self, i, item):
        k, spent_h, spent_index, addr_str, age, sequence_num, value = item
        addr_str = intern(addr_str)
        self.inputs[(intern(spent_h), spent_index)] = (addr_str, self.hash_of[i], age, sequence_num, value)
        return addr_str, ("in", i, k)

    def put_output(self, i, item):
        j, index, addr_str, is_spent, value = item
        addr_str = intern(addr_str)
        self.outputs[(self.hash_of[i], index)] = (addr_str, is_spent, value)
        return addr_str, ("out", i, j)

    def insert_whole(self, i, record):
        # explore_tx of shared/expansion.py: the addresses found are added to the graph
        _, ins, outs = record
        for item in ins:
            addr_str, ref = self.put_input(i, item)
            if addr_str not in self.addresses:
                self.addresses[addr_str] = (ref, -1)
        for item in outs:
            addr_str, ref = self.put_output(i, item)
            if addr_str not in self.addresses:
                self.addresses[addr_str] = (ref, -1)

def explore_and_insert(graph, indices, desc):
    indices = list(indices)
    records = graph.explored(explore([(i, "whole", None) for i in indices], desc), indices)
    for i, record in zip(indices, records):
        graph.insert_whole(i, record)

def select_per_node(scans, limit_mode, limit, candidates=None):
    """Distinct candidates of the frontier nodes, in order (into `candidates` if given):
    with the random node limit every node samples among its own distinct candidates,
    independently of the other nodes; a tx found by several nodes is kept once, at its
    first position. As _select of shared/expansion.py."""
    candidates = {} if candidates is None else candidates
    for found in scans:
        found = list(found)
        if limit_mode == "random node":
            found = sample(found, limit)
        candidates.update(dict.fromkeys(found))
    return candidates

def limit_hop(candidates, limit_mode, limit):
    # The random hop limit samples the distinct candidates of the hop
    return sample(candidates, limit) if limit_mode == "random hop" else list(candidates)

### Tx-based whole transaction implied addresses forward-backward expansion

def tx_whole_hop(graph, frontier, scan_func, hop, direction, limit_mode, limit, desc):
    candidates = limit_hop(select_per_node(scan(scan_func, frontier, desc, graph.idx), limit_mode, limit), limit_mode, limit)
    explore_and_insert(graph, candidates, desc)
    for i in candidates:
        graph.add_tx(i, hop, direction)
    return candidates

def expand_fw_bw_tx_whole(graph, num_for_hops, num_back_hops, limit_mode, limit):
    # Address by address: its forward txs, then its backward txs (the order of the draws)
    fw_candidates, bw_candidates = {}, {}
    for fw_found, bw_found in scan(scan_input_output_txes, [r for r, _ in graph.addresses.values()], "hop 1", graph.idx):
        select_per_node([fw_found], limit_mode, limit, fw_candidates)
        select_per_node([bw_found], limit_mode, limit, bw_candidates)
    new_fw, new_bw = limit_hop(fw_candidates, limit_mode, limit), limit_hop(bw_candidates, limit_mode, limit)

    # Every tx is explored once, the forward candidates first
    explore_and_insert(graph, dict.fromkeys(new_fw + new_bw), "hop 1")
    for i in new_fw:
        graph.add_tx(i, 1, "forward")
    for i in new_bw:
        graph.add_tx(i, 1, "backward")

    for i in range(num_for_hops - 1):
        new_fw = tx_whole_hop(graph, new_fw, scan_spending, i + 2, "forward", limit_mode, limit, f"forward hop {i + 2}")
    for i in range(num_back_hops - 1):
        new_bw = tx_whole_hop(graph, new_bw, scan_spent, i + 2, "backward", limit_mode, limit, f"backward hop {i + 2}")

### Tx-based all over expansion

def expand_all_over_tx(graph, num_hops, limit_mode, limit):
    scans = scan(scan_txes, [r for r, _ in graph.addresses.values()], "hop 1", graph.idx)
    new_txs = limit_hop(select_per_node(scans, limit_mode, limit), limit_mode, limit)
    explore_and_insert(graph, new_txs, "hop 1")
    for i in new_txs:
        graph.add_tx(i, 1)

    for i in range(num_hops - 1):
        new_txs = tx_whole_hop(graph, new_txs, scan_spent_and_spending, i + 2, None, limit_mode, limit, f"hop {i + 2}")

### Tx-based forward-backward expansion with dedicated inputs/outputs

def dedicated_candidates(groups_per_node, graph, limit_mode, limit):
    """Candidates of the dedicated variants, as _dedicated_candidates of
    shared/expansion.py: every node samples among its own txs not in the graph, and the
    txs already in the graph get the selectors of the frontier as links. Returns (tx index
    -> selectors) of the candidates and of the links."""
    groups_per_node = [dict(groups) for groups in groups_per_node]
    chosen = {}
    for node in groups_per_node:
        new = [t for t in node if t not in graph.idx]
        if limit_mode == "random node":
            new = sample(new, limit)
        chosen.update(dict.fromkeys(new))
    chosen = list(chosen)
    if limit_mode == "random hop":
        chosen = sample(chosen, limit)
    return _selectors(chosen, groups_per_node, graph)

def _selectors(chosen, groups_per_node, graph):
    candidates, links = {t: [] for t in chosen}, {}
    for node in groups_per_node:
        for t, selectors in node.items():
            if t in candidates:
                candidates[t].extend(selectors)
            elif t in graph.idx:
                links.setdefault(t, []).extend(selectors)
    return candidates, links

def explore_dedicated(graph, candidates, mode, known, desc):
    """Explores the candidates, as the dedicated variants of shared/expansion.py: forward
    (mode 'input'), the inputs of the addresses in `known` (the addresses of the graph at
    the start of the hop) and every output; backward ('output'), the outputs to the
    addresses in `known` and every input. Returns the addresses found outside `known`
    (address -> reference, in order) and the explored indices; the caller adds them."""
    indices = list(candidates)
    records = graph.explored(explore([(i, "whole", None) for i in indices], desc), indices)
    new_addresses = {}
    for i, (_, ins, outs) in zip(indices, records):
        selected = [(graph.put_input, item) for item in ins if item[3] in known] if mode == "input" else \
                   [(graph.put_output, item) for item in outs if item[2] in known]
        others = [(graph.put_output, item) for item in outs] if mode == "input" else \
                 [(graph.put_input, item) for item in ins]
        for put, item in selected:
            put(i, item)
        for put, item in others:
            addr_str, ref = put(i, item)
            if addr_str not in known:
                new_addresses.setdefault(addr_str, ref)
    return new_addresses, indices

def insert_links(graph, links, mode, desc):
    """The selected inputs ('input') or outputs ('output') of txs already in the graph."""
    indices = list(links)
    if not indices:
        return
    records = graph.explored(explore([(i, f"link {mode}", links[i]) for i in indices], desc), indices)
    for i, (_, ins, outs) in zip(indices, records):
        for item in ins:
            graph.put_input(i, item)
        for item in outs:
            graph.put_output(i, item)

def expand_fw_bw_tx_dedicated(graph, num_for_hops, num_back_hops, limit_mode, limit):
    # Seed by seed: its forward txs, sampled, then its backward txs, sampled (the order of
    # the random draws of the sequential algorithm, _dedicated_seed_candidates)
    scans = scan(scan_seed_dedicated, [r for r, _ in graph.addresses.values()], "hop 1", graph.idx)
    fw_groups, bw_groups = [dict(fw) for fw, _ in scans], [dict(bw) for _, bw in scans]
    fw_chosen, bw_chosen = {}, {}
    for fw, bw in zip(fw_groups, bw_groups):
        for node, chosen in ((fw, fw_chosen), (bw, bw_chosen)):
            new = [t for t in node if t not in graph.idx]
            if limit_mode == "random node":
                new = sample(new, limit)
            chosen.update(dict.fromkeys(new))
    fw_chosen, bw_chosen = list(fw_chosen), list(bw_chosen)
    if limit_mode == "random hop":
        fw_chosen, bw_chosen = sample(fw_chosen, limit), sample(bw_chosen, limit)
    fw_candidates, _ = _selectors(fw_chosen, fw_groups, graph)
    bw_candidates, _ = _selectors(bw_chosen, bw_groups, graph)

    known = set(graph.addresses)
    found_fw, new_fw = explore_dedicated(graph, fw_candidates, "input", known, "forward hop 1")
    found_bw, new_bw = explore_dedicated(graph, bw_candidates, "output", known, "backward hop 1")
    for found in (found_fw, found_bw):
        for addr_str, ref in found.items():
            graph.addresses.setdefault(addr_str, (ref, -1))
    for i in new_fw:
        graph.add_tx(i, 1, "forward")
    for i in new_bw:
        graph.add_tx(i, 1, "backward")

    for i in range(num_for_hops - 1):
        desc = f"forward hop {i + 2}"
        candidates, links = dedicated_candidates(scan(scan_dedicated_spending, new_fw, desc, graph.idx), graph,
                                                 limit_mode, limit)
        found, new_fw = explore_dedicated(graph, candidates, "input", set(graph.addresses), desc)
        insert_links(graph, links, "input", desc)
        for addr_str, ref in found.items():
            graph.addresses.setdefault(addr_str, (ref, -1))
        for t in new_fw:
            graph.add_tx(t, i + 2, "forward")
    for i in range(num_back_hops - 1):
        desc = f"backward hop {i + 2}"
        candidates, links = dedicated_candidates(scan(scan_dedicated_spent, new_bw, desc, graph.idx), graph,
                                                 limit_mode, limit)
        found, new_bw = explore_dedicated(graph, candidates, "output", set(graph.addresses), desc)
        insert_links(graph, links, "output", desc)
        for addr_str, ref in found.items():
            graph.addresses.setdefault(addr_str, (ref, -1))
        for t in new_bw:
            graph.add_tx(t, i + 2, "backward")

### Addr-based dedicated inputs/outputs forward-backward expansion

def addr_dedicated_hop(graph, known, frontier, forward, hop, limit_mode, limit):
    desc = f"{'forward' if forward else 'backward'} hop {hop}"
    scans = scan(scan_address_inputs if forward else scan_address_outputs, [r for r, _ in frontier.values()], desc, graph.idx)
    candidates, links = dedicated_candidates(scans, graph, limit_mode, limit)
    # Forward: the inputs of the known addresses, then every output (new addresses);
    # backward: the outputs to the known addresses, then every input (new addresses)
    found, indices = explore_dedicated(graph, candidates, "input" if forward else "output", known, desc)
    for i in indices:
        graph.add_tx(i, hop, "forward" if forward else "backward")
    insert_links(graph, links, "input" if forward else "output", desc)
    return {addr_str: (ref, -1) for addr_str, ref in found.items()}

def expand_fw_bw_addr_dedicated(graph, num_for_hops, num_back_hops, limit_mode, limit):
    addresses = graph.addresses
    news = {}
    new = addresses
    for i in range(num_for_hops):
        new = addr_dedicated_hop(graph, {**addresses, **news}, new, True, i + 1, limit_mode, limit)
        news.update(new)
    new = addresses
    for i in range(num_back_hops):
        new = addr_dedicated_hop(graph, {**addresses, **news}, new, False, i + 1, limit_mode, limit)
        news.update(new)
    addresses.update(news)

### Addr-based whole transaction implied addresses forward-backward expansion

def addr_whole_hop(graph, known, frontier, forward, hop, limit_mode, limit):
    """The txs where every frontier address is an input (forward) or an output
    (backward), with all their inputs and outputs. Returns the addresses on the other
    side (new) and on the same side (side) of the txs."""
    desc = f"{'forward' if forward else 'backward'} hop {hop}"
    candidates = {}
    for found in scan(scan_new_input_txes if forward else scan_new_output_txes, [r for r, _ in frontier.values()],
                      desc, graph.idx):
        txes = sample(found, limit) if limit_mode == "random node" else found
        candidates.update(dict.fromkeys(txes))
    if limit_mode == "random hop":
        candidates = dict.fromkeys(sample(candidates.keys(), limit))
    indices = list(candidates)
    records = graph.explored(explore([(i, "whole", None) for i in indices], desc), indices)
    new_addresses, side_addresses = {}, {}
    for i, (_, ins, outs) in zip(indices, records):
        graph.add_tx(i, hop, "forward" if forward else "backward")
        same = [(graph.put_input, item) for item in ins] if forward else [(graph.put_output, item) for item in outs]
        other = [(graph.put_output, item) for item in outs] if forward else [(graph.put_input, item) for item in ins]
        for put, item in same:
            addr_str, ref = put(i, item)
            if addr_str not in known and addr_str not in new_addresses:
                side_addresses[addr_str] = (ref, -1)
        for put, item in other:
            addr_str, ref = put(i, item)
            if addr_str not in known:
                new_addresses[addr_str] = (ref, -1)
    return new_addresses, side_addresses

def expand_fw_bw_addr_whole(graph, num_for_hops, num_back_hops, mode, limit_mode, limit):
    addresses = graph.addresses
    hop = lambda known, frontier, forward, i: addr_whole_hop(graph, known, frontier, forward, i + 1, limit_mode, limit)
    # none and opposite: every hop has a forward phase (while i < num_for_hops) followed by
    # a backward phase (while i < num_back_hops), as in shared/expansion.py
    if mode == 'none':
        new_fw, new_bw = addresses.copy(), addresses.copy()
        for i in range(max(num_for_hops, num_back_hops)):
            if i < num_for_hops:
                new_fw, side_fw = hop(addresses, new_fw, True, i)
                addresses.update({**new_fw, **side_fw})
            if i < num_back_hops:
                new_bw, side_bw = hop(addresses, new_bw, False, i)
                addresses.update({**new_bw, **side_bw})
    elif mode == 'same':
        new, side, news = addresses.copy(), {}, {}
        for i in range(num_for_hops):
            new, side = hop({**addresses, **news}, {**new, **side}, True, i)
            news.update(new)
            news.update(side)
        new, side = addresses.copy(), {}
        for i in range(num_back_hops):
            new, side = hop({**addresses, **news}, {**new, **side}, False, i)
            news.update(new)
            news.update(side)
        addresses.update(news)
    elif mode == 'opposite':
        new_fw, new_bw = addresses.copy(), addresses.copy()
        for i in range(max(num_for_hops, num_back_hops)):
            side_fw, side_bw = {}, {}
            if i < num_for_hops:
                new_fw, side_fw = hop(addresses, new_fw, True, i)
                addresses.update({**new_fw, **side_fw})
            if i < num_back_hops:
                new_bw, side_bw = hop(addresses, new_bw, False, i)
                addresses.update({**new_bw, **side_bw})
            new_fw.update(side_bw)
            new_bw.update(side_fw)
    else:
        raise ValueError(f"Unknown treatment of side addresses: {mode}")

### Addr-based all over expansion

def expand_all_over_addr(graph, num_hops, limit_mode, limit):
    addresses = graph.addresses
    new_addresses = addresses
    for i in range(num_hops):
        desc = f"hop {i + 1}"
        # Candidates repeat when frontier addresses share a tx: each tx is kept once,
        # before the random hop sampling, so a shared tx is not more likely to be chosen
        candidates = {}
        for found in scan(scan_new_txes, [r for r, _ in new_addresses.values()], desc, graph.idx):
            txes = sample(found, limit) if limit_mode == "random node" else found
            candidates.update(dict.fromkeys(txes))
        if limit_mode == "random hop":
            candidates = dict.fromkeys(sample(candidates.keys(), limit))
        indices = list(candidates)
        records = graph.explored(explore([(t, "whole", None) for t in indices], desc), indices)
        hop_addresses = {}
        for t, (_, ins, outs) in zip(indices, records):
            graph.add_tx(t, i + 1)
            for put, item in [(graph.put_input, item) for item in ins] + [(graph.put_output, item) for item in outs]:
                addr_str, ref = put(t, item)
                if addr_str not in addresses:
                    hop_addresses[addr_str] = (ref, -1)
        new_addresses = hop_addresses
        addresses.update(new_addresses)

def expand(exp_alg, addresses, txs, inputs, outputs, for_hops=None, back_hops=None, hops=None,
           limit_mode="", limit=math.inf):
    """Expands the seed addresses (addresses: address string -> (("seed", string), label))
    with the variant exp_alg (shared.expansion.VARIANTS), filling the graph in place (see
    the top of this module). Forward-backward variants use for_hops and back_hops,
    all-over variants hops; limit is ignored without limit_mode."""
    if exp_alg not in shared_expansion.VARIANTS:
        raise ValueError(f"Unknown expansion variant '{exp_alg}'; available: {', '.join(shared_expansion.VARIANTS)}")
    if not limit_mode:
        limit = math.inf
    graph = Graph(addresses, txs, inputs, outputs)
    if _CTX["verbose"]:
        print(f"Hop 1, addresses to expand: {len(addresses)}", flush=True)
    try:
        if exp_alg == "fw bw tx whole":
            expand_fw_bw_tx_whole(graph, for_hops, back_hops, limit_mode, limit)
        elif exp_alg == "fw bw tx dedicated":
            expand_fw_bw_tx_dedicated(graph, for_hops, back_hops, limit_mode, limit)
        elif exp_alg == "all over tx":
            expand_all_over_tx(graph, hops, limit_mode, limit)
        elif exp_alg.startswith("fw bw addr whole"):
            expand_fw_bw_addr_whole(graph, for_hops, back_hops, exp_alg.split()[-1], limit_mode, limit)
        elif exp_alg == "fw bw addr dedicated":
            expand_fw_bw_addr_dedicated(graph, for_hops, back_hops, limit_mode, limit)
        elif exp_alg == "all over addr":
            expand_all_over_addr(graph, hops, limit_mode, limit)
    finally:
        _close_pool()
        # The hashes are kept in the graph (txs); the caches are not needed afterwards
        shared_expansion.clear_tx_hash_cache()
    if _CTX["verbose"]:
        print(f"Final number of addrs: {len(addresses)}, txs: {len(txs)}, inputs: {len(inputs)}, "
              f"outputs: {len(outputs)}", flush=True)
