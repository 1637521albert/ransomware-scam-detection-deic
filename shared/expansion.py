# Expansion strategies of the address-transaction graph, shared by both methodologies.
#
# Inductive Multi-Instance Address Classification expands the seed addresses of a
# split jointly into one graph, and Inductive Ego-Centric Address Classification
# expands every seed address into its own egonet, with exactly the same algorithms
# (this module). Variants (direction, central unit, scope, side addresses):
#   fw bw tx whole, fw bw tx dedicated, all over tx,
#   fw bw addr whole none / same / opposite, fw bw addr dedicated, all over addr
# These are the sequential reference algorithms: the ego-centric methodology runs them
# as they are, one egonet per worker process. shared/parallel_expansion.py runs the same
# variants in parallel for one graph of many seeds (the multi-instance methodology and
# the expansion experiments), with the order of the random draws and insertions of these
# algorithms, so it gives the same graph.
#
# Addresses are identified by their address string (address_string), except the
# pay-to-pubkey addresses, whose BlockSci string is the one of the pay-to-pubkey-hash
# address of the same key: they are identified by "pubkey:" + that string, so the two
# addresses (with their own histories) are two nodes instead of one node with the
# features of whichever was found first.
#
# The expansion fills, in place:
#   addresses: address string -> (BlockSci address, label); the seeds keep their label
#              and the addresses found get new_label
#   txs:       tx hash -> BlockSci tx
#   inputs:    (spent tx hash, spent output index) -> (address string, BlockSci input, tx)
#   outputs:   (tx hash, output index) -> (tx, BlockSci output, address string)
#   tx_hops:   optional, tx hash -> (hop, direction) of the hop where the tx was added
#              to the graph: hops are numbered from 1; the direction is "forward" or
#              "backward" in the forward-backward variants ("both" for a tx of the
#              first hop found in both directions) and None in the all-over variants
# The width limit samples with the global `random` module, which the caller seeds.
# Tx hashes and address strings are memoized during an expansion (each is encoded once)
# and the caches are emptied at the start of every expansion (clear_tx_hash_cache).

import math
import random

from tqdm import tqdm

# Chain (set_chain) and settings of the running expansion (expand)
_CTX = {"chain": None, "new_label": -1, "verbose": True, "tx_hops": None}


def set_chain(chain):
    _CTX["chain"] = chain


def limit_expansion(txs, limit):
    # Python >= 3.11 only samples sequences; on older versions random.sample
    # converts dict keys to a tuple in the same order, so the result is the same
    if not isinstance(txs, (list, tuple)):
        txs = list(txs)
    return random.sample(txs, min(len(txs), limit))

# Hex-encoding a tx hash is one of the costliest BlockSci calls, and the same
# transactions are hashed over and over, so hashes are memoized by tx index.
_tx_hashes = {}


def clear_tx_hash_cache():
    _tx_hashes.clear()
    _address_strings.clear()


def tx_hash(tx):
    idx = tx.index
    h = _tx_hashes.get(idx)
    if h is None:
        h = _tx_hashes[idx] = str(tx.hash)
    return h


def input_tx_hash(inp):
    # Same as str(inp.tx.hash)
    idx = inp.tx_index
    h = _tx_hashes.get(idx)
    if h is None:
        h = _tx_hashes[idx] = str(inp.tx.hash)
    return h


def spent_tx_hash(inp):
    # Same as str(inp.spent_output.tx.hash)
    idx = inp.spent_tx_index
    h = _tx_hashes.get(idx)
    if h is None:
        h = _tx_hashes[idx] = str(inp.spent_tx.hash)
    return h


def output_tx_hash(out):
    # Same as str(out.tx.hash)
    idx = out.tx_index
    h = _tx_hashes.get(idx)
    if h is None:
        h = _tx_hashes[idx] = str(out.tx.hash)
    return h


# Encoding an address string is costly and the same address appears in many inputs and
# outputs, so strings are memoized by address (number and type, as a pubkey address
# shares its BlockSci string with its pubkey-hash address). The cache, like the tx hashes, is
# cleared at the start of every expansion. The key is made of plain values, so a parent
# process can fill the cache of the workers it forks next (remember_address_string).
_address_strings = {}
_MISSING = object()


def address_key(addr):
    return (addr.address_num, str(addr.type))


def remember_address_string(key, addr_str):
    _address_strings.setdefault(key, addr_str)


def tx_hash_cache():
    # Memoized tx hashes by tx index, to fill from a parent process
    return _tx_hashes


# Prefix of the identifier of a pay-to-pubkey address (see the top of this module)
PUBKEY_PREFIX = "pubkey:"


def _identifier(addr):
    # Address string of the address, None if it has none (e.g. bare multisig, OP_RETURN).
    # A pay-to-pubkey address is recognised by its full type ("pubkey"; a P2SH-wrapped
    # pubkey is "scripthash/pubkey" and has its own "3..." string). Only "1..." strings,
    # shared by P2PK and P2PKH, need the check.
    addr_str = getattr(addr, 'address_string', None)
    if addr_str is not None and addr_str.startswith("1") and getattr(addr, 'full_type', None) == "pubkey":
        addr_str = PUBKEY_PREFIX + addr_str
    return addr_str


def _address_string_memoized(addr):
    key = address_key(addr)
    addr_str = _address_strings.get(key, _MISSING)
    if addr_str is _MISSING:
        addr_str = _address_strings[key] = _identifier(addr)
    return addr_str


def _address_string_direct(addr):
    return _identifier(addr)


_address_string = _address_string_memoized


def address_string(addr):
    # Identifier of an address in the graph (its string, "pubkey:" + string for a
    # pay-to-pubkey address), or None. Encodes the address once instead of hasattr() +
    # repeated .address_string; falls back to encoding every time if addresses cannot be
    # identified
    global _address_string
    try:
        return _address_string(addr)
    except AttributeError:
        _address_string = _address_string_direct
        return _address_string(addr)


def lookup_tx(tx, tx_id):
    # Same as chain.tx_with_hash(tx_id) for a tx already in hand: only coinbase
    # txs can share a hash (BIP30 duplicates), so the hash index lookup is kept for them
    return _CTX["chain"].tx_with_hash(tx_id) if tx.is_coinbase else tx


def _progress(iterable):
    return tqdm(iterable, disable=not _CTX["verbose"])


def _log(*args):
    if _CTX["verbose"]:
        print(*args, flush=True)


def _new_address(addr):
    return (addr, _CTX["new_label"])

# Hops of the txs

_OPPOSITE = {"forward": "backward", "backward": "forward"}


def _record_hop(txs, tx_id, hop, direction):
    tx_hops = _CTX["tx_hops"]
    if tx_hops is None:
        return
    if tx_id not in txs:
        tx_hops[tx_id] = (hop, direction)
    elif direction in _OPPOSITE and tx_hops.get(tx_id) == (hop, _OPPOSITE[direction]):
        tx_hops[tx_id] = (hop, "both")


def add_tx(txs, tx_id, tx, hop, direction=None):
    _record_hop(txs, tx_id, hop, direction)
    txs[tx_id] = tx


def add_txs(txs, new_txs, hop, direction=None):
    # Same as txs.update(new_txs), recording the hop of the txs added
    for tx_id, tx in new_txs.items():
        add_tx(txs, tx_id, tx, hop, direction)


def explore_tx(tx, addresses, inputs, outputs, new, mode='none', selected_list=(), explored=None):
    h = tx_hash(tx)
    new[h] = tx

    # Re-exploring a whole tx rewrites the same entries, so it can be skipped
    if explored is not None:
        if tx.index in explored:
            return
        explored.add(tx.index)

    process_inputs = selected_list if mode == "input" else tx.inputs
    process_outputs = selected_list if mode == "output" else tx.outputs

    for input_tx in process_inputs:
        addr = input_tx.address
        addr_str = address_string(addr)

        if addr_str is not None:
            input_id = (spent_tx_hash(input_tx), input_tx.spent_output.index)
            inputs[input_id] = (addr_str, input_tx, tx)

            if addr_str not in addresses:
                addresses[addr_str] = _new_address(addr)

    for output_tx in process_outputs:
        addr = output_tx.address
        addr_str = address_string(addr)

        if addr_str is not None:
            outputs[(h, output_tx.index)] = (tx, output_tx, addr_str)

            if addr_str not in addresses:
                addresses[addr_str] = _new_address(addr)

# Width limit. With random node every node of the frontier samples among its own
# distinct candidates not yet in the graph, in every variant, independently of the other
# nodes (a tx chosen by a previous node is still a candidate of the next ones). A tx found by several nodes is kept
# once, at its first position, before the random hop sampling, so it is not more
# likely to be chosen.

def _distinct_new(txes, txs):
    # Distinct txs (None skipped) not in the graph, in order of appearance
    found = {}
    for tx in txes:
        if tx is not None and tx.index not in found and tx_hash(tx) not in txs:
            found[tx.index] = tx
    return list(found.values())


def _select(candidates_per_node, limit_mode, limit, candidates=None):
    # Distinct candidates of a hop (into `candidates`, tx index -> tx, if given), in order
    candidates = {} if candidates is None else candidates
    for node_candidates in candidates_per_node:
        if limit_mode == "random node":
            node_candidates = limit_expansion(node_candidates, limit)
        for tx in node_candidates:
            candidates.setdefault(tx.index, tx)
    return candidates


def _limit_hop(candidates, limit_mode, limit):
    candidates = list(candidates.values())
    return limit_expansion(candidates, limit) if limit_mode == "random hop" else candidates


# Dedicated variants. Every frontier node groups its inputs (forward) or outputs
# (backward) by tx, and samples among its own txs not in the graph. A chosen tx, at its
# first position, is explored with the inputs (forward) or outputs (backward) of the
# addresses already in the graph at the start of the hop, the frontier included, and
# every output (forward) or input (backward). The txs already in the graph get the
# inputs (outputs) of the frontier nodes as edges (links), without being explored again,
# so no frontier node loses its edge to a tx of the graph.

def _dedicated_candidates(groups_per_node, txs, limit_mode, limit):
    # groups_per_node: per frontier node, tx id -> its inputs (outputs) in the tx.
    # Returns the txs to explore and the links, tx id -> inputs (outputs)
    chosen = {}
    for groups in groups_per_node:
        new = [tx_id for tx_id in groups if tx_id not in txs]
        if limit_mode == "random node":
            new = limit_expansion(new, limit)
        chosen.update(dict.fromkeys(new))
    chosen = list(chosen)
    if limit_mode == "random hop":
        chosen = limit_expansion(chosen, limit)

    candidates = {tx_id: [] for tx_id in chosen}
    links = {}
    for groups in groups_per_node:
        for tx_id, selected in groups.items():
            if tx_id in candidates:
                candidates[tx_id].extend(selected)
            elif tx_id in txs:
                links.setdefault(tx_id, []).extend(selected)
    return candidates, links

def _dedicated_seed_candidates(fw_groups, bw_groups, txs, limit_mode, limit):
    # First hop of the tx-based dedicated variant: every seed samples its forward txs and
    # then its backward txs, so the random draws alternate between both directions
    fw_chosen, bw_chosen = {}, {}
    for fw, bw in zip(fw_groups, bw_groups):
        for groups, chosen in ((fw, fw_chosen), (bw, bw_chosen)):
            new = [tx_id for tx_id in groups if tx_id not in txs]
            if limit_mode == "random node":
                new = limit_expansion(new, limit)
            chosen.update(dict.fromkeys(new))
    fw_chosen, bw_chosen = list(fw_chosen), list(bw_chosen)
    if limit_mode == "random hop":
        fw_chosen = limit_expansion(fw_chosen, limit)
        bw_chosen = limit_expansion(bw_chosen, limit)

    result = []
    for chosen, groups_per_node in ((fw_chosen, fw_groups), (bw_chosen, bw_groups)):
        candidates = {tx_id: [] for tx_id in chosen}
        for groups in groups_per_node:
            for tx_id, selected in groups.items():
                if tx_id in candidates:
                    candidates[tx_id].extend(selected)
        result.append(candidates)
    return result

def _put_input(inputs, inp, tx):
    # Input with an address, as explore_tx; returns its address string (None without one)
    addr_str = address_string(inp.address)
    if addr_str is not None:
        inputs[(spent_tx_hash(inp), inp.spent_output.index)] = (addr_str, inp, tx)
    return addr_str

def _put_output(outputs, out, tx, tx_id):
    # Output with an address, as explore_tx; returns its address string (None without one)
    addr_str = address_string(out.address)
    if addr_str is not None:
        outputs[(tx_id, out.index)] = (tx, out, addr_str)
    return addr_str

def _of_known(items, known):
    # The inputs or outputs of a tx whose address is in `known`
    return [item for item in items if address_string(item.address) in known]

def _link_inputs(links, txs, inputs):
    for tx_id, selected in links.items():
        for inp in selected:
            _put_input(inputs, inp, txs[tx_id])

def _link_outputs(links, txs, outputs):
    for tx_id, selected in links.items():
        for out in selected:
            _put_output(outputs, out, txs[tx_id], tx_id)

### Addr-based dedicated inputs/outputs forward-backward expansion

def addr_dedicated_expand_forward(addresses, prev_addresses, txs, inputs, outputs, limit_mode=None, limit=math.inf, hop=None):
    new_addresses = {}
    groups_per_node = []

    for addr, _ in _progress(prev_addresses.values()):
        groups = {}
        for inp in addr.inputs:
            groups.setdefault(input_tx_hash(inp), []).append(inp)
        groups_per_node.append(groups)

    candidates, links = _dedicated_candidates(groups_per_node, txs, limit_mode, limit)

    for tx_id, tx_inputs in candidates.items():
        tx = lookup_tx(tx_inputs[0].tx, tx_id)
        add_tx(txs, tx_id, tx, hop, "forward")

        # The inputs of the addresses of the graph (the frontier included)
        for inp in tx.inputs:
            if address_string(inp.address) in addresses:
                _put_input(inputs, inp, tx)

        for out in tx.outputs:
            addr_str = _put_output(outputs, out, tx, tx_id)
            if addr_str is not None and addr_str not in addresses:
                new_addresses[addr_str] = _new_address(out.address)

    _link_inputs(links, txs, inputs)
    return new_addresses

def addr_dedicated_expand_backward(addresses, prev_addresses, txs, inputs, outputs, limit_mode=None, limit=math.inf, hop=None):
    new_addresses = {}
    groups_per_node = []

    for addr, _ in _progress(prev_addresses.values()):
        groups = {}
        for out in addr.outputs:
            groups.setdefault(output_tx_hash(out), []).append(out)
        groups_per_node.append(groups)

    candidates, links = _dedicated_candidates(groups_per_node, txs, limit_mode, limit)

    for tx_id, tx_outputs in candidates.items():
        tx = lookup_tx(tx_outputs[0].tx, tx_id)
        add_tx(txs, tx_id, tx, hop, "backward")

        # The outputs to the addresses of the graph (the frontier included)
        for out in tx.outputs:
            if address_string(out.address) in addresses:
                _put_output(outputs, out, tx, tx_id)

        for inp in tx.inputs:
            addr_str = _put_input(inputs, inp, tx)
            if addr_str is not None and addr_str not in addresses:
                new_addresses[addr_str] = _new_address(inp.address)

    _link_outputs(links, txs, outputs)
    return new_addresses

def expand_fw_bw_addr_dedicated(addresses, num_for_hops, num_back_hops, txs, inputs, outputs, limit_mode=None, limit=math.inf):
    news = {}

    _log("\nForward expand:")
    new = addresses
    for i in range(num_for_hops):
        new = addr_dedicated_expand_forward({**addresses, **news}, new, txs, inputs, outputs, limit_mode=limit_mode, limit=limit, hop=i + 1)
        news.update(new)

    _log("\nBackward expand:")
    new = addresses
    for i in range(num_back_hops):
        new = addr_dedicated_expand_backward({**addresses, **news}, new, txs, inputs, outputs, limit_mode=limit_mode, limit=limit, hop=i + 1)
        news.update(new)

    addresses.update(news)

### Addr-based whole transaction implied addresses forward-backward expansion

def addr_whole_expand_forward(addresses, prev_addresses, txs, inputs, outputs, limit_mode=None, limit=math.inf, hop=None):
    new_addresses = {}
    side_addresses = {}
    candidates = {}

    for addr, _ in _progress(prev_addresses.values()):
        # The txs already in the graph are not candidates, as in the tx-based variants
        in_txs = _distinct_new(addr.input_txes, txs)
        if limit_mode == "random node":
            in_txs = limit_expansion(in_txs, limit)

        for in_tx in in_txs:
            candidates.setdefault(tx_hash(in_tx), in_tx)

    if limit_mode == "random hop":
        candidates = {tx_id: candidates[tx_id] for tx_id in limit_expansion(candidates.keys(), limit)}

    for tx_id, in_tx in candidates.items():
        add_tx(txs, tx_id, in_tx, hop, "forward")

        for input_tx in in_tx.inputs:
            addr = input_tx.address
            addr_str = address_string(addr)

            if addr_str is not None:
                input_id = (spent_tx_hash(input_tx), input_tx.spent_output.index)
                inputs[input_id] = (addr_str, input_tx, in_tx)

                if addr_str not in addresses and addr_str not in new_addresses:
                    side_addresses[addr_str] = _new_address(addr)

        for output_tx in in_tx.outputs:
            addr = output_tx.address
            addr_str = address_string(addr)

            if addr_str is not None:
                outputs[(tx_id, output_tx.index)] = (in_tx, output_tx, addr_str)

                if addr_str not in addresses:
                    new_addresses[addr_str] = _new_address(addr)

    return new_addresses, side_addresses

def addr_whole_expand_backward(addresses, prev_addresses, txs, inputs, outputs, limit_mode=None, limit=math.inf, hop=None):
    new_addresses = {}
    side_addresses = {}
    candidates = {}

    for addr, _ in _progress(prev_addresses.values()):
        # The txs already in the graph are not candidates, as in the tx-based variants
        out_txs = _distinct_new(addr.output_txes, txs)
        if limit_mode == "random node":
            out_txs = limit_expansion(out_txs, limit)

        for out_tx in out_txs:
            candidates.setdefault(tx_hash(out_tx), out_tx)

    if limit_mode == "random hop":
        candidates = {tx_id: candidates[tx_id] for tx_id in limit_expansion(candidates.keys(), limit)}

    for tx_id, out_tx in candidates.items():
        add_tx(txs, tx_id, out_tx, hop, "backward")

        for output_tx in out_tx.outputs:
            addr = output_tx.address
            addr_str = address_string(addr)

            if addr_str is not None:
                outputs[(tx_id, output_tx.index)] = (out_tx, output_tx, addr_str)

                if addr_str not in addresses and addr_str not in new_addresses:
                    side_addresses[addr_str] = _new_address(addr)

        for inp in out_tx.inputs:
            addr = inp.address
            addr_str = address_string(addr)

            if addr_str is not None:
                input_id = (spent_tx_hash(inp), inp.spent_output.index)
                inputs[input_id] = (addr_str, inp, out_tx)

                if addr_str not in addresses:
                    new_addresses[addr_str] = _new_address(addr)

    return new_addresses, side_addresses

def expand_fw_bw_addr_whole(addresses, num_for_hops, num_back_hops, txs, inputs, outputs, mode, limit_mode=None, limit=math.inf):
    """mode: treatment of the side addresses (found on the same side of a tx as the
    expanded address): 'none', 'same' or 'opposite'."""
    # none and opposite: every hop has a forward phase (while i < num_for_hops) followed
    # by a backward phase (while i < num_back_hops)
    if mode == 'none':
        new_fw = addresses.copy()
        new_bw = addresses.copy()
        for i in range(max(num_for_hops, num_back_hops)):
            if i < num_for_hops:
                _log(f"\nHop {i + 1}:\n\nForward phase:")
                new_fw, side_fw = addr_whole_expand_forward(addresses, new_fw, txs, inputs, outputs, limit_mode=limit_mode, limit=limit, hop=i + 1)
                addresses.update({**new_fw, **side_fw})

            if i < num_back_hops:
                _log("\nBackward phase:")
                new_bw, side_bw = addr_whole_expand_backward(addresses, new_bw, txs, inputs, outputs, limit_mode=limit_mode, limit=limit, hop=i + 1)
                addresses.update({**new_bw, **side_bw})

            _log(f"\nTotal number of addrs: {len(addresses)}\nTotal number of txs: {len(txs)}")

    elif mode == 'same':
        new = addresses.copy()
        side = {}
        news = {}
        _log("\nForward expand:")
        for i in range(num_for_hops):
            _log(f"Hop {i} addresses to expand:", len(new) + len(side))
            new, side = addr_whole_expand_forward({**addresses, **news}, {**new, **side}, txs, inputs, outputs, limit_mode=limit_mode, limit=limit, hop=i + 1)
            news.update(new)
            news.update(side)

        _log("\nBackward expand:")
        new = addresses.copy()
        side = {}
        for i in range(num_back_hops):
            _log(f"Hop {i}, addresses to expand:", len(new) + len(side))
            new, side = addr_whole_expand_backward({**addresses, **news}, {**new, **side}, txs, inputs, outputs, limit_mode=limit_mode, limit=limit, hop=i + 1)
            news.update(new)
            news.update(side)

        addresses.update(news)

    elif mode == 'opposite':
        new_fw = addresses.copy()
        new_bw = addresses.copy()
        side_fw = {}
        side_bw = {}

        for i in range(max(num_for_hops, num_back_hops)):
            _log(f"Hop {i}, addresses to expand:", len(new_fw) + len(side_fw) + len(new_bw) + len(side_bw))

            side_fw, side_bw = {}, {}
            if i < num_for_hops:
                _log("\nForward phase:")
                new_fw, side_fw = addr_whole_expand_forward(addresses, new_fw, txs, inputs, outputs, limit_mode=limit_mode, limit=limit, hop=i + 1)
                addresses.update({**new_fw, **side_fw})

            if i < num_back_hops:
                _log("\nBackward phase:")
                new_bw, side_bw = addr_whole_expand_backward(addresses, new_bw, txs, inputs, outputs, limit_mode=limit_mode, limit=limit, hop=i + 1)
                addresses.update({**new_bw, **side_bw})

            new_fw.update(side_bw)
            new_bw.update(side_fw)

    else:
        raise ValueError(f"Unknown treatment of side addresses: {mode}")

### Addr-based all over expansion

def expand_all_over_addr(addresses, num_hops, txs, inputs, outputs, limit_mode=None, limit=math.inf):
    new_addresses = addresses

    for i in range(num_hops):
        _log(f"Hop {i + 1} addresses to expand:", len(new_addresses))
        hop_addresses = {}
        # Candidates repeat when frontier addresses share a tx: each tx is kept once
        # (by index), before the random hop sampling, so a shared tx is not more
        # likely to be chosen
        candidates = {}

        for address, (addr, label) in _progress(new_addresses.items()):
            # The txs already in the graph are not candidates, as in the tx-based variants
            txes = _distinct_new(addr.txes, txs)
            if limit_mode == 'random node':
                txes = limit_expansion(txes, limit)

            for tx in txes:
                candidates.setdefault(tx.index, tx)

        candidates = list(candidates.values())

        if limit_mode == "random hop":
            candidates = limit_expansion(candidates, limit)

        for tx in candidates:
            tx_id = tx_hash(tx)
            add_tx(txs, tx_id, tx, i + 1)

            for input_tx in tx.inputs:
                addr = input_tx.address
                addr_str = address_string(addr)

                if addr_str is not None:
                    input_id = (spent_tx_hash(input_tx), input_tx.spent_output.index)
                    inputs[input_id] = (addr_str, input_tx, tx)

                    if addr_str not in addresses:
                        hop_addresses[addr_str] = _new_address(addr)

            for output_tx in tx.outputs:
                addr = output_tx.address
                addr_str = address_string(addr)

                if addr_str is not None:
                    outputs[(tx_id, output_tx.index)] = (tx, output_tx, addr_str)

                    if addr_str not in addresses:
                        hop_addresses[addr_str] = _new_address(addr)

        new_addresses = hop_addresses
        addresses.update(new_addresses)

### Tx-based forward-backward expansion with dedicated inputs/outputs

def tx_dedicated_expand_forward(prev_txs, addresses, txs, inputs, outputs, limit_mode=None, limit=math.inf):
    new_txs = {}
    groups_per_node = []

    for tx in _progress(prev_txs.values()):
        groups = {}
        for tx_out in tx.outputs:
            out_tx = tx_out.spending_tx

            if out_tx is not None:
                groups.setdefault(tx_hash(out_tx), []).append(tx_out.spending_input)
        groups_per_node.append(groups)

    candidates, links = _dedicated_candidates(groups_per_node, txs, limit_mode, limit)
    known = set(addresses)

    for tx_id, tx_inputs in candidates.items():
        tx = lookup_tx(tx_inputs[0].tx, tx_id)
        explore_tx(tx, addresses, inputs, outputs, new_txs, mode='input', selected_list=_of_known(tx.inputs, known))

    _link_inputs(links, txs, inputs)
    return new_txs

def tx_dedicated_expand_backward(prev_txs, addresses, txs, inputs, outputs, limit_mode=None, limit=math.inf):
    new_txs = {}
    groups_per_node = []

    for tx in _progress(prev_txs.values()):
        groups = {}
        for tx_in in tx.inputs:
            in_tx = tx_in.spent_tx

            if in_tx is not None:
                groups.setdefault(tx_hash(in_tx), []).append(tx_in.spent_output)
        groups_per_node.append(groups)

    candidates, links = _dedicated_candidates(groups_per_node, txs, limit_mode, limit)
    known = set(addresses)

    for tx_id, tx_outputs in candidates.items():
        tx = lookup_tx(tx_outputs[0].tx, tx_id)
        explore_tx(tx, addresses, inputs, outputs, new_txs, mode='output', selected_list=_of_known(tx.outputs, known))

    _link_outputs(links, txs, outputs)
    return new_txs

def expand_fw_bw_tx_dedicated(addresses, num_for_hops, num_back_hops, txs, inputs, outputs, limit_mode=None, limit=math.inf):
    new_fw_txs = {}
    new_bw_txs = {}
    fw_groups = []
    bw_groups = []
    addrs = list(addresses.values()).copy()
    _log("Hop: 1, addresses to expand:", len(addresses))

    for addr, _ in _progress(addrs):
        groups = {}
        for inp in addr.inputs:
            groups.setdefault(input_tx_hash(inp), []).append(inp)
        fw_groups.append(groups)

        groups = {}
        for out in addr.outputs:
            groups.setdefault(output_tx_hash(out), []).append(out)
        bw_groups.append(groups)

    # Seed by seed, its forward txs and then its backward txs are sampled (the order of
    # the random draws)
    fw_candidates, bw_candidates = _dedicated_seed_candidates(fw_groups, bw_groups, txs, limit_mode, limit)
    known = set(addresses)

    for fw_tx_id, fw_inputs in fw_candidates.items():
        fw_tx = lookup_tx(fw_inputs[0].tx, fw_tx_id)
        explore_tx(fw_tx, addresses, inputs, outputs, new_fw_txs, mode='input', selected_list=_of_known(fw_tx.inputs, known))

    for bw_tx_id, bw_outputs in bw_candidates.items():
        bw_tx = lookup_tx(bw_outputs[0].tx, bw_tx_id)
        explore_tx(bw_tx, addresses, inputs, outputs, new_bw_txs, mode='output', selected_list=_of_known(bw_tx.outputs, known))

    add_txs(txs, new_fw_txs, 1, "forward")
    add_txs(txs, new_bw_txs, 1, "backward")

    _log("\nForward expand:")
    for i in range(num_for_hops-1):
        _log(f"Hop: {i+2}, txs to expand: {len(new_fw_txs)}")
        new_fw_txs = tx_dedicated_expand_forward(new_fw_txs, addresses, txs, inputs, outputs, limit_mode=limit_mode, limit=limit)
        add_txs(txs, new_fw_txs, i + 2, "forward")

    _log("\nBackward expand:")
    for i in range(num_back_hops-1):
        _log(f"Hop: {i+2}, txs to expand: {len(new_bw_txs)}")
        new_bw_txs = tx_dedicated_expand_backward(new_bw_txs, addresses, txs, inputs, outputs, limit_mode=limit_mode, limit=limit)
        add_txs(txs, new_bw_txs, i + 2, "backward")

### Tx-based whole transaction implied addresses forward-backward expansion

def tx_whole_expand_forward(prev_txs, addresses, txs, inputs, outputs, limit_mode=None, limit=math.inf):
    new_txs = {}
    candidates_per_node = (_distinct_new((tx_out.spending_tx for tx_out in tx.outputs), txs)
                           for tx in _progress(prev_txs.values()))

    for out_tx in _limit_hop(_select(candidates_per_node, limit_mode, limit), limit_mode, limit):
        explore_tx(out_tx, addresses, inputs, outputs, new_txs)

    return new_txs

def tx_whole_expand_backward(prev_txs, addresses, txs, inputs, outputs, limit_mode=None, limit=math.inf):
    new_txs = {}
    candidates_per_node = (_distinct_new((tx_in.spent_tx for tx_in in tx.inputs), txs)
                           for tx in _progress(prev_txs.values()))

    for in_tx in _limit_hop(_select(candidates_per_node, limit_mode, limit), limit_mode, limit):
        explore_tx(in_tx, addresses, inputs, outputs, new_txs)

    return new_txs

def expand_fw_bw_tx_whole(addresses, num_for_hops, num_back_hops, txs, inputs, outputs, limit_mode=None, limit=math.inf):
    new_fw_txs = {}
    new_bw_txs = {}
    fw_candidates = {}
    bw_candidates = {}
    addrs = list(addresses.values()).copy()
    _log("Hop: 1, addresses to expand:", len(addresses))

    # Address by address: its forward txs, then its backward txs (the order of the draws)
    for addr, _ in _progress(addrs):
        _select([list(addr.input_txes)], limit_mode, limit, fw_candidates)
        _select([list(addr.output_txes)], limit_mode, limit, bw_candidates)

    fw_candidates = _limit_hop(fw_candidates, limit_mode, limit)
    bw_candidates = _limit_hop(bw_candidates, limit_mode, limit)

    explored = set()

    for fw_tx in fw_candidates:
        explore_tx(fw_tx, addresses, inputs, outputs, new_fw_txs, explored=explored)

    for bw_tx in bw_candidates:
        explore_tx(bw_tx, addresses, inputs, outputs, new_bw_txs, explored=explored)

    add_txs(txs, new_fw_txs, 1, "forward")
    add_txs(txs, new_bw_txs, 1, "backward")

    _log("\nForward expand:")
    for i in range(num_for_hops-1):
        _log(f"Hop: {i+2}, txs to expand: {len(new_fw_txs)}")
        new_fw_txs = tx_whole_expand_forward(new_fw_txs, addresses, txs, inputs, outputs, limit_mode=limit_mode, limit=limit)
        add_txs(txs, new_fw_txs, i + 2, "forward")

    _log("\nBackward expand:")
    for i in range(num_back_hops-1):
        _log(f"Hop: {i+2}, txs to expand: {len(new_bw_txs)}")
        new_bw_txs = tx_whole_expand_backward(new_bw_txs, addresses, txs, inputs, outputs, limit_mode=limit_mode, limit=limit)
        add_txs(txs, new_bw_txs, i + 2, "backward")

### Tx-based all over expansion

def expand_all_over_tx(addresses, num_hops, txs, inputs, outputs, limit_mode=None, limit=math.inf):
    new_txs = {}
    addrs = list(addresses.values()).copy()
    _log("Hop: 0, addresses to expand:", len(addresses))

    candidates = _select(([tx for tx in addr[0].txes] for addr in _progress(addrs)), limit_mode, limit)

    for tx in _limit_hop(candidates, limit_mode, limit):
        explore_tx(tx, addresses, inputs, outputs, new_txs)

    add_txs(txs, new_txs, 1)

    for i in range(num_hops-1):
        _log(f"Hop: {i+1}, txs to expand: {len(new_txs)}")
        hop_txs = {}
        candidates_per_node = (_distinct_new([input_tx.spent_tx for input_tx in tx.inputs] +
                                             [output_tx.spending_tx for output_tx in tx.outputs], txs)
                               for tx in _progress(new_txs.values()))

        for tx in _limit_hop(_select(candidates_per_node, limit_mode, limit), limit_mode, limit):
            explore_tx(tx, addresses, inputs, outputs, hop_txs)

        new_txs = hop_txs
        add_txs(txs, new_txs, i + 2)

### Dispatcher

VARIANTS = [
    "fw bw tx whole", "fw bw tx dedicated", "all over tx",
    "fw bw addr whole none", "fw bw addr whole same", "fw bw addr whole opposite",
    "fw bw addr dedicated", "all over addr",
]


def expand(exp_alg, addresses, txs, inputs, outputs, for_hops=None, back_hops=None, hops=None,
           limit_mode=None, limit=math.inf, new_label=-1, tx_hops=None, verbose=True):
    """Expands the seed addresses in `addresses` with the given variant (see the top of
    this module). Forward-backward variants use for_hops and back_hops, all-over
    variants hops. limit is ignored without limit_mode ("random node" or "random hop")."""
    if exp_alg not in VARIANTS:
        raise ValueError(f"Unknown expansion variant '{exp_alg}'; available: {', '.join(VARIANTS)}")
    _CTX.update(new_label=new_label, verbose=verbose, tx_hops=tx_hops)
    clear_tx_hash_cache()
    if not limit_mode:
        limit = math.inf
    _log("\nInitial number of addrs:", len(addresses))
    try:
        if exp_alg == "fw bw tx whole":
            expand_fw_bw_tx_whole(addresses, for_hops, back_hops, txs, inputs, outputs, limit_mode=limit_mode, limit=limit)
        elif exp_alg == "fw bw tx dedicated":
            expand_fw_bw_tx_dedicated(addresses, for_hops, back_hops, txs, inputs, outputs, limit_mode=limit_mode, limit=limit)
        elif exp_alg == "all over tx":
            expand_all_over_tx(addresses, hops, txs, inputs, outputs, limit_mode=limit_mode, limit=limit)
        elif exp_alg.startswith("fw bw addr whole"):
            expand_fw_bw_addr_whole(addresses, for_hops, back_hops, txs, inputs, outputs, exp_alg.split()[-1], limit_mode=limit_mode, limit=limit)
        elif exp_alg == "fw bw addr dedicated":
            expand_fw_bw_addr_dedicated(addresses, for_hops, back_hops, txs, inputs, outputs, limit_mode=limit_mode, limit=limit)
        elif exp_alg == "all over addr":
            expand_all_over_addr(addresses, hops, txs, inputs, outputs, limit_mode=limit_mode, limit=limit)
    finally:
        _CTX["tx_hops"] = None
    _log("\nFinal number of addrs:", len(addresses))
    _log("Final number of txs:", len(txs))
    _log("Final number of inputs:", len(inputs))
    _log("Final number of outputs:", len(outputs))
