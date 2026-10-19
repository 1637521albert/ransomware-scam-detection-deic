# Expansion strategies of the address-transaction graph, shared by both methodologies.
#
# Inductive Multi-Instance Address Classification expands the seed addresses of a
# split jointly into one graph, and Inductive Ego-Centric Address Classification
# expands every seed address into its own egonet, with exactly the same algorithms
# (this module). Variants (direction, central unit, scope, side addresses):
#   fw bw tx whole, fw bw tx dedicated, all over tx,
#   fw bw addr whole none / same / opposite, fw bw addr dedicated, all over addr
# The multi-instance methodology runs every variant in parallel (its expansion.py), with
# the order of the random draws and insertions of these sequential algorithms, so it
# gives the same graph; the ego-centric methodology runs them as they are, one egonet
# per worker.
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
# shares its string with its pubkey-hash address). The cache, like the tx hashes, is
# cleared at the start of every expansion.
_address_strings = {}
_MISSING = object()


def _address_string_memoized(addr):
    key = (addr.address_num, addr.type)
    addr_str = _address_strings.get(key, _MISSING)
    if addr_str is _MISSING:
        addr_str = _address_strings[key] = getattr(addr, 'address_string', None)
    return addr_str


def _address_string_direct(addr):
    return getattr(addr, 'address_string', None)


_address_string = _address_string_memoized


def address_string(addr):
    # Encodes the address once instead of hasattr() + repeated .address_string; falls back
    # to encoding every time if addresses cannot be identified
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

### Addr-based dedicated inputs/outputs forward-backward expansion

def addr_dedicated_expand_forward(addresses, prev_addresses, txs, inputs, outputs, limit_mode=None, limit=math.inf, hop=None):
    new_addresses = {}
    candidates = {}

    for addr, _ in _progress(prev_addresses.values()):
        addr_candidates = {}

        for inp in addr.inputs:
            tx_id = input_tx_hash(inp)

            if tx_id not in txs:
                addr_candidates.setdefault(tx_id, []).append(inp)

        if limit_mode == "random node":
            addr_candidates = {tx_id: addr_candidates[tx_id] for tx_id in limit_expansion(addr_candidates.keys(), limit)}

        candidates.update(addr_candidates)

    if limit_mode == "random hop":
        candidates = {tx_id: candidates[tx_id] for tx_id in limit_expansion(candidates.keys(), limit)}

    for tx_id, tx_inputs in candidates.items():
        tx = lookup_tx(tx_inputs[0].tx, tx_id)
        add_tx(txs, tx_id, tx, hop, "forward")

        for inp in tx_inputs:
            input_id = (spent_tx_hash(inp), inp.spent_output.index)
            inputs[input_id] = (address_string(inp.address), inp, tx)

        for out in tx.outputs:
            addr = out.address
            addr_str = address_string(addr)
            outputs[(tx_id, out.index)] = (tx, out, addr_str)
            if addr_str is not None and addr_str not in addresses:
                new_addresses[addr_str] = _new_address(addr)

    return new_addresses

def addr_dedicated_expand_backward(addresses, prev_addresses, txs, inputs, outputs, limit_mode=None, limit=math.inf, hop=None):
    new_addresses = {}
    candidates = {}

    for addr, _ in _progress(prev_addresses.values()):
        addr_candidates = {}

        for out in addr.outputs:
            tx_id = output_tx_hash(out)

            if tx_id not in txs:
                addr_candidates.setdefault(tx_id, []).append(out)

        if limit_mode == "random node":
            addr_candidates = {tx_id: addr_candidates[tx_id] for tx_id in limit_expansion(addr_candidates.keys(), limit)}

        candidates.update(addr_candidates)

    if limit_mode == "random hop":
        candidates = {tx_id: candidates[tx_id] for tx_id in limit_expansion(candidates.keys(), limit)}

    for tx_id, tx_outputs in candidates.items():
        tx = lookup_tx(tx_outputs[0].tx, tx_id)
        add_tx(txs, tx_id, tx, hop, "backward")

        for out in tx_outputs:
            outputs[(tx_id, out.index)] = (tx, out, address_string(out.address))

        for inp in tx.inputs:
            addr = inp.address
            addr_str = address_string(addr)
            input_id = (spent_tx_hash(inp), inp.spent_output.index)
            inputs[input_id] = (addr_str, inp, tx)
            if addr_str is not None and addr_str not in addresses:
                new_addresses[addr_str] = _new_address(addr)

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
        if limit_mode == "random node":
            in_txs = limit_expansion([in_tx for in_tx in addr.input_txes], limit)
        else:
            in_txs = addr.input_txes

        for in_tx in in_txs:
            tx_id = tx_hash(in_tx)

            if tx_id not in txs:
                candidates[tx_id] = in_tx

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
        if limit_mode == "random node":
            out_txs = limit_expansion([out_tx for out_tx in addr.output_txes], limit)
        else:
            out_txs = addr.output_txes

        for out_tx in out_txs:
            tx_id = tx_hash(out_tx)

            if tx_id not in txs:
                candidates[tx_id] = out_tx

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
    if mode == 'none':
        new_fw = addresses.copy()
        new_bw = addresses.copy()
        for i in range(num_for_hops):
            _log(f"\nHop {i + 1}:\n\nForward phase:")
            new_fw, side_fw = addr_whole_expand_forward(addresses, new_fw, txs, inputs, outputs, limit_mode=limit_mode, limit=limit, hop=i + 1)
            addresses.update({**new_fw, **side_fw})

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

        for i in range(num_for_hops):
            _log(f"Hop {i}, addresses to expand:", len(new_fw) + len(side_fw) + len(new_bw) + len(side_bw))

            _log("\nForward phase:")
            new_fw, side_fw = addr_whole_expand_forward(addresses, new_fw, txs, inputs, outputs, limit_mode=limit_mode, limit=limit, hop=i + 1)
            addresses.update({**new_fw, **side_fw})

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
        candidates = []

        for address, (addr, label) in _progress(new_addresses.items()):
            if limit_mode == 'random node':
                txes = limit_expansion([tx for tx in addr.txes], limit)
            else:
                txes = addr.txes

            for tx in txes:
                if tx_hash(tx) not in txs:
                    candidates.append(tx)

        if limit_mode == "random hop":
            candidates = limit_expansion(candidates, limit)

        # Candidates repeat when frontier addresses share a tx; reprocessing
        # one rewrites the same entries, so each tx is processed once
        processed = set()

        for tx in candidates:
            if tx.index in processed:
                continue
            processed.add(tx.index)

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
    candidates = {}

    for tx in _progress(prev_txs.values()):
        tx_candidates = {}

        for tx_out in tx.outputs:
            out_tx = tx_out.spending_tx

            if out_tx is not None:
                tx_id = tx_hash(out_tx)

                if tx_id not in txs:
                    tx_candidates.setdefault(tx_id, []).append(tx_out.spending_input)

        if limit_mode == "random node":
            tx_candidates = {tx_id: tx_candidates[tx_id] for tx_id in limit_expansion(tx_candidates.keys(), limit)}

        candidates.update(tx_candidates)

    if limit_mode == "random hop":
        candidates = {tx_id: candidates[tx_id] for tx_id in limit_expansion(candidates.keys(), limit)}

    for tx_id, tx_inputs in candidates.items():
        tx = lookup_tx(tx_inputs[0].tx, tx_id)
        explore_tx(tx, addresses, inputs, outputs, new_txs, mode='input', selected_list=tx_inputs)

    return new_txs

def tx_dedicated_expand_backward(prev_txs, addresses, txs, inputs, outputs, limit_mode=None, limit=math.inf):
    new_txs = {}
    candidates = {}

    for tx in _progress(prev_txs.values()):
        tx_candidates = {}

        for tx_in in tx.inputs:
            in_tx = tx_in.spent_tx

            if in_tx is not None:
                tx_id = tx_hash(in_tx)

                if tx_id not in txs:
                    tx_candidates.setdefault(tx_id, []).append(tx_in.spent_output)

        if limit_mode == "random node":
            tx_candidates = {tx_id: tx_candidates[tx_id] for tx_id in limit_expansion(tx_candidates.keys(), limit)}

        candidates.update(tx_candidates)

    if limit_mode == "random hop":
        candidates = {tx_id: candidates[tx_id] for tx_id in limit_expansion(candidates.keys(), limit)}

    for tx_id, tx_outputs in candidates.items():
        tx = lookup_tx(tx_outputs[0].tx, tx_id)
        explore_tx(tx, addresses, inputs, outputs, new_txs, mode='output', selected_list=tx_outputs)

    return new_txs

def expand_fw_bw_tx_dedicated(addresses, num_for_hops, num_back_hops, txs, inputs, outputs, limit_mode=None, limit=math.inf):
    new_fw_txs = {}
    new_bw_txs = {}
    fw_candidates = {}
    bw_candidates = {}
    addrs = list(addresses.values()).copy()
    _log("Hop: 1, addresses to expand:", len(addresses))

    for addr, _ in _progress(addrs):
        fw_addr_candidates = {}
        for inp in addr.inputs:
            tx_id = input_tx_hash(inp)

            if tx_id not in fw_candidates:
                fw_addr_candidates.setdefault(tx_id, []).append(inp)

        if limit_mode == "random node":
            fw_addr_candidates = {tx_id: fw_addr_candidates[tx_id] for tx_id in limit_expansion(fw_addr_candidates.keys(), limit)}

        fw_candidates.update(fw_addr_candidates)

        bw_addr_candidates = {}
        for out in addr.outputs:
            tx_id = output_tx_hash(out)

            if (tx_id, out.index) not in outputs and tx_id not in bw_candidates:
                bw_addr_candidates.setdefault(tx_id, []).append(out)

        if limit_mode == "random node":
            bw_addr_candidates = {tx_id: bw_addr_candidates[tx_id] for tx_id in limit_expansion(bw_addr_candidates.keys(), limit)}

        bw_candidates.update(bw_addr_candidates)

    if limit_mode == "random hop":
        fw_candidates = {tx_id: fw_candidates[tx_id] for tx_id in limit_expansion(fw_candidates.keys(), limit)}
        bw_candidates = {tx_id: bw_candidates[tx_id] for tx_id in limit_expansion(bw_candidates.keys(), limit)}

    for fw_tx_id, fw_inputs in fw_candidates.items():
        fw_tx = lookup_tx(fw_inputs[0].tx, fw_tx_id)
        explore_tx(fw_tx, addresses, inputs, outputs, new_fw_txs, mode='input', selected_list=fw_inputs)

    for bw_tx_id, bw_outputs in bw_candidates.items():
        bw_tx = lookup_tx(bw_outputs[0].tx, bw_tx_id)
        explore_tx(bw_tx, addresses, inputs, outputs, new_bw_txs, mode='output', selected_list=bw_outputs)

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
    candidates = []
    candidate_hashes = set()

    for tx in _progress(prev_txs.values()):
        tx_candidates = []
        tx_candidate_hashes = set()

        for tx_out in tx.outputs:
            out_tx = tx_out.spending_tx

            if out_tx is not None:
                tx_id = tx_hash(out_tx)

                if tx_id not in txs and tx_id not in tx_candidate_hashes and tx_id not in candidate_hashes:
                    tx_candidates.append(out_tx)
                    tx_candidate_hashes.add(tx_id)

        if limit_mode == "random node":
            tx_candidates = limit_expansion(tx_candidates, limit)

        candidates += tx_candidates
        candidate_hashes.update(tx_hash(candidate) for candidate in tx_candidates)

    if limit_mode == "random hop":
        candidates = limit_expansion(candidates, limit)

    for out_tx in candidates:
        explore_tx(out_tx, addresses, inputs, outputs, new_txs)

    return new_txs

def tx_whole_expand_backward(prev_txs, addresses, txs, inputs, outputs, limit_mode=None, limit=math.inf):
    new_txs = {}
    candidates = []
    candidate_hashes = set()

    for tx in _progress(prev_txs.values()):
        tx_candidates = []
        tx_candidate_hashes = set()

        for tx_in in tx.inputs:
            in_tx = tx_in.spent_tx

            if in_tx is not None:
                tx_id = tx_hash(in_tx)

                if tx_id not in txs and tx_id not in tx_candidate_hashes and tx_id not in candidate_hashes:
                    tx_candidates.append(in_tx)
                    tx_candidate_hashes.add(tx_id)

        if limit_mode == "random node":
            tx_candidates = limit_expansion(tx_candidates, limit)

        candidates += tx_candidates
        candidate_hashes.update(tx_hash(candidate) for candidate in tx_candidates)

    if limit_mode == "random hop":
        candidates = limit_expansion(candidates, limit)

    for in_tx in candidates:
        explore_tx(in_tx, addresses, inputs, outputs, new_txs)

    return new_txs

def expand_fw_bw_tx_whole(addresses, num_for_hops, num_back_hops, txs, inputs, outputs, limit_mode=None, limit=math.inf):
    new_fw_txs = {}
    new_bw_txs = {}
    fw_candidates = []
    bw_candidates = []
    addrs = list(addresses.values()).copy()
    _log("Hop: 1, addresses to expand:", len(addresses))

    for addr, _ in _progress(addrs):
        if limit_mode == "random node":
            fw_addr_candidates = limit_expansion(list(addr.input_txes), limit)
            bw_addr_candidates = limit_expansion(list(addr.output_txes), limit)
        else:
            fw_addr_candidates = list(addr.input_txes)
            bw_addr_candidates = list(addr.output_txes)

        fw_candidates += fw_addr_candidates
        bw_candidates += bw_addr_candidates

    if limit_mode == "random hop":
        fw_candidates = limit_expansion(fw_candidates, limit)
        bw_candidates = limit_expansion(bw_candidates, limit)

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
    explored = set()
    addrs = list(addresses.values()).copy()
    total_txes = []
    _log("Hop: 0, addresses to expand:", len(addresses))

    for addr in _progress(addrs):
        txes = [tx for tx in addr[0].txes]

        if limit_mode == "random node" or limit_mode == "":

            if limit_mode == "random node":
                txes = limit_expansion(txes, limit)

            for tx in txes:
                explore_tx(tx, addresses, inputs, outputs, new_txs, explored=explored)

        total_txes += txes

    if limit_mode == "random hop":
        total_txes = limit_expansion(total_txes, limit)

    for tx in total_txes:
        explore_tx(tx, addresses, inputs, outputs, new_txs, explored=explored)

    add_txs(txs, new_txs, 1)

    for i in range(num_hops-1):
        _log(f"Hop: {i+1}, txs to expand: {len(new_txs)}")
        hop_txs = {}
        candidates = []
        candidate_hashes = set()

        for tx in _progress(new_txs.values()):
            tx_candidate_hashes = set()

            if limit_mode == 'random node':
                tx_candidates = []

            for input_tx in tx.inputs:
                in_tx = input_tx.spent_tx

                if in_tx is not None:
                    in_hash = tx_hash(in_tx)

                    if in_hash not in txs and in_hash not in tx_candidate_hashes and in_hash not in candidate_hashes:
                        tx_candidate_hashes.add(in_hash)

                        if limit_mode == 'random node':
                            tx_candidates.append(in_tx)

                        else:
                            candidates.append(in_tx)
                            candidate_hashes.add(in_hash)

            for output_tx in tx.outputs:
                out_tx = output_tx.spending_tx

                if out_tx is not None:
                    out_hash = tx_hash(out_tx)

                    if out_hash not in txs and out_hash not in tx_candidate_hashes and out_hash not in candidate_hashes:
                        tx_candidate_hashes.add(out_hash)

                        if limit_mode == 'random node':
                            tx_candidates.append(out_tx)

                        else:
                            candidates.append(out_tx)
                            candidate_hashes.add(out_hash)

            if limit_mode == "random node":
                tx_candidates = limit_expansion(tx_candidates, limit)
                candidates += tx_candidates
                candidate_hashes.update(tx_hash(candidate) for candidate in tx_candidates)

        if limit_mode == "random hop":
            candidates = limit_expansion(candidates, limit)

        for tx in candidates:
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
