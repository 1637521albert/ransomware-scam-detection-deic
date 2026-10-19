# STEP 1 (Methodology II) - EGONET EXPANSION (requires BlockSci)
#
# Expands an independent k-hop egonet around every seed address of each split
# (SPLITS_FILE, created by create_splits.py), using the expansion configuration
# EGO_CENTRIC_EXPANSION of config.py, and stores the egonets of each split in
# line-delimited JSON format: EGO_CENTRIC_EGONETS_DIR/<split>.jsonl
# Every expansion variant of the multi-instance methodology is available, with the
# same algorithms (shared/expansion.py). Every transaction records the hop where it
# was added to the egonet ("hop": 1 for the transactions of the seed, then 2, 3...),
# and in the forward-backward variants its direction ("hop_direction": "forward",
# "backward", or "both" for a tx of the first hop found in both directions).
#
# Every egonet is expanded from its own fixed random seed, derived from DATA_SEED
# and its seed address, so the egonets are the same in every execution
# regardless of the order or the number of parallel processes
# (config.EGO_CENTRIC_EXPANSION_WORKERS). Every egonet is one task of the forked workers
# (so a seed with a large egonet does not hold back others), and its record is built
# directly from what the expansion stored (address strings and tx hashes are not
# encoded again). Splits whose file already exists are skipped.

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# One worker per core: single-threaded numerical libraries (before importing them)
from shared.threads import limit_threads

import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import random
import math
import multiprocessing
import os
import zlib
import blocksci

import config
from shared import expansion as shared_expansion
from shared.experiment import git_commit, timestamp, write_json
from shared.parallel import blocksci_workers
from shared.paths import get_required_env_path

# --- INITIALIZE BLOCKSCI ---
BLOCKSCI_CONFIG = get_required_env_path("RSD_BLOCKSCI_CONFIG")
chain = blocksci.Blockchain(str(BLOCKSCI_CONFIG))
shared_expansion.set_chain(chain)

# Optional external mapping from address to BitcoinHeist graph id
MAPPING_PATH = PROJECT_ROOT / "Data" / "mapping_addr_bh.json"

# ## Parameters

EXPANSION = config.EGO_CENTRIC_EXPANSION
exp_alg = f"{EXPANSION['direction']}{EXPANSION['graph']}{EXPANSION['address']}{EXPANSION['side_direction']}"
limit_mode = EXPANSION["limit_mode"]
limit = EXPANSION["limit"] if limit_mode != "" else math.inf
hops = EXPANSION.get("hops", EXPANSION.get("forward_hops"))
for_hops = EXPANSION.get("forward_hops", hops)
back_hops = EXPANSION.get("backward_hops", hops)

data_path = config.EGO_CENTRIC_EGONETS_DIR


# ### Collecting addresses

def load_split_seeds():
    """Seed addresses of every split as a list of (address, label) in file order,
    and the BitcoinHeist family of every seed."""
    if not config.SPLITS_FILE.exists():
        raise FileNotFoundError(f"Missing {config.SPLITS_FILE}: run create_splits.py first.")

    splits = pd.read_csv(config.SPLITS_FILE)
    seeds = {}
    for split in config.SPLITS:
        rows = splits[splits["split"] == split]
        seeds[split] = [(addr_str, int(label)) for addr_str, label in zip(rows["address"], rows["label"])]
    meta = {addr_str: {"bh_family": family} for addr_str, family in zip(splits["address"], splits["family"])}
    return seeds, meta


# ## GRAPH EXPANSION

# ## Collecting nodes and edges features
#
# The records of an egonet are built directly from what the expansion stored (address
# strings, tx hashes and the BlockSci inputs and outputs), reading from BlockSci only the
# values that were not read yet. Addresses get even ids (the seed is 0) and transactions
# odd ids, in the order of the expansion. Inputs and outputs without an address string
# are not exported; an input whose spent tx is not in the egonet has spent_tx_id and
# spent_output_index -1.

def _sanitize(o):
    # Plain JSON values (for the BitcoinHeist metadata of the seed)
    if isinstance(o, dict):
        return {str(k): _sanitize(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_sanitize(v) for v in o]
    if o is None:
        return None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o

def address_records(addresses, seed_addr_str):
    """Address records: the seed first (id 0), then the other addresses (ids 2, 4...)."""
    rows = []
    for key, (addr, label) in addresses.items():
        # The keys are the address strings, except the seed's (the string of the splits file)
        addr_str = addr.address_string if key == seed_addr_str else key
        rows.append((addr_str, addr.full_type, label))
    if seed_addr_str not in {addr_str for addr_str, _, _ in rows}:
        rows.insert(0, (seed_addr_str, None, None))
    seed_rows = [r for r in rows if r[0] == seed_addr_str]
    other_rows = [r for r in rows if r[0] != seed_addr_str]
    records = [{"node_id": 0, "addr_str": a, "full_type": f, "class": c} for a, f, c in seed_rows]
    records += [{"node_id": (i + 1) * 2, "addr_str": a, "full_type": f, "class": c}
                for i, (a, f, c) in enumerate(other_rows)]
    return records

def tx_records(txs, tx_hops):
    """Transaction records (ids 1, 3, 5...) and the id of every tx hash."""
    with_direction = any(direction is not None for _, direction in tx_hops.values())
    records, txhash2id = [], {}
    for i, (h, tx) in enumerate(txs.items()):
        tx_id = i * 2 + 1
        txhash2id[h] = tx_id
        record = {"tx_id": tx_id, "hash": h, "block_height": tx.block_height, "fee": tx.fee,
                  "is_coinbase": tx.is_coinbase, "locktime": tx.locktime, "total_size": tx.total_size,
                  "version": tx.version, "hop": tx_hops[h][0]}
        if with_direction:
            # Direction of the hop, only in the forward-backward variants
            record["hop_direction"] = tx_hops[h][1]
        records.append(record)
    return records, txhash2id

def input_records(inputs, txhash2id):
    records = []
    for (spent_hash, spent_index), (addr_str, inp, _) in inputs.items():
        if addr_str is None:
            continue
        tx_hash = shared_expansion.input_tx_hash(inp)
        tx_id = txhash2id.get(tx_hash, -1)
        spent_tx_id = txhash2id.get(spent_hash, -1)
        spent_output_index = spent_index if spent_tx_id != -1 else -1
        records.append({"input_key": [spent_tx_id, spent_output_index], "addr_str": addr_str, "tx_hash": tx_hash,
                        "age": inp.age, "sequence_num": inp.sequence_num, "value": inp.value,
                        "spent_tx_hash": spent_hash, "spent_output_index": spent_output_index,
                        "tx_id": tx_id, "spent_tx_id": spent_tx_id})
    return records

def output_records(outputs, txhash2id):
    records = []
    for (tx_hash, index), (_, out, addr_str) in outputs.items():
        if addr_str is None:
            continue
        tx_id = txhash2id[tx_hash]
        records.append({"output_key": [tx_id, index], "tx_hash": tx_hash, "addr_str": addr_str, "index": index,
                        "is_spent": out.is_spent, "value": out.value, "tx_id": tx_id})
    return records

def load_bh_mapping(mapping_path) -> dict:
    # Optional external mapping from address to BitcoinHeist graph id
    if not Path(mapping_path).exists():
        return {}
    with open(mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)

bh_map = load_bh_mapping(MAPPING_PATH)

def run_seed_to_jsonl(seed_addr_str: str,
                      seed_tuple: tuple,
                      split_name: str,
                      bh_meta: dict):
    addr_obj, seed_label = seed_tuple

    if addr_obj is None:
        try:
            addr_obj = chain.address_from_string(seed_addr_str)
        except Exception:
            addr_obj = None
    if addr_obj is None:
        print(f"[SKIP] {seed_addr_str}: cannot build BlockSci object")
        return None

    # 1) Expansion
    addresses = {seed_addr_str: (addr_obj, seed_label)}
    txs, inputs, outputs = {}, {}, {}
    tx_hops = {}

    # Same algorithms as the multi-instance methodology. The tx hashes and address strings
    # memoized during the expansion are reused by the records, then dropped, so a worker
    # does not accumulate them
    shared_expansion.expand(exp_alg, addresses, txs, inputs, outputs, for_hops=for_hops, back_hops=back_hops,
                            hops=hops, limit_mode=limit_mode, limit=limit, new_label=0, tx_hops=tx_hops,
                            verbose=False)

    # 2) Records
    addr_records = address_records(addresses, seed_addr_str)
    tx_recs, txhash2id = tx_records(txs, tx_hops)
    in_records = input_records(inputs, txhash2id)
    out_records = output_records(outputs, txhash2id)
    shared_expansion.clear_tx_hash_cache()

    seed_bh = {}
    if bh_meta and seed_addr_str in bh_meta:
        seed_bh = {"addr_str": seed_addr_str, **bh_meta[seed_addr_str]}
        if seed_label is not None:
            seed_bh["seed_label"] = int(seed_label)

    return {
        "graph_id": bh_map.get(seed_addr_str),
        "seed": seed_addr_str,
        "class": int(seed_label) if seed_label is not None else None,
        "seed_bh": _sanitize(seed_bh),
        "sizes": {
            "n_addr": len(addr_records),
            "n_tx": len(tx_recs),
            "n_inputs": len(in_records),
            "n_outputs": len(out_records),
        },
        "addresses": addr_records,
        "transactions": tx_recs,
        "inputs": in_records,
        "outputs": out_records,
    }


# ## Exporting data

def seed_random_state(seed_addr_str):
    """Random state of one egonet, from DATA_SEED and its seed address."""
    return (config.DATA_SEED * 1000003 + zlib.crc32(seed_addr_str.encode())) % (2 ** 32)


def expand_seed(task):
    """Expands the egonet of one seed address and returns it as a JSON line (or None)."""
    seed_addr_str, seed_label, split_name = task
    state = seed_random_state(seed_addr_str)
    random.seed(state)
    np.random.seed(state)
    graph_data = run_seed_to_jsonl(seed_addr_str, (None, seed_label), split_name, bh_meta)
    return json.dumps(graph_data, ensure_ascii=False) if graph_data else None


if __name__ == "__main__":
    limit_threads()
    data_path.mkdir(parents=True, exist_ok=True)
    write_json(data_path / "expansion_config.json", {
        "created": timestamp(),
        "git_commit": git_commit(config.PROJECT_ROOT),
        "expansion": EXPANSION,
        "exp_alg": exp_alg,
        "data_seed": config.DATA_SEED,
        "random_state": "random.seed((DATA_SEED * 1000003 + crc32(seed address)) mod 2^32) before expanding each egonet",
        "splits_file": str(config.SPLITS_FILE),
    })

    split_seeds, bh_meta = load_split_seeds()
    workers = blocksci_workers(config.EGO_CENTRIC_EXPANSION_WORKERS)
    print(f"{workers} workers ({config.CPU_COUNT} cores)")

    # Forked workers share the loaded BlockSci chain; results come back in the order of the seeds
    pool = multiprocessing.get_context("fork").Pool(workers) if workers > 1 else None
    try:
        for split_name in config.SPLITS:
            jsonl_path = data_path / f"{split_name}.jsonl"
            if jsonl_path.exists():
                print(f"{split_name}: already expanded, skipping ({jsonl_path})")
                continue

            tasks = [(addr_str, label, split_name) for addr_str, label in split_seeds[split_name]]
            # One egonet per task, so a seed with a large egonet does not hold back others;
            # the lines come back in the order of the seeds
            lines = pool.imap(expand_seed, tasks, chunksize=1) if pool else map(expand_seed, tasks)

            # Written to a temporary file first, so an interrupted split is never taken as complete
            tmp_path = jsonl_path.with_name(jsonl_path.name + ".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                for line in tqdm(lines, total=len(tasks), desc=f"Expanding {split_name} ({workers} workers)"):
                    if line:
                        f.write(line + "\n")
            os.replace(str(tmp_path), str(jsonl_path))
    finally:
        if pool:
            pool.close()
            pool.join()

    print(f"\n[SUCCESS] All files successfully generated at: {data_path}")
