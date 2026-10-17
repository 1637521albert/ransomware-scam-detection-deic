# STEP 1 (Methodology II) - EGONET EXPANSION (BlockSci environment)
#
# Expands an independent k-hop egonet around every seed address of each split
# (SPLITS_FILE, created by create_splits.py), using the expansion configuration
# EGO_CENTRIC_EXPANSION of config.py, and stores the egonets of each split in
# line-delimited JSON format: EGO_CENTRIC_EGONETS_DIR/<split>.jsonl
#
# Every egonet is expanded from its own fixed random seed, derived from DATA_SEED
# and its seed address, so the egonets are the same in every execution
# regardless of the order or the number of parallel processes
# (config.EGO_CENTRIC_EXPANSION_WORKERS). Splits whose file already exists are
# skipped.

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
from shared.experiment import git_commit, timestamp, write_json
from shared.parallel import blocksci_workers
from shared.paths import get_required_env_path

# --- INITIALIZE BLOCKSCI ---
BLOCKSCI_CONFIG = get_required_env_path("RSD_BLOCKSCI_CONFIG")
chain = blocksci.Blockchain(str(BLOCKSCI_CONFIG))

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

def limit_expansion(txs, limit):
    return random.sample(txs, min(len(txs), limit))

def explore_tx(tx, addresses, inputs, outputs, new, mode='none', selected_list=()):
    new[str(tx.hash)] = tx

    process_inputs = selected_list if mode == "input" else tx.inputs
    process_outputs = selected_list if mode == "output" else tx.outputs

    for input_tx in process_inputs:
        addr = input_tx.address
        input_id = (str(input_tx.spent_output.tx.hash), input_tx.spent_output.index)
        
        if hasattr(addr, 'address_string'):
            inputs[input_id] = (input_tx.address, input_tx, input_tx.tx)
            addr_str = addr.address_string

            if addr_str not in addresses:
                addresses[addr_str] = (addr, 0)

    for output_tx in process_outputs:
        addr = output_tx.address
        output_id = (str(output_tx.tx.hash), output_tx.index)
        
        if hasattr(addr, 'address_string'):
            outputs[output_id] = (output_tx.tx, output_tx, output_tx.address)
            addr_str = addr.address_string

            if addr_str not in addresses:
                addresses[addr_str] = (addr, 0)


# ### Tx-based all over expansion

if exp_alg == "all over tx":
    def expand(addresses, num_hops, txs, inputs, outputs, limit_mode = None, limit = math.inf):
        new_txs = {}
        addrs = list(addresses.values()).copy()
        total_txes = []

        for addr in addrs:
            txes = [tx for tx in addr[0].txes]

            if limit_mode == "random node" or limit_mode == "":
                if limit_mode == "random node":
                    txes = limit_expansion(txes, limit)

                for tx in txes:
                    explore_tx(tx, addresses, inputs, outputs, new_txs)

            total_txes += txes

        if limit_mode == "random hop":
            total_txes = limit_expansion(total_txes, limit)

        for tx in total_txes:
            explore_tx(tx, addresses, inputs, outputs, new_txs)

        txs.update(new_txs)

        for i in range(num_hops-1):
            hop_txs = {}
            candidates = []
            candidate_hashes = set()

            for tx_hash, tx in new_txs.items():
                tx_candidate_hashes = set()
                if limit_mode == 'random node':
                    tx_candidates = []

                for input_tx in tx.inputs:
                    in_tx = input_tx.spent_tx

                    if in_tx is not None:
                        tx_hash = str(in_tx.hash)

                        if tx_hash not in txs and tx_hash not in tx_candidate_hashes and tx_hash not in candidate_hashes:
                            tx_candidate_hashes.add(tx_hash)
                            if limit_mode == 'random node':
                                tx_candidates.append(in_tx)
                            else:
                                candidates.append(in_tx)
                                candidate_hashes.add(tx_hash)

                for output_tx in tx.outputs:
                    out_tx = output_tx.spending_tx

                    if out_tx is not None:
                        tx_hash = str(out_tx.hash)

                        if tx_hash not in txs and tx_hash not in tx_candidate_hashes and tx_hash not in candidate_hashes:
                            tx_candidate_hashes.add(tx_hash)
                            if limit_mode == 'random node':
                                tx_candidates.append(out_tx)
                            else:
                                candidates.append(out_tx)
                                candidate_hashes.add(tx_hash)

                if limit_mode == "random node":
                    tx_candidates = limit_expansion(tx_candidates, limit)
                    candidates += tx_candidates
                    candidate_hashes.update(str(candidate.hash) for candidate in tx_candidates)

            if limit_mode == "random hop":
                candidates = limit_expansion(candidates, limit)

            for tx in candidates:
                explore_tx(tx, addresses, inputs, outputs, hop_txs)

            new_txs = hop_txs
            txs.update(new_txs)


# ## Collecting nodes and edges features

def extract_address_features(addresses):
    results = []
    address_list = list(addresses.values())
    
    for addr, label in address_list:
        info = {
            'addr_str': addr.address_string,
            'full_type': addr.full_type,
            'class': label
        }
        results.append(info)

    df = pd.DataFrame(results)
    return df

def extract_tx_features(txs):
    tx_list = list(txs.values())
    features = []
    for tx in tx_list:
        info = {
            'hash': str(tx.hash),
            'block_height': tx.block_height,
            'fee': tx.fee,
            'is_coinbase': tx.is_coinbase,
            'locktime': tx.locktime,
            'total_size': tx.total_size,
            'version': tx.version
        }
        features.append(info)
    return pd.DataFrame(features)

def extract_input_features(inputs):
    features = []
    for input_id, inp in inputs.items():
        inp = inp[1]
        if hasattr(inp.address, 'address_string'):
            info = {
                'addr_str': inp.address.address_string,
                'tx_hash': str(inp.tx.hash),
                'age': inp.age,
                'sequence_num': inp.sequence_num,
                'value': inp.value,
                'spent_tx_hash': input_id[0],
                'spent_output_index': input_id[1]
            }
            features.append(info)
    return pd.DataFrame(features)

def extract_output_features(outputs):
    output_list = list(outputs.values())
    features = []
    for out in output_list:
        out = out[1]
        if hasattr(out.address, 'address_string'):
            info = {
                'tx_hash': str(out.tx.hash),
                'addr_str': out.address.address_string,
                'index': out.index,
                'is_spent': out.is_spent,
                'value': out.value
            }
            features.append(info)
    return pd.DataFrame(features)

def _build_even_ids_for_addresses(addr_df: pd.DataFrame, seed_addr_str: str):
    if addr_df is None or addr_df.empty:
        return addr_df, {}
    df = addr_df.copy()
    if "addr_str" not in df.columns and "address" in df.columns:
        df = df.rename(columns={"address": "addr_str"})
    df["addr_str"] = df["addr_str"].astype(str)

    if seed_addr_str not in set(df["addr_str"]):
        seed_row = {c: None for c in df.columns}
        seed_row["addr_str"] = seed_addr_str
        df = pd.concat([pd.DataFrame([seed_row]), df], ignore_index=True)

    seed_mask = (df["addr_str"] == seed_addr_str)
    seed_part = df[seed_mask].reset_index(drop=True)
    rest_part = df[~seed_mask].reset_index(drop=True)

    seed_part["node_id"] = 0
    rest_part["node_id"] = (rest_part.index + 1) * 2  

    out = pd.concat([seed_part, rest_part], ignore_index=True)
    addr2id = dict(zip(out["addr_str"].astype(str), out["node_id"].astype(int)))
    return out, addr2id

def _build_odd_ids_for_transactions(tx_df: pd.DataFrame):
    if tx_df is None or tx_df.empty:
        return tx_df, {}
    df = tx_df.copy()
    df["hash"] = df["hash"].astype(str)
    df = df.reset_index(drop=True)
    df["tx_id"] = df.index * 2 + 1
    txhash2id = dict(zip(df["hash"].astype(str), df["tx_id"].astype(int)))
    return df, txhash2id

def _to_py_scalar(v):
    if v is None or v is pd.NA:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if np.isnan(f) else f
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v

def _sanitize(o):
    if isinstance(o, dict):
        return {str(k): _sanitize(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_sanitize(v) for v in o]
    return _to_py_scalar(o)


KEEP_ADDR = ["node_id", "addr_str","full_type","class",
             "bh_year","bh_day","bh_length","bh_weight","bh_count",
             "bh_looped","bh_neighbors","bh_income","bh_label_raw"]

KEEP_TX   = ["tx_id", "hash","block_height","fee","is_coinbase","locktime",
             "total_size","version"]

KEEP_IN   = ["input_key", "addr_str","tx_hash","age","sequence_num","value",
             "spent_tx_hash","spent_output_index","tx_id","spent_tx_id"]

KEEP_OUT  = ["output_key", "tx_hash","addr_str","index","is_spent","value","tx_id"]

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

    if exp_alg.startswith('fw bw'):
        expand(addresses, for_hops, back_hops, txs, inputs, outputs,
               limit_mode=limit_mode,
               limit=(limit if (isinstance(limit_mode, str) and limit_mode != "") else math.inf))
    else:
        expand(addresses, hops, txs, inputs, outputs,
               limit_mode=limit_mode,
               limit=(limit if (isinstance(limit_mode, str) and limit_mode != "") else math.inf))

    # 2) Features
    addr_df = extract_address_features(addresses)   
    tx_df   = extract_tx_features(txs)              
    in_df   = extract_input_features(inputs)        
    out_df  = extract_output_features(outputs)      

    # 3) Address Enrichment
    if addr_df is None or addr_df.empty:
        addr_df = pd.DataFrame(columns=["addr_str", "full_type", "class"])
    if "addr_str" not in addr_df.columns and "address" in addr_df.columns:
        addr_df = addr_df.rename(columns={"address": "addr_str"})
    addr_df["addr_str"] = addr_df["addr_str"].astype(str)

    addr_df["seed_label"] = addr_df["addr_str"].map(lambda a: addresses.get(a, (None, seed_label))[1])
    
    seed_bh = {}
    if bh_meta and seed_addr_str in bh_meta:
        seed_bh = {"addr_str": seed_addr_str, **bh_meta[seed_addr_str]}
        if seed_label is not None:
            seed_bh["seed_label"] = int(seed_label)

    addr_df, addr2id = _build_even_ids_for_addresses(addr_df, seed_addr_str)
    addr_df = addr_df[["node_id", "addr_str", "full_type", "class"]]

    tx_df, txhash2id = _build_odd_ids_for_transactions(tx_df)

    # --- Inputs ---
    if in_df is not None and not in_df.empty:
        in_df = in_df.copy()
        in_df["tx_id"] = in_df["tx_hash"].astype(str).map(txhash2id)
        in_df["tx_id"] = in_df["tx_id"].fillna(-1).astype(int)
        in_df["spent_tx_id"] = in_df["spent_tx_hash"].astype(str).map(txhash2id)
        in_df["spent_tx_id"] = in_df["spent_tx_id"].fillna(-1).astype(int)
        in_df["spent_output_index"] = in_df.get("spent_output_index", pd.Series(index=in_df.index, dtype=np.float64))
        in_df["spent_output_index"] = in_df["spent_output_index"].fillna(-1).astype(int)
        in_df.loc[in_df["spent_tx_id"] == -1, "spent_output_index"] = -1
        in_df["input_key"] = list(zip(in_df["spent_tx_id"].astype(int), in_df["spent_output_index"].astype(int)))

    # --- Outputs ---
    if out_df is not None and not out_df.empty:
        out_df = out_df.copy()
        out_df["tx_id"] = out_df["tx_hash"].astype(str).map(txhash2id).astype(int)
        out_df["output_key"] = [[_to_py_scalar(x), _to_py_scalar(y)]
                                for x, y in zip(out_df["tx_id"], out_df["index"])]

    # 5) Common Metadata
    for df_ in (addr_df, tx_df, in_df, out_df):
        if df_ is None or df_.empty: 
            continue
        df_["seed"]       = seed_addr_str
        df_["split"]      = split_name
        df_["exp_alg"]    = exp_alg
        df_["limit_mode"] = (limit_mode or "none")
        if exp_alg.startswith('fw bw'):
            df_["forward_hops"] = for_hops; df_["backward_hops"] = back_hops
        else:
            df_["hops"] = hops

    # 6) Column Selection
    def pick(df_, cols):
        if df_ is None or df_.empty:
            return pd.DataFrame(columns=cols, dtype=object)
        keep = [c for c in cols if c in df_.columns]
        return df_[keep]

    addr_df = pick(addr_df, KEEP_ADDR)
    tx_df   = pick(tx_df,   KEEP_TX)
    in_df   = pick(in_df,   KEEP_IN)
    out_df  = pick(out_df,  KEEP_OUT)

    # 7) Serialization -> JSONL
    def rec(df_):
        if df_ is None or df_.empty:
            return []
        df2 = df_.astype(object)
        df2 = df2.where(pd.notna(df2), None)
        return df2.to_dict(orient="records")

    graph_json = {
        "graph_id": bh_map.get(seed_addr_str),  
        "seed": seed_addr_str,
        "class": int(seed_label) if seed_label is not None else None,
        "seed_bh": seed_bh,
        "sizes": {
            "n_addr": len(addr_df),
            "n_tx":   0 if tx_df is None else len(tx_df),
            "n_inputs": 0 if in_df is None else len(in_df),
            "n_outputs": 0 if out_df is None else len(out_df)
        },
        "addresses":    rec(addr_df),
        "transactions": rec(tx_df),
        "inputs":       rec(in_df),
        "outputs":      rec(out_df)
    }

    graph_json_py = _sanitize(graph_json)
    return graph_json_py


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
            lines = pool.imap(expand_seed, tasks, chunksize=8) if pool else map(expand_seed, tasks)

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
