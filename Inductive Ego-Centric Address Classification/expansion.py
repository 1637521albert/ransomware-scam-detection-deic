#!/usr/bin/env python
# coding: utf-8

# # RANSOMWARE ADDRESSES DETECTION
# 
# ## Import libraries
# 

import sys
from pathlib import Path

# --- PROJECT ROOT RESOLUTION ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import random
import math
import os
import blocksci

from shared.paths import (
    build_run_dir,
    build_run_name,
    ensure_split_dirs,
    get_required_env_path,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# --- INITIALIZE BLOCKSCI & ENVIRONMENT PATHS ---
BLOCKSCI_CONFIG = get_required_env_path("RSD_BLOCKSCI_CONFIG")
BITCOINHEIST_CSV = get_required_env_path("RSD_BITCOINHEIST_CSV")
chain = blocksci.Blockchain(str(BLOCKSCI_CONFIG))

# Relative path for external mappings
MAPPING_PATH = PROJECT_ROOT / "Data" / "mapping_addr_bh.json"


# ## Parameters
# 

def get_parameters():
    """
    print("Type of seed addresses:\n---------------------------------")
    print("1. Licit\n2. Illicit\n3. Licit and Illicit (50/50)")
    seed_options = {"1": "licit", "2": "illicit", "3": "licit and illicit"}
    seed = seed_options[input("Option: ")]
    """

    print("\nDirection of the expansion:\n---------------------------------")
    print("1. Forward Backward\n2. All over")
    direction_options = {"1": "fw bw", "2": "all over"}
    direction = direction_options[input("Option: ")]

    print("\nApproach of the expansion:\n---------------------------------")
    print("1. Transaction-based\n2. Address-based")
    graph_options = {"1": " tx", "2": " addr"}
    graph = graph_options[input("Option: ")]

    if direction == "fw bw":
        print("\nTransaction addresses proportion of the expansion:\n---------------------------------")
        print("1. Whole\n2. Dedicated")
        address_options = {"1": " whole", "2": " dedicated"}
        address = address_options[input("Option: ")]

        if address == " whole" and graph == " addr":
            print("\nSide addresses direction of the expansion:\n---------------------------------")
            print("1. None\n2. Same\n3. Opposite")
            side_direction_options = {"1": " none", "2": " same", "3": " opposite"}
            side_direction = side_direction_options[input("Option: ")]
        else:
            side_direction = ""
    else:
        address = ""
        side_direction = ""

    exp_alg = f'{direction}{graph}{address}{side_direction}'

    print("\nLimit mode:\n---------------------------------")
    print("1. Random node\n2. Random hop\n3. None")
    limit_mode_options = {"1": "random node", "2": "random hop", "3": ""}
    limit_mode = limit_mode_options[input("Option: ")]

    if limit_mode != "":
        print("\nLimit value:\n---------------------------------")
        limit = int(input("Option: "))
        
    else:
        limit = "no"

    if exp_alg.startswith('fw bw'):
        print("\nNumber of forward hops:\n---------------------------------")
        for_hops = int(input("Option: "))
        
        print("\nNumber of backward hops:\n---------------------------------")
        back_hops = int(input("Option: "))
        hops = for_hops

    else:
        print("\nNumber of hops:\n---------------------------------")
        hops = int(input("Option: "))
        for_hops = hops
        back_hops = hops

    print("\nNumber of train samples:\n---------------------------------")
    train_samples = int(input("Option: "))

    print("\nNumber of validation samples:\n---------------------------------")
    val_samples = int(input("Option: "))

    print("\nNumber of test samples:\n---------------------------------")
    test_samples = int(input("Option: "))

    # ── Illicit proportions per split setup
    def _read_p(prompt, default=0.5):
        s = input(prompt).strip()
        if s == "":
            return float(default)
        p = float(s)
        if not (0.0 <= p <= 1.0):
            raise ValueError("Proportion must be between 0 and 1.")
        return p

    print("\nIllicit proportion setup:\n---------------------------------")
    print("1. Same proportion for all splits (default 0.5)")
    print("2. Custom proportion per split")
    prop_mode = input("Option: ").strip()

    if prop_mode == "2":
        print("\nProportion of ILLICIT seeds in TRAIN (0..1, blank=0.5):")
        p_illicit['train'] = _read_p("p_illicit_train = ", 0.5)
        print("\nProportion of ILLICIT seeds in VAL (0..1, blank=0.5):")
        p_illicit['val']   = _read_p("p_illicit_val = ", 0.5)
        print("\nProportion of ILLICIT seeds in TEST (0..1, blank=0.5):")
        p_illicit['test']  = _read_p("p_illicit_test = ", 0.5)
    else:
        print("\nProportion of ILLICIT seeds for ALL splits (0..1, blank=0.5):")
        p_all = _read_p("p_illicit_all = ", 0.5)
        p_illicit = p_all

    space = "" if limit_mode == "" else " "

    # --- RELATIVE RUN DIRECTORY GENERATION ---
    run_name = build_run_name(
        train_samples,
        val_samples,
        test_samples,
        exp_alg,
        hops,
        limit,
        space,
        limit_mode,
    )
    data_path = build_run_dir(__file__, run_name)
    ensure_split_dirs(data_path)

    config = {
        "direction": direction,
        "graph": graph}

    if address != "":
        config["address"] = address

    if side_direction != "":
        config["side_direction"] = side_direction 

    config["limit_mode"] = limit_mode

    if limit_mode != "":
        config["limit"] = limit   

    if exp_alg.startswith('fw bw'):
        config["forward_hops"] = for_hops
        config["backward_hops"] = back_hops

    else:
        config["hops"] = hops

    config["train_samples"] = train_samples
    config["val_samples"] = val_samples
    config["test_samples"] = test_samples

    config_path = data_path / "graph_config.json"
    with open(config_path, "w") as config_file:
        json.dump(config, config_file, indent=4)
        
    return space, exp_alg, limit_mode, limit, for_hops, back_hops, hops, train_samples, val_samples, test_samples, data_path

space, exp_alg, limit_mode, limit, for_hops, back_hops, hops, train_samples, val_samples, test_samples, data_path = get_parameters()


# ### Collecting addresses

def get_data(
    bitcoinheist_csv: str,
    n_train: int, n_val: int, n_test: int,
    p_illicit_train: float = 0.5,
    p_illicit_val: float = 0.5,
    p_illicit_test: float = 0.5,
    seed: int = None,
    chunksize: int = 55000,
):
    """
    Reads bitcoinheist.csv in chunks and randomly selects the necessary addresses
    to cover all splits (train/val/test) with given proportions. 
    Loads the corresponding BlockSci object for each selected address.
    """
    def _targets(n_total, p):
        n_il = int(round(n_total * float(p)))
        n_li = n_total - n_il
        return n_li, n_il

    need_li_tr, need_il_tr = _targets(n_train, p_illicit_train)
    need_li_va, need_il_va = _targets(n_val,   p_illicit_val)
    need_li_te, need_il_te = _targets(n_test,  p_illicit_test)

    need_licit   = need_li_tr + need_li_va + need_li_te
    need_illicit = need_il_tr + need_il_va + need_il_te

    rng = random.Random(seed)
    seen = set()

    addresses  = {}  
    bh_meta    = {}
    label_map  = {}
    licit_done = 0
    illicit_done = 0

    usecols = ["address","label","year","day","length","weight","count","looped","neighbors","income"]
    for chunk in pd.read_csv(bitcoinheist_csv, usecols=usecols, chunksize=chunksize, iterator=True):
        chunk = chunk[~chunk["address"].isna()].copy()
        if chunk.empty:
            continue

        chunk["address"] = chunk["address"].astype(str).str.strip()
        chunk = chunk[~chunk["address"].isin(seen)].drop_duplicates(subset="address", keep="first")
        if chunk.empty:
            continue

        chunk["seed_label"] = (chunk["label"].astype(str).str.lower() != "white").astype(int)
        chunk = chunk.sample(frac=1.0, random_state=rng.randint(0, 2**31-1))  # randomize within chunk

        for _, row in chunk.iterrows():
            addr_str = row["address"]
            if addr_str in seen:
                continue

            label_bin = int(row["seed_label"])
            if label_bin == 0 and licit_done >= need_licit:
                continue
            if label_bin == 1 and illicit_done >= need_illicit:
                continue

            try:
                addr_obj = chain.address_from_string(addr_str)
                if addr_obj is None:
                    continue
            except Exception:
                continue

            seen.add(addr_str)
            meta = {
                "bh_year":      row.get("year", None),
                "bh_day":       row.get("day", None),
                "bh_length":    row.get("length", None),
                "bh_weight":    row.get("weight", None),
                "bh_count":     row.get("count", None),
                "bh_looped":    row.get("looped", None),
                "bh_neighbors": row.get("neighbors", None),
                "bh_income":    row.get("income", None),
                "bh_label_raw": row.get("label", None),
            }
            addresses[addr_str] = (addr_obj, label_bin)
            bh_meta[addr_str]   = meta
            label_map[addr_str] = label_bin

            if label_bin == 0:
                licit_done += 1
            else:
                illicit_done += 1

            if licit_done >= need_licit and illicit_done >= need_illicit:
                break

        if licit_done >= need_licit and illicit_done >= need_illicit:
            break
            
    total = need_licit + need_illicit
    print(f"[get_data] selected licit={licit_done}/{total}, illicit={illicit_done}/{total} per split")
    return addresses, bh_meta, label_map


addresses, bh_meta, label_map = get_data(
    bitcoinheist_csv=str(BITCOINHEIST_CSV),
    n_train=train_samples, n_val=val_samples, n_test=test_samples,
    p_illicit_train=0.5, p_illicit_val=0.5, p_illicit_test=0.5,
    seed=42, chunksize=50_000
)


# ### Splitting addresses

def split_addresses(addresses: dict,
                    train_samples: int, val_samples: int, test_samples: int,
                    p_illicit_train: float = 0.5, p_illicit_val: float = 0.5, p_illicit_test: float = 0.5,
                    RANDOMIZE: bool = False, seed: int = None):
    licit_pool   = [a for a,(_,l) in addresses.items() if l==0]
    illicit_pool = [a for a,(_,l) in addresses.items() if l==1]

    if RANDOMIZE:
        rng = random.Random(seed)
        rng.shuffle(licit_pool)
        rng.shuffle(illicit_pool)

    def take_first(pool, n):
        n = max(0, min(n, len(pool)))
        sel = pool[:n]
        del pool[:n]
        return sel

    def compute_targets(n_total, p):
        p = 0.5 if p is None else float(p)
        n_il = int(round(n_total * p))
        n_li = n_total - n_il
        return n_li, n_il

    def build(n_total, p_illicit):
        n_li, n_il = compute_targets(n_total, p_illicit)
        if n_li > len(licit_pool) or n_il > len(illicit_pool):
            raise ValueError(
                f"Insufficient addresses for split (li={n_li}, il={n_il}) "
                f"with licit_pool={len(licit_pool)}, illicit_pool={len(illicit_pool)}"
            )
        sel_li = take_first(licit_pool,   n_li)
        sel_il = take_first(illicit_pool, n_il)
        out = {}
        for a in sel_li: out[a] = addresses[a]
        for a in sel_il: out[a] = addresses[a]
        return out

    train_addresses = build(train_samples, p_illicit_train)
    val_addresses   = build(val_samples,   p_illicit_val)
    test_addresses  = build(test_samples,  p_illicit_test)
    return train_addresses, val_addresses, test_addresses

train_addresses, val_addresses, test_addresses = split_addresses(
    addresses, train_samples, val_samples, test_samples,
    p_illicit_train=0.5, p_illicit_val=0.5, p_illicit_test=0.5,
    RANDOMIZE=True
)

print("-" * 40)
print(f"SIZE VERIFICATION:")
print(f"  Total pool addresses : {len(addresses)}")
print(f"  Train addresses size : {len(train_addresses)} (Expected: {train_samples})")
print(f"  Val addresses size   : {len(val_addresses)} (Expected: {val_samples})")
print(f"  Test addresses size  : {len(test_addresses)} (Expected: {test_samples})")
print("-" * 40)


# ## GRAPH EXPANSION

def limit_expansion(txs, limit):
    return random.sample(txs, min(len(txs), limit))

def explore_tx(tx, addresses, inputs, outputs, new, mode = 'none', selected_list = []):
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

            for tx_hash, tx in new_txs.items():
                if limit_mode == 'random node':
                    tx_candidates = []

                for input_tx in tx.inputs:
                    in_tx = input_tx.spent_tx

                    if in_tx is not None:
                        tx_hash = str(in_tx.hash)

                        if tx_hash not in txs and tx_hash not in hop_txs:
                            if limit_mode == 'random node':
                                tx_candidates.append(in_tx)
                            else:
                                candidates.append(in_tx)

                for output_tx in tx.outputs:
                    out_tx = output_tx.spending_tx

                    if out_tx is not None:
                        tx_hash = str(out_tx.hash)

                        if tx_hash not in txs and tx_hash not in hop_txs:
                            if limit_mode == 'random node':
                                tx_candidates.append(out_tx)
                            else:
                                candidates.append(out_tx)

                if limit_mode == "random node":
                    candidates += limit_expansion(tx_candidates, limit)

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

def split_jsonl_path(split_name, p_illicit):
    pct = int(round((0.5 if p_illicit is None else p_illicit)*100))
    return os.path.join(data_path, "_aggregates", f"{split_name}_p{pct}.jsonl")

def load_bh_mapping(mapping_path: str) -> dict:
    with open(mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)

bh_map = load_bh_mapping(str(MAPPING_PATH))

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

def split_jsonl_path(split_name, p_illicit):
    pct = int(round((0.5 if p_illicit is None else p_illicit)*100))
    return os.path.join(data_path, f"{split_name}_p{pct}.jsonl")

for split_name, addr_dict, p_illicit in [
    ("train", train_addresses, 0.5),
    ("val",   val_addresses,   0.5),
    ("test",  test_addresses,  0.5),
]:
    jsonl_path = split_jsonl_path(split_name, p_illicit)
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for seed_addr_str, seed_tuple in tqdm(addr_dict.items(), desc=f"Expanding {split_name}"):
            graph_data = run_seed_to_jsonl(seed_addr_str, seed_tuple, split_name, bh_meta)
            if graph_data:
                f.write(json.dumps(graph_data, ensure_ascii=False) + "\n")

print(f"\n[SUCCESS] All files successfully generated at: {data_path}")
