#!/usr/bin/env python
# coding: utf-8

import os
import json
import math
import re
import tempfile
import gc
import multiprocessing
from pathlib import Path
from typing import Dict, Any, Tuple
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
import networkx as nx
from tqdm import tqdm
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.paths import get_required_env_path

# --- CONFIGURATION ---
MAX_GRAPHS   = None    # None = all
USE_LCC      = False   
RADIUS       = 6       
LIMIT_NODES  = None    
BETW_K       = None    # None = Exacte (Lent). Posa 50 per Aproximat (Ràpid).
RANDOM_SEED  = 42

# --- SEED MODE ---
ONLY_SEED    = True    

EGO_DIR = Path(get_required_env_path("RSD_HETEROGENEOUS_EGONETS"))

# --- SHARED MEMORY MANAGEMENT ---
_worker_data = {}

def init_worker(sa, ta, sb, tb, ptrs, cent, addr_idx2str, N_ADDR, RADIUS_VAL, LIMIT_VAL, SCHEMAS_VAL):
    """Initializes shared memory in each worker process."""
    _worker_data['sa'] = sa
    _worker_data['ta'] = ta
    _worker_data['sb'] = sb
    _worker_data['tb'] = tb
    _worker_data['ptrs'] = ptrs
    _worker_data['cent'] = cent
    _worker_data['addr_idx2str'] = addr_idx2str
    _worker_data['N_ADDR'] = N_ADDR
    _worker_data['RADIUS'] = RADIUS_VAL
    _worker_data['LIMIT_NODES'] = LIMIT_VAL
    _worker_data['SCHEMAS'] = SCHEMAS_VAL

# --- HELPER FUNCTIONS ---
def sanitize_filename(s: str) -> str:
    if s is None: return "unknown"
    return re.sub(r'[^A-Za-z0-9._-]', '_', str(s))

def atomic_write_gpickle(G, target_path: Path):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(target_path.parent), prefix=".tmp_", suffix=".gpickle")
    os.close(fd)
    try:
        nx.write_gpickle(G, tmp)        
        os.replace(tmp, str(target_path))  
    except Exception:
        try: os.remove(tmp)
        except: pass

def select_directory(base_path):
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
    directoris = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    if not directoris:
        return None
    print("Available subdirectories:")
    for i, d in enumerate(directoris, 1):
        print(f"{i}. {d}")
    while True:
        try:
            seleccio = int(input(f"Select a directory (1-{len(directoris)}): "))
            if 1 <= seleccio <= len(directoris):
                return directoris[seleccio - 1]
        except ValueError:
            pass
    return None

def json_graph_to_heterodata(gj: Dict[str, Any]) -> Tuple[HeteroData, dict, dict, dict, dict]:
    data = HeteroData()
    addrs = gj.get("addresses", [])
    txs   = gj.get("transactions", [])
    
    addr2loc = {}
    idx2addr = {}
    for i, r in enumerate(addrs):
        addr_str = r.get("addr_str")
        addr2loc[addr_str] = i
        idx2addr[i] = addr_str
        
    txid2loc = {}
    idx2tx = {}
    for i, r in enumerate(txs):
        tx_id = int(r.get("tx_id"))
        txid2loc[tx_id] = i
        idx2tx[i] = tx_id

    addr_classes = {}
    for r in addrs:
        if "class" in r:
            addr_classes[r["addr_str"]] = r["class"]

    inputs  = gj.get("inputs", [])
    outputs = gj.get("outputs", [])

    a2t_src, a2t_dst, a2t_vals = [], [], [] 
    for r in inputs:
        addr_s = r.get("addr_str")
        tx_id  = int(r.get("tx_id"))
        if addr_s in addr2loc and tx_id in txid2loc:
            a2t_src.append(addr2loc[addr_s])
            a2t_dst.append(txid2loc[tx_id])
            a2t_vals.append(float(r.get("value", 0)))

    t2a_src, t2a_dst, t2a_vals = [], [], []
    for r in outputs:
        tx_id  = int(r.get("tx_id"))
        addr_s = r.get("addr_str")
        if tx_id in txid2loc and addr_s in addr2loc:
            t2a_src.append(txid2loc[tx_id])
            t2a_dst.append(addr2loc[addr_s])
            t2a_vals.append(float(r.get("value", 0)))

    if len(a2t_src) > 0:
        data['addr','input','tx'].edge_index = torch.tensor([a2t_src, a2t_dst], dtype=torch.long)
        data['addr','input','tx'].edge_attr  = torch.tensor(a2t_vals, dtype=torch.float)
    else:
        data['addr','input','tx'].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data['addr','input','tx'].edge_attr  = torch.tensor([], dtype=torch.float)

    if len(t2a_src) > 0:
        data['tx','output','addr'].edge_index = torch.tensor([t2a_src, t2a_dst], dtype=torch.long)
        data['tx','output','addr'].edge_attr  = torch.tensor(t2a_vals, dtype=torch.float)
    else:
        data['tx','output','addr'].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data['tx','output','addr'].edge_attr  = torch.tensor([], dtype=torch.float)

    return data, idx2addr, idx2tx, addr_classes, addr2loc

def hetero_to_nx(data: HeteroData, idx2addr: dict, addr_classes: dict, idx2tx: dict) -> nx.DiGraph:
    G = nx.DiGraph()
    for idx, addr_str in idx2addr.items():
        cls = addr_classes.get(addr_str, "unknown")
        G.add_node(f"addr_{idx}", type="addr", label=cls, original_id=addr_str)
    for idx, tx_val in idx2tx.items():
        G.add_node(f"tx_{idx}", type="tx", original_id=tx_val)

    if ('addr', 'input', 'tx') in data.edge_types:
        edge_index = data['addr', 'input', 'tx'].edge_index
        vals = data['addr', 'input', 'tx'].edge_attr.tolist() if 'edge_attr' in data['addr', 'input', 'tx'] else [0]*edge_index.shape[1]
        src, dst = edge_index.tolist()
        for s, t, v in zip(src, dst, vals):
            G.add_edge(f"addr_{s}", f"tx_{t}", type="input", value=v)

    if ('tx', 'output', 'addr') in data.edge_types:
        edge_index = data['tx', 'output', 'addr'].edge_index
        vals = data['tx', 'output', 'addr'].edge_attr.tolist() if 'edge_attr' in data['tx', 'output', 'addr'] else [0]*edge_index.shape[1]
        src, dst = edge_index.tolist()
        for s, t, v in zip(src, dst, vals):
            G.add_edge(f"tx_{s}", f"addr_{t}", type="output", value=v)
    return G

def build_centralities_from_nx(G_work: nx.DiGraph, betw_k, rand_seed) -> dict:
    addr_nodes = [n for n,d in G_work.nodes(data=True) if d['type']=='addr']
    addr2loc = {n:i for i,n in enumerate(addr_nodes)}
    
    try: pr = nx.pagerank(G_work, alpha=0.85, max_iter=200)
    except: pr = {n: 0.0 for n in G_work.nodes()}
    
    UG = G_work.to_undirected()
    clust = nx.clustering(UG)
    
    try:
        k = min(betw_k, UG.number_of_nodes()) if betw_k else None
        betw = nx.betweenness_centrality(UG, normalized=True, k=k, seed=rand_seed)
    except: betw = {n: 0.0 for n in G_work.nodes()}

    try:
        ecc = {}
        for C in nx.connected_components(UG):
            ecc.update(nx.eccentricity(UG.subgraph(C)))
    except: ecc = {n: 0 for n in G_work.nodes()}

    cent = {'pagerank':{}, 'clustering':{}, 'betweenness':{}, 'eccentricity':{}}
    for n, idx in addr2loc.items():
        cent['pagerank'][idx] = pr.get(n, 0.0)
        cent['clustering'][idx] = clust.get(n, 0.0)
        cent['betweenness'][idx] = betw.get(n, 0.0)
        cent['eccentricity'][idx] = ecc.get(n, 0)
    return cent

def nx_to_hetero_lcc_preserve_ids(G):
    addr_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'addr']
    tx_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'tx']
    
    addr_map = {n: i for i, n in enumerate(addr_nodes)}
    tx_map = {n: i for i, n in enumerate(tx_nodes)}
    
    addr_strs = [G.nodes[n].get('original_id', 'unknown') for n in addr_nodes]
    labels = []
    for n in addr_nodes:
        l = G.nodes[n].get('label', 0)
        try: labels.append(int(l))
        except: labels.append(0) 
            
    src_in, dst_in, src_out, dst_out = [], [], [], []
    for u, v, d in G.edges(data=True):
        t = d.get('type')
        if t == 'input' and u in addr_map and v in tx_map:
            src_in.append(addr_map[u]); dst_in.append(tx_map[v])
        elif t == 'output' and u in tx_map and v in addr_map:
            src_out.append(tx_map[u]); dst_out.append(addr_map[v])
                
    data = HeteroData()
    data['addr'].num_nodes = len(addr_nodes)
    data['addr'].y = torch.tensor(labels, dtype=torch.long)
    data['tx'].num_nodes = len(tx_nodes)
    
    if src_in: data['addr','input','tx'].edge_index = torch.tensor([src_in, dst_in], dtype=torch.long)
    else: data['addr','input','tx'].edge_index = torch.zeros((2, 0), dtype=torch.long)
        
    if src_out: data['tx','output','addr'].edge_index = torch.tensor([src_out, dst_out], dtype=torch.long)
    else: data['tx','output','addr'].edge_index = torch.zeros((2, 0), dtype=torch.long)
        
    return data, addr_strs

def build_csr(src, dst, N):
    order = np.argsort(src)
    src_s, dst_s = src[order], dst[order]
    counts = np.bincount(src_s, minlength=N)
    indptr = np.concatenate(([0], np.cumsum(counts)))
    return dst_s.astype(np.int32), indptr.astype(np.int32)

def bfs_addr_tx(centre_idx, nbr_at, ptr_at, nbr_at_rev, ptr_at_rev, nbr_ta, ptr_ta, nbr_ta_rev, ptr_ta_rev, RADIUS):
    vis = {'addr':{centre_idx}, 'tx':set()}
    current_addr = {centre_idx}
    for _ in range(RADIUS):
        next_tx = set()
        for idx in current_addr:
            start, end = ptr_at[idx], ptr_at[idx+1]
            for t in nbr_at[start:end]:
                if t not in vis['tx']: vis['tx'].add(t); next_tx.add(t)
            start, end = ptr_at_rev[idx], ptr_at_rev[idx+1]
            for t in nbr_at_rev[start:end]:
                if t not in vis['tx']: vis['tx'].add(t); next_tx.add(t)
        if not next_tx: break

        next_addr = set()
        for idx in next_tx:
            start, end = ptr_ta[idx], ptr_ta[idx+1]
            for a in nbr_ta[start:end]:
                if a not in vis['addr']: vis['addr'].add(a); next_addr.add(a)
            start, end = ptr_ta_rev[idx], ptr_ta_rev[idx+1]
            for a in nbr_ta_rev[start:end]:
                if a not in vis['addr']: vis['addr'].add(a); next_addr.add(a)
        if not next_addr: break
        current_addr = next_addr
    return vis

def build_egonet_index(vis, centre_idx, sa, ta, sb, tb):
    G_ego = nx.DiGraph()
    vis_addr = list(vis['addr'])
    vis_tx = list(vis['tx'])
    for a in vis_addr: G_ego.add_node(a * 2, type='addr', original_id=a)
    for t in vis_tx: G_ego.add_node(t * 2 + 1, type='tx', original_id=t)

    mask_in = np.isin(sa, vis_addr) & np.isin(ta, vis_tx)
    for a, t in zip(sa[mask_in], ta[mask_in]): G_ego.add_edge(a * 2, t * 2 + 1, type='input')
    mask_out = np.isin(sb, vis_tx) & np.isin(tb, vis_addr)
    for t, a in zip(sb[mask_out], tb[mask_out]): G_ego.add_edge(t * 2 + 1, a * 2, type='output')
    return G_ego

def count_peeling_chains(ego, centre_idx, global_tx_in_deg, global_tx_out_deg, depth=2):
    total_chains = 0
    def dfs(current_node, current_step):
        nonlocal total_chains
        for _, tx_node, dt in ego.out_edges(current_node, data=True):
            if dt.get('type') == 'input':
                t_idx = ego.nodes[tx_node]['original_id']
                global_in = global_tx_in_deg[t_idx] if t_idx < len(global_tx_in_deg) else 0
                global_out = global_tx_out_deg[t_idx] if t_idx < len(global_tx_out_deg) else 0
                
                if global_in != 1 or global_out != 2:
                    continue
                    
                if current_step + 1 == depth:
                    total_chains += 1
                    continue 
                    
                for _, next_addr, da in ego.out_edges(tx_node, data=True):
                    if da.get('type') == 'output' and next_addr != current_node:
                        dfs(next_addr, current_step + 1)
                        
    dfs(centre_idx, 0)
    return total_chains

def count_metapaths_and_features(ego, centre_idx_original, cent, global_tx_in_deg, global_tx_out_deg):
    cnt = Counter()
    centre_node_id = centre_idx_original * 2
    
    if centre_node_id not in ego:
        feats = {name: 0 for name in SCHEMAS.values()}
        feats.update({'in_deg':0, 'out_deg':0, 'peeling_2':0, 'peeling_3':0, 
                      'density':0.0, 'eccentricity':0, 
                      'n_nodes': ego.number_of_nodes(), 'n_edges': ego.number_of_edges()})
        for k in ['pagerank','clustering','betweenness','eccentricity']:
            feats[k] = cent[k].get(centre_idx_original, 0.0)
        return feats

    for t in ego.successors(centre_node_id):
        if ego[centre_node_id][t]['type'] != 'input': continue
        deg = ego.out_degree(t)
        if deg == 1: cnt['star_out'] += 1
        elif deg == 2: cnt['2star_out'] += 1
        elif deg >= 3: cnt['split_2+'] += 1
            
    for t in ego.predecessors(centre_node_id):
        if ego[t][centre_node_id]['type'] != 'output': continue
        deg = ego.in_degree(t)
        if deg == 1: cnt['star_in'] += 1     
        elif deg == 2: cnt['2star_in'] += 1  
        elif deg >= 3: cnt['merge_2+'] += 1  
            
    feats = {name: cnt[name] for name in SCHEMAS.values()}
    feats['in_deg'] = ego.in_degree(centre_node_id)
    feats['out_deg'] = ego.out_degree(centre_node_id)
    
    feats['peeling_2'] = count_peeling_chains(ego, centre_node_id, global_tx_in_deg, global_tx_out_deg, depth=2)
    feats['peeling_3'] = count_peeling_chains(ego, centre_node_id, global_tx_in_deg, global_tx_out_deg, depth=3)
    
    for k in ['pagerank','clustering','betweenness','eccentricity']:
        feats[k] = cent[k].get(centre_idx_original, 0.0)
        
    try: feats['density'] = nx.density(ego)
    except: feats['density'] = 0.0
    
    try:
        ud = ego.to_undirected()
        if centre_node_id in ud:
            comp = ud.subgraph(nx.node_connected_component(ud, centre_node_id))
            feats['eccentricity'] = nx.eccentricity(comp, centre_node_id)
        else: feats['eccentricity'] = 0
    except: feats['eccentricity'] = 0
    
    feats['n_nodes'] = ego.number_of_nodes()
    feats['n_edges'] = ego.number_of_edges()
    
    return feats

# --- TASCA TOTAL PER A CADA WORKER INDEPENDENT ---
def process_single_line_task(args):
    line, line_idx, split_guess, out_dir_str, radius_val, limit_val, betw_k_val, rand_seed = args
    
    try:
        gj = json.loads(line)
        seed_str = str(gj.get("seed") or f"g{line_idx}")

        data_h, idx2addr, idx2tx, addr_classes, addr2loc = json_graph_to_heterodata(gj)
        G = hetero_to_nx(data_h, idx2addr, addr_classes, idx2tx)

        if G.number_of_nodes() == 0:
            return []

        cent = build_centralities_from_nx(G, betw_k_val, rand_seed)
        data_final, addr_strs = nx_to_hetero_lcc_preserve_ids(G)
        
        N_ADDR = data_final['addr'].num_nodes
        labels = data_final['addr'].y.numpy()
        
        sa, ta = data_final['addr','input','tx'].edge_index.numpy() if ('addr','input','tx') in data_final.edge_types else (np.array([], dtype=np.int32), np.array([], dtype=np.int32))
        sb, tb = data_final['tx','output','addr'].edge_index.numpy() if ('tx','output','addr') in data_final.edge_types else (np.array([], dtype=np.int32), np.array([], dtype=np.int32))
        
        nbr_at, ptr_at = build_csr(sa, ta, N_ADDR)
        nbr_at_rev, ptr_at_rev = build_csr(tb, sb, N_ADDR)
        nbr_ta, ptr_ta = build_csr(sb, tb, data_final['tx'].num_nodes)
        nbr_ta_rev, ptr_ta_rev = build_csr(ta, sa, data_final['tx'].num_nodes)
        ptrs = (nbr_at, ptr_at, nbr_at_rev, ptr_at_rev, nbr_ta, ptr_ta, nbr_ta_rev, ptr_ta_rev)

        global_tx_in_deg = np.bincount(ta) if len(ta) > 0 else np.array([])
        global_tx_out_deg = np.bincount(sb) if len(sb) > 0 else np.array([])

        targets = []
        if ONLY_SEED:
            try:
                seed_idx = addr_strs.index(seed_str)
                targets = [(seed_idx, labels[seed_idx])]
            except ValueError:
                if N_ADDR > 0: targets = [(0, labels[0])]
        else:
            pass # Lògica de data augmentation
            
        rows = []
        for centre_idx, label in targets:
            vis = bfs_addr_tx(centre_idx, ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7], radius_val)
            ego = build_egonet_index(vis, centre_idx, sa, ta, sb, tb)
            
            centre_node_id = centre_idx * 2
            if limit_val and ego.number_of_nodes() > limit_val:
                deg = sorted(ego.degree(), key=lambda x: x[1], reverse=True)
                nodes = {n for n,_ in deg[:limit_val]}
                if centre_node_id not in nodes: nodes.add(centre_node_id)
                ego = ego.subgraph(nodes).copy()
                
            addr_str_target = addr_strs[centre_idx] if centre_idx < len(addr_strs) else str(centre_idx)
            safe_seed = sanitize_filename(seed_str)
            safe_node = sanitize_filename(addr_str_target)
            fname = f"{int(label)}_{safe_seed}_{safe_node}.gpickle"
            
            atomic_write_gpickle(ego, Path(out_dir_str) / fname)
            
            row = count_metapaths_and_features(ego, centre_idx, cent, global_tx_in_deg, global_tx_out_deg)
            row['label'] = int(label)
            row['addr_str'] = addr_str_target
            row['seed_str'] = seed_str
            row['split'] = split_guess
            rows.append(row)
            
        return rows

    except Exception as e:
        return f"ERROR in line {line_idx}: {e}"

# --- MAIN BLOCK ---
if __name__ == "__main__":
    directori = select_directory(EGO_DIR)
    
    if directori:
        RES_DIR = EGO_DIR / directori
        jsonl_paths = sorted(list(RES_DIR.glob("*_p*.jsonl")))
        
        if not jsonl_paths:
            print(f"No JSONL files found in: {RES_DIR}")
        else:
            all_rows = []
            
            # --- CREEM UN ÚNIC POOL PER A TOTA L'EXECUCIÓ ---
            # Deixem 4 nuclis lliures per al sistema operatiu
            n_workers = max(1, multiprocessing.cpu_count() - 4)
            print(f"\nStarting Pool amb {n_workers} workers...")
            
            with multiprocessing.Pool(processes=n_workers) as pool:
            
                for fpath in jsonl_paths:
                    split_guess = "train"
                    if "val" in fpath.name: split_guess = "val"
                    elif "test" in fpath.name: split_guess = "test"
                    
                    out_dir = RES_DIR / split_guess
                    out_dir.mkdir(exist_ok=True)
                    
                    print(f"\nProcessing: {fpath.name}")
                    
                    print("Comptant línies del fitxer per a la barra de progrés...")
                    with open(fpath, 'r') as f:
                        total_lines = sum(1 for _ in f)
                        
                    def task_generator(filepath, split_val, out_dir_val, rad_val, lim_val, betw_val, rand_val):
                        with open(filepath, 'r') as f_gen:
                            for idx, line in enumerate(f_gen):
                                yield (line, idx, split_val, str(out_dir_val), rad_val, lim_val, betw_val, rand_val)
                    
                    tasks_gen = task_generator(fpath, split_guess, out_dir, RADIUS, LIMIT_NODES, BETW_K, RANDOM_SEED)
                    
                    for result in tqdm(pool.imap_unordered(process_single_line_task, tasks_gen, chunksize=1), total=total_lines, desc="Graphs", smoothing=0.1, mininterval=1.0):
                        if isinstance(result, list):
                            all_rows.extend(result)
                        elif isinstance(result, str) and "ERROR" in result:
                            print(f"\n[WORKER ERROR] {result}")
            
            if all_rows:
                df = pd.DataFrame(all_rows)
                out_csv = RES_DIR / f"full_dataset_{RADIUS}.csv"
                df.to_csv(out_csv, index=False)
                print(f"\n[SUCCESS] Full dataset saved to: {out_csv}")
                print(f"Total rows: {len(df)}")
            else:
                print("\n[WARNING] No rows generated.")
    else:
        print("Operation cancelled.")
