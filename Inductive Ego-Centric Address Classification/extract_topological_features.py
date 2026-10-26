# STEP 2 (Methodology II) - TOPOLOGICAL FEATURE EXTRACTION
#
# Computes the motif counts and graph metrics of every egonet stored in
# EGO_CENTRIC_EGONETS_DIR/<split>.jsonl (written by expansion.py), in parallel
# worker processes, and saves one table per split as soon as it is complete:
# EGO_CENTRIC_FEATURES_DIR/<split>.csv (one row per seed address, with its label
# and split), with <split>.json describing it.
#
# The features of a split are reused only if they were computed by the same code (this
# file, by its syntax tree) and feature parameters from the same egonets (the
# fingerprint of the expansion of the split, EGO_CENTRIC_EGONETS_DIR/<split>.meta.json):
# their fingerprint, saved in <split>.json with the expansion configuration of the
# egonets. Otherwise they are extracted again. The tabular models identify their data by
# the content of the feature files, and record this expansion configuration.
#
# Every egonet is one task of a pool of worker processes, which read it from its
# file. The egonets of all the pending splits are queued in a single stream, split
# after split and the largest first within each split (by the size of their JSON
# line), so the workers are not idle at the end of a split and the slowest egonets do
# not finish last. Only the centralities of the seed address are computed: PageRank
# needs the whole egonet, exact betweenness is computed with igraph (C library, the
# same values as networkx up to floating-point rounding) and eccentricity only for
# the seed. The graphs are built directly from the JSON records (a networkx graph for
# the centralities and integer arrays for the radius-limited egonet), and every worker
# runs single-threaded numerical libraries, as there is one worker per core.

import json
import multiprocessing
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# One worker per core: single-threaded numerical libraries (before importing them)
from shared.threads import limit_threads

import igraph
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from shared.experiment import code_fingerprint, digest, git_commit, read_json, timestamp, write_json

# Configuration (config.EGO_CENTRIC_FEATURES)
RADIUS = config.EGO_CENTRIC_FEATURES["radius"]
BETW_K = config.EGO_CENTRIC_FEATURES["betweenness_k"]  # None = exact betweenness
RANDOM_SEED = config.DATA_SEED
SAVE_EGONETS = config.EGO_CENTRIC_FEATURES["save_egonets"]

# Single-hop motifs counted around the seed (see count_metapaths_and_features):
# transactions spending from the seed by number of outputs, and transactions
# paying the seed by number of inputs
SCHEMAS = {
    "B1": "star_out",   # 1 output
    "B2": "2star_out",  # 2 outputs
    "B3+": "split_2+",  # 3 or more outputs
    "A1": "star_in",    # 1 input
    "A2": "2star_in",   # 2 inputs
    "A3+": "merge_2+",  # 3 or more inputs
}

# Helper functions

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

def json_graph(gj: Dict[str, Any]):
    """Addresses, transactions and edges of an egonet record. Edges refer to the
    position of their address and transaction in the record (the last one, if an
    address or a tx appears twice), in the order of the record."""
    addrs = gj.get("addresses", [])
    txs   = gj.get("transactions", [])

    idx2addr = [r.get("addr_str") for r in addrs]
    addr2loc = {addr_str: i for i, addr_str in enumerate(idx2addr)}
    idx2tx = [int(r.get("tx_id")) for r in txs]
    txid2loc = {tx_id: i for i, tx_id in enumerate(idx2tx)}
    addr_classes = {r["addr_str"]: r["class"] for r in addrs if "class" in r}

    in_pairs = []
    for r in gj.get("inputs", []):
        addr_s = r.get("addr_str")
        tx_id  = int(r.get("tx_id"))
        if addr_s in addr2loc and tx_id in txid2loc:
            in_pairs.append((addr2loc[addr_s], txid2loc[tx_id]))

    out_pairs = []
    for r in gj.get("outputs", []):
        tx_id  = int(r.get("tx_id"))
        addr_s = r.get("addr_str")
        if tx_id in txid2loc and addr_s in addr2loc:
            out_pairs.append((txid2loc[tx_id], addr2loc[addr_s]))

    return idx2addr, idx2tx, addr_classes, in_pairs, out_pairs

def build_digraph(idx2addr, idx2tx, addr_classes, in_pairs, out_pairs) -> nx.DiGraph:
    """Directed address-transaction graph of the egonet (repeated edges collapse)."""
    G = nx.DiGraph()
    for idx, addr_str in enumerate(idx2addr):
        cls = addr_classes.get(addr_str, "unknown")
        G.add_node(f"addr_{idx}", type="addr", label=cls, original_id=addr_str)
    for idx, tx_val in enumerate(idx2tx):
        G.add_node(f"tx_{idx}", type="tx", original_id=tx_val)
    G.add_edges_from((f"addr_{s}", f"tx_{t}", {"type": "input"}) for s, t in in_pairs)
    G.add_edges_from((f"tx_{s}", f"addr_{t}", {"type": "output"}) for s, t in out_pairs)
    return G

def edge_arrays(pairs):
    """Source and target arrays of the distinct edges, in the order in which the
    directed graph lists them: by source, and by first appearance for each source."""
    distinct = sorted(dict.fromkeys(pairs), key=lambda pair: pair[0])
    if not distinct:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    edges = np.array(distinct, dtype=np.int64)
    return edges[:, 0].copy(), edges[:, 1].copy()

def address_labels(idx2addr, addr_classes):
    labels = []
    for addr_str in idx2addr:
        try: labels.append(int(addr_classes.get(addr_str, "unknown")))
        except: labels.append(0)
    return np.array(labels, dtype=np.int64)

def betweenness_of(UG, node, betw_k, rand_seed):
    """Normalized betweenness of one node of an undirected graph: exact with igraph, or
    the sampled approximation of networkx with betw_k sources."""
    if betw_k:
        k = min(betw_k, UG.number_of_nodes())
        return nx.betweenness_centrality(UG, normalized=True, k=k, seed=rand_seed).get(node, 0.0)
    nodes = list(UG.nodes())
    loc = {n: i for i, n in enumerate(nodes)}
    g = igraph.Graph(n=len(nodes), edges=[(loc[u], loc[v]) for u, v in UG.edges()], directed=False)
    raw = g.betweenness(vertices=[loc[node]], directed=False)[0]
    # Same normalization as networkx: every pair of nodes counted once, divided by the
    # number of pairs of the other nodes (n-1)(n-2)/2
    n = len(nodes)
    return raw * 2.0 / ((n - 1) * (n - 2)) if n > 2 else 0.0

def eccentricity_of(UG, node):
    # Eccentricity within the connected component of the node
    try: return nx.eccentricity(UG.subgraph(nx.node_connected_component(UG, node)), node)
    except: return 0

def build_centralities_from_nx(G_work: nx.DiGraph, target_locs, betw_k, rand_seed):
    """PageRank and betweenness of the target addresses (their position among
    the address nodes of the egonet), and the undirected graph. Only the values of the
    targets are used, so the others are not computed (PageRank needs the whole graph).
    The eccentricity is only used if the target is not in its radius-limited egonet
    (eccentricity_of)."""
    addr_nodes = [n for n,d in G_work.nodes(data=True) if d['type']=='addr']
    targets = {idx: addr_nodes[idx] for idx in target_locs}

    try: pr = nx.pagerank(G_work, alpha=0.85, max_iter=200)
    except: pr = {n: 0.0 for n in G_work.nodes()}

    UG = G_work.to_undirected()

    cent = {'pagerank':{}, 'betweenness':{}, 'eccentricity':{}}
    for idx, n in targets.items():
        cent['pagerank'][idx] = pr.get(n, 0.0)
        try: cent['betweenness'][idx] = betweenness_of(UG, n, betw_k, rand_seed)
        except: cent['betweenness'][idx] = 0.0
    return cent, UG, targets

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
                      'density':0.0, 'eccentricity':0})
        for k in ['pagerank','betweenness','eccentricity']:
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

    for k in ['pagerank','betweenness','eccentricity']:
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

    return feats

# Features of one egonet (one line of a .jsonl file), computed by an independent worker

def read_line(path, offset, length):
    with open(path, 'rb') as f:
        f.seek(offset)
        return f.read(length).decode('utf-8')

def line_offsets(path):
    """(offset, length) of every line of a file."""
    offsets = []
    position = 0
    with open(path, 'rb') as f:
        for line in f:
            offsets.append((position, len(line)))
            position += len(line)
    return offsets

def process_single_line_task(args):
    path, offset, length, line_idx, split_guess, out_dir_str, radius_val, betw_k_val, rand_seed = args
    return split_guess, process_egonet(read_line(path, offset, length), line_idx, split_guess, out_dir_str,
                                       radius_val, betw_k_val, rand_seed)

def process_egonet(line, line_idx, split_guess, out_dir_str, radius_val, betw_k_val, rand_seed):
    try:
        gj = json.loads(line)
        seed_str = str(gj.get("seed") or f"g{line_idx}")

        idx2addr, idx2tx, addr_classes, in_pairs, out_pairs = json_graph(gj)
        G = build_digraph(idx2addr, idx2tx, addr_classes, in_pairs, out_pairs)

        if G.number_of_nodes() == 0:
            return []

        addr_strs = idx2addr
        N_ADDR = len(idx2addr)
        N_TX = len(idx2tx)
        labels = address_labels(idx2addr, addr_classes)

        sa, ta = edge_arrays(in_pairs)
        sb, tb = edge_arrays(out_pairs)

        nbr_at, ptr_at = build_csr(sa, ta, N_ADDR)
        nbr_at_rev, ptr_at_rev = build_csr(tb, sb, N_ADDR)
        nbr_ta, ptr_ta = build_csr(sb, tb, N_TX)
        nbr_ta_rev, ptr_ta_rev = build_csr(ta, sa, N_TX)
        ptrs = (nbr_at, ptr_at, nbr_at_rev, ptr_at_rev, nbr_ta, ptr_ta, nbr_ta_rev, ptr_ta_rev)

        global_tx_in_deg = np.bincount(ta) if len(ta) > 0 else np.array([])
        global_tx_out_deg = np.bincount(sb) if len(sb) > 0 else np.array([])

        # The features are computed for the seed address (the first address if it is
        # not in the largest connected component)
        targets = []
        if seed_str in addr_strs:
            seed_idx = addr_strs.index(seed_str)
            targets = [(seed_idx, labels[seed_idx])]
        elif N_ADDR > 0:
            targets = [(0, labels[0])]

        cent, UG, target_nodes = build_centralities_from_nx(G, [centre_idx for centre_idx, _ in targets], betw_k_val, rand_seed)

        rows = []
        for centre_idx, label in targets:
            vis = bfs_addr_tx(centre_idx, ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7], radius_val)
            ego = build_egonet_index(vis, centre_idx, sa, ta, sb, tb)

            addr_str_target = addr_strs[centre_idx] if centre_idx < len(addr_strs) else str(centre_idx)
            safe_seed = sanitize_filename(seed_str)
            safe_node = sanitize_filename(addr_str_target)
            fname = f"{int(label)}_{safe_seed}_{safe_node}.gpickle"

            if SAVE_EGONETS:
                atomic_write_gpickle(ego, Path(out_dir_str) / fname)

            # The eccentricity in the whole egonet is only used if the target is not in
            # its radius-limited egonet
            if centre_idx * 2 not in ego:
                cent['eccentricity'][centre_idx] = eccentricity_of(UG, target_nodes[centre_idx])

            row = count_metapaths_and_features(ego, centre_idx, cent, global_tx_in_deg, global_tx_out_deg)
            row['label'] = int(label)
            row['addr_str'] = addr_str_target
            row['seed_str'] = seed_str
            row['split'] = split_guess
            rows.append(row)

        return rows

    except Exception as e:
        return f"ERROR in line {line_idx}: {e}"

def egonets_meta(split):
    """Fingerprint and expansion configuration of the complete egonets of a split, or None."""
    egonets_dir = config.EGO_CENTRIC_EGONETS_DIR
    meta_path = egonets_dir / f"{split}.meta.json"
    if not meta_path.exists() or not (egonets_dir / f"{split}.jsonl").exists():
        return None
    return read_json(meta_path)


def features_fingerprint(code, egonets_fingerprint):
    """What the features of a split are computed from (see the top of this file)."""
    return digest({"code": code, "egonets": egonets_fingerprint, "radius": RADIUS, "betweenness_k": BETW_K,
                   "data_seed": RANDOM_SEED})


def main():
    limit_threads()
    egonets_dir = config.EGO_CENTRIC_EGONETS_DIR
    features_dir = config.EGO_CENTRIC_FEATURES_DIR
    features_dir.mkdir(parents=True, exist_ok=True)
    code = code_fingerprint({Path(__file__).resolve(): None})

    pending, metas = [], {}
    for split in config.SPLITS:
        meta = egonets_meta(split)
        if meta is None:
            raise FileNotFoundError(f"Missing or incomplete egonets of {split} ({egonets_dir}): run expansion.py first.")
        out_csv = features_dir / f"{split}.csv"
        info_path = out_csv.with_suffix(".json")
        fingerprint = features_fingerprint(code, meta["fingerprint"])
        saved = read_json(info_path).get("fingerprint") if out_csv.exists() and info_path.exists() else None
        if saved == fingerprint:
            print(f"{split}: features already extracted from these egonets with this code, skipping ({out_csv})")
            continue
        if out_csv.exists():
            print(f"{split}: the existing features come from other egonets, code or parameters "
                  f"({saved or 'unknown'} instead of {fingerprint}), extracting again")
            out_csv.unlink()
        metas[split] = {**meta, "features_fingerprint": fingerprint}
        pending.append(split)
    if not pending:
        return

    # One stream of tasks: split after split, the largest egonets first within each split
    tasks, totals = [], {}
    for split in pending:
        fpath = egonets_dir / f"{split}.jsonl"
        out_dir = egonets_dir / split
        if SAVE_EGONETS:
            out_dir.mkdir(exist_ok=True)
        offsets = line_offsets(fpath)
        totals[split] = len(offsets)
        order = sorted(range(len(offsets)), key=lambda i: -offsets[i][1])
        tasks += [(str(fpath), offsets[i][0], offsets[i][1], i, split, str(out_dir), RADIUS, BETW_K, RANDOM_SEED)
                  for i in order]
        print(f"{split}: {len(offsets)} egonets")

    n_workers = config.EGO_CENTRIC_FEATURES["workers"]
    print(f"\nStarting a pool of {n_workers} workers...")

    rows = {split: [] for split in pending}
    received = {split: 0 for split in pending}
    errors = {split: 0 for split in pending}
    with multiprocessing.Pool(processes=n_workers) as pool:
        results = pool.imap_unordered(process_single_line_task, tasks, chunksize=1)
        for split, result in tqdm(results, total=len(tasks), desc="Egonets", smoothing=0.1, mininterval=1.0):
            received[split] += 1
            if isinstance(result, list):
                rows[split].extend(result)
            elif isinstance(result, str) and "ERROR" in result:
                errors[split] += 1
                print(f"\n[WORKER ERROR] {split}: {result}")
            if received[split] == totals[split]:
                save_split(split, rows.pop(split), totals[split], errors[split], features_dir, metas[split])


def save_split(split, rows, total_lines, n_errors, features_dir, meta):
    """Saves the features of a split once all its egonets are processed."""
    out_csv = features_dir / f"{split}.csv"
    if not rows:
        raise RuntimeError(f"No rows generated for {split}.")

    # Rows arrive in completion order, so they are sorted to make the file deterministic
    df = pd.DataFrame(rows).sort_values("addr_str").reset_index(drop=True)

    # Written to a temporary file first, so an interrupted split is never taken as complete
    tmp_csv = out_csv.with_name(out_csv.name + ".tmp")
    df.to_csv(tmp_csv, index=False)
    os.replace(str(tmp_csv), str(out_csv))

    # Written after the table: with its fingerprint, it marks the split as complete
    write_json(out_csv.with_suffix(".json"), {
        "fingerprint": meta["features_fingerprint"],
        "egonets_fingerprint": meta["fingerprint"],
        "expansion": meta["expansion"],
        "exp_alg": meta["exp_alg"],
        "created": timestamp(),
        "git_commit": git_commit(config.PROJECT_ROOT),
        "features": config.EGO_CENTRIC_FEATURES,
        "betweenness": "exact, igraph" if not config.EGO_CENTRIC_FEATURES["betweenness_k"] else "sampled, networkx",
        "split": split,
        "egonets": total_lines,
        "rows": int(len(df)),
        "errors": n_errors,
    })
    print(f"\n[SUCCESS] Features of {len(df)} {split} seed addresses saved to: {out_csv}")
    if n_errors:
        print(f"[WARNING] {n_errors} {split} egonets could not be processed.")


if __name__ == "__main__":
    main()
