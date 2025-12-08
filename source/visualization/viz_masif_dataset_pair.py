#!/usr/bin/env python3
import os
import argparse
import yaml
import numpy as np
import pymesh
from scipy.spatial import cKDTree
from itertools import zip_longest

# ----------------------------
# Utils
# ----------------------------

def load_config(path: str) -> dict:
    """Load YAML config file and handle compatibility."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    exp = cfg.get("exp")
    if exp:
        for k, v in list(cfg.items()):
            if isinstance(v, str):
                cfg[k] = v.replace("{exp}", str(exp))
        if isinstance(cfg.get("ppi_search"), dict):
            for k, v in list(cfg["ppi_search"].items()):
                if isinstance(v, str):
                    cfg["ppi_search"][k] = v.replace("{exp}", str(exp))
    if "ppi_search" not in cfg or not cfg["ppi_search"]:
        raise ValueError("Config missing required key: ppi_search")
    if "ply_file_template" not in cfg or not cfg["ply_file_template"]:
        raise ValueError("Config missing required key: ply_file_template")
    return cfg

def write_triangle_ply(path, V, F, rgb01):
    """
    Write an ASCII PLY triangle mesh with per-vertex colors.
    V: (N,3), F: (M,3), rgb01: (N,3) in [0,1]
    """
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int32)
    C = (np.clip(np.asarray(rgb01, dtype=np.float64), 0.0, 1.0) * 255.0).astype(np.uint8)
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError(f"V must be (N,3), got {V.shape}")
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError(f"F must be (M,3), got {F.shape}")
    if C.shape != (V.shape[0], 3):
        raise ValueError(f"rgb01 must be (N,3), got {C.shape}, N={V.shape[0]}")

    nV, nF = V.shape[0], F.shape[0]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {nV}\n")
        f.write("property double x\nproperty double y\nproperty double z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {nF}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for i in range(nV):
            x, y, z = V[i]
            r, g, b = C[i]
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")
        for i in range(nF):
            a, b, c = F[i]
            f.write(f"3 {int(a)} {int(b)} {int(c)}\n")

def parse_name_entry(s: str):
    """
    Cache name format is: f"{ppi_pdb_id_id}_{pid}_{ii}"
    ppi_pdb_id_id contains underscores, so we use rsplit('_',2)
    Returns: (pdb_id, pid, idx_int)
    """
    pdb_id, pid, idx = s.rsplit("_", 2)
    return pdb_id, pid, int(idx)

def build_vertex_adjacency(num_v: int, F: np.ndarray):
    """
    Build adjacency list from face connectivity (F).
    """
    adj = [set() for _ in range(num_v)]
    F = np.asarray(F, dtype=np.int32)
    for (a, b, c) in F:
        adj[a].add(b); adj[a].add(c)
        adj[b].add(a); adj[b].add(c)
        adj[c].add(a); adj[c].add(b)
    return adj

def k_hop_expand(seeds, adj, hops: int):
    """
    Expand from seed vertices using k-hop neighbors.
    """
    seeds = set(int(x) for x in seeds)
    if hops <= 0:
        return seeds
    frontier = set(seeds)
    visited = set(seeds)
    for _ in range(hops):
        nxt = set()
        for u in frontier:
            nxt.update(adj[u])
        nxt -= visited
        visited |= nxt
        frontier = nxt
        if not frontier:
            break
    return visited

def extract_submesh_by_vertices(V, F, keep_vertices_set, C):
    """
    Extract a submesh consisting of only the vertices in keep_vertices_set,
    and re-map the vertices to create a compact index.
    Also, update the color (C) array for the submesh.
    """
    keep = np.array(sorted(list(keep_vertices_set)), dtype=np.int64)

    # Check if the keep_vertices_set is empty or invalid
    if len(keep) == 0:
        return None, None, None

    keep_mask = np.zeros((V.shape[0],), dtype=bool)
    keep_mask[keep] = True

    F = np.asarray(F, dtype=np.int32)
    tri_keep = keep_mask[F].all(axis=1)
    F_sel = F[tri_keep]

    if F_sel.size == 0:
        return None, None, None  # No valid faces for the submesh

    # Get the vertices involved in the selected faces
    used_old = np.unique(F_sel.reshape(-1))
    V_new = V[used_old]
    
    # old -> new mapping
    remap = {int(old): i for i, old in enumerate(used_old)}
    F_new = np.vectorize(remap.get)(F_sel).astype(np.int32)

    # Rebuild colors (C) for the submesh
    C_new = C[used_old]
    return V_new, F_new, C_new

def reduce_sc_to_vertex(sc):
    """
    Reduce shape complementarity (SC) to vertex level.
    Common case: (2, N, 10) -> take sc[0] and median over ring -> (N,)
    Also handles (N, 10) / (N,)
    """
    sc = np.asarray(sc)
    sc = np.nan_to_num(sc)
    sc = np.squeeze(sc)
    if sc.ndim == 3:
        sc0 = sc[0]              # (N, 10) percentile 25
        return np.median(sc0, axis=1)
    if sc.ndim == 2:
        return np.median(sc, axis=1)
    if sc.ndim == 1:
        return sc
    raise ValueError(f"Unexpected sc shape: {sc.shape}")

# ----------------------------
# Core per-pdb_id visualization
# ----------------------------

def visualize_one_pdb_id(cfg, pdb_id, outdir, also_gray=False, hops_p1=0, hops_p2=0, use_cache=True, seed=0):
    """
    Visualize one protein pdb_id, coloring the p1 and p2 positive and negative samples.
    For each p1_pos (positive sample), generate three PLY files: p1, p2 positives, p2 negatives.
    If --also_gray is set, generate additional full gray meshes.
    """
    params = cfg["ppi_search"]
    ply_tpl = cfg["ply_file_template"]

    fields = pdb_id.split("_")
    if len(fields) < 3:
        raise ValueError(f"pdb_id must look like PDB_A_B, got {pdb_id}")
    pdbid, chain1, chain2 = fields[0], fields[1], fields[2]

    p1_ply = ply_tpl.format(pdbid, chain1)
    p2_ply = ply_tpl.format(pdbid, chain2)

    if not os.path.exists(p1_ply) or not os.path.exists(p2_ply):
        raise FileNotFoundError(f"Missing ply: p1={p1_ply} or p2={p2_ply}")

    cache_dir = params.get("cache_dir", None)
    precomp_root = params["masif_precomputation_dir"]
    precomp_dir = os.path.join(precomp_root, pdb_id)

    # Load meshes via PyMesh to keep vertex ordering consistent with MaSIF precompute
    m1 = pymesh.load_mesh(p1_ply)
    m2 = pymesh.load_mesh(p2_ply)
    V1, F1 = np.asarray(m1.vertices), np.asarray(m1.faces)
    V2, F2 = np.asarray(m2.vertices), np.asarray(m2.faces)

    # --- Get indices: p1_pos (k1), p2_neg (k_neg2) from cache if available ---
    p1_pos = None
    p2_neg = None

    if use_cache and cache_dir and os.path.isdir(cache_dir):
        pos_names_path = os.path.join(cache_dir, "pos_names.npy")
        neg_names_path = os.path.join(cache_dir, "neg_names.npy")
        if os.path.exists(pos_names_path):
            pos_names = np.load(pos_names_path, allow_pickle=True)
            p1_pos = [idx for (pp, pid, idx) in (parse_name_entry(str(x)) for x in pos_names)
                      if pp == pdb_id and pid == "p1"]
            p1_pos = np.array(sorted(set(p1_pos)), dtype=np.int64) if len(p1_pos) else np.array([], dtype=np.int64)
        if os.path.exists(neg_names_path):
            neg_names = np.load(neg_names_path, allow_pickle=True)
            p2_neg = [idx for (pp, pid, idx) in (parse_name_entry(str(x)) for x in neg_names)
                      if pp == pdb_id and pid == "p2"]
            p2_neg = np.array(sorted(set(p2_neg)), dtype=np.int64) if len(p2_neg) else np.array([], dtype=np.int64)

    # --- Fallback: recompute like cache script (if cache missing) ---
    if p1_pos is None:
        sc_path = os.path.join(precomp_dir, "p1_sc_labels.npy")
        if not os.path.exists(sc_path):
            raise FileNotFoundError(f"Missing {sc_path} (needed to recompute samples)")
        sc = np.load(sc_path, allow_pickle=True)
        labels = reduce_sc_to_vertex(sc)  # (N1,)
        min_sc = float(params["min_sc_filt"])
        max_sc = float(params["max_sc_filt"])
        pos_labels = np.where((labels < max_sc) & (labels > min_sc))[0]
        rng = np.random.default_rng(seed)
        K = int(float(params["pos_surf_accept_probability"]) * len(pos_labels))
        if K < 1:
            p1_pos = np.array([], dtype=np.int64)
        else:
            sel = rng.permutation(len(pos_labels))[:K]
            l = pos_labels[sel]
            # then contact filtering
            kdt = cKDTree(V2)
            d, r = kdt.query(V1[l])
            contact = np.where(d < float(params["pos_interface_cutoff"]))[0]
            p1_pos = l[contact].astype(np.int64)

    # --- Recover p2 positives (k2) from p1 positives via nearest neighbor ---
    if len(p1_pos) > 0:
        kdt2 = cKDTree(V2)
        d, k2 = kdt2.query(V1[p1_pos])
        # optionally enforce cutoff (should already be true if cache)
        cutoff = float(params.get("pos_interface_cutoff", 1.5))
        good = np.where(d < cutoff)[0]
        p1_pos = p1_pos[good]
        p2_pos = k2[good].astype(np.int64)
    else:
        p2_pos = np.array([], dtype=np.int64)

    if p2_neg is None:
        # fallback recompute negatives like cache
        if len(p2_pos) == 0:
            p2_neg = np.array([], dtype=np.int64)
        else:
            kdt = cKDTree(V1[p1_pos] if len(p1_pos) else V1)
            dneg, _ = kdt.query(V2)
            far = np.where(dneg > float(params["pos_interface_cutoff"]))[0]
            rng = np.random.default_rng(seed)
            rng.shuffle(far)
            p2_neg = far[: len(p2_pos)].astype(np.int64)

    # --- Build colors ---
    gray = np.array([0.75, 0.75, 0.75], dtype=np.float64)
    green = np.array([0.10, 1.00, 0.10], dtype=np.float64)  # p1 positives
    red   = np.array([1.00, 0.20, 0.20], dtype=np.float64)  # p2 positives
    blue  = np.array([0.20, 0.35, 1.00], dtype=np.float64)  # p2 negatives
    magenta = np.array([1.00, 0.20, 1.00], dtype=np.float64)  # overlap (rare)

    # --- Write outputs ---
    pdb_id_out = os.path.join(outdir, pdb_id)
    os.makedirs(pdb_id_out, exist_ok=True)

    # If --also_gray is set, save the full gray mesh for p1
    if also_gray:
        p1_gray_path = os.path.join(pdb_id_out, f"{pdb_id}_p1_full_gray.ply")
        write_triangle_ply(p1_gray_path, V1, F1, np.tile(gray[None, :], (V1.shape[0], 1)))
        p2_gray_path = os.path.join(pdb_id_out, f"{pdb_id}_p2_full_gray.ply")
        write_triangle_ply(p2_gray_path, V2, F2, np.tile(gray[None, :], (V2.shape[0], 1)))

    # Loop through each p1_pos, p2_pos, and p2_neg to save three files for each
    for idx, (idx_p1, idx_p2, idx_neg) in enumerate(zip_longest(p1_pos, p2_pos, p2_neg, fillvalue=None)):

        C1 = np.tile(gray[None, :], (V1.shape[0], 1))
        C2 = np.tile(gray[None, :], (V2.shape[0], 1))
        C3 = np.tile(gray[None, :], (V2.shape[0], 1))
        C1[idx_p1] = green
        C2[idx_p2] = red
        C3[idx_neg] = blue

        # Prepare output directory for each idx
        idx_out_dir = os.path.join(pdb_id_out, f"{idx}")
        os.makedirs(idx_out_dir, exist_ok=True)

        # Processing p1_pos elements
        if idx_p1 is not None:
            p1_col_path = os.path.join(idx_out_dir, f"{pdb_id}_p1_{idx_p1}_train_samples.ply")
            adj1 = build_vertex_adjacency(V1.shape[0], F1)
            keep1 = k_hop_expand([idx_p1], adj1, hops=1)  # 1-hop neighborhood
            V1s, F1s, C1s = extract_submesh_by_vertices(V1, F1, keep1, C1)
            write_triangle_ply(p1_col_path, V1s, F1s, C1s)

        # Processing p2_pos elements
        if idx_p2 is not None:
            p2_col_path = os.path.join(idx_out_dir, f"{pdb_id}_p2_{idx_p2}_train_samples.ply")
            adj2 = build_vertex_adjacency(V2.shape[0], F2)
            keep2 = k_hop_expand([idx_p2], adj2, hops=1)  # 1-hop neighborhood
            V2s, F2s, C2s = extract_submesh_by_vertices(V2, F2, keep2, C2)
            write_triangle_ply(p2_col_path, V2s, F2s, C2s)

        # Processing p2_neg elements
        if idx_neg is not None:
            p2_neg_col_path = os.path.join(idx_out_dir, f"{pdb_id}_p2_neg_{idx_neg}_train_samples.ply")
            adj2 = build_vertex_adjacency(V2.shape[0], F2)
            keep2 = k_hop_expand([idx_neg], adj2, hops=1)  # 1-hop neighborhood
            V2s, F2s, C3s = extract_submesh_by_vertices(V2, F2, keep2, C3)
            write_triangle_ply(p2_neg_col_path, V2s, F2s, C3s)

        print(f"Processed idx: {idx}, p1_pos: {idx_p1}, p2_pos: {idx_p2}, p2_neg: {idx_neg}")
    
    print(f"[OK] {pdb_id}")


def read_list_file(path: str):
    """
    Read a list file, ignoring blank lines and comments (#...).
    Each line typically looks like: PDBID_A_B
    """
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    return out

# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Visualize MaSIF-search cached training samples: p1 positives (green), p2 positives (red), p2 negatives (blue)."
    )
    ap.add_argument("-c", "--config", required=True, help="YAML config")
    ap.add_argument("--pdb_id", action="append", default=[], help="ppi pdb_id id like 1A0G_A_B (can repeat)")
    ap.add_argument("--list", default=None, help="Optional list file of pdb_ids to visualize (one per line)")
    ap.add_argument("--outdir", default="./viz_out/dataset", help="Output directory")
    ap.add_argument("--also_gray", action="store_true", help="Also save full gray meshes (p1 and p2)")
    ap.add_argument("--hops_p1", type=int, default=0, help="If >0, also export p1 patch-only mesh (k-hop around p1 positives)")
    ap.add_argument("--hops_p2", type=int, default=0, help="If >0, also export p2 patch-only mesh (k-hop around p2 pos+neg)")
    ap.add_argument("--no_cache", action="store_true", help="Do not use cache_dir/pos_names.npy,neg_names.npy (recompute from SC)")
    ap.add_argument("--seed", type=int, default=0, help="Seed for recompute fallback")
    ap.add_argument("--max_pdb_ids", type=int, default=0, help="Limit number of pdb_ids (0=all)")
    args = ap.parse_args()

    cfg = load_config(args.config)

    pdb_ids = []
    if args.list is not None:
        pdb_ids.extend(read_list_file(args.list))
    pdb_ids.extend(args.pdb_id)

    if not pdb_ids:
        # default: visualize training_list
        tlist = cfg["ppi_search"].get("training_list", None)
        if not tlist:
            raise ValueError("No --pdb_id/--list given and config has no ppi_search.training_list")
        pdb_ids = read_list_file(tlist)

    if args.max_pdb_ids and args.max_pdb_ids > 0:
        pdb_ids = pdb_ids[: args.max_pdb_ids]

    os.makedirs(args.outdir, exist_ok=True)

    for pdb_id in pdb_ids:
        visualize_one_pdb_id(
            cfg=cfg,
            pdb_id=pdb_id,
            outdir=args.outdir,
            also_gray=args.also_gray,
            hops_p1=args.hops_p1,
            hops_p2=args.hops_p2,
            use_cache=(not args.no_cache),
            seed=args.seed,
        )

if __name__ == "__main__":
    main()

