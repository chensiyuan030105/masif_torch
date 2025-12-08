#!/usr/bin/env python3
import os
import argparse
import numpy as np
import yaml
import pymesh
from scipy.spatial import cKDTree
from itertools import zip_longest

# ----------------------------
# Utils
# ----------------------------

def load_config(path: str) -> dict:
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


def robust01(x, lo=5.0, hi=95.0):
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x)
    a = np.percentile(x, lo)
    b = np.percentile(x, hi)
    if b - a < 1e-12:
        return np.zeros_like(x)
    y = (x - a) / (b - a)
    return np.clip(y, 0.0, 1.0)


def enhance_contrast(t01, cscale=1.0, gamma=1.0):
    t = np.asarray(t01, dtype=np.float64)
    if cscale != 1.0:
        t = np.clip((t - 0.5) * float(cscale) + 0.5, 0.0, 1.0)
    if gamma != 1.0:
        t = np.power(t, float(gamma))
    return np.clip(t, 0.0, 1.0)


def red_blue_ramp(t01):
    # t=0 (small dist)->red, t=1 (large dist)->blue
    t = np.asarray(t01, dtype=np.float64)
    r = 1.0 - t
    g = np.full_like(t, 0.2)
    b = t
    return np.stack([r, g, b], axis=1)


def write_triangle_ply_rgb(path, V, F, rgb01):
    """
    ASCII PLY triangle mesh with per-vertex colors.
    Supports F empty (0,3): will write vertices-only PLY (still valid PLY).
    """
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int32).reshape(-1, 3) if F is not None else np.zeros((0, 3), dtype=np.int32)
    C = (np.clip(np.asarray(rgb01, dtype=np.float64), 0, 1) * 255.0).astype(np.uint8)

    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError(f"V must be (N,3), got {V.shape}")
    if C.shape != (V.shape[0], 3):
        raise ValueError(f"rgb01 must be (N,3) matching V, got {C.shape}")

    nV, nF = V.shape[0], F.shape[0]
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
            
def build_vertex_adjacency(F, nV):
    adj = [set() for _ in range(nV)]
    F = np.asarray(F, dtype=np.int64)
    for a, b, c in F:
        adj[a].update((b, c))
        adj[b].update((a, c))
        adj[c].update((a, b))
    return adj


def k_ring_vertices(seed, adj, k):
    seed = int(seed)
    visited = {seed}
    frontier = {seed}
    for _ in range(int(k)):
        nxt = set()
        for v in frontier:
            nxt |= adj[v]
        nxt -= visited
        visited |= nxt
        frontier = nxt
        if not frontier:
            break
    return np.array(sorted(visited), dtype=np.int64)


def extract_submesh(V, F, keep_vids, face_rule="all"):
    """
    keep_vids: 1D int array of old vertex indices to keep (as region core)
    face_rule:
      - 'all': keep a face only if ALL 3 vertices in keep_vids (clean patch)
      - 'any': keep a face if ANY vertex in keep_vids (bigger patch)
    Returns: (V_new, F_new, old_to_new_map_dict)
    """
    V = np.asarray(V)
    F = np.asarray(F, dtype=np.int64)
    keep_vids = np.asarray(keep_vids, dtype=np.int64)
    keep_mask = np.zeros((V.shape[0],), dtype=bool)
    keep_mask[keep_vids] = True

    tri_keep = keep_mask[F]  # (nF,3) bool
    if face_rule == "any":
        fmask = np.any(tri_keep, axis=1)
    else:
        fmask = np.all(tri_keep, axis=1)

    F_sel = F[fmask]
    if F_sel.size == 0:
        used_old = np.array(sorted(set(int(x) for x in keep_vids.tolist())), dtype=np.int64)
        V_new = V[used_old]
        F_new = np.zeros((0, 3), dtype=np.int32)
        old_to_new = {int(old): i for i, old in enumerate(used_old)}
        return V_new, F_new, old_to_new

    used_old = np.unique(F_sel.reshape(-1))
    V_new = V[used_old]
    remap = {int(old): i for i, old in enumerate(used_old)}
    F_new = np.vectorize(lambda x: remap[int(x)], otypes=[np.int32])(F_sel).astype(np.int32)
    return V_new, F_new, remap


def main():
    ap = argparse.ArgumentParser(
        description="Outputs 5 PLYs: p2 full descdist (unchanged), p1 target patch, p2 min patch, plus p1 full gray and p2 full gray."
    )
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--pdb_id", required=True, help="ppi_pair_id, e.g. 1A1U_A_C")

    ap.add_argument("--use", choices=["p1_str__p2_flip", "p1_flip__p2_str"], default="p1_str__p2_flip")
    ap.add_argument("--target", choices=["sc_center", "idx"], default="sc_center")
    ap.add_argument("--target_idx", type=int, default=None)

    # keep p2 full descdist behavior as before
    ap.add_argument("--norm_lo", type=float, default=5.0)
    ap.add_argument("--norm_hi", type=float, default=95.0)
    ap.add_argument("--cscale", type=float, default=2.0)
    ap.add_argument("--gamma", type=float, default=0.7)

    # local patch size
    ap.add_argument("--p1_rings", type=int, default=1, help="k-ring neighborhood for p1 target patch")
    ap.add_argument("--p2_rings", type=int, default=1, help="k-ring neighborhood for p2 min patch")
    ap.add_argument("--face_rule", choices=["all", "any"], default="all")

    ap.add_argument("--outdir", default=None, help="Output root dir; will create <outdir>/<pdb_id>/")
    ap.add_argument("--save_dist", action="store_true", help="Also save p2 distances as .npy")

    args = ap.parse_args()

    cfg = load_config(args.config)
    params = cfg["ppi_search"]
    desc_dir = params["desc_dir"]
    precomp_root = params["masif_precomputation_dir"]

    fields = args.pdb_id.split("_")
    if len(fields) < 3:
        raise ValueError("--pdb_id must look like PDB_A_B")
    pdbid, chain1, chain2 = fields[0], fields[1], fields[2]

    p1_ply = cfg["ply_file_template"].format(pdbid, chain1)
    p2_ply = cfg["ply_file_template"].format(pdbid, chain2)

    pdesc_dir = os.path.join(desc_dir, args.pdb_id)
    precomp_dir = os.path.join(precomp_root, args.pdb_id)

    if args.use == "p1_str__p2_flip":
        p1_desc_path = os.path.join(pdesc_dir, "p1_desc_straight.npy")
        p2_desc_path = os.path.join(pdesc_dir, "p2_desc_flipped.npy")
    else:
        p1_desc_path = os.path.join(pdesc_dir, "p1_desc_flipped.npy")
        p2_desc_path = os.path.join(pdesc_dir, "p2_desc_straight.npy")

    for p in (p1_ply, p2_ply, p1_desc_path, p2_desc_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")

    # load descriptors
    desc1 = np.load(p1_desc_path)  # (N1,D)
    desc2 = np.load(p2_desc_path)  # (N2,D)
    if desc1.ndim != 2 or desc2.ndim != 2 or desc1.shape[1] != desc2.shape[1]:
        raise ValueError(f"Descriptor mismatch: desc1 {desc1.shape}, desc2 {desc2.shape}")

    # load meshes
    m1 = pymesh.load_mesh(p1_ply)
    m2 = pymesh.load_mesh(p2_ply)
    V1, F1 = m1.vertices, m1.faces
    V2, F2 = m2.vertices, m2.faces

    if V1.shape[0] != desc1.shape[0]:
        raise ValueError(f"p1 vertex mismatch: mesh={V1.shape[0]} desc={desc1.shape[0]}")
    if V2.shape[0] != desc2.shape[0]:
        raise ValueError(f"p2 vertex mismatch: mesh={V2.shape[0]} desc={desc2.shape[0]}")

    # choose p1 target vertex
    if args.target == "idx":
        if args.target_idx is None:
            raise ValueError("--target idx requires --target_idx")
        t_idx = int(args.target_idx)
    else:
        sc_path = os.path.join(precomp_dir, "p1_sc_labels.npy")
        if not os.path.exists(sc_path):
            raise FileNotFoundError(f"Missing SC labels: {sc_path}")
        sc = np.load(sc_path)
        sc0 = np.nan_to_num(sc[0])
        per_v = np.median(sc0, axis=1)
        t_idx = int(np.argmax(per_v))

    if not (0 <= t_idx < desc1.shape[0]):
        raise ValueError(f"target idx out of range: {t_idx}")

    # compute p2 distances
    t_desc = desc1[t_idx]
    d = np.linalg.norm(desc2 - t_desc[None, :], axis=1)
    min_i = int(np.argmin(d))

    # === p2 full descdist (UNCHANGED) ===
    t01 = robust01(d, lo=args.norm_lo, hi=args.norm_hi)
    t01 = enhance_contrast(t01, cscale=args.cscale, gamma=args.gamma)
    p2_rgb_full = red_blue_ramp(t01)

    # === p1 patch (only small region) ===
    adj1 = build_vertex_adjacency(F1, V1.shape[0])
    keep1 = k_ring_vertices(t_idx, adj1, args.p1_rings)
    V1p, F1p, map1 = extract_submesh(V1, F1, keep1, face_rule=args.face_rule)

    p1_patch_rgb = np.tile(np.array([[0.75, 0.75, 0.75]], dtype=np.float64), (V1p.shape[0], 1))
    if t_idx in map1:
        p1_patch_rgb[map1[t_idx]] = np.array([0.1, 1.0, 0.1], dtype=np.float64)

    # === p2 min patch (only small region) ===
    adj2 = build_vertex_adjacency(F2, V2.shape[0])
    keep2 = k_ring_vertices(min_i, adj2, args.p2_rings)
    V2p, F2p, map2 = extract_submesh(V2, F2, keep2, face_rule=args.face_rule)

    p2_min_patch_rgb = np.tile(np.array([[0.75, 0.75, 0.75]], dtype=np.float64), (V2p.shape[0], 1))
    if min_i in map2:
        p2_min_patch_rgb[map2[min_i]] = np.array([0.1, 1.0, 0.1], dtype=np.float64)

    # === full gray meshes (ALL triangles) ===
    p1_full_gray = np.tile(np.array([[0.75, 0.75, 0.75]], dtype=np.float64), (V1.shape[0], 1))
    p2_full_gray = np.tile(np.array([[0.75, 0.75, 0.75]], dtype=np.float64), (V2.shape[0], 1))

    # output dir
    if args.outdir is None:
        outdir = os.path.abspath(".")
    else:
        outdir = os.path.join(args.outdir, args.pdb_id)
    os.makedirs(outdir, exist_ok=True)

    base = f"{args.pdb_id}__{args.use}__target{t_idx}"

    out_p2_full_desc = os.path.join(outdir, base + "__p2_descdist.ply")                 # unchanged
    out_p1_patch     = os.path.join(outdir, base + f"__p1_patch_r{args.p1_rings}.ply")
    out_p2_min_patch = os.path.join(outdir, base + f"__p2_minpatch_v{min_i}_r{args.p2_rings}.ply")

    out_p1_gray_full = os.path.join(outdir, base + "__p1_full_gray.ply")               # full mesh gray
    out_p2_gray_full = os.path.join(outdir, base + "__p2_full_gray.ply")               # full mesh gray

    # write
    write_triangle_ply_rgb(out_p2_full_desc, V2, F2, p2_rgb_full)
    write_triangle_ply(out_p1_patch, V1p, F1p, p1_patch_rgb)
    write_triangle_ply(out_p2_min_patch, V2p, F2p, p2_min_patch_rgb)
    write_triangle_ply_rgb(out_p1_gray_full, V1, F1, p1_full_gray)
    write_triangle_ply_rgb(out_p2_gray_full, V2, F2, p2_full_gray)

    print("[OK] wrote p2 full descdist (unchanged):", out_p2_full_desc)
    print("[OK] wrote p1 target patch:", out_p1_patch)
    print("[OK] wrote p2 min patch   :", out_p2_min_patch)
    print("[OK] wrote p1 full gray   :", out_p1_gray_full)
    print("[OK] wrote p2 full gray   :", out_p2_gray_full)
    print(f"[INFO] p1 target idx={t_idx}, p2 min idx={min_i}, min dist={d[min_i]:.6g}")

    if args.save_dist:
        out_npy = os.path.join(outdir, base + "__p2_descdist.npy")
        np.save(out_npy, d.astype(np.float32))
        print("[OK] wrote dist npy:", out_npy)


if __name__ == "__main__":
    main()



