#!/usr/bin/env python3
import os
import argparse
import numpy as np
import yaml

def load_npy(path):
    return np.load(path, allow_pickle=True)

def load_config(path: str) -> dict:
    """Load YAML config and expand '{exp}' placeholders if present."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    exp = cfg.get("exp")
    if exp:
        # Expand top-level strings
        for k, v in list(cfg.items()):
            if isinstance(v, str):
                cfg[k] = v.replace("{exp}", str(exp))
        # Expand nested ppi_search strings
        if isinstance(cfg.get("ppi_search"), dict):
            for k, v in list(cfg["ppi_search"].items()):
                if isinstance(v, str):
                    cfg["ppi_search"][k] = v.replace("{exp}", str(exp))

    # Minimal keys needed for this script
    if "ply_file_template" not in cfg or not cfg["ply_file_template"]:
        raise ValueError("Config missing required key: ply_file_template")
    if "ppi_search" not in cfg or not cfg["ppi_search"]:
        raise ValueError("Config missing required key: ppi_search")
    if "masif_precomputation_dir" not in cfg["ppi_search"] or not cfg["ppi_search"]["masif_precomputation_dir"]:
        raise ValueError("Config missing required key: ppi_search.masif_precomputation_dir")

    return cfg

def parse_pdb_id(pdb_id: str):
    """
    Parse pdb_id name like '1A0G_A_B' or '1A0G_A'.
    Returns (pdb_upper, chain1, chain2_or_None).
    """
    fields = pdb_id.strip().split("_")
    if len(fields) < 2:
        raise ValueError("pdb_id must look like PDB_CHAIN or PDB_CHAIN1_CHAIN2 (e.g. 1A0G_A_B)")
    pdb = fields[0].upper()
    c1 = fields[1]
    c2 = fields[2] if len(fields) >= 3 and fields[2] != "" else None
    return pdb, c1, c2

def get_paths_from_cfg(cfg: dict, pdb_id: str, pid: str):
    """
    Build:
      - precomp_dir: <masif_precomputation_dir>/<pdb_id>
      - ply_path:    ply_file_template.format(PDB, chain_for_pid)
    """
    pdb, c1, c2 = parse_pdb_id(pdb_id)
    precomp_root = cfg["ppi_search"]["masif_precomputation_dir"]
    precomp_dir = os.path.join(precomp_root, pdb_id)

    if pid == "p1":
        chain = c1
    else:
        if c2 is None:
            raise ValueError(f"pdb_id '{pdb_id}' has no second chain, but pid=p2 was requested.")
        chain = c2

    ply_path = cfg["ply_file_template"].format(pdb, chain)
    return precomp_dir, ply_path

def reduce_sc_to_vertex(sc, N):
    """Reduce sc to shape (N,) from shapes like (2,N,10), (1,N,10), (N,10), (N,)"""
    sc = np.asarray(sc, dtype=float)
    sc = np.nan_to_num(sc)
    sc = np.squeeze(sc)

    if sc.ndim == 3:
        # (B,N,K) -> (B,N)
        sc = np.median(sc, axis=2)
        # (B,N) -> (N,)
        sc = sc.max(axis=0) if sc.shape[0] > 1 else sc[0]
    elif sc.ndim == 2:
        # (N,K) -> (N,)
        sc = np.median(sc, axis=1)
    elif sc.ndim == 1:
        pass
    else:
        raise ValueError(f"Unexpected sc shape: {sc.shape}")

    if sc.shape[0] != N:
        raise ValueError(f"sc length mismatch: sc={sc.shape}, N={N}")
    return sc

def reduce_mask_to_vertex(mask, N):
    """Reduce mask to (N,) from shapes like (N,200), (1,N,200), (N,) etc."""
    m = np.asarray(mask)
    m = np.squeeze(m)

    if m.ndim == 1 and m.shape[0] == N:
        return m.astype(bool)

    if m.ndim == 2:
        if m.shape[0] == N:
            return np.any(m > 0, axis=1)
        if m.shape[1] == N:
            return np.any(m > 0, axis=0)

    # Fallback: treat all valid
    return np.ones((N,), dtype=bool)

def normalize01(x):
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def write_facecolored_ply(path, V, F, face_rgb_uint8):
    """
    Write ASCII PLY with per-face color (red/green/blue on face element).
    Many viewers support it (MeshLab/CloudCompare often do).
    """
    V = np.asarray(V, dtype=np.float32)
    F = np.asarray(F, dtype=np.int32)
    C = np.asarray(face_rgb_uint8, dtype=np.uint8)
    assert V.shape[1] == 3 and F.shape[1] == 3 and C.shape[1] == 3
    nV, nF = V.shape[0], F.shape[0]

    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {nV}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {nF}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(nV):
            f.write(f"{V[i,0]} {V[i,1]} {V[i,2]}\n")
        for i in range(nF):
            a, b, c = F[i]
            r, g, b2 = C[i]
            f.write(f"3 {a} {b} {c} {int(r)} {int(g)} {int(b2)}\n")

def main():
    ap = argparse.ArgumentParser(
        description="Color a triangle mesh by MaSIF labels and export a face-colored PLY (reads paths from --config)."
    )
    ap.add_argument("--config", "-c", required=True, help="Path to YAML config (must include ply_file_template and ppi_search.masif_precomputation_dir).")
    ap.add_argument("--pdb_id", required=True, help="Entry name, e.g. 1A0G_A_B")
    ap.add_argument("--pid", default="p1", choices=["p1", "p2"], help="Which side labels to use (p1 uses chain1; p2 uses chain2).")
    ap.add_argument("--mode", default="iface", choices=["iface", "sc"], help="Color mode.")
    ap.add_argument("--face_rule", default="any", choices=["any", "mean", "majority"],
                    help="How to aggregate vertex labels to faces (iface mode).")
    ap.add_argument("--outdir", default=None, help="Output Directory (default: same as precomputation dir).")
    ap.add_argument("--show", action="store_true", help="Show mesh with Open3D (requires GUI).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    precomp_dir, ply_path = get_paths_from_cfg(cfg, args.pdb_id, args.pid)

    import open3d as o3d

    # Load triangle mesh from PLY (must contain triangles).
    mesh = o3d.io.read_triangle_mesh(ply_path)
    if mesh.is_empty():
        raise RuntimeError(f"Failed to load mesh: {ply_path}")
    if len(mesh.triangles) == 0:
        raise RuntimeError(f"Loaded mesh has no triangles: {ply_path}")

    V = np.asarray(mesh.vertices, dtype=np.float32)
    F = np.asarray(mesh.triangles, dtype=np.int32)
    N = V.shape[0]

    # Load (optional) mask and reduce to per-vertex boolean.
    mask_path = os.path.join(precomp_dir, f"{args.pid}_mask.npy")
    if os.path.exists(mask_path):
        mask1d = reduce_mask_to_vertex(load_npy(mask_path), N)
    else:
        mask1d = np.ones((N,), dtype=bool)

    if args.mode == "iface":
        lab_path = os.path.join(precomp_dir, f"{args.pid}_iface_labels.npy")
        lab = load_npy(lab_path)
        lab = np.squeeze(np.asarray(lab))

        # Reduce possible shapes to (N,)
        if lab.ndim == 2 and lab.shape[0] == 1:
            lab = lab[0]
        if lab.ndim == 2:
            # (N,K) -> (N,), take max
            lab = np.max(np.nan_to_num(lab), axis=1)

        if lab.ndim != 1 or lab.shape[0] != N:
            raise ValueError(f"iface label shape mismatch: got {lab.shape}, expected (N,) with N={N}")

        vscore = (lab > 0.5).astype(np.float32)
        tri_vs = vscore[F]  # (nF, 3)

        if args.face_rule == "any":
            tscore = (np.max(tri_vs, axis=1) > 0.5).astype(np.float32)
        elif args.face_rule == "majority":
            tscore = (np.sum(tri_vs > 0.5, axis=1) >= 2).astype(np.float32)
        else:  # mean
            tscore = (np.mean(tri_vs, axis=1) > 0.5).astype(np.float32)

        # Red for interface faces, gray otherwise.
        face_rgb = np.zeros((F.shape[0], 3), dtype=np.float32) + 0.6
        face_rgb[tscore > 0.5] = np.array([1.0, 0.2, 0.2], dtype=np.float32)

    else:
        sc_path = os.path.join(precomp_dir, f"{args.pid}_sc_labels.npy")
        sc = load_npy(sc_path)
        sc = reduce_sc_to_vertex(sc, N)
        sc = normalize01(sc)

        tri_sc = sc[F]  # (nF, 3)
        tscore = normalize01(np.mean(tri_sc, axis=1))

        # Blue->red ramp.
        face_rgb = np.zeros((F.shape[0], 3), dtype=np.float32)
        face_rgb[:, 0] = tscore
        face_rgb[:, 2] = 1.0 - tscore
        face_rgb[:, 1] = 0.2

    # Darken faces touching any masked-out vertex.
    tri_mask = mask1d[F]  # (nF, 3)
    bad_face = ~np.all(tri_mask, axis=1)
    face_rgb[bad_face] *= 0.2

    out_dir = os.path.join(args.outdir, f"{args.pdb_id}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.pdb_id}__{args.pid}__{args.mode}.ply")
    out_path = os.path.abspath(out_path)
    write_facecolored_ply(out_path, V, F, (np.clip(face_rgb, 0, 1) * 255).astype(np.uint8))
    print("[OK] wrote:", out_path)
    print("[INFO] used ply:", ply_path)
    print("[INFO] used precomp_dir:", precomp_dir)

    if args.show:
        mesh.triangle_colors = o3d.utility.Vector3dVector(face_rgb.astype(np.float64))
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh])

if __name__ == "__main__":
    main()
