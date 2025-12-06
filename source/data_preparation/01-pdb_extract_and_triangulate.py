#!/usr/bin/env python3
import os
import sys
import shutil
import argparse
import yaml
import numpy as np

from IPython.core.debugger import set_trace

# Local includes (unchanged)
from triangulation.computeMSMS import computeMSMS
from triangulation.fixmesh import fix_mesh
import pymesh
from input_output.extractPDB import extractPDB
from input_output.save_ply import save_ply
from input_output.protonate import protonate
from triangulation.computeHydrophobicity import computeHydrophobicity
from triangulation.computeCharges import computeCharges, assignChargesToNewMesh
from triangulation.computeAPBS import computeAPBS
from triangulation.compute_normal import compute_normal
from sklearn.neighbors import KDTree


def load_config(path: str) -> dict:
    """Load YAML config and do a small amount of compatibility handling."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Optional: expand "{exp}" placeholders if present
    exp = cfg.get("exp")
    if exp:
        for k, v in list(cfg.items()):
            if isinstance(v, str):
                cfg[k] = v.replace("{exp}", str(exp))
        if isinstance(cfg.get("ppi_search"), dict):
            for k, v in list(cfg["ppi_search"].items()):
                if isinstance(v, str):
                    cfg["ppi_search"][k] = v.replace("{exp}", str(exp))

    # Required keys for this script
    required = ["raw_pdb_dir", "tmp_dir", "ply_chain_dir", "pdb_chain_dir", "ppi_search"]
    for k in required:
        if k not in cfg or not cfg[k]:
            raise ValueError(f"Config missing required key: {k}")

    if "training_list" not in cfg["ppi_search"] or not cfg["ppi_search"]["training_list"]:
        raise ValueError("Config missing required key: ppi_search.training_list")

    return cfg


def ensure_dirs(cfg: dict):
    """Create directories if they do not exist."""
    os.makedirs(cfg["tmp_dir"], exist_ok=True)
    os.makedirs(cfg["raw_pdb_dir"], exist_ok=True)
    os.makedirs(cfg["ply_chain_dir"], exist_ok=True)
    os.makedirs(cfg["pdb_chain_dir"], exist_ok=True)


def read_list(list_path: str) -> list[str]:
    """
    Read a list file, ignore blank lines and comments (#...).
    Each line typically looks like: PDBID_A_B
    """
    if not os.path.exists(list_path):
        raise FileNotFoundError(f"List file not found: {list_path}")

    out = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    return out


def parse_entry(entry: str):
    """
    Parse a line like '1abc_A_B' -> ('1abc', 'A', 'B')
    If only one chain exists '1abc_A' -> ('1abc','A',None)
    """
    parts = entry.split("_")
    pdb_id = parts[0].strip().lower()
    chain_a = parts[1].strip() if len(parts) > 1 else None
    chain_b = parts[2].strip() if len(parts) > 2 else None
    return pdb_id, chain_a, chain_b


def output_paths(cfg: dict, pdb_id: str, chain_id: str):
    """
    Define final output locations (where we consider 'already processed').
    We treat processed = both files exist and are non-empty:
      - ply_chain_dir/PDBID_CHAIN.ply
      - pdb_chain_dir/PDBID_CHAIN.pdb
    """
    tag = f"{pdb_id.upper()}_{chain_id}"
    ply_final = os.path.join(cfg["ply_chain_dir"], f"{tag}.ply")
    pdb_final = os.path.join(cfg["pdb_chain_dir"], f"{tag}.pdb")
    return ply_final, pdb_final, tag


def already_done(ply_path: str, pdb_path: str) -> bool:
    """Return True if both outputs exist and are non-empty."""
    return (
        os.path.exists(ply_path) and os.path.getsize(ply_path) > 0 and
        os.path.exists(pdb_path) and os.path.getsize(pdb_path) > 0
    )


def process_one_chain(cfg: dict, pdb_id: str, chain_id: str, force: bool = False) -> bool:
    """
    Process one (PDBID, chain) into a surface ply and chain pdb.
    Returns True if work was done, False if skipped.
    """
    ply_final, pdb_final, tag = output_paths(cfg, pdb_id, chain_id)

    # Skip if outputs already exist
    if (not force) and already_done(ply_final, pdb_final):
        print(f"[INFO] Already processed, skipping: {tag}")
        return False

    # Raw PDB expected from your downloader: RAW/PDBID.pdb (uppercase)
    raw_pdb = os.path.join(cfg["raw_pdb_dir"], f"{pdb_id.upper()}.pdb")
    if not os.path.exists(raw_pdb):
        print(f"[ERROR] Raw PDB not found (run download first?): {raw_pdb}")
        return False

    tmp_dir = cfg["tmp_dir"]

    # Protonate into tmp (keep per-pdb file to reuse)
    protonated_file = os.path.join(tmp_dir, f"{pdb_id.upper()}.pdb")
    if (not os.path.exists(protonated_file)) or os.path.getsize(protonated_file) == 0 or force:
        print(f"[INFO] Protonating PDB: {raw_pdb} -> {protonated_file}")
        protonate(raw_pdb, protonated_file)
    else:
        print(f"[INFO] Using existing protonated PDB: {protonated_file}")

    pdb_filename = protonated_file

    # Extract chain into tmp
    out_prefix = os.path.join(tmp_dir, f"{pdb_id.upper()}_{chain_id}")
    chain_pdb_tmp = out_prefix + ".pdb"
    print(f"[INFO] Extracting chain {chain_id} -> {chain_pdb_tmp}")
    extractPDB(pdb_filename, chain_pdb_tmp, chain_id)

    # Compute MSMS
    print(f"[INFO] Computing MSMS surface for: {chain_pdb_tmp}")
    try:
        vertices1, faces1, normals1, names1, areas1 = computeMSMS(chain_pdb_tmp, protonate=True)
        print(f"[INFO] MSMS computed: vertices={len(vertices1)}, faces={len(faces1)}")
    except Exception as e:
        print(f"[ERROR] MSMS failed for {chain_pdb_tmp}")
        print(e)
        set_trace()
        raise

    # Charges / Hydrophobicity
    if cfg.get("use_hbond", False):
        print("[INFO] Computing charges (HBond feature)...")
        vertex_hbond = computeCharges(out_prefix, vertices1, names1)
    else:
        vertex_hbond = None

    if cfg.get("use_hphob", False):
        print("[INFO] Computing hydrophobicity...")
        vertex_hphobicity = computeHydrophobicity(names1)
    else:
        vertex_hphobicity = None

    # Regularize mesh
    mesh = pymesh.form_mesh(vertices1, faces1)
    mesh_res = float(cfg.get("mesh_res", 1.0))
    print(f"[INFO] Fixing mesh with resolution {mesh_res} ...")
    regular_mesh = fix_mesh(mesh, mesh_res)
    print(f"[INFO] Regular mesh vertices: {len(regular_mesh.vertices)}, faces: {len(regular_mesh.faces)}")

    vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)

    if cfg.get("use_hbond", False) and vertex_hbond is not None:
        vertex_hbond = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hbond, cfg)

    if cfg.get("use_hphob", False) and vertex_hphobicity is not None:
        vertex_hphobicity = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hphobicity, cfg)

    if cfg.get("use_apbs", False):
        print("[INFO] Computing APBS charges...")
        vertex_charges = computeAPBS(regular_mesh.vertices, chain_pdb_tmp, out_prefix)
    else:
        vertex_charges = None

    # Interface (optional)
    iface = np.zeros(len(regular_mesh.vertices))
    if cfg.get("compute_iface", False):
        print("[INFO] Computing interface surface...")
        v3, f3, _, _, _ = computeMSMS(pdb_filename, protonate=True)
        mesh_full = pymesh.form_mesh(v3, f3)
        v3 = mesh_full.vertices
        kdt = KDTree(v3)
        d, _ = kdt.query(regular_mesh.vertices)
        d = np.square(d)
        iface_v = np.where(d >= 2.0)[0]
        iface[iface_v] = 1.0

    # Save in tmp then copy to final dirs
    ply_tmp = out_prefix + ".ply"
    print(f"[INFO] Saving mesh to: {ply_tmp}")
    save_ply(
        ply_tmp,
        regular_mesh.vertices,
        regular_mesh.faces,
        normals=vertex_normal,
        charges=vertex_charges,
        normalize_charges=True,
        hbond=vertex_hbond,
        hphob=vertex_hphobicity,
        iface=iface,
    )

    print("[INFO] Copying results to output directories...")
    shutil.copy(ply_tmp, ply_final)
    shutil.copy(chain_pdb_tmp, pdb_final)

    print(f"[✅ DONE] Processed {tag}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Batch process PDB chains from ppi_search.training_list: extract chain, build surface, save ply."
    )
    parser.add_argument(
        "-c", "--config",
        required=True,
        help="Path to YAML config file (converted from masif_opts).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if outputs already exist.",
    )
    parser.add_argument(
        "--chains",
        choices=["A", "B", "both"],
        default="both",
        help="Which chain(s) to process from each line 'PDBID_A_B'. Default: both.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    list_path = cfg["ppi_search"]["training_list"]
    entries = read_list(list_path)

    print(f"[INFO] training_list: {list_path}")
    print(f"[INFO] Entries: {len(entries)}")

    done = 0
    skipped = 0
    failed = 0

    for entry in entries:
        pdb_id, chain_a, chain_b = parse_entry(entry)

        # Decide which chains to run for this entry
        chains_to_run = []
        if args.chains in ("A", "both") and chain_a:
            chains_to_run.append(chain_a)
        if args.chains in ("B", "both") and chain_b and chain_b != chain_a:
            chains_to_run.append(chain_b)

        # If only one chain in file and user asked both, still run it
        if not chains_to_run and chain_a:
            chains_to_run = [chain_a]

        for chain_id in chains_to_run:
            try:
                did = process_one_chain(cfg, pdb_id, chain_id, force=args.force)
                if did:
                    done += 1
                else:
                    skipped += 1
            except Exception as e:
                failed += 1
                print(f"[ERROR] Failed for {pdb_id.upper()}_{chain_id}: {e}")

    print("[INFO] Batch completed.")
    print(f"[INFO] Done: {done}, Skipped: {skipped}, Failed: {failed}")


if __name__ == "__main__":
    main()
