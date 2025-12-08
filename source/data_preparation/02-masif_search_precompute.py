#!/usr/bin/env python3
import os
import time
import argparse
import warnings
import numpy as np
import yaml
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)

from ..masif_modules.read_data_from_surface import (
    read_data_from_surface,
    compute_shape_complementarity,
)

np.random.seed(0)

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

    # Minimal required keys for this script
    required = ["ply_file_template", "ppi_search"]
    for k in required:
        if k not in cfg or not cfg[k]:
            raise ValueError(f"Config missing required key: {k}")

    if "training_list" not in cfg["ppi_search"] or not cfg["ppi_search"]["training_list"]:
        raise ValueError("Config missing required key: ppi_search.training_list")

    if "masif_precomputation_dir" not in cfg["ppi_search"] or not cfg["ppi_search"]["masif_precomputation_dir"]:
        raise ValueError("Config missing required key: ppi_search.masif_precomputation_dir")

    return cfg


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


def info_arr(name, arr):
    """Print basic array stats for debugging."""
    if arr is None:
        print(f"[INFO] {name}: None")
        return
    try:
        print(
            f"[INFO] {name}: shape={arr.shape}, dtype={arr.dtype}, "
            f"min={np.nanmin(arr):.4g}, max={np.nanmax(arr):.4g}"
        )
    except Exception:
        try:
            print(f"[INFO] {name}: type={type(arr)}, len={len(arr)}")
        except Exception:
            print(f"[INFO] {name}: type={type(arr)}")


def save_npy(path_no_ext, arr):
    """Save numpy array to <path_no_ext>.npy using np.save()."""
    np.save(path_no_ext, arr)
    print(f"[SAVE] {path_no_ext}.npy  (saved)")


def expected_outputs(pids: list[str], two_chains: bool) -> list[str]:
    """
    Return list of expected output basenames (with .npy extension) inside my_precomp_dir.
    """
    outs = []
    for pid in pids:
        outs += [
            f"{pid}_rho_wrt_center.npy",
            f"{pid}_theta_wrt_center.npy",
            f"{pid}_input_feat.npy",
            f"{pid}_mask.npy",
            f"{pid}_list_indices.npy",
            f"{pid}_iface_labels.npy",
            f"{pid}_X.npy",
            f"{pid}_Y.npy",
            f"{pid}_Z.npy",
        ]
    if two_chains:
        outs += ["p1_sc_labels.npy", "p2_sc_labels.npy"]
    return outs


def outputs_exist(precomp_dir: str, pids: list[str], two_chains: bool) -> bool:
    """
    Check whether all expected output .npy files exist and are non-empty.
    """
    for fn in expected_outputs(pids, two_chains):
        p = os.path.join(precomp_dir, fn)
        if (not os.path.exists(p)) or os.path.getsize(p) == 0:
            return False
    return True


def build_ply_paths(cfg: dict, ppi_pair_id: str):
    """
    Given ppi_pair_id like '1abc_A_B', return:
      - fields (split by _)
      - ply_file dict {'p1': ..., 'p2': ... (optional)}
      - pids list ['p1'] or ['p1','p2']
    """
    fields = ppi_pair_id.split("_")
    if len(fields) < 2:
        raise ValueError("ppi_pair_id should look like PDBID_CHAIN or PDBID_CHAIN1_CHAIN2")

    pdb = fields[0].upper()
    c1 = fields[1]
    ply_file = {"p1": cfg["ply_file_template"].format(pdb, c1)}
    if len(fields) >= 3 and fields[2] != "":
        c2 = fields[2]
        ply_file["p2"] = cfg["ply_file_template"].format(pdb, c2)
        pids = ["p1", "p2"]
    else:
        pids = ["p1"]

    return fields, ply_file, pids


def run_one(cfg: dict, ppi_pair_id: str, force: bool = False, verbose: bool = True) -> str:
    """
    Run precomputation for one ppi_pair_id.
    Returns: "done" | "skipped" | "failed"
    """
    params = cfg["ppi_search"]

    my_precomp_dir = os.path.join(params["masif_precomputation_dir"], ppi_pair_id)
    os.makedirs(my_precomp_dir, exist_ok=True)
    if verbose:
        print(f"\n[INPUT] ppi_pair_id={ppi_pair_id}")
        print(f"[DIR] precomputation_dir={my_precomp_dir}")

    try:
        fields, ply_file, pids = build_ply_paths(cfg, ppi_pair_id)
    except Exception as e:
        print(f"[ERROR] Bad entry '{ppi_pair_id}': {e}")
        return "failed"

    two_chains = (len(pids) > 1)

    # Skip if already computed
    if (not force) and outputs_exist(my_precomp_dir, pids, two_chains):
        print(f"[INFO] Outputs already exist, skipping: {ppi_pair_id}")
        return "skipped"

    # Check PLY existence
    for pid in pids:
        if verbose:
            print(f"[FILE] {pid} ply = {ply_file[pid]}")
        if not os.path.exists(ply_file[pid]):
            print(f"[ERROR] PLY not found for {pid}: {ply_file[pid]}")
            return "failed"

    rho, theta, mask = {}, {}, {}
    input_feat, neigh_indices, iface_labels, verts = {}, {}, {}, {}

    # Read data from surface(s)
    if verbose:
        print("[INFO] Reading data from input ply surface files...")
    t0 = time.time()
    try:
        for pid in pids:
            t_start = time.time()
            (
                input_feat[pid],
                rho[pid],
                theta[pid],
                mask[pid],
                neigh_indices[pid],
                iface_labels[pid],
                verts[pid],
            ) = read_data_from_surface(ply_file[pid], params)

            if verbose:
                print(f"[OK] read_data_from_surface({pid}) in {time.time() - t_start:.2f}s")
                info_arr(f"{pid}.verts", verts[pid])
                info_arr(f"{pid}.input_feat", input_feat[pid])
                info_arr(f"{pid}.rho", rho[pid])
                info_arr(f"{pid}.theta", theta[pid])
                info_arr(f"{pid}.mask", mask[pid])
                info_arr(f"{pid}.neigh_indices", neigh_indices[pid])
                info_arr(f"{pid}.iface_labels", iface_labels[pid])

        if verbose:
            print(f"[INFO] Total surface read time: {time.time() - t0:.2f}s")
    except Exception as e:
        print(f"[ERROR] read_data_from_surface failed for {ppi_pair_id}: {e}")
        return "failed"

    # Shape complementarity (only if two chains)
    if two_chains:
        if verbose:
            print("[INFO] Computing shape complementarity (p1 vs p2)...")
        try:
            t_sc = time.time()
            p1_sc_labels, p2_sc_labels = compute_shape_complementarity(
                ply_file["p1"],
                ply_file["p2"],
                neigh_indices["p1"],
                neigh_indices["p2"],
                rho["p1"],
                rho["p2"],
                mask["p1"],
                mask["p2"],
                params,
            )
            if verbose:
                print(f"[OK] shape complementarity computed in {time.time() - t_sc:.2f}s")
                info_arr("p1_sc_labels", p1_sc_labels)
                info_arr("p2_sc_labels", p2_sc_labels)

            save_npy(os.path.join(my_precomp_dir, "p1_sc_labels"), p1_sc_labels)
            save_npy(os.path.join(my_precomp_dir, "p2_sc_labels"), p2_sc_labels)
        except Exception as e:
            print(f"[ERROR] shape complementarity failed for {ppi_pair_id}: {e}")
            return "failed"

    # Save precomputed arrays
    if verbose:
        print("[INFO] Saving precomputed arrays...")
    try:
        for pid in pids:
            save_npy(os.path.join(my_precomp_dir, f"{pid}_rho_wrt_center"), rho[pid])
            save_npy(os.path.join(my_precomp_dir, f"{pid}_theta_wrt_center"), theta[pid])
            save_npy(os.path.join(my_precomp_dir, f"{pid}_input_feat"), input_feat[pid])
            save_npy(os.path.join(my_precomp_dir, f"{pid}_mask"), mask[pid])
            save_npy(os.path.join(my_precomp_dir, f"{pid}_list_indices"), neigh_indices[pid])
            save_npy(os.path.join(my_precomp_dir, f"{pid}_iface_labels"), iface_labels[pid])

            save_npy(os.path.join(my_precomp_dir, f"{pid}_X"), verts[pid][:, 0])
            save_npy(os.path.join(my_precomp_dir, f"{pid}_Y"), verts[pid][:, 1])
            save_npy(os.path.join(my_precomp_dir, f"{pid}_Z"), verts[pid][:, 2])
    except Exception as e:
        print(f"[ERROR] Saving npy failed for {ppi_pair_id}: {e}")
        return "failed"

    print(f"[DONE] Precomputation complete for {ppi_pair_id}")
    return "done"


def main():
    parser = argparse.ArgumentParser(
        description="Batch precompute MaSIF PPI-search arrays from ppi_search.training_list."
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to YAML config file (converted from masif_opts).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if outputs already exist.",
    )
    parser.add_argument(
        "--list",
        default=None,
        help="Optional override for the list file path (otherwise uses ppi_search.training_list).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Less verbose output.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    list_path = args.list if args.list is not None else cfg["ppi_search"]["training_list"]
    entries = read_list(list_path)

    print(f"[INFO] Using list: {list_path}")
    print(f"[INFO] Entries: {len(entries)}")

    done = skipped = failed = 0
    for entry in entries:
        status = run_one(cfg, entry, force=args.force, verbose=(not args.quiet))
        if status == "done":
            done += 1
        elif status == "skipped":
            skipped += 1
        else:
            failed += 1

    print("\n[INFO] Batch completed.")
    print(f"[INFO] Done: {done}, Skipped: {skipped}, Failed: {failed}")


if __name__ == "__main__":
    main()
