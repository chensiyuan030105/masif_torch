#!/usr/bin/env python3
import os
import argparse
import yaml

from Bio.PDB import PDBList
from input_output.protonate import protonate


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

    # Basic required keys
    for k in ["raw_pdb_dir", "tmp_dir", "ppi_search"]:
        if k not in cfg or not cfg[k]:
            raise ValueError(f"Config missing required key: {k}")

    if "training_list" not in cfg["ppi_search"] or not cfg["ppi_search"]["training_list"]:
        raise ValueError("Config missing required key: ppi_search.training_list")

    return cfg


def ensure_dirs(cfg: dict):
    """Create directories if they do not exist."""
    os.makedirs(cfg["raw_pdb_dir"], exist_ok=True)
    os.makedirs(cfg["tmp_dir"], exist_ok=True)


def read_training_list(list_path: str) -> list[str]:
    """
    Read training_list txt file.
    Expected each non-empty line to look like 'PDBID_A_B' (chains may vary).
    Lines starting with '#' are treated as comments.
    """
    if not os.path.exists(list_path):
        raise FileNotFoundError(f"training_list file not found: {list_path}")

    items: list[str] = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            items.append(s)

    return items


def extract_pdb_id(entry: str) -> str:
    """Extract PDB ID from a line like '1abc_A_B' -> '1abc'."""
    return entry.split("_")[0].strip().lower()


def download_and_protonate(pdb_id: str, cfg: dict, force: bool = False) -> bool:
    """
    Download a PDB into tmp_dir, rename to PDBID.pdb, protonate, and save to raw_pdb_dir.
    Returns True if work was done, False if skipped.
    """
    out_pdb = os.path.join(cfg["raw_pdb_dir"], f"{pdb_id.upper()}.pdb")

    # Skip if output already exists (unless --force is provided)
    if (not force) and os.path.exists(out_pdb) and os.path.getsize(out_pdb) > 0:
        print(f"[INFO] Output already exists, skipping: {out_pdb}")
        return False

    tmp_dir = cfg["tmp_dir"]
    pdbl = PDBList()

    print(f"[INFO] Downloading PDB: {pdb_id}")
    downloaded = pdbl.retrieve_pdb_file(pdb_id, pdir=tmp_dir, file_format="pdb")
    print(f"[INFO] Downloaded to: {downloaded}")

    # Rename to a consistent filename in tmp: PDBID.pdb
    tmp_pdb = os.path.join(tmp_dir, f"{pdb_id.upper()}.pdb")
    if os.path.exists(tmp_pdb):
        os.remove(tmp_pdb)
    os.replace(downloaded, tmp_pdb)

    # Protonate (adds hydrogens / prepares charges for downstream steps)
    print(f"[INFO] Protonating: {tmp_pdb} -> {out_pdb}")
    protonate(tmp_pdb, out_pdb)
    print(f"[INFO] Saved protonated PDB: {out_pdb}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Batch download PDBs from ppi_search.training_list, protonate, save into raw_pdb_dir."
    )
    parser.add_argument(
        "-c", "--config",
        required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if the output PDB file already exists.",
    )
    args = parser.parse_args()

    print("[INFO] Starting batch PDB download script...")

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    training_list_path = cfg["ppi_search"]["training_list"]
    entries = read_training_list(training_list_path)

    # Extract PDB IDs, keep order, de-duplicate
    seen = set()
    pdb_ids: list[str] = []
    for e in entries:
        pid = extract_pdb_id(e)
        if pid and pid not in seen:
            seen.add(pid)
            pdb_ids.append(pid)

    print(f"[INFO] training_list: {training_list_path}")
    print(f"[INFO] Unique PDB IDs to process: {len(pdb_ids)}")

    done = 0
    skipped = 0
    failed = 0

    for pdb_id in pdb_ids:
        try:
            did_work = download_and_protonate(pdb_id, cfg, force=args.force)
            if did_work:
                done += 1
            else:
                skipped += 1
        except Exception as e:
            failed += 1
            print(f"[ERROR] Failed for {pdb_id}: {e}")

    print("[INFO] Batch completed.")
    print(f"[INFO] Done: {done}, Skipped: {skipped}, Failed: {failed}")


if __name__ == "__main__":
    main()
