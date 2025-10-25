#!/usr/bin/python
import Bio
from Bio.PDB import * 
import sys
import importlib
import os

from default_config.masif_opts import masif_opts
# Local includes
from input_output.protonate import protonate

if len(sys.argv) <= 1: 
    print("Usage: "+sys.argv[0]+" PDBID_A_B")
    print("A or B are the chains to include in this pdb.")
    sys.exit(1)

print("[INFO] Starting PDB download script...")

if not os.path.exists(masif_opts['raw_pdb_dir']):
    print(f"[INFO] Creating raw PDB directory at {masif_opts['raw_pdb_dir']}")
    os.makedirs(masif_opts['raw_pdb_dir'])

if not os.path.exists(masif_opts['tmp_dir']):
    print(f"[INFO] Creating temporary directory at {masif_opts['tmp_dir']}")
    os.mkdir(masif_opts['tmp_dir'])

in_fields = sys.argv[1].split('_')
pdb_id = in_fields[0]
print(f"[INFO] PDB ID extracted: {pdb_id}")

import os
from Bio.PDB import PDBList

pdb_id = pdb_id.lower()
tmp_dir = masif_opts['tmp_dir']
os.makedirs(tmp_dir, exist_ok=True)

pdbl = PDBList()
pdb_filename = pdbl.retrieve_pdb_file(pdb_id, pdir=tmp_dir, file_format='pdb')

print(f"[INFO] PDB file downloaded to: {pdb_filename}")

upper_path = os.path.join(tmp_dir, f"{pdb_id.upper()}.pdb")
os.rename(pdb_filename, upper_path)
pdb_filename = upper_path

##### Protonate with reduce, if hydrogens included.
# - Always protonate as this is useful for charges. If necessary ignore hydrogens later.
protonated_file = masif_opts['raw_pdb_dir']+"/"+pdb_id.upper()+".pdb"
print(f"[INFO] Protonating PDB file: {pdb_filename}")
protonate(pdb_filename, protonated_file)
print(f"[INFO] Protonated PDB file saved to: {protonated_file}")

pdb_filename = protonated_file
print("[INFO] PDB download and processing completed.")