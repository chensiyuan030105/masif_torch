#!/usr/bin/python
import numpy as np
import os
import Bio
import shutil
from Bio.PDB import * 
import sys
import importlib
from IPython.core.debugger import set_trace

# Local includes
from default_config.masif_opts import masif_opts
from triangulation.computeMSMS import computeMSMS
from triangulation.fixmesh import fix_mesh
import pymesh
from input_output.extractPDB import extractPDB
from input_output.save_ply import save_ply
from input_output.read_ply import read_ply
from input_output.protonate import protonate
from triangulation.computeHydrophobicity import computeHydrophobicity
from triangulation.computeCharges import computeCharges, assignChargesToNewMesh
from triangulation.computeAPBS import computeAPBS
from triangulation.compute_normal import compute_normal
from sklearn.neighbors import KDTree

if len(sys.argv) <= 1: 
    print("Usage: {config} "+sys.argv[0]+" PDBID_A")
    sys.exit(1)

# ---------------------------
# 📝 Step 1: Parse input
# ---------------------------
in_fields = sys.argv[1].split("_")
pdb_id = in_fields[0]
chain_ids1 = in_fields[1]
print(f"[INFO] Processing PDB: {pdb_id}, Chains: {chain_ids1}")

if (len(sys.argv) > 2) and (sys.argv[2] == 'masif_ligand'):
    pdb_filename = os.path.join(masif_opts["ligand"]["assembly_dir"], pdb_id + ".pdb")
else:
    pdb_filename = masif_opts['raw_pdb_dir'] + pdb_id + ".pdb"

if not os.path.exists(pdb_filename):
    print(f"[ERROR] PDB file not found: {pdb_filename}")
    sys.exit(1)

tmp_dir = masif_opts['tmp_dir']
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

protonated_file = os.path.join(tmp_dir, f"{pdb_id}.pdb")
print(f"[INFO] Protonating PDB: {pdb_filename} -> {protonated_file}")
protonate(pdb_filename, protonated_file)
pdb_filename = protonated_file

# ---------------------------
# 📝 Step 2: Extract chains
# ---------------------------
out_filename1 = os.path.join(tmp_dir, f"{pdb_id}_{chain_ids1}")
print(f"[INFO] Extracting chains to: {out_filename1}.pdb")
extractPDB(pdb_filename, out_filename1 + ".pdb", chain_ids1)

# ---------------------------
# 📝 Step 3: Compute MSMS
# ---------------------------
print(f"[INFO] Computing MSMS surface for: {out_filename1}.pdb")
try:
    vertices1, faces1, normals1, names1, areas1 = computeMSMS(out_filename1 + ".pdb", protonate=True)
    print(f"[INFO] MSMS computed: vertices={len(vertices1)}, faces={len(faces1)}")
except Exception as e:
    print(f"[ERROR] MSMS failed for {out_filename1}.pdb")
    print(e)
    set_trace()

# ---------------------------
# 📝 Step 4: Charges / Hydrophobicity
# ---------------------------
if masif_opts['use_hbond']:
    print(f"[INFO] Computing charges...")
    vertex_hbond = computeCharges(out_filename1, vertices1, names1)

if masif_opts['use_hphob']:
    print(f"[INFO] Computing hydrophobicity...")
    vertex_hphobicity = computeHydrophobicity(names1)

# ---------------------------
# 📝 Step 5: Regularize mesh
# ---------------------------
mesh = pymesh.form_mesh(vertices1, faces1)
# mesh = pymesh.Mesh(vertices1, faces1)
print(f"[INFO] Fixing mesh with resolution {masif_opts['mesh_res']} ...")
regular_mesh = fix_mesh(mesh, masif_opts['mesh_res'])
print(f"[INFO] Regular mesh vertices: {len(regular_mesh.vertices)}, faces: {len(regular_mesh.faces)}")

vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)

if masif_opts['use_hbond']:
    vertex_hbond = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hbond, masif_opts)

if masif_opts['use_hphob']:
    vertex_hphobicity = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hphobicity, masif_opts)

if masif_opts['use_apbs']:
    print(f"[INFO] Computing APBS charges...")
    vertex_charges = computeAPBS(regular_mesh.vertices, out_filename1 + ".pdb", out_filename1)
else:
    vertex_charges = None

# ---------------------------
# 📝 Step 6: Interface (optional)
# ---------------------------
iface = np.zeros(len(regular_mesh.vertices))
if 'compute_iface' in masif_opts and masif_opts['compute_iface']:
    print(f"[INFO] Computing interface surface...")
    v3, f3, _, _, _ = computeMSMS(pdb_filename, protonate=True)
    mesh = pymesh.form_mesh(v3, f3)
    v3 = mesh.vertices
    kdt = KDTree(v3)
    d, r = kdt.query(regular_mesh.vertices)
    d = np.square(d)
    iface_v = np.where(d >= 2.0)[0]
    iface[iface_v] = 1.0

# ---------------------------
# 📝 Step 7: Save results
# ---------------------------
ply_out = out_filename1 + ".ply"
print(f"[INFO] Saving mesh to: {ply_out}")
save_ply(ply_out,
         regular_mesh.vertices,
         regular_mesh.faces,
         normals=vertex_normal,
         charges=vertex_charges,
         normalize_charges=True,
         hbond=(vertex_hbond if masif_opts['use_hbond'] else None),
         hphob=(vertex_hphobicity if masif_opts['use_hphob'] else None),
         iface=iface)

if not os.path.exists(masif_opts['ply_chain_dir']):
    os.makedirs(masif_opts['ply_chain_dir'])
if not os.path.exists(masif_opts['pdb_chain_dir']):
    os.makedirs(masif_opts['pdb_chain_dir'])

print(f"[INFO] Copying results to output directories...")
shutil.copy(ply_out, masif_opts['ply_chain_dir'])
shutil.copy(out_filename1 + '.pdb', masif_opts['pdb_chain_dir'])

print(f"[✅ DONE] Processed PDB {pdb_id}, chain {chain_ids1}")
