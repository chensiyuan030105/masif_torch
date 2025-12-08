# Header variables and parameters.
import pymesh
import sys
import os
import time
import numpy as np
from IPython.core.debugger import set_trace
from sklearn import metrics
import importlib
import torch
import argparse
import yaml
from sklearn.neighbors import NearestNeighbors

from ..masif_modules.MaSIF_ppi_search import MaSIF_ppi_search
from ..masif_modules.inference_ppi_search import inference_ppi_search
from ..masif_modules.train_ppi_search import compute_val_test_desc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

# --- Helper function to log array shapes and sample values ---
def log_indices(name, arr, logfile, sample_count=5, max_elements=200):
    logfile.write(f"[INFO] {name} shape: {arr.shape}\n")

    if arr is None:
        logfile.write(f"       [WARNING] {name} is None\n")
        return

    if len(arr) == 0:
        logfile.write(f"       [WARNING] {name} is empty\n")
        return

    # Determine how many samples to show
    sample_count = min(sample_count, len(arr))
    display_count = min(len(arr), max_elements)

    # Print the first few values (truncated if too large)
    logfile.write(f"       Example {name} values (first {sample_count} of {display_count} shown):\n")
    logfile.write(f"       {arr[:display_count]}\n")

    # Warn if the array was truncated
    if len(arr) > max_elements:
        logfile.write(f"       ... [truncated, total length={len(arr)}]\n")


def log_array(name, arr, logfile, sample_count=5, max_elements=200):
    logfile.write(f"[INFO] {name} shape: {arr.shape}\n")

    if arr is None:
        logfile.write(f"       [WARNING] {name} is None\n")
        return

    if len(arr) == 0:
        logfile.write(f"       [WARNING] {name} is empty\n")
        return

    # For multidimensional arrays, print the first few rows
    display_count = min(len(arr), max_elements)
    logfile.write(f"       Example {name} values (first {sample_count} rows of {display_count} shown):\n")
    logfile.write(f"       {arr[:display_count]}\n")

    if len(arr) > max_elements:
        logfile.write(f"       ... [truncated, total length={len(arr)}]\n")

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

    if "ppi_search" not in cfg or not cfg["ppi_search"]:
        raise ValueError("Config missing required key: ppi_search")
    return cfg

# Apply mask to input_feat
def mask_input_feat(input_feat, mask):
    mymask = np.where(np.array(mask) == 0.0)[0]
    return np.delete(input_feat, mymask, axis=2)

def compute_roc_auc(pos, neg):
    labels = np.concatenate([np.ones((len(pos))), np.zeros((len(neg)))])
    dist_pairs = np.concatenate([pos, neg])
    return metrics.roc_auc_score(labels, dist_pairs)

def main():

    parser = argparse.ArgumentParser(description="Train MaSIF ppi_search from cached arrays.")
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--checkpoint",
        default="model_epoch_3500.pt",
        help="Checkpoint filename inside ppi_search.model_dir (or an absolute path).",
    )
    args = parser.parse_args()

    masif_opts = load_config(args.config)
    params = masif_opts["ppi_search"]
    parent_in_dir = params["masif_precomputation_dir"]
    desc_dir = params["desc_dir"]
    os.makedirs(desc_dir, exist_ok=True)
    logfile = open(os.path.join(params["desc_dir"], "log.txt"), "w+")

    idx_count = 0
    all_pos_dists = []
    all_neg_dists = []
    all_pos_dists_pos_neg = []
    all_neg_dists_pos_neg = []
    eval_list = []

    load_iter = 0
    benchmark_list = params["benchmark_list"]
    eval_list = open(benchmark_list).readlines()
    eval_list = [x.rstrip() for x in eval_list]
    print("eval_list =", eval_list)

    # Build model
    model = MaSIF_ppi_search(
        params["max_distance"],
        n_thetas=16,
        n_rhos=5,
        n_rotations=16,
        device=device,
        feat_mask=params["feat_mask"],
    )

    # Resolve checkpoint path
    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(params["model_dir"], ckpt_path)

    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    for count, ppi_pair_id in enumerate(eval_list):
        print("count =", count, ", ppi_pair_id =", ppi_pair_id)

        # Construct the output directory for the current PPI pair
        in_dir = parent_in_dir + ppi_pair_id + "/"
        out_desc_dir = os.path.join(params["desc_dir"], ppi_pair_id)
        os.makedirs(out_desc_dir, exist_ok=True)
        target_file = os.path.join(out_desc_dir, 'p1_desc_straight.npy')

        # Parse PPI pair ID into components (PDB ID and chain identifiers)
        pdbid = ppi_pair_id.split("_")[0]
        chain1 = ppi_pair_id.split("_")[1]
        if len(ppi_pair_id.split("_")) > 2:
            chain2 = ppi_pair_id.split("_")[2]
        else:
            chain2 = ''
        
        print("pbdid =", pdbid, ", chain1 =", chain1, ", chain2 =", chain2)

        # Read shape complementarity (SC) labels if a second chain exists
        if chain2 != '':
            labels = np.load(os.path.join(in_dir, "p1_sc_labels.npy"))
            print("labels.shape =", labels.shape)
            mylabels = labels[0]
            labels = np.median(mylabels, axis=1)

            # Filter vertices by SC value range
            pos_labels = np.where(
                (labels > params["min_sc_filt"]) & (labels < params["max_sc_filt"])
            )[0]
            l = pos_labels

            # Log first few indices to inspect filtering result
            if len(pos_labels) > 0:
                sample_count = min(10, len(pos_labels))
        else:
            l = []
        
        if len(l) > 0 and chain2 != "":
            ply_fn1 = masif_opts['ply_file_template'].format(pdbid, chain1)
            v1 = pymesh.load_mesh(ply_fn1).vertices[l]

            ply_fn2 = masif_opts['ply_file_template'].format(pdbid, chain2 )
            v2 = pymesh.load_mesh(ply_fn2).vertices

            # For each point in v1, find the closest point in v2.
            nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(v2)
            d, r = nbrs.kneighbors(v1)
            d = np.squeeze(d, axis=1)
            r = np.squeeze(r, axis=1)

            contact_points = np.where(d < params["pos_interface_cutoff"])[0]
            if len(contact_points) > 0:
                k1 = l[contact_points]  # contact points protein 1
                k2 = r[contact_points]  # contact points protein 2
                assert len(k1) == len(k2)
            else:
                l = []

        pid = "p1"
        # Load precomputed geometric features
        p1_rho = np.load(in_dir + pid + "_rho_wrt_center.npy")
        p1_theta = np.load(in_dir + pid + "_theta_wrt_center.npy")
        p1_feat = np.load(in_dir + pid + "_input_feat.npy")
        p1_mask = np.load(in_dir + pid + "_mask.npy")
        # Apply feature masking
        p1_feat = mask_input_feat(p1_feat, params["feat_mask"])
        idx1 = np.array(range(len(p1_rho)))

        p1_rho   = torch.from_numpy(p1_rho).to(device=device, dtype=torch.float32)
        p1_theta = torch.from_numpy(p1_theta).to(device=device, dtype=torch.float32)
        p1_feat  = torch.from_numpy(p1_feat).to(device=device, dtype=torch.float32)
        p1_mask  = torch.from_numpy(p1_mask).to(device=device, dtype=torch.float32)
        idx1  = torch.from_numpy(idx1).to(device=device, dtype=torch.long)

        desc1_str = compute_val_test_desc(
            model,
            idx1,
            p1_rho,
            p1_theta,
            p1_feat,
            p1_mask,
            batch_size=32,
            flip=False,
        )
        desc1_flip = compute_val_test_desc(
            model,
            idx1,
            p1_rho,
            p1_theta,
            p1_feat,
            p1_mask,
            batch_size=32,
            flip=True,
        )

        if chain2 != "":
            pid = "p2"
            # Load precomputed geometric features
            p2_rho = np.load(os.path.join(in_dir, pid + "_rho_wrt_center.npy"))
            p2_theta = np.load(os.path.join(in_dir, pid + "_theta_wrt_center.npy"))
            p2_feat = np.load(os.path.join(in_dir, pid + "_input_feat.npy"))
            p2_mask = np.load(os.path.join(in_dir, pid + "_mask.npy"))
            # Apply feature masking
            p2_feat = mask_input_feat(p2_feat, params["feat_mask"])
            idx2 = np.array(range(len(p2_rho)))

            p2_rho   = torch.from_numpy(p2_rho).to(device=device, dtype=torch.float32)
            p2_theta = torch.from_numpy(p2_theta).to(device=device, dtype=torch.float32)
            p2_feat  = torch.from_numpy(p2_feat).to(device=device, dtype=torch.float32)
            p2_mask  = torch.from_numpy(p2_mask).to(device=device, dtype=torch.float32)
            idx2  = torch.from_numpy(idx2).to(device=device, dtype=torch.long)

            # --- Compute descriptors for protein 2 ---
            desc2_str = compute_val_test_desc(
                model,
                idx2,
                p2_rho,
                p2_theta,
                p2_feat,
                p2_mask,
                batch_size=32,
                flip=False,
            )
            desc2_flip = compute_val_test_desc(
                model,
                idx2,
                p2_rho,
                p2_theta,
                p2_feat,
                p2_mask,
                batch_size=32,
                flip=True,
            )

        desc1_str = desc1_str.detach().cpu().numpy()
        desc1_flip = desc1_flip.detach().cpu().numpy()
        idx1 = idx1.detach().cpu().numpy()
        
        # Save descriptors
        np.save(os.path.join(out_desc_dir, "p1_desc_straight.npy"), desc1_str)
        np.save(os.path.join(out_desc_dir, "p1_desc_flipped.npy"), desc1_flip)

        if chain2 != "":

            desc2_str = desc2_str.detach().cpu().numpy()
            desc2_flip = desc2_flip.detach().cpu().numpy()
            idx2 = idx2.detach().cpu().numpy()

            np.save(os.path.join(out_desc_dir, "p2_desc_straight.npy"), desc2_str)
            np.save(os.path.join(out_desc_dir, "p2_desc_flipped.npy"), desc2_flip)

        # === Compute ROC AUC for sanity check and statistics ===
        if chain2 != "" and len(l) > 0:

            # --- Generate negative samples ---
            np.random.shuffle(idx1)
            kneg1 = idx1[: len(k1)]
            np.random.shuffle(idx2)
            kneg2 = idx2[: len(k2)]

            # === First ROC AUC: fully random negative pairs ===
            pos_dists = np.sqrt(np.sum(np.square(desc1_str[k1] - desc2_flip[k2]), axis=1))
            neg_dists = np.sqrt(np.sum(np.square(desc1_str[kneg1] - desc2_flip[kneg2]), axis=1))
            roc_auc = 1.0 - compute_roc_auc(pos_dists, neg_dists)
            print("roc_auc =", roc_auc)

            # --- Save distances ---
            all_pos_dists.append(pos_dists)
            all_neg_dists.append(neg_dists)

            # === Second ROC AUC: fixed chain1 positive, random chain2 negative ===
            np.random.shuffle(idx2)
            kneg2 = idx2[: len(k2)]
            pos_dists = np.sqrt(np.sum(np.square(desc1_str[k1] - desc2_flip[k2]), axis=1))
            neg_dists = np.sqrt(np.sum(np.square(desc1_str[k1] - desc2_flip[kneg2]), axis=1))
            roc_auc = 1.0 - compute_roc_auc(pos_dists, neg_dists)
            print("roc_auc =", roc_auc)

            # --- Save distances ---
            all_pos_dists_pos_neg.append(pos_dists)
            all_neg_dists_pos_neg.append(neg_dists)

    if len(all_pos_dists) > 0:
        all_pos_dists = np.concatenate(all_pos_dists, axis=0)
        all_neg_dists = np.concatenate(all_neg_dists, axis=0)

        roc_auc = 1.0 - compute_roc_auc(all_pos_dists, all_neg_dists)
        np.save(params["desc_dir"] + "/all_pos_dists.npy", all_pos_dists)
        np.save(params["desc_dir"] + "/all_neg_dists.npy", all_neg_dists)

        all_pos_dists_pos_neg = np.concatenate(all_pos_dists_pos_neg, axis=0)
        all_neg_dists_pos_neg = np.concatenate(all_neg_dists_pos_neg, axis=0)
        roc_auc = 1.0 - compute_roc_auc(all_pos_dists_pos_neg, all_neg_dists_pos_neg)
        np.save(params["desc_dir"] + "/all_pos_dists_pos_neg.npy", all_pos_dists_pos_neg)
        np.save(params["desc_dir"] + "/all_neg_dists_pos_neg.npy", all_neg_dists_pos_neg)

if __name__ == "__main__":
    main()