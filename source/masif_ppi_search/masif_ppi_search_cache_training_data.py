#!/usr/bin/env python3
# Header variables and parameters.
import sys
import os
import argparse
import yaml
import numpy as np
import pymesh
from IPython.core.debugger import set_trace
from scipy.spatial import cKDTree

"""
masif_ppi_search_cache_training_data.py: Function to cache all the training data for MaSIF-search. 
This function extracts all the positive pairs and a random number of negative surfaces.
Pablo Gainza - LPDI STI EPFL 2019
Released under an Apache License 2.0
"""


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
    if "ply_file_template" not in cfg or not cfg["ply_file_template"]:
        raise ValueError("Config missing required key: ply_file_template")
    return cfg


def apply_overrides(params: dict, overrides: list[str]):
    """
    Apply overrides like: ["max_sc_filt=1.0", "batch_size=16"] to params dict.
    Values are parsed as int/float/bool/str in a simple way.
    """
    def parse_value(v: str):
        vl = v.lower()
        if vl in ("true", "false"):
            return vl == "true"
        try:
            if "." in v or "e" in vl:
                return float(v)
            return int(v)
        except Exception:
            return v

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Bad override '{item}', expected key=value")
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        params[k] = parse_value(v)
        print(f"[OVERRIDE] Setting params['{k}'] = {params[k]} ({type(params[k]).__name__})")


def main():
    parser = argparse.ArgumentParser(description="Cache training data for MaSIF ppi_search from precomputation outputs.")
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override ppi_search params: --override key=value (can be repeated).",
    )
    args = parser.parse_args()

    masif_opts = load_config(args.config)
    params = masif_opts["ppi_search"]
    apply_overrides(params, args.override)

    if "pids" not in params:
        params["pids"] = ["p1", "p2"]

    parent_in_dir = params["masif_precomputation_dir"]

    # Accumulators
    binder_rho_wrt_center = []
    binder_theta_wrt_center = []
    binder_input_feat = []
    binder_mask = []

    pos_rho_wrt_center = []
    pos_theta_wrt_center = []
    pos_input_feat = []
    pos_mask = []

    neg_rho_wrt_center = []
    neg_theta_wrt_center = []
    neg_input_feat = []
    neg_mask = []

    np.random.seed(0)
    training_idx = []
    val_idx = []
    test_idx = []
    pos_names = []
    neg_names = []

    training_list = [x.rstrip() for x in open(params["training_list"]).readlines()]
    testing_list = [x.rstrip() for x in open(params["testing_list"]).readlines()]

    idx_count = 0
    for count, ppi_pair_id in enumerate(os.listdir(parent_in_dir)):
        if ppi_pair_id not in testing_list and ppi_pair_id not in training_list:
            continue

        in_dir = os.path.join(parent_in_dir, ppi_pair_id) + "/"
        print(ppi_pair_id)

        # Read binder and pos.
        train_val = np.random.random()

        # Read binder first, which is p1.
        try:
            labels = np.load(in_dir + "p1_sc_labels.npy")
            # Take the median of the percentile 25 shape complementarity.
            mylabels = labels[0]
            labels = np.median(mylabels, axis=1)
        except Exception as e:
            print("Could not open " + in_dir + "p1_sc_labels.npy: " + str(e))
            continue

        # Read the corresponding ply files.
        fields = ppi_pair_id.split("_")
        ply_fn1 = masif_opts["ply_file_template"].format(fields[0], fields[1])
        ply_fn2 = masif_opts["ply_file_template"].format(fields[0], fields[2])

        # pos_labels: points > max_sc_filt and > min_sc_filt.
        pos_labels = np.where((labels < params["max_sc_filt"]) & (labels > params["min_sc_filt"]))[0]
        K = int(params["pos_surf_accept_probability"] * len(pos_labels))
        if K < 1:
            continue

        l = np.arange(len(pos_labels))
        np.random.shuffle(l)
        l = l[:K]
        l = pos_labels[l]

        v1 = pymesh.load_mesh(ply_fn1).vertices[l]
        v2 = pymesh.load_mesh(ply_fn2).vertices

        # For each point in v1, find the closest point in v2.
        kdt = cKDTree(v2)
        d, r = kdt.query(v1)

        # Contact points: those within a cutoff distance.
        contact_points = np.where(d < params["pos_interface_cutoff"])[0]
        try:
            k1 = l[contact_points]
            print("k1 =", k1)
        except Exception:
            set_trace()
        k2 = r[contact_points]

        # For negatives, get points in v2 far from p1.
        try:
            kdt = cKDTree(v1)
            dneg, rneg = kdt.query(v2)
        except Exception:
            set_trace()
        k_neg2 = np.where(dneg > params["pos_interface_cutoff"])[0]

        assert len(k1) == len(k2)
        n_pos = len(k1)

        # Binder is p1
        pid = "p1"
        for ii in k1:
            pos_names.append(f"{ppi_pair_id}_{pid}_{ii}")

        rho_wrt_center = np.load(in_dir + pid + "_rho_wrt_center.npy")
        theta_wrt_center = np.load(in_dir + pid + "_theta_wrt_center.npy")
        input_feat = np.load(in_dir + pid + "_input_feat.npy")
        mask = np.load(in_dir + pid + "_mask.npy")

        binder_rho_wrt_center.append(rho_wrt_center[k1])
        binder_theta_wrt_center.append(theta_wrt_center[k1])
        binder_input_feat.append(input_feat[k1])
        binder_mask.append(mask[k1])

        # Read pos, which is p2.
        pid = "p2"
        rho_wrt_center = np.load(in_dir + pid + "_rho_wrt_center.npy")
        theta_wrt_center = np.load(in_dir + pid + "_theta_wrt_center.npy")
        input_feat = np.load(in_dir + pid + "_input_feat.npy")
        mask = np.load(in_dir + pid + "_mask.npy")

        pos_rho_wrt_center.append(rho_wrt_center[k2])
        pos_theta_wrt_center.append(theta_wrt_center[k2])
        pos_input_feat.append(input_feat[k2])
        pos_mask.append(mask[k2])

        # Get a set of negatives from p2.
        np.random.shuffle(k_neg2)
        k_neg2 = k_neg2[: len(k2)]
        assert len(k_neg2) == n_pos

        neg_rho_wrt_center.append(rho_wrt_center[k_neg2])
        neg_theta_wrt_center.append(theta_wrt_center[k_neg2])
        neg_input_feat.append(input_feat[k_neg2])
        neg_mask.append(mask[k_neg2])

        for ii in k_neg2:
            neg_names.append(f"{ppi_pair_id}_{pid}_{ii}")

        # Training, validation or test?
        if ppi_pair_id in training_list:
            if train_val <= params["range_val_samples"]:
                training_idx = np.append(training_idx, np.arange(idx_count, idx_count + n_pos))
            else:
                val_idx = np.append(val_idx, np.arange(idx_count, idx_count + n_pos))

        if ppi_pair_id in testing_list:
            test_idx = np.append(test_idx, np.arange(idx_count, idx_count + n_pos))

        idx_count += n_pos

    os.makedirs(params["cache_dir"], exist_ok=True)

    binder_rho_wrt_center = np.concatenate(binder_rho_wrt_center, axis=0)
    binder_theta_wrt_center = np.concatenate(binder_theta_wrt_center, axis=0)
    binder_input_feat = np.concatenate(binder_input_feat, axis=0)
    binder_mask = np.concatenate(binder_mask, axis=0)

    pos_rho_wrt_center = np.concatenate(pos_rho_wrt_center, axis=0)
    pos_theta_wrt_center = np.concatenate(pos_theta_wrt_center, axis=0)
    pos_input_feat = np.concatenate(pos_input_feat, axis=0)
    pos_mask = np.concatenate(pos_mask, axis=0)
    np.save(os.path.join(params["cache_dir"], "pos_names.npy"), pos_names)

    neg_rho_wrt_center = np.concatenate(neg_rho_wrt_center, axis=0)
    neg_theta_wrt_center = np.concatenate(neg_theta_wrt_center, axis=0)
    neg_input_feat = np.concatenate(neg_input_feat, axis=0)
    neg_mask = np.concatenate(neg_mask, axis=0)
    np.save(os.path.join(params["cache_dir"], "neg_names.npy"), neg_names)

    print(f"Read {len(neg_rho_wrt_center)} negative shapes")
    print(f"Read {len(pos_rho_wrt_center)} positive shapes")

    # Save binder
    np.save(os.path.join(params["cache_dir"], "binder_rho_wrt_center.npy"), binder_rho_wrt_center)
    np.save(os.path.join(params["cache_dir"], "binder_theta_wrt_center.npy"), binder_theta_wrt_center)
    np.save(os.path.join(params["cache_dir"], "binder_input_feat.npy"), binder_input_feat)
    np.save(os.path.join(params["cache_dir"], "binder_mask.npy"), binder_mask)

    # Save indices + pos
    np.save(os.path.join(params["cache_dir"], "pos_training_idx.npy"), training_idx)
    np.save(os.path.join(params["cache_dir"], "pos_val_idx.npy"), val_idx)
    np.save(os.path.join(params["cache_dir"], "pos_test_idx.npy"), test_idx)
    np.save(os.path.join(params["cache_dir"], "pos_rho_wrt_center.npy"), pos_rho_wrt_center)
    np.save(os.path.join(params["cache_dir"], "pos_theta_wrt_center.npy"), pos_theta_wrt_center)
    np.save(os.path.join(params["cache_dir"], "pos_input_feat.npy"), pos_input_feat)
    np.save(os.path.join(params["cache_dir"], "pos_mask.npy"), pos_mask)

    # Save indices + neg (same split indices as pos)
    np.save(os.path.join(params["cache_dir"], "neg_training_idx.npy"), training_idx)
    np.save(os.path.join(params["cache_dir"], "neg_val_idx.npy"), val_idx)
    np.save(os.path.join(params["cache_dir"], "neg_test_idx.npy"), test_idx)
    np.save(os.path.join(params["cache_dir"], "neg_rho_wrt_center.npy"), neg_rho_wrt_center)
    np.save(os.path.join(params["cache_dir"], "neg_theta_wrt_center.npy"), neg_theta_wrt_center)
    np.save(os.path.join(params["cache_dir"], "neg_input_feat.npy"), neg_input_feat)
    np.save(os.path.join(params["cache_dir"], "neg_mask.npy"), neg_mask)

    # (Already saved above, but keep for compatibility)
    np.save(os.path.join(params["cache_dir"], "neg_names.npy"), neg_names)


if __name__ == "__main__":
    main()



