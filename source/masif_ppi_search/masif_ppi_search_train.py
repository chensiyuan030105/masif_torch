#!/usr/bin/env python3
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader

from masif_modules.MaSIF_ppi_search import MaSIF_ppi_search
from masif_modules.train_ppi_search import train_ppi_search
from masif_modules.dataloader import PpiSearchCachedDataset


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


"""
masif_ppi_search_train.py: Entry function to train the MaSIF-search neural network.
Pablo Gainza - LPDI STI EPFL 2019
Released under an Apache License 2.0
"""


def main():
    parser = argparse.ArgumentParser(description="Train MaSIF ppi_search from cached arrays.")
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--wandb_project", default="masif_search", help="Weights&Biases project name.")
    parser.add_argument("--wandb_name", default="masif_search", help="Weights&Biases run name.")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging.")
    args = parser.parse_args()

    masif_opts = load_config(args.config)
    args.wandb_name = masif_opts["exp"]
    params = masif_opts["ppi_search"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    learning_rate = params["learning_rate"]
    print("Learning rate:", learning_rate)

    cache_dir = params["cache_dir"]

    # --- Load binder arrays ---
    binder_rho_wrt_center = np.load(os.path.join(cache_dir, "binder_rho_wrt_center.npy"))
    binder_theta_wrt_center = np.load(os.path.join(cache_dir, "binder_theta_wrt_center.npy"))
    binder_input_feat = np.load(os.path.join(cache_dir, "binder_input_feat.npy"))
    binder_mask = np.load(os.path.join(cache_dir, "binder_mask.npy"))
    binder_input_feat = mask_input_feat(binder_input_feat, params["feat_mask"])

    # --- Load positive arrays ---
    pos_training_idx = (np.load(os.path.join(cache_dir, "pos_training_idx.npy"))).astype(int)
    pos_val_idx      = (np.load(os.path.join(cache_dir, "pos_val_idx.npy"))).astype(int)
    pos_test_idx     = (np.load(os.path.join(cache_dir, "pos_test_idx.npy"))).astype(int)
    pos_rho_wrt_center   = np.load(os.path.join(cache_dir, "pos_rho_wrt_center.npy"))
    pos_theta_wrt_center = np.load(os.path.join(cache_dir, "pos_theta_wrt_center.npy"))
    pos_input_feat       = np.load(os.path.join(cache_dir, "pos_input_feat.npy"))
    pos_mask             = np.load(os.path.join(cache_dir, "pos_mask.npy"))
    pos_input_feat = mask_input_feat(pos_input_feat, params["feat_mask"])
    pos_names = np.load(os.path.join(cache_dir, "pos_names.npy"))

    # --- Load negative arrays ---
    neg_training_idx = (np.load(os.path.join(cache_dir, "neg_training_idx.npy"))).astype(int)
    neg_val_idx      = (np.load(os.path.join(cache_dir, "neg_val_idx.npy"))).astype(int)
    neg_test_idx     = (np.load(os.path.join(cache_dir, "neg_test_idx.npy"))).astype(int)
    neg_rho_wrt_center   = np.load(os.path.join(cache_dir, "neg_rho_wrt_center.npy"))
    neg_theta_wrt_center = np.load(os.path.join(cache_dir, "neg_theta_wrt_center.npy"))
    neg_input_feat       = np.load(os.path.join(cache_dir, "neg_input_feat.npy"))
    neg_mask             = np.load(os.path.join(cache_dir, "neg_mask.npy"))
    neg_input_feat = mask_input_feat(neg_input_feat, params["feat_mask"])

    # Default pids
    if "pids" not in params:
        params["pids"] = ["p1", "p2"]

    # --- Build model ---
    model = MaSIF_ppi_search(
        params["max_distance"],
        n_thetas=16,
        n_rhos=5,
        n_rotations=16,
        device=device,
        feat_mask=params["feat_mask"],
    )

    # Ensure model output directory exists
    os.makedirs(params["model_dir"], exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- W&B ---
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={"learning_rate": learning_rate},
        )
        wandb_obj = wandb
    else:
        wandb_obj = None

    # --- Datasets / Dataloaders ---
    train_dataset = PpiSearchCachedDataset(
        binder_rho_wrt_center, binder_theta_wrt_center, binder_input_feat, binder_mask,
        pos_rho_wrt_center,    pos_theta_wrt_center,    pos_input_feat,    pos_mask,
        neg_rho_wrt_center,    neg_theta_wrt_center,    neg_input_feat,    neg_mask,
        pos_training_idx, pos_val_idx, pos_test_idx,
        neg_training_idx, neg_val_idx, neg_test_idx,
        split="train",
    )

    val_dataset = PpiSearchCachedDataset(
        binder_rho_wrt_center, binder_theta_wrt_center, binder_input_feat, binder_mask,
        pos_rho_wrt_center,    pos_theta_wrt_center,    pos_input_feat,    pos_mask,
        neg_rho_wrt_center,    neg_theta_wrt_center,    neg_input_feat,    neg_mask,
        pos_training_idx, pos_val_idx, pos_test_idx,
        neg_training_idx, neg_val_idx, neg_test_idx,
        split="val",
    )

    test_dataset = PpiSearchCachedDataset(
        binder_rho_wrt_center, binder_theta_wrt_center, binder_input_feat, binder_mask,
        pos_rho_wrt_center,    pos_theta_wrt_center,    pos_input_feat,    pos_mask,
        neg_rho_wrt_center,    neg_theta_wrt_center,    neg_input_feat,    neg_mask,
        pos_training_idx, pos_val_idx, pos_test_idx,
        neg_training_idx, neg_val_idx, neg_test_idx,
        split="test",
    )

    train_dataloader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
    # If you want validation, set this to a DataLoader instead of None
    val_dataloader = None  # DataLoader(val_dataset, batch_size=params["batch_size"], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # --- Train ---
    train_ppi_search(
        model,
        params,
        optimizer,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        wandb=wandb_obj,
    )


if __name__ == "__main__":
    main()


