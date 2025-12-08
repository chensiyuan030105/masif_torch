#!/usr/bin/env python3
# Header variables and parameters.
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)

import os
import math
import argparse
import yaml
import numpy as np
import pymesh
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn import metrics

from ..masif_modules.MaSIF_ppi_search import MaSIF_ppi_search
from ..masif_modules.dataloader import PpiSearchCachedDataset
from ..masif_modules.inference_ppi_search import inference_ppi_search

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device =", device)
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

    if "ppi_search" not in cfg or not cfg["ppi_search"]:
        raise ValueError("Config missing required key: ppi_search")
    return cfg


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _minmax01(v, eps=1e-8):
    v = v.astype(np.float32, copy=False)
    mn, mx = v.min(), v.max()
    return (v - mn) / (mx - mn + eps)


def plot_desc_lines(results, out_png="desc_lines.png",
                    which_result=0, max_samples=16, strip_height=8, cmap="turbo"):
    """
    desc expected shape: [4B, D] in order [pos, binder, neg, neg2]
    Output: each 'type' is one line (strip). Total lines = 4 * max_samples.
    """
    desc = _to_numpy(results[which_result]["desc"])
    if desc.ndim == 1:
        desc = desc[None, :]

    N, D = desc.shape
    if N % 4 != 0:
        raise ValueError(f"Expect desc shape [4B, D], got N={N} not divisible by 4")

    B = N // 4
    B_plot = min(B, int(max_samples)) if max_samples is not None else B

    pos    = desc[0:B]
    binder = desc[B:2*B]
    neg    = desc[2*B:3*B]
    neg2   = desc[3*B:4*B]

    groups = [pos, binder, neg, neg2]
    names  = ["pos", "binder", "neg", "neg2"]

    nrows = 4 * B_plot
    fig, axes = plt.subplots(nrows, 1, figsize=(12, nrows * 0.6), constrained_layout=True)
    if nrows == 1:
        axes = [axes]

    row = 0
    for i in range(B_plot):
        for g, name in enumerate(names):
            v01 = _minmax01(groups[g][i])                 # [D]
            strip = np.tile(v01[None, :], (strip_height, 1))  # [H, D]

            ax = axes[row]
            ax.imshow(strip, vmin=0, vmax=1, cmap=cmap, interpolation="nearest", aspect="auto")
            ax.axis("off")
            ax.set_title(f"sample {i} - {name}", fontsize=10, loc="left")
            row += 1

    plt.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_png}")


# Apply mask to input_feat
def mask_input_feat(input_feat, mask):
    mymask = np.where(np.array(mask) == 0.0)[0]
    return np.delete(input_feat, mymask, axis=2)


def compute_roc_auc(pos, neg):
    labels = np.concatenate([np.ones((len(pos))), np.zeros((len(neg)))])
    dist_pairs = np.concatenate([pos, neg])
    return metrics.roc_auc_score(labels, dist_pairs)


def main():
    parser = argparse.ArgumentParser(description="Run MaSIF ppi_search inference from cached arrays and plot descriptors.")
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--checkpoint",
        default="model_epoch_3500.pt",
        help="Checkpoint filename inside ppi_search.model_dir (or an absolute path).",
    )
    parser.add_argument("--out_png", default="desc_lines.png", help="Output PNG path.")
    parser.add_argument("--max_samples", type=int, default=8, help="Max samples to visualize.")
    parser.add_argument("--strip_height", type=int, default=10, help="Height of each descriptor strip.")
    args = parser.parse_args()

    masif_opts = load_config(args.config)
    params = masif_opts["ppi_search"]

    # Ensure desc_dir exists + open logfile like your original
    os.makedirs(params["desc_dir"], exist_ok=True)
    logfile = open(os.path.join(params["desc_dir"], "log.txt"), "w+")
    logfile.write("inference started\n")
    logfile.flush()

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

    cache_dir = params["cache_dir"]

    # Load binder arrays
    binder_rho_wrt_center = np.load(os.path.join(cache_dir, "binder_rho_wrt_center.npy"))
    binder_theta_wrt_center = np.load(os.path.join(cache_dir, "binder_theta_wrt_center.npy"))
    binder_input_feat = np.load(os.path.join(cache_dir, "binder_input_feat.npy"))
    binder_mask = np.load(os.path.join(cache_dir, "binder_mask.npy"))
    binder_input_feat = mask_input_feat(binder_input_feat, params["feat_mask"])

    # Load positive arrays
    pos_training_idx = (np.load(os.path.join(cache_dir, "pos_training_idx.npy"))).astype(int)
    pos_val_idx = (np.load(os.path.join(cache_dir, "pos_val_idx.npy"))).astype(int)
    pos_test_idx = (np.load(os.path.join(cache_dir, "pos_test_idx.npy"))).astype(int)
    pos_rho_wrt_center = np.load(os.path.join(cache_dir, "pos_rho_wrt_center.npy"))
    pos_theta_wrt_center = np.load(os.path.join(cache_dir, "pos_theta_wrt_center.npy"))
    pos_input_feat = np.load(os.path.join(cache_dir, "pos_input_feat.npy"))
    pos_mask = np.load(os.path.join(cache_dir, "pos_mask.npy"))
    pos_input_feat = mask_input_feat(pos_input_feat, params["feat_mask"])
    pos_names = np.load(os.path.join(cache_dir, "pos_names.npy"))

    # Load negative arrays
    neg_training_idx = (np.load(os.path.join(cache_dir, "neg_training_idx.npy"))).astype(int)
    neg_val_idx = (np.load(os.path.join(cache_dir, "neg_val_idx.npy"))).astype(int)
    neg_test_idx = (np.load(os.path.join(cache_dir, "neg_test_idx.npy"))).astype(int)
    neg_rho_wrt_center = np.load(os.path.join(cache_dir, "neg_rho_wrt_center.npy"))
    neg_theta_wrt_center = np.load(os.path.join(cache_dir, "neg_theta_wrt_center.npy"))
    neg_input_feat = np.load(os.path.join(cache_dir, "neg_input_feat.npy"))
    neg_mask = np.load(os.path.join(cache_dir, "neg_mask.npy"))
    neg_input_feat = mask_input_feat(neg_input_feat, params["feat_mask"])

    # Dataset / loader
    test_dataset = PpiSearchCachedDataset(
        binder_rho_wrt_center, binder_theta_wrt_center, binder_input_feat, binder_mask,
        pos_rho_wrt_center,    pos_theta_wrt_center,    pos_input_feat,    pos_mask,
        neg_rho_wrt_center,    neg_theta_wrt_center,    neg_input_feat,    neg_mask,
        pos_training_idx, pos_val_idx, pos_test_idx,
        neg_training_idx, neg_val_idx, neg_test_idx,
        split="test",
    )
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Inference
    results = inference_ppi_search(model, params, test_dataloader)

    # Plot
    plot_desc_lines(
        results,
        out_png=args.out_png,
        max_samples=args.max_samples,
        strip_height=args.strip_height,
    )

    logfile.write("inference done\n")
    logfile.close()


if __name__ == "__main__":
    main()
