import time
import math
from sklearn import metrics
import numpy as np
import sys
import os
from IPython.core.debugger import set_trace
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Features and theta are flipped for the binder in construct_batch (except for hydrophobicity).
def construct_batch(
    binder_rho, binder_theta, binder_feat, binder_mask,
    pos_rho,    pos_theta,    pos_feat,    pos_mask,
    neg_rho,    neg_theta,    neg_feat,    neg_mask,
):
    """
    Inputs are torch tensors on CPU/GPU.
    Expected shapes (typical):
      *_rho, *_theta: [B, K]
      *_feat:         [B, K, C]
      *_mask:         [B, K]
    Returns:
      batch_rho:   [4B, K, 1]
      batch_theta: [4B, K, 1]
      batch_feat:  [4B, K, C]
      batch_mask:  [4B, K, 1]
    """

    # Expand rho/theta to [B, K, 1]
    batch_rho_binder   = binder_rho.unsqueeze(2)
    batch_theta_binder = binder_theta.unsqueeze(2)
    batch_feat_binder  = binder_feat
    batch_mask_binder  = binder_mask

    batch_rho_pos   = pos_rho.unsqueeze(2)
    batch_theta_pos = pos_theta.unsqueeze(2)
    batch_feat_pos  = pos_feat
    batch_mask_pos  = pos_mask

    # --- Flip binder features (negate everything, except hydrophobicity hack) ---
    batch_feat_binder = -batch_feat_binder

    # If last channel is hydrophobicity, undo its negation (matches original hack)
    C = batch_feat_binder.shape[2]
    if C == 5 or C == 3:
        batch_feat_binder[..., -1] = -batch_feat_binder[..., -1]

    # Flip binder theta: theta -> 2*pi - theta  (theta has shape [B, K, 1])
    twopi = 2.0 * math.pi
    batch_theta_binder = twopi - batch_theta_binder

    batch_rho_neg   = neg_rho.unsqueeze(2)
    batch_theta_neg = neg_theta.unsqueeze(2)
    batch_feat_neg  = neg_feat
    batch_mask_neg  = neg_mask

    # --- neg_2 is a copy of binder (as in original code) ---
    batch_rho_neg_2   = batch_rho_binder.clone()
    batch_theta_neg_2 = batch_theta_binder.clone()
    batch_feat_neg_2  = batch_feat_binder.clone()
    batch_mask_neg_2  = batch_mask_binder.clone()

    # Concatenate along batch dimension
    batch_rho = torch.cat([batch_rho_pos, batch_rho_binder, batch_rho_neg, batch_rho_neg_2], dim=0)
    batch_theta = torch.cat([batch_theta_pos, batch_theta_binder, batch_theta_neg, batch_theta_neg_2], dim=0)
    batch_feat = torch.cat([batch_feat_pos, batch_feat_binder, batch_feat_neg, batch_feat_neg_2], dim=0)

    # mask: [B, K] -> concat -> [4B, K] -> expand -> [4B, K, 1]
    batch_mask = torch.cat([batch_mask_pos, batch_mask_binder, batch_mask_neg, batch_mask_neg_2], dim=0).unsqueeze(2)

    return batch_rho, batch_theta, batch_feat, batch_mask

def compute_dists(descs1, descs2):
    dists = np.sqrt(np.sum(np.square(descs1 - descs2), axis=1))
    return dists

def compute_roc_auc(pos, neg):
    labels = np.concatenate([np.ones((len(pos))), np.zeros((len(neg)))])
    dist_pairs = np.concatenate([pos, neg])
    return metrics.roc_auc_score(labels, dist_pairs)

# --- Log the shapes and example values of the copied indices ---
def log_indices(name, arr, logfile, sample_count=5):
    logfile.write(f"[INFO] {name} shape: {arr.shape}\n")
    if len(arr) > 0:
        sample_count = min(sample_count, len(arr))
        logfile.write(f"       Example {name} values (first {sample_count}): {arr[:sample_count]}\n")
    else:
        logfile.write(f"       [WARNING] {name} is empty\n")

# Randomly pick
def inference_ppi_search(
    model,
    params,
    test_dataloader
):

    model.eval()
    results = []  # each item: {"desc": ..., "loss": ..., "score": ...}
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            (binder_rho, binder_theta, binder_feat, binder_mask,
            pos_rho, pos_theta, pos_feat, pos_mask,
            neg_rho, neg_theta, neg_feat, neg_mask) = batch

            # ---- to device ----
            binder_rho   = binder_rho.to(device, non_blocking=True)
            binder_theta = binder_theta.to(device, non_blocking=True)
            binder_feat  = binder_feat.to(device, non_blocking=True)
            binder_mask  = binder_mask.to(device, non_blocking=True)

            pos_rho   = pos_rho.to(device, non_blocking=True)
            pos_theta = pos_theta.to(device, non_blocking=True)
            pos_feat  = pos_feat.to(device, non_blocking=True)
            pos_mask  = pos_mask.to(device, non_blocking=True)

            neg_rho   = neg_rho.to(device, non_blocking=True)
            neg_theta = neg_theta.to(device, non_blocking=True)
            neg_feat  = neg_feat.to(device, non_blocking=True)
            neg_mask  = neg_mask.to(device, non_blocking=True)

            batch_rho, batch_theta, batch_feat, batch_mask = construct_batch(
                binder_rho, binder_theta, binder_feat, binder_mask,
                pos_rho, pos_theta, pos_feat, pos_mask,
                neg_rho, neg_theta, neg_feat, neg_mask,
            )

            keep_prob = 0.5

            desc, loss, score = model(
                rho=batch_rho,
                theta=batch_theta,
                feat=batch_feat,
                mask=batch_mask,
                keep_prob=keep_prob
            )

            # store (move to cpu so list doesn't hold GPU memory)
            results.append({
                "desc":  desc.detach().cpu(),
                "loss":  loss.detach().cpu() if torch.is_tensor(loss) else loss,
                "score": score.detach().cpu() if torch.is_tensor(score) else score,
            })

    return results