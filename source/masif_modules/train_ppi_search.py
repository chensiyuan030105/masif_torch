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
import math

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

def construct_batch_val_test(
    c_idx, rho, theta, feat, mask, flip=False
):
    batch_rho = rho[c_idx].unsqueeze(2)
    batch_theta = theta[c_idx].unsqueeze(2)
    batch_feat = feat[c_idx]
    batch_mask = mask[c_idx].unsqueeze(2)
    # Flip features and theta (except hydrophobicity)
    if flip:
        batch_feat = -batch_feat
        batch_theta = 2 * np.pi - batch_theta
        assert len(batch_feat.shape) == 3
        # Hydrophobicity is not flipped. -- FIx this.
        if batch_feat.shape[2] == 5 or batch_feat.shape[2] == 3:
            batch_feat[:, :, -1] = -batch_feat[:, :, -1]

    return batch_rho, batch_theta, batch_feat, batch_mask

def compute_val_test_desc(
    model,
    idx,
    rho,
    theta,
    feat,
    mask,
    batch_size=100,
    flip=False,
):  
    with torch.no_grad():
        model.eval()
        all_descs = []
        num_batches = (idx.numel() + batch_size - 1) // batch_size
        # Compute all desc for positive shapes.
        for kk in tqdm(range(num_batches)):
            # idx: torch.Tensor, shape [N]
            start = kk * batch_size
            end = min((kk + 1) * batch_size, idx.numel())

            pos = torch.arange(start, end, device=idx.device)
            c_idx = idx.index_select(0, pos)

            batch_rho, batch_theta, batch_feat, batch_mask = construct_batch_val_test(
                c_idx, rho, theta, feat, mask, flip=flip
            )

            keep_prob = 0.5
            desc, loss, score = model(
                rho=batch_rho,
                theta=batch_theta,
                feat=batch_feat,
                mask=batch_mask,
                keep_prob=keep_prob
            )
            desc = desc.squeeze()
            if desc.dim() == 1:
                desc = desc.unsqueeze(0)
            all_descs.append(desc)
        if len(all_descs) > 1:
            all_descs = torch.cat(all_descs, dim=0)
        else:
            all_descs = all_descs[0]
        return all_descs

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
def train_ppi_search(
    model,
    params,
    optimizer,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    wandb=None,
):

    num_epochs          = params["num_epochs"]
    save_epoch          = params["save_epoch"]
    out_dir             = params["model_dir"]
    
    log_path = os.path.join(out_dir, "log.txt")
    logfile = open(log_path, "w") 

    model.train()

    # ---- put this right after wandb.init(...) ----
    wandb.define_metric("global_step")
    wandb.define_metric("epoch")

    # loss vs step
    wandb.define_metric("train/loss_step", step_metric="global_step")

    # loss vs epoch (log once per epoch)
    wandb.define_metric("train/loss_epoch", step_metric="epoch")

    global_step = 0

    for epoch in range(num_epochs + 1):  # num_epochs = params["epoch"] or whatever you use
        epoch_loss_sum = 0.0
        epoch_batches = 0

        for step_in_epoch, batch in enumerate(train_dataloader):
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

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_val = float(loss.detach().cpu().item())

            # ---- per-step logging (x-axis = global_step) ----
            wandb.log(
                {
                    "train/loss_step": math.log10(loss_val),
                    "epoch": epoch,
                    "step_in_epoch": step_in_epoch,
                    "global_step": global_step,
                },
                step=global_step,
            )

            epoch_loss_sum += loss_val
            epoch_batches += 1
            global_step += 1

        # ---- per-epoch logging (x-axis = epoch) ----
        if epoch_batches > 0:
            wandb.log(
                {
                    "train/loss_epoch": math.log10(epoch_loss_sum / epoch_batches),
                    "epoch": epoch,
                    "global_step": global_step,  # optional, handy for reference
                }
            )



        if epoch % save_epoch == 0:
            with torch.no_grad():
                model.eval()

                for batch in test_dataloader:
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

                    desc = desc.detach().cpu().numpy()

                    n_patches = desc.shape[0] // 4
                    pos_desc = desc[0:n_patches]
                    binder_desc = desc[n_patches:2*n_patches]
                    neg_desc = desc[2*n_patches:3*n_patches]
                    neg_desc_2 = desc[3*n_patches:4*n_patches]

                    # Compute val ROC AUC.
                    pos_dists = compute_dists(pos_desc, binder_desc)
                    neg_dists = compute_dists(neg_desc, neg_desc_2)
                    roc_auc = 1 - compute_roc_auc(pos_dists, neg_dists)

                    logfile.write("Iteration {} validation roc auc: {}\n".format(epoch, roc_auc))
                    logfile.write("Mean validation positive score: {} ".format(np.mean(pos_dists)))
                    logfile.write("Mean validation negative score: {} ".format(np.mean(neg_dists)))
                    logfile.flush()

                logfile.write(">>> Saving model.\n")
                print(">>> Saving model.")

                output_model = os.path.join(out_dir, f"model_epoch_{epoch}.pt")
                torch.save(model.state_dict(), output_model)
                msg = f">>> Epoch {epoch}: Saved model and test results.\n"
                logfile.write(msg)
                print(msg)

            model.train()

    wandb.finish()