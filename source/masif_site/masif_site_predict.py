"""
masif_site_predict_torch.py: Evaluate proteins on MaSIF-site (PyTorch version)
Author: Adapted from Pablo Gainza's TensorFlow version (EPFL 2019)
Converted to PyTorch by Siyuan Chen, 2025
"""

import os
import sys
import time
import importlib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from masif_modules.MaSIF_site import MaSIF_site
from default_config.masif_opts import masif_opts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device =", device)

# === Mask helper ===
def mask_input_feat(input_feat, mask):
    """Remove feature channels where mask == 0."""
    mymask = np.where(np.array(mask) == 0.0)[0]
    return np.delete(input_feat, mymask, axis=2)


# === Main ===
def main():
    params = masif_opts["site"]

    # Load custom parameters from .py file
    custom_params_file = sys.argv[1]
    custom_params = importlib.import_module(custom_params_file, package=None).custom_params
    for key, val in custom_params.items():
        print(f"Setting {key} to {val}")
        params[key] = val

    parent_in_dir = params["masif_precomputation_dir"]

    # Determine input proteins
    if len(sys.argv) == 3:
        ppi_pair_ids = [sys.argv[2]]
    elif len(sys.argv) == 4 and sys.argv[2] == "-l":
        with open(sys.argv[3]) as f:
            ppi_pair_ids = [line.strip() for line in f if line.strip()]
    else:
        print("Usage: python masif_site_predict_torch.py config [pdb_chain | -l list.txt]")
        sys.exit(1)

    # === Initialize model ===
    print("\n[INFO] Initializing MaSIF-site (PyTorch)...")
    print("params[\"n_conv_layers\"] =", params["n_conv_layers"])
    model = MaSIF_site(
        params["max_distance"],
        n_thetas=4,
        n_rhos=3,
        n_rotations=4,
        device=device,
        feat_mask=params["feat_mask"],
        n_conv_layers=params["n_conv_layers"],
    ).cuda()

    # === Print model parameter information ===
    total_params = 0
    print("\n[DEBUG] Model Parameters:")
    print("-" * 60)
    for name, param in model.named_parameters():
        shape = tuple(param.shape)
        numel = param.numel()
        total_params += numel
        print(f"Name: {name}")
        print(f"Shape: {shape}")
        # print(f"Number of elements: {numel}")
        # print(f"Values (first few): {param.flatten()[:10].detach().cpu().numpy()}")
        # print("-" * 60)

    print(f"\n[INFO] Total number of trainable parameters: {total_params:,}")

    load_pretrained_model = False

    if load_pretrained_model:
        ckpt_path = os.path.join(params["model_dir"], "masif_site_from_tf.pt")
        # ckpt_path = os.path.join(params["model_dir"], "model_step_100.pt")
        print(f"[INFO] Loading model weights from {ckpt_path}")

        # Load checkpoint (TensorFlow converted)
        state_dict = torch.load(ckpt_path)

        # # --- ✅ Remap TF-style keys to PyTorch-style keys ---
        # mapped_state_dict = {}
        # for k, v in state_dict.items():
        #     new_k = k
        #     if "/weights" in new_k:
        #         new_k = new_k.replace("/weights", ".weight")
        #         mapped_state_dict[new_k] = v.T
        #     if "/biases" in new_k:
        #         new_k = new_k.replace("/biases", ".bias")
        #         mapped_state_dict[new_k] = v.T

        # --- ✅ Remap TensorFlow-style keys to PyTorch-style keys ---
        mapped_state_dict = {}
        unrecognized = []

        for k, v in state_dict.items():
            new_k = k
            matched = False

            # Convert TensorFlow parameter naming to PyTorch convention
            if "/weights" in new_k:
                new_k = new_k.replace("/weights", ".weight")
                mapped_state_dict[new_k] = v.T  # Transpose to match PyTorch weight layout
                matched = True
            elif "/biases" in new_k:
                new_k = new_k.replace("/biases", ".bias")
                mapped_state_dict[new_k] = v.T  # Transpose bias just in case (safe for 1D)
                matched = True
            else:
                mapped_state_dict[new_k] = v    # Transpose bias just in case (safe for 1D)
                matched = True

            # Keep track of parameters that don’t match any rule
            if not matched:
                unrecognized.append(k)

        # --- ⚠️ Print unrecognized parameters ---
        print("unrecognized ="  , unrecognized)
        if unrecognized:
            print(f"[⚠️ Unrecognized {len(unrecognized)} params]")
            for name in unrecognized:
                v = state_dict[name]
                print(f"  {name:40s}  {tuple(v.shape)}")

        # --- ✅ Keep only parameters that exist in the PyTorch model ---
        model_state_dict = {
            k: v for k, v in mapped_state_dict.items() if k in model.state_dict()
        }


        print("\n✅ Keys loaded into the model:")
        for k, v in model_state_dict.items():
            shape = tuple(v.shape)
            v_cpu = v.detach().cpu()
            preview = v_cpu.flatten()[:5].numpy()
            print(f"  {k:40s}  shape={str(shape):<20}  values={preview}")


        # # --- 🚫 Print ignored parameters (keys not found in the model) ---
        # ignored_keys = [k for k in mapped_state_dict.keys() if k not in model_state_dict]
        # if ignored_keys:
        #     print(f"\n⚠️ Keys ignored (not in model): {len(ignored_keys)}")
        #     for k in ignored_keys:
        #         v = mapped_state_dict[k]
        #         print(f"  {k:40s}  {tuple(v.shape)}")

        # # --- 🔍 Extra check: print parameters whose shapes don’t match ---
        # print("\n🔍 Checking for shape mismatches...")
        # for k, v in model.state_dict().items():
        #     if k in mapped_state_dict and v.shape != mapped_state_dict[k].shape:
        #         print(f"[Shape mismatch] {k}: model {tuple(v.shape)}, ckpt {tuple(mapped_state_dict[k].shape)}")


        # --- ✅ Load with strict=False to allow partial load ---
        model.load_state_dict(model_state_dict, strict=False)
        model.eval()
    
    else:
        # Build the checkpoint path
        ckpt_path = os.path.join(params["model_dir"], "model_step_3600.pt")

        # Load checkpoint from file
        # Use map_location='cpu' if you don't have GPU, or 'cuda' if you do
        checkpoint = torch.load(ckpt_path, map_location='cuda')

        # If checkpoint is a full dictionary (contains model_state_dict)
        # otherwise, it's likely just the model's state_dict itself
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        # Switch the model to evaluation mode (important for dropout / batchnorm)
        model.eval()

        print("✅ Pretrained model loaded from:", ckpt_path)

    print("\n[INFO] ✅ Model weights successfully loaded.")

    # === Prepare output directory ===
    os.makedirs(params["out_pred_dir"], exist_ok=True)

    # === Evaluate each PPI ===
    for ppi_pair_id in ppi_pair_ids:
        print(f"\n[INFO] Evaluating {ppi_pair_id}")
        in_dir = os.path.join(parent_in_dir, ppi_pair_id)

        fields = ppi_pair_id.split("_")
        if len(fields) < 2:
            continue
        pdbid, chain1 = fields[0], fields[1]
        pids, chains = ["p1"], [chain1]
        if len(fields) == 3 and fields[2] != "":
            pids.append("p2")
            chains.append(fields[2])

        for ix, pid in enumerate(pids):
            pdb_chain_id = f"{pdbid}_{chains[ix]}"
            print(f"  -> Processing {pdb_chain_id}")

            # Load precomputed numpy files
            try:
                rho = np.load(f"{in_dir}/{pid}_rho_wrt_center.npy")
                theta = np.load(f"{in_dir}/{pid}_theta_wrt_center.npy")
                input_feat = np.load(f"{in_dir}/{pid}_input_feat.npy")
                mask = np.load(f"{in_dir}/{pid}_mask.npy")
                indices = np.load(f"{in_dir}/{pid}_list_indices.npy", allow_pickle=True)
            except FileNotFoundError as e:
                print(f"    [WARN] Missing file: {e.filename}")
                continue

            input_feat = mask_input_feat(input_feat, params["feat_mask"])
            print(f"    Total patches: {len(mask)}")

            # Convert to torch tensors
            rho_t = torch.tensor(rho, dtype=torch.float32).cuda()
            theta_t = torch.tensor(theta, dtype=torch.float32).cuda()
            feat_t = torch.tensor(input_feat, dtype=torch.float32).cuda()
            mask_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(2).cuda()

            # Run inference
            tic = time.time()
            with torch.no_grad():
                score, training_loss, eval_labels, full_score = model(rho_t, theta_t, feat_t, mask_t, indices_tensor=indices)
            toc = time.time()

            print(f"    Predicted {len(full_score)} patch full_score in {toc - tic:.3f}s")

            # Save output
            out_path = os.path.join(params["out_pred_dir"], f"pred_{pdbid}_{chains[ix]}.npy")
            print("full_score.shape =", full_score.shape)
            print("full_score =", full_score.cpu().numpy())
            np.save(out_path, full_score.cpu().numpy())
            print(f"    Saved to {out_path}")


if __name__ == "__main__":
    main()
