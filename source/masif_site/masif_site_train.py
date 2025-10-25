# Header variables and parameters.
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import importlib
import sys
from default_config.masif_opts import masif_opts

"""
masif_site_train.py: Entry function to train MaSIF-site.
Pablo Gainza - LPDI STI EPFL 2019
This file is part of MaSIF.
Released under an Apache License 2.0
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device =", device)
params = masif_opts["site"]

if len(sys.argv) > 1:
    custom_params_file = sys.argv[1]
    custom_params = importlib.import_module(custom_params_file, package=None)
    custom_params = custom_params.custom_params

    for key in custom_params:
        print("Setting {} to {} ".format(key, custom_params[key]))
        params[key] = custom_params[key]

# Apply mask to input_feat (using PyTorch tensors)
def mask_input_feat(input_feat, mask):
    mymask = torch.where(mask == 0.0)[0]
    return torch.index_select(input_feat, 2, mymask)

if "pids" not in params:
    params["pids"] = ["p1", "p2"]

# Build the neural network model using PyTorch
from masif_modules.MaSIF_site import MaSIF_site

if "n_theta" in params:
    learning_obj = MaSIF_site(
        params["max_distance"],
        n_thetas=params["n_theta"],
        n_rhos=params["n_rho"],
        n_rotations=params["n_rotations"],
        device=device,
        feat_mask=params["feat_mask"],
        n_conv_layers=params["n_conv_layers"],
    )
else:
    learning_obj = MaSIF_site(
        params["max_distance"],
        n_thetas=4,
        n_rhos=3,
        n_rotations=4,
        device=device,
        feat_mask=params["feat_mask"],
        n_conv_layers=params["n_conv_layers"],
    )

# Prepare for training
from masif_modules.train_masif_site import train_masif_site

print(params["feat_mask"])
if not os.path.exists(params["model_dir"]):
    os.makedirs(params["model_dir"])
# else:
#     # Load existing network.
#     print('Reading pre-trained network')
#     checkpoint = torch.load(params['model_dir'] + 'model.pth')
#     learning_obj.load_state_dict(checkpoint['model_state_dict'])
#     learning_obj.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     print("Model and optimizer states loaded.")

# Initialize optimizer
# optimizer = optim.Adam(learning_obj.parameters(), lr=params["learning_rate"])

# 1️⃣ Define optimizer
# Initial learning rate is 0.001 (used for the first 500 steps)
optimizer = optim.Adam(learning_obj.parameters(), lr=0.01)

# 2️⃣ Define custom learning rate schedule function
def lr_lambda(step):
    """
    Returns a scaling factor for the learning rate depending on the training step.
    - For the first 500 steps: keep lr = 0.001 (scale = 1.0)
    - After 500 steps: reduce lr to 0.0001 (scale = 0.1)
    """
    if step < 500:
        return 1.0   # no change
    elif step < 5000:
        return 0.1   # scale down by 10x
    else:
        return 0.01   # scale down by 10x

# 3️⃣ Create scheduler
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

# 4️⃣ Pass both optimizer and scheduler to your training function
train_masif_site(learning_obj, params, optimizer, scheduler)

