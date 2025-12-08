import numpy as np
from pathlib import Path
import glob
from scipy.spatial import cKDTree
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os
import time
import pickle
import sys
import random
import torch
import logging

from .score_nn import ScoreNN  # Your PyTorch ScoreNN class

"""
train_evaluation_network.py: Train a neural network to score protein complex alignments (based on MaSIF)
Freyr Sverrisson - LPDI STI EPFL 2019
Released under an Apache License 2.0
"""

log_dir = "models/nn_score"
os.makedirs(log_dir, exist_ok=True)

np.random.seed(42)
torch.manual_seed(42)          # CPU
torch.cuda.manual_seed(42)     # GPU (single GPU)
torch.cuda.manual_seed_all(42) # GPU (all GPUs, if using multi-GPU)

data_dir = "transformation_data/"

with open(
    "../lists/training_10.txt"
) as f:
    training_list = f.read().splitlines()

with open(
    "../lists/testing_10.txt"
) as f:
    testing_list = f.read().splitlines()

n_positives = 1 # Number of correctly aligned to train on
n_negatives = 200 # Number of incorrectly aligned
max_rmsd = 5.0
max_npoints = 200
n_features = 3
data_list = glob.glob(data_dir+'*')
data_list = [
    d
    for d in data_list
    if (os.path.exists(d + "/" + "features.npy")) and d.split("/")[-1] in training_list
]
    
all_features = np.empty(
    (len(data_list) * (n_positives + n_negatives), max_npoints, n_features)
)
all_labels = np.empty((len(data_list) * (n_positives + n_negatives), 1))
all_scores = np.empty((len(data_list) * (n_positives + n_negatives), 1))
all_npoints = []
all_idxs = []
all_nsources = []

n_samples = 0  # Initialize number of samples

# Loading data into memory
for i, d in enumerate(data_list):
    if (i % 100 == 0) and (i == 0):
        start = time.time()
    elif i % 100 == 0:
        end = time.time()
        start = time.time()
    
    source_patch_rmsds = np.load(d + "/" + "source_patch_rmsds.npy")

    positive_alignments = np.where(source_patch_rmsds < max_rmsd)[0]
    negative_alignments = np.where(source_patch_rmsds >= max_rmsd)[0]

    # Skip if insufficient alignments
    if len(positive_alignments) == 0 or len(negative_alignments) < n_negatives:
        continue

    # Randomly choose positives and negatives
    chosen_positives = np.random.choice(positive_alignments, n_positives, replace=False)
    chosen_negatives = np.random.choice(negative_alignments, n_negatives, replace=False)
    chosen_alignments = np.concatenate([chosen_positives, chosen_negatives])

    try:
        features = np.load(d + "/" + "features.npy", encoding="latin1", allow_pickle=True)
    except:
        continue
    
    n_sources = len(features)
    features = features[chosen_alignments]
    features_trimmed = np.zeros((len(chosen_alignments), max_npoints, n_features))
    
    # Limit the number of points in each alignment to 'max_npoints'
    for j, f in enumerate(features):
        # f.shape[0] is the number of points in the current alignment
        # f.shape[1] is the number of features per point (n_features)

        if f.shape[0] <= max_npoints:
            # If the alignment has fewer points than the maximum allowed, 
            # copy all points into the trimmed array
            features_trimmed[j, :f.shape[0], :f.shape[1]] = f

        else:
            # If the alignment has more points than max_npoints,
            # randomly select a subset of points
            selected_rows = np.random.choice(f.shape[0], max_npoints, replace=False)
            features_trimmed[j, :, :f.shape[1]] = f[selected_rows]

    labels = np.array(
        (source_patch_rmsds[chosen_alignments] < max_rmsd).astype(int)
    ).reshape(-1, 1)
    # Log the shape and first few values of labels

    all_features[
        n_samples : n_samples + len(chosen_alignments), :, :
    ] = features_trimmed

    all_labels[n_samples : n_samples + len(chosen_alignments)] = labels

    n_samples += len(chosen_alignments)

all_features = all_features[:n_samples]

all_labels = all_labels[:n_samples]

all_idxs = np.concatenate(
    [
        (n_positives + n_negatives) * [i]
        for i in range(int(all_features.shape[0] / (n_positives + n_negatives)))
    ]
)

# Instantiate model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ScoreNN(device=device)

# Move data to torch tensors
features_tensor = torch.tensor(all_features, dtype=torch.float32)
labels_tensor = torch.tensor(all_labels, dtype=torch.long)  # sparse labels

# Training parameters
batch_size = 32
epochs = 10000
learning_rate = 1e-4

# Compute class weights
class_weights = torch.tensor([1.0 / n_negatives, 1.0 / n_positives], dtype=torch.float32).to(device)

# Dataset and DataLoader
from torch.utils.data import TensorDataset, DataLoader
dataset = TensorDataset(features_tensor, labels_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
model.train()
for epoch in range(epochs):
    epoch_loss = 0.0
    for batch_features, batch_labels in dataloader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)
        batch_labels = batch_labels.squeeze(1)

        optimizer.zero_grad()
        outputs = model(batch_features)  # Forward pass
        loss = criterion(outputs, batch_labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update parameters

        epoch_loss += loss.item() * batch_features.size(0)

    avg_loss = epoch_loss / len(dataset)

    if epoch % 100 == 0:

        # Save the trained model
        torch.save(model.state_dict(), f"models/nn_score/model_{epoch}.pt")
