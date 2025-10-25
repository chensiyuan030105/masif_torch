import time
import os
from sklearn import metrics
import numpy as np
from IPython.core.debugger import set_trace
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import torch.optim as optim
from sklearn import metrics

# Apply mask to input_feat
def mask_input_feat(input_feat, mask):
    mymask = np.where(np.array(mask) == 0.0)[0]
    return np.delete(input_feat, mymask, axis=2)


def pad_indices(indices, max_verts):
    padded_ix = np.zeros((len(indices), max_verts), dtype=int)
    for patch_ix in range(len(indices)):
        padded_ix[patch_ix] = np.concatenate(
            [indices[patch_ix], [patch_ix] * (max_verts - len(indices[patch_ix]))]
        )
    return padded_ix


# Run masif site on a protein, on a previously trained network.
def run_masif_site(
    params, learning_obj, rho_wrt_center, theta_wrt_center, input_feat, mask, indices
):
    indices = pad_indices(indices, mask.shape[1])
    mask = np.expand_dims(mask, 2)
    feed_dict = {
        learning_obj.rho_coords: rho_wrt_center,
        learning_obj.theta_coords: theta_wrt_center,
        learning_obj.input_feat: input_feat,
        learning_obj.mask: mask,
        learning_obj.indices_tensor: indices,
    }

    score = learning_obj.session.run([learning_obj.full_score], feed_dict=feed_dict)
    return score


def compute_roc_auc(pos, neg):
    labels = np.concatenate([np.ones((len(pos))), np.zeros((len(neg)))])
    dist_pairs = np.concatenate([pos, neg])
    return metrics.roc_auc_score(labels, dist_pairs)


def train_masif_site(
    learning_obj,
    params,
    optimizer,
    scheduler,
    batch_size=100,
    num_iterations=1000000,
    num_iter_test=1000,
    batch_size_val_test=50,
):

    # Open training list.

    list_training_loss = []
    list_training_auc = []
    list_validation_auc = []
    iter_time = []
    best_val_auc = 0

    out_dir = params["model_dir"]
    logfile = open(out_dir + "log.txt", "w")
    for key in params:
        logfile.write("{}: {}\n".format(key, params[key]))

    training_list = open(params["training_list"]).readlines()
    training_list = [x.rstrip() for x in training_list]

    testing_list = open(params["testing_list"]).readlines()
    testing_list = [x.rstrip() for x in testing_list]

    data_dirs = os.listdir(params["masif_precomputation_dir"])
    np.random.shuffle(data_dirs)
    data_dirs = data_dirs
    n_val = len(data_dirs) // 10
    val_dirs = set(data_dirs[(len(data_dirs) - n_val) :])

    for num_iter in range(num_iterations):
        # Start training epoch:
        list_training_loss = []
        list_training_auc = []
        list_val_auc = []
        list_val_pos_labels = []
        list_val_neg_labels = []
        list_val_names = []
        list_training_acc = []
        list_val_acc = []
        logfile.write("Starting epoch {}".format(num_iter + 1))
        print("Starting epoch {}".format(num_iter + 1))
        tic = time.time()
        all_training_labels = []
        all_training_scores = []
        all_val_labels = []
        all_val_scores = []
        all_test_labels = []
        all_test_scores = []
        count_proteins = 0

        list_test_auc = []
        list_test_names = []
        list_test_acc = []
        all_test_labels = []
        all_test_scores = []

        for ppi_pair_id in data_dirs:
            mydir = params["masif_precomputation_dir"] + ppi_pair_id + "/"
            pdbid = ppi_pair_id.split("_")[0]
            chains1 = ppi_pair_id.split("_")[1]
            if len(ppi_pair_id.split("_")) > 2:
                chains2 = ppi_pair_id.split("_")[2]
            else: 
                chains2 = ''
            pids = []
            if pdbid + "_" + chains1 in training_list:
                pids.append("p1")
            if pdbid + "_" + chains2 in training_list:
                pids.append("p2")
            for pid in pids:
                try:
                    iface_labels = np.load(mydir + pid + "_iface_labels.npy")
                except:
                    continue
                if len(iface_labels) > 8000:
                    continue
                if (
                    np.sum(iface_labels) > 0.75 * len(iface_labels)
                    or np.sum(iface_labels) < 30
                ):
                    continue
                count_proteins += 1

                rho_wrt_center = np.load(mydir + pid + "_rho_wrt_center.npy")
                theta_wrt_center = np.load(mydir + pid + "_theta_wrt_center.npy")
                input_feat = np.load(mydir + pid + "_input_feat.npy")
                if np.sum(params["feat_mask"]) < 5:
                    input_feat = mask_input_feat(input_feat, params["feat_mask"])
                mask = np.load(mydir + pid + "_mask.npy")
                mask = np.expand_dims(mask, 2)
                indices = np.load(mydir + pid + "_list_indices.npy", encoding="latin1", allow_pickle=True)
                # indices is (n_verts x <30), it should be
                # indices = pad_indices(indices, mask.shape[1])
                tmp = np.zeros((len(iface_labels), 2))
                for i in range(len(iface_labels)):
                    if iface_labels[i] == 1:
                        tmp[i, 0] = 1
                    else:
                        tmp[i, 1] = 1
                iface_labels_dc = tmp
                logfile.flush()
                pos_labels = np.where(iface_labels == 1)[0]
                neg_labels = np.where(iface_labels == 0)[0]
                np.random.shuffle(neg_labels)
                np.random.shuffle(pos_labels)
                # Scramble neg idx, and only get as many as pos_labels to balance the training.
                if params["n_conv_layers"] == 1:
                    n = min(len(pos_labels), len(neg_labels))
                    n = min(n, batch_size // 2)
                    subset = np.concatenate([neg_labels[:n], pos_labels[:n]])

                    rho_wrt_center = rho_wrt_center[subset]
                    theta_wrt_center = theta_wrt_center[subset]
                    input_feat = input_feat[subset]
                    mask = mask[subset]
                    iface_labels_dc = iface_labels_dc[subset]
                    indices = indices[subset]
                    pos_labels = range(0, n)
                    neg_labels = range(n, n * 2)
                else:
                    n = min(len(pos_labels), len(neg_labels))
                    neg_labels = neg_labels[:n]
                    pos_labels = pos_labels[:n]

                # Prepare the data as PyTorch tensors
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                rho_wrt_center = torch.tensor(rho_wrt_center).to(device)
                theta_wrt_center = torch.tensor(theta_wrt_center).to(device)
                input_feat = torch.tensor(input_feat).to(device)
                mask = torch.tensor(mask).to(device)
                iface_labels_dc = torch.tensor(iface_labels_dc).to(device)
                pos_labels = torch.tensor(pos_labels).to(device)
                neg_labels = torch.tensor(neg_labels).to(device)
                # indices = torch.tensor(indices).to(device)
                
                if ppi_pair_id in val_dirs:
                    # Validation Phase
                    logfile.write(f"Validating on {ppi_pair_id} {pid}\n")
                    
                    learning_obj.eval()  # Set model to evaluation mode (disables dropout)
                    
                    with torch.no_grad():  # No gradients needed for validation
                        # Forward pass
                        score, training_loss, eval_labels, full_score = learning_obj(
                            rho_wrt_center, 
                            theta_wrt_center, 
                            input_feat, 
                            mask, 
                            pos_idx=pos_labels,
                            neg_idx=neg_labels,
                            labels=iface_labels_dc,
                            indices_tensor=indices)
                        # score = outputs['score']
                        # eval_labels = outputs['labels']
                        # training_loss = outputs['loss']
                        
                        # Calculate AUC
                        auc = metrics.roc_auc_score(
                            eval_labels[:, 0].detach().cpu().numpy(), 
                            score.detach().cpu().numpy()
                        )

                        # Log and store validation results
                        list_val_pos_labels.append(torch.sum(iface_labels_dc).item())
                        list_val_neg_labels.append(len(iface_labels_dc) - torch.sum(iface_labels_dc).item())
                        list_val_auc.append(auc)
                        list_val_names.append(ppi_pair_id)
                        all_val_labels = np.concatenate([all_val_labels, eval_labels[:, 0].cpu().numpy()])
                        all_val_scores = np.concatenate([all_val_scores, score.cpu().numpy()])

                else:
                    # Training Phase
                    logfile.write(f"Training on {ppi_pair_id} {pid}\n")
                    
                    learning_obj.train()  # Set model to training mode (enables dropout)
                    
                    optimizer.zero_grad()  # Zero the gradients before the forward pass
                    
                    # Forward pass
                    score, training_loss, eval_labels, full_score = learning_obj(
                        rho_wrt_center, 
                        theta_wrt_center, 
                        input_feat, 
                        mask, 
                        pos_idx=pos_labels,
                        neg_idx=neg_labels,
                        labels=iface_labels_dc,
                        indices_tensor=indices)
                    
                    print(f"Training loss: {training_loss.item()}")
                    with open("./training_log.txt", "a") as f:
                        f.write(f"Training loss: {training_loss.item()}\n")
                    # Backpropagation and optimizer step
                    training_loss.backward()  # Compute gradients
                    optimizer.step()  # Update the model parameters
                    scheduler.step()       # Update learning rate according to schedule

                    # Optional: print current learning rate
                    if (num_iter + 1) % 1 == 0:
                        current_lr = optimizer.param_groups[0]['lr']
                        print(f"Step {num_iter + 1}: learning rate = {current_lr:.6f}")            

                    # Calculate AUC
                    auc = metrics.roc_auc_score(
                        eval_labels[:, 0].detach().cpu().numpy(), 
                        score.detach().cpu().numpy()
                    )
                    # Log and store training results
                    list_training_auc.append(auc)
                    list_training_loss.append(training_loss.item())  # Add scalar loss value
                    
                    # Store training metrics
                    all_training_labels = np.concatenate([all_training_labels, eval_labels[:, 0].detach().cpu().numpy()])
                    all_training_scores = np.concatenate([all_training_scores, score.detach().cpu().numpy()])
                
                logfile.flush()  # Ensure logs are written out

        # Run testing cycle.
        for ppi_pair_id in data_dirs:
            mydir = params["masif_precomputation_dir"] + ppi_pair_id + "/"
            pdbid = ppi_pair_id.split("_")[0]
            chains1 = ppi_pair_id.split("_")[1]
            if len(ppi_pair_id.split("_")) > 2:
                chains2 = ppi_pair_id.split("_")[2]
            else: 
                chains2 = ''
            pids = []
            if pdbid + "_" + chains1 in testing_list:
                pids.append("p1")
            if pdbid + "_" + chains2 in testing_list:
                pids.append("p2")
            for pid in pids:
                logfile.write("Testing on {} {}\n".format(ppi_pair_id, pid))
                try:
                    iface_labels = np.load(mydir + pid + "_iface_labels.npy")
                except:
                    continue
                if len(iface_labels) > 20000:
                    continue
                if (
                    np.sum(iface_labels) > 0.75 * len(iface_labels)
                    or np.sum(iface_labels) < 30
                ):
                    continue
                count_proteins += 1

                rho_wrt_center = np.load(mydir + pid + "_rho_wrt_center.npy")
                theta_wrt_center = np.load(mydir + pid + "_theta_wrt_center.npy")
                input_feat = np.load(mydir + pid + "_input_feat.npy")
                if np.sum(params["feat_mask"]) < 5:
                    input_feat = mask_input_feat(input_feat, params["feat_mask"])
                mask = np.load(mydir + pid + "_mask.npy")
                mask = np.expand_dims(mask, 2)
                indices = np.load(mydir + pid + "_list_indices.npy", encoding="latin1", allow_pickle=True)
                # indices is (n_verts x <30), it should be
                # indices = pad_indices(indices, mask.shape[1])
                tmp = np.zeros((len(iface_labels), 2))
                for i in range(len(iface_labels)):
                    if iface_labels[i] == 1:
                        tmp[i, 0] = 1
                    else:
                        tmp[i, 1] = 1
                iface_labels_dc = tmp
                logfile.flush()
                pos_labels = np.where(iface_labels == 1)[0]
                neg_labels = np.where(iface_labels == 0)[0]

                # Prepare the data as PyTorch tensors
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                rho_wrt_center = torch.tensor(rho_wrt_center).to(device)
                theta_wrt_center = torch.tensor(theta_wrt_center).to(device)
                input_feat = torch.tensor(input_feat).to(device)
                mask = torch.tensor(mask).to(device)
                iface_labels_dc = torch.tensor(iface_labels_dc).to(device)
                pos_labels = torch.tensor(pos_labels).to(device)
                neg_labels = torch.tensor(neg_labels).to(device)
                # indices = torch.tensor(indices).to(device)
                
                learning_obj.eval()  # Set model to evaluation mode (disables dropout)
                    
                with torch.no_grad():  # No gradients needed for validation
                    # Forward pass
                    score, training_loss, eval_labels, full_score = learning_obj(
                        rho_wrt_center, 
                        theta_wrt_center, 
                        input_feat, 
                        mask, 
                        pos_idx=pos_labels,
                        neg_idx=neg_labels,
                        labels=iface_labels_dc,
                        indices_tensor=indices)

                auc = metrics.roc_auc_score(
                    iface_labels, 
                    score.detach().cpu().numpy()
                )
                list_test_auc.append(auc)
                list_test_names.append((ppi_pair_id, pid))
                all_test_labels.append(iface_labels)
                all_test_scores.append(score.detach().cpu().numpy())

        outstr = "Epoch ran on {} proteins\n".format(count_proteins)
        outstr += "Per protein AUC mean (training): {:.4f}; median: {:.4f} for iter {}\n".format(
            np.mean(list_training_auc), np.median(list_training_auc), num_iter + 1
        )
        outstr += "Per protein AUC mean (validation): {:.4f}; median: {:.4f} for iter {}\n".format(
            np.mean(list_val_auc), np.median(list_val_auc), num_iter + 1
        )
        outstr += "Per protein AUC mean (test): {:.4f}; median: {:.4f} for iter {}\n".format(
            np.mean(list_test_auc), np.median(list_test_auc), num_iter + 1
        )
        flat_all_test_labels = np.concatenate(all_test_labels, axis=0)
        flat_all_test_scores = np.concatenate(all_test_scores, axis=0)
        outstr += "Testing auc (all points): {:.2f}".format(
            metrics.roc_auc_score(flat_all_test_labels, flat_all_test_scores)
        )
        outstr += "Epoch took {:2f}s\n".format(time.time() - tic)
        logfile.write(outstr + "\n")
        print(outstr)
        print("np.mean(list_val_auc) =", np.mean(list_val_auc))
        print("best_val_auc =", best_val_auc)

        if (num_iter + 1) % 100 == 0 and num_iter + 1 > 0:

            output_model = os.path.join(out_dir, f"model_step_{num_iter + 1}.pt")

            torch.save({
                'step': num_iter + 1,
                'model_state_dict': learning_obj.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, output_model)

            msg = f">>> Step {num_iter + 1}: Saved model and test results.\n"
            logfile.write(msg)
            print(msg)

    logfile.close()
