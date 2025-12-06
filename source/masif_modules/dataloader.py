import numpy as np
import torch
from torch.utils.data import Dataset

class PpiSearchCachedDataset(Dataset):
    def __init__(
        self,
        binder_rho, binder_theta, binder_feat, binder_mask,
        pos_rho, pos_theta, pos_feat, pos_mask,
        neg_rho, neg_theta, neg_feat, neg_mask,
        pos_training_idx, pos_val_idx, pos_test_idx,
        neg_training_idx, neg_val_idx, neg_test_idx,
        split="train",
    ):
        assert split in ("train", "val", "test")

        # ---- convert ALL arrays to torch once (CPU) ----
        self.binder_rho   = torch.as_tensor(binder_rho,   dtype=torch.float32)
        self.binder_theta = torch.as_tensor(binder_theta, dtype=torch.float32)
        self.binder_feat  = torch.as_tensor(binder_feat,  dtype=torch.float32)
        self.binder_mask  = torch.as_tensor(binder_mask,  dtype=torch.float32)

        self.pos_rho   = torch.as_tensor(pos_rho,   dtype=torch.float32)
        self.pos_theta = torch.as_tensor(pos_theta, dtype=torch.float32)
        self.pos_feat  = torch.as_tensor(pos_feat,  dtype=torch.float32)
        self.pos_mask  = torch.as_tensor(pos_mask,  dtype=torch.float32)

        self.neg_rho   = torch.as_tensor(neg_rho,   dtype=torch.float32)
        self.neg_theta = torch.as_tensor(neg_theta, dtype=torch.float32)
        self.neg_feat  = torch.as_tensor(neg_feat,  dtype=torch.float32)
        self.neg_mask  = torch.as_tensor(neg_mask,  dtype=torch.float32)

        # ---- pick split indices, and store as torch.long ----
        if split == "train":
            pos_idx = pos_training_idx
            neg_idx = neg_training_idx
        elif split == "val":
            pos_idx = pos_val_idx
            neg_idx = neg_val_idx
        else:
            pos_idx = pos_test_idx
            neg_idx = neg_test_idx

        self.pos_idx = torch.as_tensor(np.asarray(pos_idx, dtype=np.int64), dtype=torch.long)
        self.neg_idx = torch.as_tensor(np.asarray(neg_idx, dtype=np.int64), dtype=torch.long)

        if len(self.pos_idx) != len(self.neg_idx):
            raise ValueError(f"pos_idx and neg_idx length mismatch for split={split}: "
                             f"{len(self.pos_idx)} vs {len(self.neg_idx)}")

    def __len__(self):
        return int(self.pos_idx.numel())

    def __getitem__(self, i):
        p = int(self.pos_idx[i].item())
        n = int(self.neg_idx[i].item())

        return (
            self.binder_rho[p], self.binder_theta[p], self.binder_feat[p], self.binder_mask[p],
            self.pos_rho[p],    self.pos_theta[p],    self.pos_feat[p],    self.pos_mask[p],
            self.neg_rho[n],    self.neg_theta[n],    self.neg_feat[n],    self.neg_mask[n],
        )



