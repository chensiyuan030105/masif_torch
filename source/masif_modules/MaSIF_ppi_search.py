import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MaSIF_ppi_search(nn.Module):

    """
    The neural network model to classify two patches into binders or not binders. 
    """

    def __init__(
        self,
        max_rho,
        n_thetas=16,
        n_rhos=5,
        n_rotations=16,
        feat_mask=(1.0, 1.0, 1.0, 1.0, 1.0),
        device=None,
        eps=1e-5,
    ):
        super().__init__()
        self.device = device
        self.max_rho = max_rho
        self.n_thetas = n_thetas
        self.n_rhos = n_rhos
        self.sigma_rho_init = (
            max_rho / 8
        )  # in MoNet was 0.005 with max radius=0.04 (i.e. 8 times smaller)
        self.sigma_theta_init = 1.0  # 0.25
        self.n_rotations = n_rotations
        self.n_feat = int(sum(feat_mask))
        self.n_labels = 2
        self.eps = eps
        
        initial_coords = self.compute_initial_coordinates()  # shape (n_gauss, 2)
        mu_rho_initial = torch.tensor(initial_coords[:, 0], dtype=torch.float32).unsqueeze(0).to(self.device).to(torch.float64)
        mu_theta_initial = torch.tensor(initial_coords[:, 1], dtype=torch.float32).unsqueeze(0).to(self.device).to(torch.float64)

        self.mu_rho = nn.ParameterList()
        self.mu_theta = nn.ParameterList()
        self.sigma_rho = nn.ParameterList()
        self.sigma_theta = nn.ParameterList()

        for i in range(self.n_feat):
            setattr(self, f"mu_rho_{i}", nn.Parameter(mu_rho_initial.clone()))
            setattr(self, f"mu_theta_{i}", nn.Parameter(mu_theta_initial.clone()))
            setattr(self, f"sigma_rho_{i}", nn.Parameter(torch.ones_like(mu_rho_initial) * self.sigma_rho_init))
            setattr(self, f"sigma_theta_{i}", nn.Parameter(torch.ones_like(mu_theta_initial) * self.sigma_theta_init))
            
        self.global_desc = []

        for i in range(self.n_feat):
            b_conv = torch.zeros(
                self.n_thetas * self.n_rhos,
                device=self.device,
                dtype=torch.float64
            )
            setattr(self, f"b_conv_{i}", nn.Parameter(b_conv))

        for i in range(self.n_feat):
            W_conv = torch.empty(
                self.n_thetas * self.n_rhos, self.n_thetas * self.n_rhos,
                device=self.device, dtype=torch.float64
            )
            nn.init.xavier_uniform_(W_conv)
            W_conv = nn.Parameter(W_conv)
            setattr(self, f"W_conv_{i}", W_conv)

        self.fully_connected = torch.nn.Linear(
            self.n_thetas * self.n_rhos * self.n_feat,  # in_features
            self.n_thetas * self.n_rhos                 # out_features
        ).to(self.device).to(torch.float64)

    # --------------------------
    # Utilities
    # --------------------------
    @staticmethod
    def compute_initial_coordinates_static(n_thetas, n_rhos, max_rho):
        range_rho = [0.0, max_rho]
        range_theta = [0, 2 * math.pi]
        grid_rho = [range_rho[0] + (i + 1) * (range_rho[1] - range_rho[0]) / n_rhos for i in range(n_rhos)]
        grid_theta = [range_theta[0] + i * (range_theta[1] - range_theta[0]) / n_thetas for i in range(n_thetas)]
        coords = []
        for t in grid_theta:
            for r in grid_rho:
                coords.append((r, t))
        import numpy as _np

        return _np.array(coords, dtype=_np.float32)  # (n_gauss, 2)

    def compute_initial_coordinates(self):
        return self.compute_initial_coordinates_static(self.n_thetas, self.n_rhos, self.max_rho)

    def count_number_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"Total number parameters: {total}")
        return total

    def frobenius_norm(self, tensor: torch.Tensor):
        return torch.sqrt(torch.sum(tensor * tensor))

    # --------------------------
    # Core inference block (per-spec equivalent of TF inference function)
    # --------------------------
    def inference(self, rho_coords, theta_coords, feat, mask, W_conv, b_conv, mu_rho, sigma_rho, mu_theta, sigma_theta, mean_gauss_activation=True):

        n_samples = rho_coords.shape[0]   # batch size
        n_vertices = rho_coords.shape[1]  # number of surface points (vertices)
        n_feat = feat.shape[2]      # number of input features

        all_conv_feat = []

        # Convert all tensors to float64 (double precision)
        rho_coords = rho_coords.to(self.device).to(torch.float64)
        theta_coords = theta_coords.to(self.device).to(torch.float64)
        feat = feat.to(self.device).to(torch.float64)
        mask = mask.to(self.device).to(torch.float64)
        W_conv = W_conv.to(self.device).to(torch.float64)
        b_conv = b_conv.to(self.device).to(torch.float64)
        mu_rho = mu_rho.to(self.device).to(torch.float64)
        sigma_rho = sigma_rho.to(self.device).to(torch.float64)
        mu_theta = mu_theta.to(self.device).to(torch.float64)
        sigma_theta = sigma_theta.to(self.device).to(torch.float64)

        for k in range(self.n_rotations):
            # Flatten rho and theta coordinates: [batch_size * n_vertices, 1]
            rho_coords_ = rho_coords.reshape(-1, 1)
            thetas_coords_ = theta_coords.reshape(-1, 1)

            # Apply rotation on theta
            thetas_coords_ += k * 2 * math.pi / self.n_rotations
            thetas_coords_ = torch.remainder(thetas_coords_, 2 * math.pi)

            # Gaussian activation on rho and theta
            rho_coords_ = torch.exp(
                -torch.pow(rho_coords_ - mu_rho, 2) / (torch.pow(sigma_rho, 2) + self.eps)
            )
            thetas_coords_ = torch.exp(
                -torch.pow(thetas_coords_ - mu_theta, 2) / (torch.pow(sigma_theta, 2) + self.eps)
            )

            # Element-wise product: [batch_size*n_vertices, n_gauss]
            gauss_activations = rho_coords_ * thetas_coords_
            gauss_activations = gauss_activations.reshape(n_samples, n_vertices, -1)

            # Apply mask
            gauss_activations = gauss_activations * mask

            # Normalize activations if needed
            if mean_gauss_activation:
                gauss_activations = gauss_activations / (
                    torch.sum(gauss_activations, dim=1, keepdim=True) + self.eps
                )

            # Expand dimensions for broadcasting
            gauss_activations = gauss_activations.unsqueeze(2)  # [B, V, 1, n_gauss]
            feat_ = feat.unsqueeze(3)               # [B, V, n_feat, 1]

            # Multiply features with activations and aggregate
            gauss_desc = gauss_activations * feat_
            gauss_desc = torch.sum(gauss_desc, dim=1)  # [B, n_feat, n_gauss]
            gauss_desc = gauss_desc.reshape(n_samples, self.n_thetas * self.n_rhos * n_feat)

            # Linear transformation: [B, n_gauss]
            conv_feat = torch.matmul(gauss_desc, W_conv) + b_conv
            all_conv_feat.append(conv_feat)

        # Max-pooling over rotations
        all_conv_feat = torch.stack(all_conv_feat, dim=0)  # [n_rotations, B, n_gauss]
        conv_feat, _ = torch.max(all_conv_feat, dim=0)     # [B, n_gauss]
        conv_feat = F.relu(conv_feat)

        return conv_feat

    # Data loss
    # Values above 10 are ignored.
    def compute_data_loss(self, pos_thresh=0.0, neg_thresh=10.0):
        # Split global descriptors into 4 parts
        self.global_desc_pos = self.global_desc[0:self.n_patches]
        self.global_desc_binder = self.global_desc[self.n_patches:2*self.n_patches]
        self.global_desc_neg = self.global_desc[2*self.n_patches:3*self.n_patches]
        self.global_desc_neg_2 = self.global_desc[3*self.n_patches:4*self.n_patches]

        # Compute squared distances
        pos_distances = torch.sum((self.global_desc_binder - self.global_desc_pos) ** 2, dim=1)
        neg_distances = torch.sum((self.global_desc_neg - self.global_desc_neg_2) ** 2, dim=1)

        # Store scores (optional, for monitoring)
        self.score = torch.cat([pos_distances, neg_distances], dim=0)

        # Apply thresholds and ReLU
        pos_distances = F.relu(pos_distances - pos_thresh)
        neg_distances = F.relu(-neg_distances + neg_thresh)

        # Compute mean and std
        pos_mean = torch.mean(pos_distances)
        pos_std = torch.std(pos_distances, unbiased=False)
        neg_mean = torch.mean(neg_distances)
        neg_std = torch.std(neg_distances, unbiased=False)

        # Final data loss
        data_loss = pos_mean + pos_std + neg_mean + neg_std

        return data_loss

    # --------------------------
    # Forward: corresponds to entire pipeline until full_score
    # --------------------------
    def forward(self, rho, theta, feat, mask, keep_prob=None):
        
        self.global_desc = []
        initial_coords = self.compute_initial_coordinates()  # shape (n_gauss, 2)
        mu_rho_initial = torch.tensor(initial_coords[:, 0], dtype=torch.float32).unsqueeze(0)
        mu_theta_initial = torch.tensor(initial_coords[:, 1], dtype=torch.float32).unsqueeze(0)

        for i in range(self.n_feat):
            # Extract the i-th feature channel and add an extra dimension at axis=2
            my_feat = feat[:, :, i].unsqueeze(2)  # equivalent to tf.expand_dims(..., 2)

            # Retrieve the corresponding learnable parameters for feature i
            W_conv = getattr(self, f"W_conv_{i}")
            b_conv = getattr(self, f"b_conv_{i}")
            mu_rho = getattr(self, f"mu_rho_{i}")
            sigma_rho = getattr(self, f"sigma_rho_{i}")
            mu_theta = getattr(self, f"mu_theta_{i}")
            sigma_theta = getattr(self, f"sigma_theta_{i}")

            # Apply the inference function to compute the output descriptor
            out = self.inference(
                rho,
                theta,
                my_feat,
                mask,
                W_conv,
                b_conv,
                mu_rho,
                sigma_rho,
                mu_theta,
                sigma_theta
            )

            # Store the descriptor for this feature channel
            self.global_desc.append(out)

        self.global_desc = torch.stack(self.global_desc, dim=1)  # [B, n_rotations, ...]
        self.global_desc_stack = self.global_desc

        # Flatten to [B, n_thetas * n_rhos * n_feat]
        self.global_desc = self.global_desc.reshape(
            -1, self.n_thetas * self.n_rhos * self.n_feat
        )

        self.global_desc_reshape = self.global_desc

        self.global_desc = self.fully_connected(self.global_desc)

        # Compute loss
        self.n_patches = self.global_desc.shape[0] // 4
        self.data_loss = self.compute_data_loss()

        return self.global_desc, self.data_loss, self.score

