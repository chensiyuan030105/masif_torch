import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MaSIF_site(nn.Module):
    """
    PyTorch reimplementation of your TF1 MaSIF_site class.
    - Keeps the same high-level architecture: gaussian-based local pooling,
      per-feature linear projection (W_conv, b_conv), rotation max-pooling,
      then MLP refinement. Supports multiple conv layers.
    - Input shapes (expected):
        input_feat:  (batch, n_vertices, n_feat)
        rho_coords:  (batch, n_vertices, 1)  -- radial distances per vertex
        theta_coords:(batch, n_vertices, 1)  -- angular coords per vertex (radians)
        mask:        (batch, n_vertices, 1)  -- 0/1 mask
        indices_tensor: (batch, max_verts)   -- int indices to gather neighbors
    """

    def __init__(
        self,
        max_rho,
        n_thetas=16,
        n_rhos=5,
        n_rotations=16,
        feat_mask=(1.0, 1.0, 1.0, 1.0, 1.0),
        n_conv_layers=1,
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
        self.n_conv_layers = n_conv_layers
        
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
            
        if n_conv_layers > 1:
            self.mu_rho_l2 = nn.Parameter(mu_rho_initial.clone())
            self.mu_theta_l2 = nn.Parameter(mu_theta_initial.clone())
            self.sigma_rho_l2 = nn.Parameter(torch.ones_like(mu_rho_initial) * self.sigma_rho_init)
            self.sigma_theta_l2 = nn.Parameter(torch.ones_like(mu_theta_initial) * self.sigma_theta_init)

        if n_conv_layers > 2:
            self.mu_rho_l3 = nn.Parameter(mu_rho_initial.clone())
            self.mu_theta_l3 = nn.Parameter(mu_theta_initial.clone())
            self.sigma_rho_l3 = nn.Parameter(torch.ones_like(mu_rho_initial) * self.sigma_rho_init)
            self.sigma_theta_l3 = nn.Parameter(torch.ones_like(mu_theta_initial) * self.sigma_theta_init)

        if n_conv_layers > 3:
            self.mu_rho_l4 = nn.Parameter(mu_rho_initial.clone())
            self.mu_theta_l4 = nn.Parameter(mu_theta_initial.clone())
            self.sigma_rho_l4 = nn.Parameter(torch.ones_like(mu_rho_initial) * self.sigma_rho_init)
            self.sigma_theta_l4 = nn.Parameter(torch.ones_like(mu_theta_initial) * self.sigma_theta_init)

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


        # --- Check if second convolutional layer is needed ---
        if n_conv_layers > 1:
            # Initialize weights and bias for the second conv layer
            W_conv_l2 = torch.empty(
                self.n_feat * self.n_thetas * self.n_rhos,
                self.n_thetas * self.n_rhos * self.n_feat,
                device=self.device,
                dtype=torch.float64
            )
            nn.init.xavier_uniform_(W_conv_l2)
            self.W_conv_l2 = nn.Parameter(W_conv_l2)

            b_conv_l2 = torch.zeros(
                self.n_thetas * self.n_rhos * self.n_feat,
                device=self.device,
                dtype=torch.float64
            )
            self.b_conv_l2 = nn.Parameter(b_conv_l2)


        # --- Check if third convolutional layer is needed ---
        if n_conv_layers > 2:
            # Initialize weights and bias for the third conv layer
            W_conv_l3 = torch.empty(
                self.n_thetas * self.n_rhos * self.n_feat,
                self.n_thetas * self.n_rhos * self.n_feat,
                device=self.device,
                dtype=torch.float64
            )
            nn.init.xavier_uniform_(W_conv_l3)
            self.W_conv_l3 = nn.Parameter(W_conv_l3)

            b_conv_l3 = torch.zeros(
                self.n_thetas * self.n_rhos * self.n_feat,
                device=self.device,
                dtype=torch.float64
            )
            self.b_conv_l3 = nn.Parameter(b_conv_l3)


        # --- Check if fourth convolutional layer is needed ---
        if n_conv_layers > 3:
            # Initialize weights and bias for the fourth conv layer
            W_conv_l4 = torch.empty(
                self.n_thetas * self.n_rhos * self.n_thetas * self.n_rhos,
                self.n_thetas * self.n_rhos * self.n_thetas * self.n_rhos,
                device=self.device,
                dtype=torch.float64
            )
            nn.init.xavier_uniform_(W_conv_l4)
            self.W_conv_l4 = nn.Parameter(W_conv_l4)

            b_conv_l4 = torch.zeros(
                self.n_thetas * self.n_rhos * self.n_thetas * self.n_rhos,
                device=self.device,
                dtype=torch.float64
            )
            self.b_conv_l4 = nn.Parameter(b_conv_l4)

        self.fully_connected = torch.nn.Linear(
            self.n_thetas * self.n_rhos * self.n_feat,  # in_features
            self.n_thetas * self.n_rhos                 # out_features
        ).to(self.device).to(torch.float64)
        self.fully_connected_1 = torch.nn.Linear(
            self.n_thetas * self.n_rhos, 
            self.n_feat
        ).to(self.device).to(torch.float64)
        self.fully_connected_2 = torch.nn.Linear(
            self.n_feat,
            self.n_thetas
        ).to(self.device).to(torch.float64)
        self.fully_connected_3 = torch.nn.Linear(
            self.n_thetas,
            self.n_labels
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
    def inference(self, rho_coords, theta_coords, input_feat, mask, W_conv, b_conv, mu_rho, sigma_rho, mu_theta, sigma_theta, mean_gauss_activation=True):
        print(">>> inference <<<")
        n_samples = rho_coords.shape[0]   # batch size
        n_vertices = rho_coords.shape[1]  # number of surface points (vertices)
        n_feat = input_feat.shape[2]      # number of input features

        all_conv_feat = []

        # Convert all tensors to float64 (double precision)
        rho_coords = rho_coords.to(self.device).to(torch.float64)
        theta_coords = theta_coords.to(self.device).to(torch.float64)
        input_feat = input_feat.to(self.device).to(torch.float64)
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
            input_feat_ = input_feat.unsqueeze(3)               # [B, V, n_feat, 1]

            # Multiply features with activations and aggregate
            gauss_desc = gauss_activations * input_feat_
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

    # --------------------------
    # Forward: corresponds to entire pipeline until full_score
    # --------------------------
    def forward(self, rho_coords, theta_coords, input_feat, mask, pos_idx=None, neg_idx=None, labels=None, indices_tensor=None, keep_prob=None):
        
        self.global_desc = []
        initial_coords = self.compute_initial_coordinates()  # shape (n_gauss, 2)
        mu_rho_initial = torch.tensor(initial_coords[:, 0], dtype=torch.float32).unsqueeze(0)
        mu_theta_initial = torch.tensor(initial_coords[:, 1], dtype=torch.float32).unsqueeze(0)
        max_len = max(len(idx) for idx in indices_tensor)
        padded_indices = [idx + [0]*(max_len - len(idx)) for idx in indices_tensor]
        indices_tensor = torch.tensor(padded_indices, dtype=torch.long).to(self.device)

        for i in range(self.n_feat):
            # Extract the i-th feature channel and add an extra dimension at axis=2
            my_input_feat = input_feat[:, :, i].unsqueeze(2)  # equivalent to tf.expand_dims(..., 2)

            # Retrieve the corresponding learnable parameters for feature i
            W_conv = getattr(self, f"W_conv_{i}")
            b_conv = getattr(self, f"b_conv_{i}")
            mu_rho = getattr(self, f"mu_rho_{i}")
            sigma_rho = getattr(self, f"sigma_rho_{i}")
            mu_theta = getattr(self, f"mu_theta_{i}")
            sigma_theta = getattr(self, f"sigma_theta_{i}")

            # Apply the inference function to compute the output descriptor
            out = self.inference(
                rho_coords,
                theta_coords,
                my_input_feat,
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

        self.global_desc = torch.relu(self.fully_connected(self.global_desc))
        self.global_desc = torch.relu(self.fully_connected_1(self.global_desc))

        # Do a second convolutional layer
        if self.n_conv_layers > 1:
            # (1) Gather features: same as tf.gather(global_desc, indices_tensor)
            self.global_desc = torch.gather(
                self.global_desc.unsqueeze(0).expand(len(indices_tensor), -1, -1),  # [batch, 2377, 5]
                1,  # gather along the vertex dimension
                indices_tensor.unsqueeze(-1).expand(-1, -1, self.global_desc.size(-1))  # [batch, max_len, 5]
            )

            # (2) Run second inference with new weights
            self.global_desc = self.inference(
                rho_coords,
                theta_coords,
                self.global_desc,
                mask,
                self.W_conv_l2,
                self.b_conv_l2,
                self.mu_rho_l2,
                self.sigma_rho_l2,
                self.mu_theta_l2,
                self.sigma_theta_l2,
            )  # shape: [batch_size, n_feat * n_thetas * n_rhos]

            batch_size = self.global_desc.shape[0]

            # (3) Reshape to [B, n_feat, n_thetas * n_rhos]
            self.global_desc = self.global_desc.view(batch_size, self.n_feat, self.n_thetas * self.n_rhos)

            # (4) Reduce mean over last dimension
            self.global_desc = torch.mean(self.global_desc, dim=2)

            # Save final shape
            self.global_desc_shape = self.global_desc.shape

        # Do a third convolutional layer
        if self.n_conv_layers > 2:
            # (1) Gather features: same as tf.gather(global_desc, indices_tensor)
            self.global_desc = torch.gather(
                self.global_desc.unsqueeze(0).expand(len(indices_tensor), -1, -1),  # [batch, 2377, 5]
                1,  # gather along the vertex dimension
                indices_tensor.unsqueeze(-1).expand(-1, -1, self.global_desc.size(-1))  # [batch, max_len, 5]
            )

            # (2) Run third inference with new weights
            self.global_desc = self.inference(
                rho_coords,
                theta_coords,
                self.global_desc,
                mask,
                self.W_conv_l3,
                self.b_conv_l3,
                self.mu_rho_l3,
                self.sigma_rho_l3,
                self.mu_theta_l3,
                self.sigma_theta_l3,
            )  # shape: [batch_size, n_feat * n_thetas * n_rhos]
            
            batch_size = self.global_desc.shape[0]

            # (3) Reshape to [B, n_feat, n_thetas * n_rhos]
            self.global_desc = self.global_desc.view(batch_size, self.n_feat, self.n_thetas * self.n_rhos)

            # (4) Reduce mean over last dimension
            self.global_desc = torch.mean(self.global_desc, dim=2)

        # Do a fourth convolutional layer
        if self.n_conv_layers > 3:
            # (1) Gather features: same as tf.gather(global_desc, indices_tensor)
            self.global_desc = torch.gather(
                self.global_desc.unsqueeze(0).expand(len(indices_tensor), -1, -1),  # [batch, 2377, 5]
                1,  # gather along the vertex dimension
                indices_tensor.unsqueeze(-1).expand(-1, -1, self.global_desc.size(-1))  # [batch, max_len, 5]
            )

            # (2) Run fourth inference with new weights
            self.global_desc = self.inference(
                rho_coords,
                theta_coords,
                self.global_desc,
                mask,
                self.W_conv_l4,
                self.b_conv_l4,
                self.mu_rho_l4,
                self.sigma_rho_l4,
                self.mu_theta_l4,
                self.sigma_theta_l4,
            )  # shape: [batch_size, n_feat * n_thetas * n_rhos]
            
            batch_size = self.global_desc.shape[0]

            # (3) Reshape to [B, n_feat, n_thetas * n_rhos]
            self.global_desc = self.global_desc.view(
                batch_size, 
                self.n_thetas * self.n_rhos, 
                self.n_thetas * self.n_rhos
            )

            self.global_desc = self.global_desc.max(dim=2).values
            self.global_desc_shape = self.global_desc.shape
        
        self.global_desc = F.relu(self.fully_connected_2(self.global_desc))
        self.logits = self.fully_connected_3(self.global_desc)

        # Initialize empty eval tensors
        self.eval_labels = torch.empty((0,), dtype=torch.float32, device=self.device)
        self.eval_logits = torch.empty((0, self.logits.shape[1]), dtype=self.logits.dtype, device=self.device)

        # Only process if labels exist
        if labels is not None:
            # Handle pos_idx
            if pos_idx is not None and len(pos_idx) > 0:
                pos_idx_tensor = torch.tensor(pos_idx, dtype=torch.long, device=self.device)
                eval_labels_pos = labels[pos_idx_tensor]
                eval_logits_pos = self.logits[pos_idx_tensor]
            else:
                eval_labels_pos = torch.empty((0, *labels.shape[1:]), dtype=labels.dtype, device=self.device)
                eval_logits_pos = torch.empty((0, *self.logits.shape[1:]), dtype=self.logits.dtype, device=self.device)

            # Handle neg_idx
            if neg_idx is not None and len(neg_idx) > 0:
                neg_idx_tensor = torch.tensor(neg_idx, dtype=torch.long, device=self.device)
                eval_labels_neg = labels[neg_idx_tensor]
                eval_logits_neg = self.logits[neg_idx_tensor]
            else:
                eval_labels_neg = torch.empty((0, *labels.shape[1:]), dtype=labels.dtype, device=self.device)
                eval_logits_neg = torch.empty((0, *self.logits.shape[1:]), dtype=self.logits.dtype, device=self.device)

            # Concatenate along the first dimension
            self.eval_labels = torch.cat([eval_labels_pos, eval_labels_neg], dim=0)
            self.eval_logits = torch.cat([eval_logits_pos, eval_logits_neg], dim=0)

        # Keep a copy of first eval logits
        self.eval_logits_first = self.eval_logits

        # Compute binary cross-entropy loss with logits
        # Only compute if eval_labels is not empty
        if self.eval_labels.numel() > 0:
            # Ensure labels are float32
            labels_float = self.eval_labels.float()
            self.data_loss = F.binary_cross_entropy_with_logits(
                input=self.eval_logits, target=labels_float, reduction='none'
            ).mean()
        else:
            self.data_loss = None  # or torch.tensor(0.0, device=self.eval_logits.device)

        # Apply sigmoid to get probabilities
        self.eval_logits = torch.sigmoid(self.eval_logits)

        # Compute eval_score (take first column if multi-class)
        if self.eval_logits.ndim > 1 and self.eval_logits.shape[1] > 0:
            self.eval_score = self.eval_logits[:, 0].squeeze()
        else:
            self.eval_score = self.eval_logits.squeeze()

        # Apply sigmoid to logits
        self.full_logits = torch.sigmoid(self.logits)  # same as tf.nn.sigmoid

        # Squeeze extra dimensions if needed and take first column
        self.full_score = self.full_logits.squeeze(dim=1)[:, 0] if self.full_logits.ndim > 1 else self.full_logits

        return self.eval_score, self.data_loss, self.eval_labels, self.full_score