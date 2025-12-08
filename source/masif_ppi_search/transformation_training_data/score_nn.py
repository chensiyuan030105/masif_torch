import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class ScoreNN(nn.Module):
    def __init__(self, device='cuda'):
        super(ScoreNN, self).__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Define layers
        self.conv1 = nn.Conv1d(3, 8, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(8)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(16)
        self.conv3 = nn.Conv1d(16, 32, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.conv4 = nn.Conv1d(32, 64, kernel_size=1)
        self.bn4 = nn.BatchNorm1d(64)
        self.conv5 = nn.Conv1d(64, 128, kernel_size=1)
        self.bn5 = nn.BatchNorm1d(128)
        self.conv6 = nn.Conv1d(128, 256, kernel_size=1)
        self.bn6 = nn.BatchNorm1d(256)

        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 8)
        self.fc6 = nn.Linear(8, 4)
        self.fc7 = nn.Linear(4, 2)

        self.to(self.device)

    def forward(self, x):
        # x shape: (batch_size, n_points, n_features) => (batch_size, 3, 200) after transpose
        x = x.transpose(1, 2).to(self.device).float()  # Conv1d expects (B, C, L)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))

        x = torch.mean(x, dim=2)  # GlobalAveragePooling1D
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.softmax(self.fc7(x), dim=1)
        return x

    def print_layer_weights(self):
        logging.info("Printing model layer weights...")
        for name, param in self.named_parameters():
            logging.info(f"{name}: shape={param.shape}")

    def restore_model(self, path):
        """Restore weights from a saved PyTorch checkpoint (.pt or .pth)"""
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, map_location=self.device))
            logging.info(f"Model weights restored from {path}")
            self.print_layer_weights()
        else:
            logging.warning(f"Checkpoint {path} not found. Model initialized randomly.")

    def train_model(self, features, labels, n_negatives, n_positives, epochs=50, batch_size=32, lr=1e-4):
        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        dataset = TensorDataset(features, labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Compute class weights
        class_weights = torch.tensor([1.0/n_negatives, 1.0/n_positives], dtype=torch.float32).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_features, batch_labels in loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.forward(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_features.size(0)

            avg_loss = epoch_loss / len(dataset)
            logging.info(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_loss:.6f}")

    def eval_model(self, features):
        self.eval()
        features = torch.tensor(features, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.forward(features)
        return outputs.cpu().numpy()


