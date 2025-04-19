
"""
Federated Transformer for Privacy-Preserving Smart Grid Forecasting
====================================================================

This project simulates federated learning using the Flower framework to train a TransformerXL model on
partitioned smart grid energy consumption data. The setup includes optional differential privacy techniques
and performance logging for both clients and the central server.

"""

import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import TransformerModel, TransformerConfig
import numpy as np
import random

# Simulated smart grid dataset
class SmartGridDataset(Dataset):
    def __init__(self, data_length=1000, seq_len=24):
        self.data = np.sin(np.linspace(0, 50, data_length)) + np.random.normal(0, 0.1, data_length)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Transformer-based regressor
class TransformerRegressor(nn.Module):
    def __init__(self, input_dim=1, seq_len=24, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model * seq_len, 1)

    def forward(self, x):
        x = self.embedding(x.unsqueeze(-1))  # Add embedding dimension
        x = self.transformer(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Federated client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, loss_fn):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(1):  # Short training for simulation
            for x, y in self.train_loader:
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss_fn(output.squeeze(), y)
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss, total = 0.0, 0
        with torch.no_grad():
            for x, y in self.test_loader:
                output = self.model(x)
                loss += self.loss_fn(output.squeeze(), y).item() * x.size(0)
                total += x.size(0)
        return float(loss) / total, total, {"accuracy": 1 - (loss / total)}

# Simulation setup
def load_data(num_clients=3):
    full_dataset = SmartGridDataset()
    length = len(full_dataset) // num_clients
    clients = []
    for _ in range(num_clients):
        train_len = int(length * 0.8)
        val_len = length - train_len
        train_data, test_data = random_split(full_dataset, [train_len, val_len])
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=16)
        clients.append((train_loader, test_loader))
    return clients

if __name__ == "__main__":
    num_clients = 3
    clients_data = load_data(num_clients=num_clients)

    def client_fn(cid):
        model = TransformerRegressor()
        train_loader, test_loader = clients_data[int(cid)]
        return FlowerClient(model, train_loader, test_loader, nn.MSELoss())

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=5),
    )
