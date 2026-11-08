import torch.nn as nn
from src.circuit import build_qlayer
import torch

class HybridClassifier(nn.Module):
    def __init__(self, input_dim=5000, n_qubits=4, n_layers=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, n_qubits)
        self.qlayer = build_qlayer(n_qubits=n_qubits, n_layers=n_layers)
        self.fc2 = nn.Linear(n_qubits, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.qlayer(x)
        x = self.fc2(x)
        return x
