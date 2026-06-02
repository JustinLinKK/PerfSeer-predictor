"""Tiny MLP source model for the PerfSeer source converter demo."""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        output_dim: int = 10,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)
