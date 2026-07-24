"""Small representative model used by CPU predictor tests."""

import torch
from torch import nn


class TinyConv(nn.Module):
    def __init__(self, channels: int = 8) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(channels * 2, 10)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.flatten(self.features(inputs), 1))


def build_model(channels: int = 8) -> nn.Module:
    return TinyConv(channels)
