"""Model whose BatchNorm label is absent from the deployed student vocabulary."""

import torch
from torch import nn


class TinyBatchNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.features(inputs)


def build_model() -> nn.Module:
    return TinyBatchNorm()
