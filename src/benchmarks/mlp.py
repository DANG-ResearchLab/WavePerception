# =============================================================================
# src/models/mlp.py — ShallowMLP model
# =============================================================================
"""
Shallow multilayer perceptron baseline for SWH regression.

The raw 4-channel input is flattened to a single 200,704-D vector, then
processed by two hidden layers (512, 128) with ReLU and dropout. ~102.8M
trainable parameters — the largest baseline by far, but with no spatial
inductive bias.

This is the baseline that establishes how much value spatial structure
(convolutions, attention) adds: if a flatten-then-MLP achieves similar
accuracy to CNN/ViT, the visual structure doesn't matter much.
"""

import torch
import torch.nn as nn


class ShallowMLP(nn.Module):
    """
    Three-layer MLP that operates on the FLATTENED 4-channel polarimetric image.

    Architecture:
        Input  (B, 4, 224, 224)
          flatten         -> (B, 200704)
          Linear(200704 -> hidden1) + ReLU + Dropout
          Linear(hidden1 ->  hidden2) + ReLU + Dropout
          Linear(hidden2 ->       1)
        Output (B,) scalar Hs in cm

    Defaults from the rejected paper: hidden1=512, hidden2=128, dropout=0.3.
    """

    def __init__(
        self,
        in_channels: int = 4,
        image_size: int = 224,
        hidden1: int = 512,
        hidden2: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.image_size  = image_size
        flat_dim = in_channels * image_size * image_size   # 4 * 224 * 224 = 200,704

        self.fc1 = nn.Linear(flat_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, 224, 224) -> (B, 200704)
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x.squeeze(-1)