# =============================================================================
# src/models/cnn.py — SimpleCNN model
# =============================================================================
"""
Convolutional baseline for SWH regression (from the rejected paper).
Raw 4-channel input [I0, I45, I90, I135], output a scalar Hs in cm.
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """4-block CNN with BatchNorm, MaxPool, AdaptiveAvgPool, MLP head.
    ~250k parameters."""

    def __init__(self, in_channels: int = 4, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),  nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),  nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.regressor(x)
        return x.squeeze(-1)