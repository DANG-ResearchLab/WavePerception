# =============================================================================
# src/models/resnet.py — ResNet34 regressor for polarimetric SWH estimation
# =============================================================================
"""
ResNet34 baseline (He et al., 2016) adapted for 4-channel polarimetric input
and scalar regression.

Architecture follows the rejected manuscript's specification:
  - First 7x7 conv layer modified: in_channels 3 -> 4
  - Original 1000-way classification head replaced with a 1-unit linear head
  - All other blocks (4 residual stages with [3, 4, 6, 3] blocks) unchanged
  - From-scratch initialization (no ImageNet pretraining), consistent with the
    other from-scratch baselines (SimpleCNN, ShallowMLP) and PAMS-ViT.

Total trainable parameters: ~21.3M.
"""

from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models


class ResNet34Regressor(nn.Module):
    """
    ResNet34 with 4-channel polarimetric input and a scalar regression head.

    Args:
        in_channels: Number of input channels (default 4 for raw I0/I45/I90/I135).
        pretrained:  Whether to load ImageNet pretrained weights for the
                     unchanged layers. Default False for fair comparison with
                     the other from-scratch baselines.
        kaiming_init_new_layers: Apply Kaiming-normal init to the modified
                                 first-conv weights when pretrained=True,
                                 to preserve activation variance.

    Forward I/O:
        Input:  (B, in_channels, 224, 224)
        Output: (B,) scalar Hs estimate in cm
    """

    def __init__(
        self,
        in_channels: int = 4,
        pretrained: bool = False,
        kaiming_init_new_layers: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.pretrained  = pretrained

        # Build torchvision ResNet34 backbone
        if pretrained:
            weights = tv_models.ResNet34_Weights.IMAGENET1K_V1
            backbone = tv_models.resnet34(weights=weights)
        else:
            backbone = tv_models.resnet34(weights=None)

        # ---- Modification 1: replace first conv layer for 4-channel input ----
        original_conv1 = backbone.conv1
        new_conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=(original_conv1.bias is not None),
        )
        if pretrained and kaiming_init_new_layers:
            # Initialize new 4-channel conv with Kaiming-normal to preserve
            # activation variance during early training.
            nn.init.kaiming_normal_(new_conv1.weight,
                                    mode="fan_out",
                                    nonlinearity="relu")
        backbone.conv1 = new_conv1

        # ---- Modification 2: replace classification head with regression head ----
        feat_dim = backbone.fc.in_features   # 512 for ResNet34
        backbone.fc = nn.Linear(feat_dim, 1)

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # backbone returns (B, 1); squeeze last dim -> (B,)
        return self.backbone(x).squeeze(-1)