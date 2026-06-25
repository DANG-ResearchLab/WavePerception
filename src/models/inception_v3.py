# =============================================================================
# src/models/inception_v3.py — InceptionV3 regressor for polarimetric SWH
# =============================================================================
"""
InceptionV3 baseline (Szegedy et al., 2016) adapted for 4-channel polarimetric
input and scalar regression, matching the rejected manuscript's Section II-B-4.

Architectural specification (from rejected manuscript, page 6-7, lines 134-151
and Table I, total params = 21.8M):
  - First conv layer modified: in_channels 3 -> 4 (Kaiming-normal init)
  - Inception-A (Mixed_5b-5d), Inception-B (Mixed_6a-6e), Inception-C
    (Mixed_7a-7c) modules unchanged
  - Factorized 1x7/7x1 convolutions retained
  - Final 2048-channel feature map -> GAP -> Linear(2048, 1) regression head
  - Auxiliary classifier (aux_logits) DISABLED for single-output regression
  - Input resolution: 224x224 (matching all other LOWTO baselines for fair
    comparison; PyTorch InceptionV3 accepts arbitrary input sizes ≥ ~75)
  - transform_input = False (our per-fold normalization replaces ImageNet stats)
  - From-scratch initialization (no ImageNet pretraining)
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


class InceptionV3Regressor(nn.Module):
    """
    InceptionV3 with 4-channel polarimetric input and a scalar regression head.

    Forward I/O:
        Input:  (B, 4, 224, 224)
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

        # Build torchvision InceptionV3 backbone
        # - aux_logits=False     : disable auxiliary classifier (single output)
        # - transform_input=False: skip ImageNet-style internal normalization
        # - init_weights=True    : standard init for from-scratch training
        if pretrained:
            weights = tv_models.Inception_V3_Weights.IMAGENET1K_V1
            backbone = tv_models.inception_v3(
                weights=weights,
                aux_logits=True,           # required when loading weights
                transform_input=False,
                init_weights=False,
            )
            # Disable auxiliary classifier post-hoc by setting AuxLogits to identity
            backbone.AuxLogits = None
            backbone.aux_logits = False
        else:
            backbone = tv_models.inception_v3(
                weights=None,
                aux_logits=False,
                transform_input=False,
                init_weights=True,
            )

        # ---- Modification 1: replace first conv layer for 4-channel input ----
        # The first conv in InceptionV3 is at backbone.Conv2d_1a_3x3.conv
        original_conv = backbone.Conv2d_1a_3x3.conv
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=(original_conv.bias is not None),
        )
        if pretrained and kaiming_init_new_layers:
            nn.init.kaiming_normal_(new_conv.weight,
                                    mode="fan_out",
                                    nonlinearity="relu")
        backbone.Conv2d_1a_3x3.conv = new_conv

        # ---- Modification 2: replace classification head with regression head ----
        feat_dim = backbone.fc.in_features   # 2048 for InceptionV3
        backbone.fc = nn.Linear(feat_dim, 1)

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # When aux_logits=False, backbone returns a single tensor (B, 1)
        out = self.backbone(x)
        # Guard against returning a tuple if aux_logits was enabled accidentally
        if isinstance(out, tuple):
            out = out[0]
        return out.squeeze(-1)