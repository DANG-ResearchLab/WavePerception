# =============================================================================
# src/models/pams_vit.py — Polarization-Aware Multi-Scale Vision Transformer
# =============================================================================
"""
PAMS-ViT: Polarization-Aware Multi-Scale Vision Transformer for SWH estimation.

This is the proposed architecture. Three core components:
  (1) Learnable Stokes Fusion (LSF)         — currently DISABLED (lsf_enabled=False)
  (2) Multi-Scale Polarimetric Tokenization (MSPT) — 16x16, 32x32, 56x56 patches
  (3) Transformer encoder + regression head

Current configuration: lsf_enabled=False, so raw [I0, I45, I90, I135] (4 channels)
is fed directly to MSPT. When the LSF block is activated in a future ablation,
set `lsf_enabled=True` in the config; MSPT will then receive 8 fused channels
instead of 4 raw ones.

References:
  - Vaswani et al., "Attention is All You Need," NeurIPS 2017.
  - Dosovitskiy et al., "An Image is Worth 16x16 Words: ...," ICLR 2021.
"""

import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# (1) Learnable Stokes Fusion (LSF) — optional, currently inactive
# -----------------------------------------------------------------------------
class LearnableStokesFusion(nn.Module):
    """
    Two 1x1 convolutional layers with GELU, expanding then compressing
    the polarimetric input.
        4 channels  -> 16 channels (1x1 conv, GELU)
       16 channels  ->  8 channels (1x1 conv, GELU)
    """

    def __init__(self, in_channels: int = 4, mid_channels: int = 16, out_channels: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        self.act   = nn.GELU()
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x


# -----------------------------------------------------------------------------
# (2) Multi-Scale Polarimetric Tokenization (MSPT)
# -----------------------------------------------------------------------------
class MultiScalePolarimetricTokenization(nn.Module):
    """
    Three parallel Conv2d projections at patch sizes 16x16, 32x32, 56x56.
    Each yields a token sequence; concatenated along the token axis to form
    a multi-scale token set.

    For 224x224 inputs:
      - 16x16 -> 14x14 = 196 tokens (fine scale, ripples)
      - 32x32 ->  7x 7 =  49 tokens (mid scale, slope variation)
      - 56x56 ->  4x 4 =  16 tokens (coarse scale, envelope)
      Total: 261 tokens per image.
    """

    def __init__(self, in_channels: int, embed_dim: int = 64,
                 patch_sizes: tuple = (16, 32, 56)):
        super().__init__()
        self.patch_sizes = patch_sizes
        self.embed_dim   = embed_dim
        self.projections = nn.ModuleList([
            nn.Conv2d(in_channels, embed_dim, kernel_size=ps, stride=ps)
            for ps in patch_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) where C is in_channels and H = W = 224 (assumed)
        Returns: (B, N_total, embed_dim) where N_total = 196 + 49 + 16 = 261
        """
        token_sequences = []
        for ps, proj in zip(self.patch_sizes, self.projections):
            # Safe cropping: trim H, W to nearest multiple of patch size
            B, C, H, W = x.shape
            H_crop = (H // ps) * ps
            W_crop = (W // ps) * ps
            x_cropped = x[:, :, :H_crop, :W_crop]
            # (B, embed_dim, H/ps, W/ps)
            tokens = proj(x_cropped)
            # Flatten spatial dims -> (B, embed_dim, N_s) -> (B, N_s, embed_dim)
            tokens = tokens.flatten(2).transpose(1, 2)
            token_sequences.append(tokens)
        # Concatenate along token axis
        return torch.cat(token_sequences, dim=1)


# -----------------------------------------------------------------------------
# (3) Transformer encoder block
# -----------------------------------------------------------------------------
class TransformerEncoderBlock(nn.Module):
    """
    Pre-norm transformer encoder: LayerNorm -> MHSA -> residual -> LayerNorm -> MLP -> residual.
    """

    def __init__(self, embed_dim: int, n_heads: int = 4,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, n_heads,
                                           dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MHSA + residual
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        # MLP + residual
        x = x + self.mlp(self.norm2(x))
        return x


# -----------------------------------------------------------------------------
# PAMS-ViT full model
# -----------------------------------------------------------------------------
class PAMSViT(nn.Module):
    """
    Polarization-Aware Multi-Scale Vision Transformer.

    Pipeline (with lsf_enabled=False, current default):
        Raw 4-channel input (B, 4, 224, 224)
          [LSF skipped]
          MSPT             -> (B, 261, embed_dim)
          + special token  -> (B, 262, embed_dim)
          + positional enc
          Transformer x N  -> (B, 262, embed_dim)
          Take special-token output
          Regression head  -> (B,) scalar Hs in cm

    With lsf_enabled=True:
          LSF: 4 -> 16 -> 8 channels
          MSPT receives 8-channel input
          rest identical
    """

    def __init__(
        self,
        in_channels: int = 4,
        image_size: int = 224,
        embed_dim: int = 64,
        patch_sizes: tuple = (16, 32, 56),
        n_transformer_blocks: int = 4,
        n_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        head_hidden: int = 64,
        lsf_enabled: bool = False,
        lsf_mid_channels: int = 16,
        lsf_out_channels: int = 8,
    ):
        super().__init__()
        self.lsf_enabled = lsf_enabled
        self.embed_dim   = embed_dim

        # ---- (1) LSF (optional) ----
        if lsf_enabled:
            self.lsf = LearnableStokesFusion(
                in_channels=in_channels,
                mid_channels=lsf_mid_channels,
                out_channels=lsf_out_channels,
            )
            mspt_in_channels = lsf_out_channels
        else:
            self.lsf = None
            mspt_in_channels = in_channels

        # ---- (2) MSPT ----
        self.mspt = MultiScalePolarimetricTokenization(
            in_channels=mspt_in_channels,
            embed_dim=embed_dim,
            patch_sizes=patch_sizes,
        )

        # Compute total number of tokens (for positional encoding)
        n_tokens = sum((image_size // ps) ** 2 for ps in patch_sizes)
        self.n_tokens = n_tokens   # 196 + 49 + 16 = 261 for image_size=224

        # Special (CLS-like) token + positional encoding
        self.special_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed     = nn.Parameter(torch.zeros(1, n_tokens + 1, embed_dim))
        nn.init.trunc_normal_(self.special_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed,     std=0.02)

        # ---- (3) Transformer encoder ----
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(n_transformer_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # ---- Regression head ----
        self.head = nn.Sequential(
            nn.Linear(embed_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (Optional) LSF
        if self.lsf is not None:
            x = self.lsf(x)
        # MSPT -> token sequence (B, N, D)
        tokens = self.mspt(x)
        B = tokens.shape[0]
        # Prepend special token
        spc = self.special_token.expand(B, -1, -1)
        tokens = torch.cat([spc, tokens], dim=1)
        # Add positional encoding
        tokens = tokens + self.pos_embed[:, : tokens.shape[1], :]
        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        # Take special-token output
        cls_out = tokens[:, 0]
        # Regression head
        return self.head(cls_out).squeeze(-1)