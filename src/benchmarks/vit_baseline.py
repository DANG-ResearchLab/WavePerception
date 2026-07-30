# =============================================================================
# src/models/vit_baseline.py — Vanilla Vision Transformer (ViT) for SWH regression
# =============================================================================
"""
Vanilla ViT baseline reconstructed from the rejected manuscript (Section II-B-5).

Architectural specification (from rejected manuscript, page 7, lines 152-169
and Table I, total params = 183k):
  - Patch size:           16 x 16  -> 196 patches at 224 input resolution
  - Embedding dim D:      64
  - CLS-like token:       1 trainable, prepended to patch sequence
  - Total tokens:         197 (196 + 1)
  - Transformer blocks:   4 (POST-norm: x = LN(x + sublayer(x)) equivalent in
                             the original ViT formulation, but here we follow
                             the rejected code: x' = attn(LN(x)) + x; x'' = MLP(LN(x')) + x'
                             which is the PRE-norm convention. The rejected
                             manuscript code uses this formulation; we follow
                             it exactly for fair reconstruction.)
  - Attention heads:      4 (D / 16, standard choice)
  - MLP hidden dim:       64 (mlp_ratio = 1.0; gives 183k total params as in
                             the rejected manuscript Table I)
  - Regression head:      LayerNorm -> Linear(D, 64) -> GELU -> Linear(64, 1)

This implementation reproduces the rejected manuscript's ViT baseline as
faithfully as possible from its published description and source snippet,
so that the LOWTO comparison against PAMS-ViT isolates the contribution
of multi-scale tokenization rather than architectural differences in the
transformer encoder itself.
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Single-scale patch embedding via strided convolution."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)       # (B, D, H', W')
        x = x.flatten(2)              # (B, D, N)
        x = x.transpose(1, 2)         # (B, N, D)
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Pre-norm transformer block, matching the rejected manuscript's code:
        x = attn(LN(x)) + x
        x = mlp(LN(x))  + x
    """

    def __init__(self, embed_dim: int, n_heads: int, mlp_hidden: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = attn_out + residual

        residual = x
        x = self.mlp(self.ln2(x)) + residual
        return x


class MLPHead(nn.Module):
    """Regression head: LayerNorm -> Linear -> GELU -> Linear -> scalar."""

    def __init__(self, embed_dim: int, hidden: int = 64):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim)
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        return self.regressor(x)


class VanillaViT(nn.Module):
    """
    Vanilla single-scale Vision Transformer for SWH regression.

    Forward I/O:
        Input:  (B, in_channels, image_size, image_size)
        Output: (B,) scalar Hs estimate in cm
    """

    def __init__(
        self,
        in_channels: int = 4,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 64,
        n_transformer_blocks: int = 4,
        n_heads: int = 4,
        mlp_hidden: int = 64,
        head_hidden: int = 64,
    ):
        super().__init__()
        assert image_size % patch_size == 0, \
            f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"

        self.in_channels = in_channels
        self.image_size  = image_size
        self.patch_size  = patch_size
        self.embed_dim   = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        n_patches = (image_size // patch_size) ** 2  # 196 for 224/16

        # CLS-like special token + learnable positional encoding
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, embed_dim))
        # Initialize with small-std normal, consistent with standard ViT practice
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder stack
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                n_heads=n_heads,
                mlp_hidden=mlp_hidden,
            )
            for _ in range(n_transformer_blocks)
        ])

        # Regression head
        self.mlp_head = MLPHead(embed_dim, hidden=head_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patch embedding -> (B, N, D)
        x = self.patch_embed(x)
        B = x.size(0)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)               # (B, N+1, D)

        # Add positional encoding
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Take CLS token output
        cls_out = x[:, 0]

        # Regression head -> (B, 1) -> (B,)
        return self.mlp_head(cls_out).squeeze(-1)