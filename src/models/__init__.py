# =============================================================================
# src/models/__init__.py — Model registry
# =============================================================================
"""
Add new models by importing them and registering in MODEL_REGISTRY.
The experiment runner calls build_model(name, params) — no hardcoded
model construction anywhere else.
"""

from typing import Callable, Dict
import torch.nn as nn

from src.models.cnn          import SimpleCNN
from src.models.mlp          import ShallowMLP
from src.models.pams_vit     import PAMSViT
from src.models.resnet       import ResNet34Regressor
from src.models.vit_baseline import VanillaViT
from src.models.inception_v3 import InceptionV3Regressor       # ← নতুন import


MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    "simple_cnn":   SimpleCNN,
    "shallow_mlp":  ShallowMLP,
    "pams_vit":     PAMSViT,
    "resnet34":     ResNet34Regressor,
    "vit_baseline": VanillaViT,
    "inception_v3": InceptionV3Regressor,                       # ← নতুন registry line
}


def build_model(name: str, params: dict) -> nn.Module:
    """Instantiate a model by its registry name with the given kwargs."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}. "
                         f"Available: {list(MODEL_REGISTRY.keys())}")
    cls = MODEL_REGISTRY[name]
    return cls(**params)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)