# =============================================================================
# src/train_utils.py — Training loop, validation, early stopping
# =============================================================================
"""
Reusable training utilities. Device-aware (CPU or CUDA), config-driven, with
the min_epochs safeguard that prevents an untrained network's lucky early
validation score from being captured as 'best'.
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# =============================================================================
# Early stopping
# =============================================================================
class EarlyStopping:
    """
    Monitors a validation metric; signals stop and tracks best weights.
    Best tracking is gated by `min_epoch_allowed` to ignore the lucky-but-
    untrained early epochs.
    """

    def __init__(self, patience: int = 10, delta: float = 1e-4,
                 mode: str = "min"):
        self.patience    = int(patience)
        self.delta       = float(delta)
        self.mode        = mode
        self.best_score  = float("inf") if mode == "min" else float("-inf")
        self.best_epoch  = -1
        self.best_state  = None
        self.counter     = 0
        self.should_stop = False

    def _is_improvement(self, score: float) -> bool:
        if self.mode == "min":
            return score < (self.best_score - self.delta)
        return score > (self.best_score + self.delta)

    def step(self, score: float, epoch: int, model: nn.Module,
             allow_best: bool = True) -> bool:
        """
        Update with the latest val score.

        Parameters
        ----------
        allow_best : bool
            If False, the score is observed but cannot be marked as best
            (used while epoch < min_epochs). Prevents capturing early-epoch
            artifacts.

        Returns
        -------
        bool : True if patience exhausted (training should stop).
        """
        if not allow_best:
            return False

        if self._is_improvement(score):
            self.best_score = score
            self.best_epoch = epoch
            # CPU copy to avoid pinning GPU memory unnecessarily
            self.best_state = {k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()}
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# =============================================================================
# Optimizer / scheduler factory
# =============================================================================
def build_training_components(model: nn.Module, training_cfg: dict
                              ) -> Tuple[torch.optim.Optimizer,
                                         torch.optim.lr_scheduler._LRScheduler,
                                         EarlyStopping]:
    """
    Build fresh optimizer + scheduler + early_stopper from the training config.
    Called once per fold so state is reset.
    """
    opt_cfg = training_cfg["optimizer"]
    sch_cfg = training_cfg["scheduler"]
    es_cfg  = training_cfg["early_stopping"]

    if opt_cfg["name"].lower() != "adamw":
        raise NotImplementedError(f"optimizer {opt_cfg['name']!r} not supported")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg["lr"]),
        weight_decay=float(opt_cfg["weight_decay"]),
        betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        eps=float(opt_cfg.get("eps", 1e-8)),
    )

    if sch_cfg["name"].lower() != "reduce_lr_on_plateau":
        raise NotImplementedError(f"scheduler {sch_cfg['name']!r} not supported")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=sch_cfg.get("mode", "min"),
        factor=float(sch_cfg["factor"]),
        patience=int(sch_cfg["patience"]),
        threshold=float(sch_cfg.get("threshold", 1e-4)),
        threshold_mode=sch_cfg.get("threshold_mode", "rel"),
        min_lr=float(sch_cfg.get("min_lr", 1e-7)),
    )

    early_stopper = EarlyStopping(
        patience=int(es_cfg["patience"]),
        delta=float(es_cfg["delta"]),
        mode=es_cfg.get("mode", "min"),
    )
    return optimizer, scheduler, early_stopper


# =============================================================================
# Single-epoch passes
# =============================================================================
def train_one_epoch(
    model: nn.Module, loader: DataLoader,
    optimizer: torch.optim.Optimizer, criterion: nn.Module,
    device: torch.device, grad_clip_norm: float | None = 1.0,
) -> float:
    """One training epoch. Returns sample-weighted mean loss."""
    model.train()
    total_loss, total_samples = 0.0, 0
    for images, targets in loader:
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        preds = model(images)
        loss  = criterion(preds, targets)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        bs = images.size(0)
        total_loss    += loss.item() * bs
        total_samples += bs
    return total_loss / total_samples


def validate_one_epoch(
    model: nn.Module, loader: DataLoader, criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """One eval pass. Returns (mean_loss, preds_np, targets_np)."""
    model.eval()
    total_loss, total_samples = 0.0, 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, targets in loader:
            images  = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            preds   = model(images)
            loss    = criterion(preds, targets)

            bs = images.size(0)
            total_loss    += loss.item() * bs
            total_samples += bs
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

    mean_loss = total_loss / total_samples
    preds_np   = np.concatenate(all_preds,   axis=0)
    targets_np = np.concatenate(all_targets, axis=0)
    return mean_loss, preds_np, targets_np


# =============================================================================
# Full training loop
# =============================================================================
def train_model(
    model: nn.Module,
    train_loader: DataLoader, val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: nn.Module,
    early_stopper: EarlyStopping,
    device: torch.device,
    max_epochs: int = 60,
    min_epochs: int = 5,
    grad_clip_norm: float | None = 1.0,
    fold_label: str = "",
) -> dict:
    """
    Full training loop with validation, LR scheduling, and early stopping.

    min_epochs: Best-tracking is gated until epoch >= min_epochs to avoid
    capturing the lucky early validation loss of an untrained network.
    """
    history = {"epoch": [], "train_loss": [], "val_loss": [],
               "lr": [], "epoch_time": []}

    label = f" [{fold_label}]" if fold_label else ""
    logger.info(f"Training start{label}: max_epochs={max_epochs}, "
                f"min_epochs={min_epochs}")
    logger.info(f"{'Epoch':>5} | {'Train Loss':>11} | {'Val Loss':>11} | "
                f"{'LR':>9} | {'Time(s)':>7} | Notes")

    t0 = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        t_epoch = time.perf_counter()

        train_loss = train_one_epoch(model, train_loader, optimizer,
                                     criterion, device, grad_clip_norm)
        val_loss, _, _ = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.perf_counter() - t_epoch
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)
        history["epoch_time"].append(epoch_time)

        # Gate best-tracking until min_epochs
        allow_best = epoch >= min_epochs
        is_improvement = (
            allow_best and
            val_loss < (early_stopper.best_score - early_stopper.delta)
        )
        should_stop = early_stopper.step(val_loss, epoch, model,
                                          allow_best=allow_best)

        if not allow_best:
            notes = f"warmup ({epoch}/{min_epochs})"
        elif is_improvement:
            notes = f"↓ best ({val_loss:.6f})"
        else:
            notes = f"no improve ({early_stopper.counter}/{early_stopper.patience})"

        logger.info(f"{epoch:>5} | {train_loss:>11.6f} | {val_loss:>11.6f} | "
                    f"{current_lr:>9.2e} | {epoch_time:>7.1f} | {notes}")

        if should_stop:
            logger.info(f"Early stopping at epoch {epoch}. "
                        f"Best epoch: {early_stopper.best_epoch} "
                        f"(val_loss={early_stopper.best_score:.6f}).")
            break

    total_time = time.perf_counter() - t0

    # Restore best weights (back on the correct device)
    if early_stopper.best_state is not None:
        model.load_state_dict({k: v.to(device)
                               for k, v in early_stopper.best_state.items()})
        logger.info(f"Restored best weights from epoch {early_stopper.best_epoch}. "
                    f"Total training time: {total_time/60:.2f} min.")
    else:
        logger.warning("No best state captured (training ended without improvement).")

    history["total_train_time_s"] = total_time
    return history