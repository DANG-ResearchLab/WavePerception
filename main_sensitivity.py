#!/usr/bin/env python3
# =============================================================================
# main_sensitivity.py — Single-fold hyperparameter sensitivity runner
# =============================================================================
"""
Lightweight runner for single-fold hyperparameter sensitivity analysis on
PAMS-ViT (full LSF). Uses the same machinery as main.py / experiments/lowto.py
but runs only ONE fold with optional hyperparameter overrides from CLI.

Output: outputs/sensitivity/<run_name>_<timestamp>/
  - effective_config.yaml
  - run_log.txt
  - result.json
  - predictions.csv

Usage:
    python main_sensitivity.py --config configs/sensitivity_pamsvit_fullLSF.yaml \
        --fold 5 --run_name default_baseline
    python main_sensitivity.py --config configs/sensitivity_pamsvit_fullLSF.yaml \
        --fold 5 --embed_dim 32 --run_name embed_dim_32
    python main_sensitivity.py --config configs/sensitivity_pamsvit_fullLSF.yaml \
        --fold 5 --lr 1e-3 --run_name lr_1e-3
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.utils import (set_seed, get_device, load_config, create_run_dir,
                       save_config_snapshot, setup_logging,
                       collect_environment_info)
from src.splits import load_splits, build_fold_assignment, get_fold_dataframes
from src.stats import compute_channel_stats
from src.dataset import PolarimetricWaveDataset
from src.train_utils import build_training_components, train_model, \
    validate_one_epoch
from src.metrics import compute_metrics
from src.models import build_model


def parse_args():
    p = argparse.ArgumentParser(
        description="PAMS-ViT single-fold sensitivity runner")
    p.add_argument("--config", type=str, required=True,
                   help="Base YAML config")
    p.add_argument("--fold", type=int, required=True,
                   help="Single fold index (0..n_folds-1)")
    p.add_argument("--embed_dim", type=int, default=None,
                   help="Override model.params.embed_dim")
    p.add_argument("--n_blocks", type=int, default=None,
                   help="Override model.params.n_transformer_blocks")
    p.add_argument("--lr", type=float, default=None,
                   help="Override training.optimizer.lr")
    p.add_argument("--run_name", type=str, required=True,
                   help="Unique identifier for this sensitivity run")
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    # -------------------------------------------------------------------------
    # Apply hyperparameter overrides
    # -------------------------------------------------------------------------
    if args.embed_dim is not None:
        config["model"]["params"]["embed_dim"] = args.embed_dim
    if args.n_blocks is not None:
        config["model"]["params"]["n_transformer_blocks"] = args.n_blocks
    if args.lr is not None:
        config["training"]["optimizer"]["lr"] = float(args.lr)

    # Tag the run name into the config for snapshot
    config["experiment"]["name"] = f"{config['experiment']['name']}__{args.run_name}"

    # -------------------------------------------------------------------------
    # Run directory
    # -------------------------------------------------------------------------
    runs_root = config.get("output", {}).get("runs_root", "outputs/sensitivity")
    run_dir = create_run_dir(
        base_dir=runs_root,
        experiment_name=args.run_name,
        timestamp_format=config.get("output", {}).get(
            "timestamp_format", "%Y%m%d_%H%M%S"),
    )

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    setup_logging(log_file=run_dir / "run_log.txt", level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("PAMS-ViT Sensitivity Analysis — Single Fold")
    logger.info("=" * 60)
    logger.info(f"Run name:      {args.run_name}")
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Config (base): {args.config}")
    logger.info(f"Fold:          {args.fold}")
    logger.info(f"Overrides:")
    logger.info(f"  embed_dim:   {args.embed_dim}")
    logger.info(f"  n_blocks:    {args.n_blocks}")
    logger.info(f"  lr:          {args.lr}")
    logger.info(f"Effective model params: {config['model']['params']}")
    logger.info(f"Effective optimizer lr: "
                f"{config['training']['optimizer']['lr']}")
    logger.info(f"Environment:   {collect_environment_info()}")

    # -------------------------------------------------------------------------
    # Seed + device
    # -------------------------------------------------------------------------
    set_seed(int(config["experiment"]["seed"]))
    device = get_device(config.get("device", "auto"))
    logger.info(f"Device:        {device}")

    # -------------------------------------------------------------------------
    # Snapshot effective config
    # -------------------------------------------------------------------------
    save_config_snapshot(config, run_dir,
                         filename="effective_config.yaml")

    # -------------------------------------------------------------------------
    # Load splits and resolve single fold
    # -------------------------------------------------------------------------
    splits_csv = config["data"]["splits_csv"]
    splits_df, splits_hash = load_splits(splits_csv)
    logger.info(f"Loaded splits.csv: {splits_csv}")
    logger.info(f"SHA-256: {splits_hash}")
    logger.info(f"Total frames: {len(splits_df)}")

    # Build / verify fold assignments
    split_cfg = config["split"]
    n_folds = int(split_cfg["n_folds"])
    fold_col = split_cfg["fold_column"]
    val_rule = split_cfg.get("val_rule", "cyclic_next")

    # if fold_col not in splits_df.columns:
    #     splits_df = build_fold_assignment(splits_df, n_folds=n_folds,
    #                                       fold_column=fold_col)

    # if args.fold < 0 or args.fold >= n_folds:
    #     raise ValueError(f"--fold must be in [0, {n_folds-1}], got {args.fold}")

    # train_df, val_df, test_df = get_fold_dataframes(
    #     splits_df, fold=args.fold, n_folds=n_folds,
    #     fold_column=fold_col, val_rule=val_rule,
    # )

    if args.fold < 0 or args.fold >= n_folds:
        raise ValueError(f"--fold must be in [0, {n_folds-1}], got {args.fold}")

    # Build all fold assignments dict (keyed by fold index 0..n_folds-1)
    fold_assignments = build_fold_assignment(
        splits_df,
        fold_column=fold_col,
        val_rule=val_rule,
    )

    # Pick the assignment for our chosen fold
    fold_assign = fold_assignments[args.fold]
    logger.info(f"Fold {args.fold} assignment: "
                f"test_wt={fold_assign['test_wt']}, "
                f"val_wt={fold_assign['val_wt']}, "
                f"train_wts ({len(fold_assign['train_wts'])}): "
                f"{fold_assign['train_wts']}")

    # Get train/val/test DataFrames
    train_df, val_df, test_df = get_fold_dataframes(splits_df, fold_assign)
    
    logger.info(f"Train frames: {len(train_df)}  "
                f"(wave trains: {sorted(train_df['wave_train'].unique())})")
    logger.info(f"Val frames:   {len(val_df)}  "
                f"(wave trains: {sorted(val_df['wave_train'].unique())})")
    logger.info(f"Test frames:  {len(test_df)}  "
                f"(wave trains: {sorted(test_df['wave_train'].unique())})")

    test_hs = float(test_df['hs_cm'].iloc[0])
    logger.info(f"Test Hs:      {test_hs} cm")

    # -------------------------------------------------------------------------
    # Channel statistics (train only, leakage-free)
    # -------------------------------------------------------------------------
    dataset_dir = config["data"]["dataset_root"]
    image_size = int(config["data"].get("image_size", 224))

    # logger.info("Computing per-channel statistics on training partition ...")
    # stats = compute_channel_stats(
    #     df=train_df,
    #     dataset_dir=dataset_dir,
    #     image_size=image_size,
    # )
    # means = stats["means"]
    # stds = stats["stds"]
    # logger.info(f"Means: {means.tolist()}")
    # logger.info(f"Stds:  {stds.tolist()}")

    logger.info("Computing per-channel statistics on training partition ...")
    means, stds = compute_channel_stats(
        train_df=train_df,
        dataset_dir=dataset_dir,
        image_size=image_size,
    )
    logger.info(f"Means: {means.tolist()}")
    logger.info(f"Stds:  {stds.tolist()}")

    # Save channel stats
    np.savez(run_dir / "channel_stats.npz", means=means, stds=stds)

    # -------------------------------------------------------------------------
    # Datasets and loaders
    # -------------------------------------------------------------------------
    train_ds = PolarimetricWaveDataset(train_df, dataset_dir, means, stds,
                                       image_size=image_size)
    val_ds = PolarimetricWaveDataset(val_df, dataset_dir, means, stds,
                                     image_size=image_size)
    test_ds = PolarimetricWaveDataset(test_df, dataset_dir, means, stds,
                                      image_size=image_size)

    bs = int(config["training"]["batch_size"])
    nw = int(config["training"].get("num_workers", 4))
    pin = bool(config["training"].get("pin_memory", True))

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=pin, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=nw, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False,
                             num_workers=nw, pin_memory=pin)

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    model = build_model(config["model"]["name"], config["model"]["params"])
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {config['model']['name']} "
                f"({n_params:,} trainable params)")

    # -------------------------------------------------------------------------
    # Training components
    # -------------------------------------------------------------------------
    optimizer, scheduler, early_stopper = build_training_components(
        model, config["training"]
    )
    criterion = nn.MSELoss()

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------
    max_epochs = int(config["training"].get("max_epochs", 60))
    min_epochs = int(config["training"].get("min_epochs", 20))
    grad_clip = config["training"].get("grad_clip_norm", 1.0)

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        early_stopper=early_stopper,
        device=device,
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        grad_clip_norm=grad_clip,
        fold_label=f"sensitivity fold={args.fold}",
    )

    # Save training history
    pd.DataFrame({
        k: history[k] for k in ("epoch", "train_loss", "val_loss",
                                "lr", "epoch_time")
    }).to_csv(run_dir / "training_history.csv", index=False)

    # -------------------------------------------------------------------------
    # Evaluate on test set (best weights already restored by train_model)
    # -------------------------------------------------------------------------
    test_loss, test_preds, test_targets = validate_one_epoch(
        model, test_loader, criterion, device
    )
    metrics = compute_metrics(test_targets, test_preds)
    logger.info(f"Test loss:    {test_loss:.6f}")
    logger.info(f"Test RMSE:    {metrics['rmse']:.4f} cm")
    logger.info(f"Test MAE:     {metrics['mae']:.4f} cm")

    # Save predictions
    pd.DataFrame({
        "target": test_targets, "prediction": test_preds,
    }).to_csv(run_dir / "predictions.csv", index=False)

    # -------------------------------------------------------------------------
    # Result JSON
    # -------------------------------------------------------------------------
    result = {
        "run_name": args.run_name,
        "fold": args.fold,
        "test_hs": test_hs,
        "hyperparameters": {
            "embed_dim": config["model"]["params"].get("embed_dim"),
            "n_blocks": config["model"]["params"].get(
                "n_transformer_blocks"),
            "lr": config["training"]["optimizer"]["lr"],
        },
        "n_params": int(n_params),
        "best_epoch": int(early_stopper.best_epoch),
        "best_val_loss": float(early_stopper.best_score),
        "test_loss": float(test_loss),
        "test_metrics": {k: float(v) for k, v in metrics.items()},
        "splits_hash": splits_hash,
        "training_time_s": float(history.get("total_train_time_s", 0.0)),
    }
    with open(run_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"\nResult saved: {run_dir / 'result.json'}")
    logger.info(f"Done. (run_dir = {run_dir})")


if __name__ == "__main__":
    main()