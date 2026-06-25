# =============================================================================
# src/experiments/lowto.py — LOWTO experiment orchestrator
# =============================================================================
"""
Runs N-fold Leave-One-Wave-Train-Out cross-validation. Resumable: any fold
whose fold_summary.json exists is skipped, so SLURM time limits and crashes
can be recovered from without losing completed work.
"""

from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import PolarimetricWaveDataset
from src.splits import (load_splits, build_fold_assignment,
                         get_fold_dataframes, verify_leakage_free)
from src.stats import compute_channel_stats
from src.metrics import compute_metrics, compute_pooled_metrics
from src.models import build_model, count_parameters
from src.train_utils import (build_training_components, train_model,
                              validate_one_epoch)
from src.utils import set_seed

logger = logging.getLogger(__name__)


# =============================================================================
# Loader builder (replaces the notebook's build_loaders_for_fold)
# =============================================================================
def _build_loaders(train_df, val_df, test_df, means, stds,
                   dataset_dir, image_size, batch_size, num_workers,
                   pin_memory, seed):
    train_ds = PolarimetricWaveDataset(train_df, dataset_dir, means, stds, image_size)
    val_ds   = PolarimetricWaveDataset(val_df,   dataset_dir, means, stds, image_size)
    test_ds  = PolarimetricWaveDataset(test_df,  dataset_dir, means, stds, image_size)

    gen = torch.Generator()
    gen.manual_seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=False, generator=gen,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
    )
    return train_loader, val_loader, test_loader


# =============================================================================
# Single fold
# =============================================================================
def run_one_fold(
    fold_k: int, splits_df: pd.DataFrame, splits_sha: str,
    assignment: dict, config: dict, device: torch.device,
    folds_dir: Path,
) -> dict:
    """Run one full LOWTO fold end-to-end; save artifacts; return summary."""
    fold_start = time.perf_counter()
    fold_dir   = folds_dir / f"fold_{fold_k:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    test_wt   = assignment["test_wt"]
    val_wt    = assignment["val_wt"]
    train_wts = assignment["train_wts"]

    test_hs = splits_df.loc[splits_df["wave_train"] == test_wt, "hs_cm"].iloc[0]
    val_hs  = splits_df.loc[splits_df["wave_train"] == val_wt,  "hs_cm"].iloc[0]

    logger.info("=" * 72)
    logger.info(f"FOLD {fold_k} | test={test_wt} (Hs={test_hs:.1f}) | "
                f"val={val_wt} (Hs={val_hs:.1f})")
    logger.info("=" * 72)

    # ---- 1. Partition DataFrames ----
    train_df, val_df, test_df = get_fold_dataframes(splits_df, assignment)
    verify_leakage_free(train_df, val_df, test_df)
    logger.info(f"  train={len(train_df)} frames, val={len(val_df)}, "
                f"test={len(test_df)}")

    # ---- 2. Leakage-free channel stats from THIS fold's train set ----
    dataset_dir = Path(config["data"]["dataset_root"])
    image_size  = int(config["data"]["image_size"])
    means, stds = compute_channel_stats(train_df, dataset_dir, image_size)

    # ---- 3. Loaders ----
    train_cfg = config["training"]
    train_loader, val_loader, test_loader = _build_loaders(
        train_df, val_df, test_df, means, stds,
        dataset_dir=dataset_dir, image_size=image_size,
        batch_size=int(train_cfg["batch_size"]),
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=bool(train_cfg.get("pin_memory", False)),
        seed=int(config["experiment"]["seed"]),
    )

    # ---- 4. Fresh model + optimizer/scheduler/early_stopper ----
    set_seed(int(config["experiment"]["seed"]))   # deterministic init each fold
    model = build_model(config["model"]["name"], config["model"]["params"])
    model = model.to(device)
    logger.info(f"  Model: {config['model']['name']} "
                f"({count_parameters(model):,} trainable params)")

    optimizer, scheduler, early_stopper = build_training_components(model, train_cfg)
    criterion = nn.MSELoss(reduction="mean")

    # ---- 5. Train ----
    history = train_model(
        model, train_loader, val_loader,
        optimizer, scheduler, criterion, early_stopper, device,
        max_epochs=int(train_cfg["max_epochs"]),
        min_epochs=int(train_cfg.get("min_epochs", 5)),
        grad_clip_norm=train_cfg.get("grad_clip_norm", None),
        fold_label=f"fold {fold_k} | test Hs={test_hs:.1f}",
    )

    # ---- 6. Evaluate on test (best weights already restored) ----
    test_loss, test_preds, test_targets = validate_one_epoch(
        model, test_loader, criterion, device,
    )
    metrics = compute_metrics(test_targets, test_preds)
    logger.info(f"  TEST: RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  "
                f"Bias={metrics['bias']:+.4f}  MAPE={metrics['mape']:.2f}%")

    # ---- 7. Save artifacts ----
    # 7a. Per-sample predictions
    pd.DataFrame({
        "fold":        fold_k,
        "wave_train":  test_df["wave_train"].values,
        "frame_index": test_df["frame_index"].values,
        "true_hs":     test_targets,
        "pred_hs":     test_preds,
        "residual":    test_preds - test_targets,
        "abs_error":   np.abs(test_preds - test_targets),
    }).to_csv(fold_dir / "test_predictions.csv", index=False)

    # 7b. Channel stats
    with open(fold_dir / "channel_stats.json", "w") as f:
        json.dump({"fold": fold_k, "train_wave_trains": train_wts,
                   "means": means.tolist(), "stds": stds.tolist()}, f, indent=2)

    # 7c. Training history
    with open(fold_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # 7d. Best checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "best_epoch":       early_stopper.best_epoch,
        "best_val_loss":    float(early_stopper.best_score),
        "model_name":       config["model"]["name"],
        "model_params":     config["model"]["params"],
        "fold":             fold_k,
    }, fold_dir / "best_model.pt")

    # 7e. Fold summary
    fold_time = time.perf_counter() - fold_start
    summary = {
        "fold":           fold_k,
        "test_wave_train": test_wt, "test_hs": float(test_hs),
        "val_wave_train":  val_wt,  "val_hs":  float(val_hs),
        "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
        "best_epoch":    early_stopper.best_epoch,
        "best_val_loss": float(early_stopper.best_score),
        "metrics":       {k: v for k, v in metrics.items() if not k.startswith("_")},
        "actual_epochs": len(history["epoch"]),
        "fold_time_s":   fold_time,
        "splits_sha256": splits_sha,
    }
    with open(fold_dir / "fold_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"  Saved fold {fold_k} artifacts ({fold_time/60:.1f} min)")

    # Free memory before next fold (matters more on GPU)
    del model, optimizer, scheduler, early_stopper
    del train_loader, val_loader, test_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return summary


# =============================================================================
# Master loop (resumable)
# =============================================================================
def run_lowto(config: dict, device: torch.device, run_dir: Path) -> None:
    """Run all N folds. Resumable: skip folds whose summary already exists."""
    set_seed(int(config["experiment"]["seed"]))

    splits_df, splits_sha = load_splits(config["data"]["splits_csv"])

    assignment = build_fold_assignment(
        splits_df,
        fold_column=config["split"]["fold_column"],
        val_rule=config["split"]["val_rule"],
    )
    n_folds = len(assignment)
    expected = int(config["split"].get("n_folds", n_folds))
    if n_folds != expected:
        raise ValueError(f"n_folds mismatch: config={expected}, splits.csv={n_folds}")

    folds_dir = run_dir / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)

    session_start = time.perf_counter()
    ran, skipped = [], []

    for k in range(n_folds):
        fold_dir = folds_dir / f"fold_{k:02d}"
        summary_path = fold_dir / "fold_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                prev = json.load(f)
            logger.info(f"SKIP fold {k} (already done): "
                        f"RMSE={prev['metrics']['rmse']:.4f}")
            skipped.append(k)
            continue
        try:
            run_one_fold(k, splits_df, splits_sha, assignment[k],
                          config, device, folds_dir)
            ran.append(k)
        except Exception as e:
            logger.exception(f"FOLD {k} FAILED: {e}")
            logger.error("Stopping master loop. Resubmit to resume.")
            return

    session_time = time.perf_counter() - session_start
    logger.info(f"Session done. Ran: {ran}, Skipped: {skipped}, "
                f"Time: {session_time/60:.1f} min")

    # All folds done → run aggregation
    completed = sorted([k for k in range(n_folds)
                        if (folds_dir / f"fold_{k:02d}" / "fold_summary.json").exists()])
    if len(completed) == n_folds:
        logger.info("All folds complete — running aggregation.")
        _aggregate(folds_dir, run_dir / "aggregate", n_folds)
    else:
        remaining = [k for k in range(n_folds) if k not in completed]
        logger.info(f"Folds remaining: {remaining}. Resubmit to continue.")


def _aggregate(folds_dir: Path, agg_dir: Path, n_folds: int) -> None:
    """Pool predictions across folds; compute mean ± std and pooled R²."""
    agg_dir.mkdir(parents=True, exist_ok=True)

    per_fold = []
    pooled_rows = []
    for k in range(n_folds):
        fold_dir = folds_dir / f"fold_{k:02d}"
        with open(fold_dir / "fold_summary.json") as f:
            summary = json.load(f)
        preds_df = pd.read_csv(fold_dir / "test_predictions.csv")

        per_fold.append({
            "fold":    k,
            "metrics": summary["metrics"],
            "y_true":  preds_df["true_hs"].values,
            "y_pred":  preds_df["pred_hs"].values,
        })
        pooled_rows.append(preds_df.assign(fold=k))

    # Pooled predictions CSV
    pooled = pd.concat(pooled_rows, ignore_index=True)
    pooled.to_csv(agg_dir / "pooled_predictions.csv", index=False)

    # Aggregated metrics
    agg = compute_pooled_metrics(per_fold)
    with open(agg_dir / "summary_metrics.json", "w") as f:
        json.dump(agg, f, indent=2, default=float)

    # Per-wave-train table (one row per fold)
    table = pd.DataFrame([
        {"fold": k,
         "test_hs":  per_fold[k]["y_true"][0],
         **per_fold[k]["metrics"]}
        for k in range(n_folds)
    ])
    table.to_csv(agg_dir / "per_wavetrain_table.csv", index=False)

    logger.info(f"Aggregation done. Pooled RMSE={agg['pooled']['rmse']:.4f}, "
                f"Pooled R²={agg['pooled']['r2']:.4f}, "
                f"Mean per-fold RMSE={agg['per_fold_mean']['rmse']:.4f} "
                f"± {agg['per_fold_std']['rmse']:.4f}")