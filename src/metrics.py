# =============================================================================
# src/metrics.py — Regression metrics for SWH estimation
# =============================================================================
"""
Pure functions for the six metrics agreed on for the resubmission:
    RMSE (cm), MAE (cm), MSE (cm²), R² (dimensionless),
    Bias (cm, signed), MAPE (%).

Includes a pooled-prediction aggregator for LOWTO: per-fold R² is undefined
when each fold's test set has a single Hs value, but R² computed over the
pooled predictions from all 12 folds (12 distinct Hs values) is meaningful.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8
) -> dict:
    """
    Six regression metrics from 1-D arrays of true and predicted Hs (cm).

    Bias convention: mean(pred - true). Positive => over-prediction.
    MAPE is asymmetric and undefined as y_true -> 0; the eps guard prevents
    division-by-zero. Safe here because Hs > 0 always.
    R² is set to NaN (with _r2_valid=False) when all y_true are equal.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    residuals = y_pred - y_true                       # signed (pred - true)

    mse  = float(np.mean(residuals ** 2))             # cm²
    rmse = float(np.sqrt(mse))                        # cm
    mae  = float(np.mean(np.abs(residuals)))          # cm
    bias = float(np.mean(residuals))                  # cm  (+over / -under)
    mape = float(100.0 * np.mean(np.abs(residuals) / (np.abs(y_true) + eps)))

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < eps:
        r2, r2_valid = float("nan"), False
    else:
        r2, r2_valid = float(1.0 - ss_res / ss_tot), True

    return {
        "rmse": rmse, "mae": mae, "mse": mse,
        "r2": r2, "bias": bias, "mape": mape,
        "_r2_valid": r2_valid,
        "_n_samples": int(len(y_true)),
    }


def compute_pooled_metrics(
    per_fold_results: Sequence[dict],
) -> dict:
    """
    Aggregate metrics across LOWTO folds.

    Two distinct aggregations are produced:
      1. mean ± std of per-fold metrics (e.g., mean RMSE across 12 folds).
      2. POOLED metrics computed on the concatenated predictions/targets
         across all folds. Pooled R² is the key LOWTO output.

    Parameters
    ----------
    per_fold_results : sequence of dicts
        Each dict must have keys 'y_true' (np.ndarray) and 'y_pred' (np.ndarray)
        for that fold's test set, plus a 'metrics' dict from compute_metrics.

    Returns
    -------
    dict with keys:
        'per_fold_mean'  : dict of mean of each metric across folds
        'per_fold_std'   : dict of std of each metric across folds
        'pooled'         : dict of metrics on concatenated y_true/y_pred
        'n_folds'        : number of folds aggregated
        'n_samples_total': total number of test predictions pooled
    """
    metric_names = ["rmse", "mae", "mse", "bias", "mape"]
    # Per-fold mean ± std (skip R² because it's NaN per fold)
    per_fold_arrays = {m: np.array([r["metrics"][m] for r in per_fold_results])
                       for m in metric_names}
    per_fold_mean = {m: float(v.mean()) for m, v in per_fold_arrays.items()}
    per_fold_std  = {m: float(v.std(ddof=1)) if len(v) > 1 else 0.0
                     for m, v in per_fold_arrays.items()}

    # Pool predictions and targets, then compute metrics on the pool
    y_true_all = np.concatenate([r["y_true"] for r in per_fold_results])
    y_pred_all = np.concatenate([r["y_pred"] for r in per_fold_results])
    pooled = compute_metrics(y_true_all, y_pred_all)

    return {
        "per_fold_mean": per_fold_mean,
        "per_fold_std":  per_fold_std,
        "pooled":        pooled,
        "n_folds":       len(per_fold_results),
        "n_samples_total": int(len(y_true_all)),
    }