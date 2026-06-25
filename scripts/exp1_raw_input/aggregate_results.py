# =============================================================================
# scripts/aggregate_results.py — Post-LOWTO plots
# =============================================================================
"""
Usage:
    python scripts/aggregate_results.py --run_dir outputs/runs/lowto_cnn_v1_20260528_140523

Produces:
    aggregate/parity_pooled.png       — predicted vs true (12 distinct Hs)
    aggregate/residual_pooled.png     — residual vs true Hs
    aggregate/box_per_wavetrain.png   — RMSE distribution per fold
    aggregate/rmse_vs_hs.png          — per-fold RMSE as function of Hs
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    agg_dir = run_dir / "aggregate"
    fig_dir = agg_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    pooled = pd.read_csv(agg_dir / "pooled_predictions.csv")
    per_wt = pd.read_csv(agg_dir / "per_wavetrain_table.csv")
    with open(agg_dir / "summary_metrics.json") as f:
        summary = json.load(f)

    # ---- 1. Pooled parity plot ----
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pooled["true_hs"], pooled["pred_hs"],
               alpha=0.4, s=15, edgecolors="none")
    lim = [0, 3.3]
    ax.plot(lim, lim, "r--", label="Ideal (1:1)")
    ax.set_xlabel("True Hs (cm)")
    ax.set_ylabel("Predicted Hs (cm)")
    ax.set_title(f"Pooled Parity (LOWTO, n={len(pooled)})\n"
                 f"Pooled R² = {summary['pooled']['r2']:.4f}, "
                 f"RMSE = {summary['pooled']['rmse']:.4f} cm")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / "parity_pooled.png", dpi=200)
    plt.close(fig)

    # ---- 2. Residual plot ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(pooled["true_hs"], pooled["residual"],
               alpha=0.4, s=15, edgecolors="none")
    ax.axhline(0, color="r", linestyle="--")
    ax.set_xlabel("True Hs (cm)")
    ax.set_ylabel("Residual (pred − true, cm)")
    ax.set_title("Pooled Residuals across LOWTO folds")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "residual_pooled.png", dpi=200)
    plt.close(fig)

    # ---- 3. Box plot per wave train ----
    fig, ax = plt.subplots(figsize=(10, 5))
    hs_sorted = sorted(pooled["true_hs"].unique())
    data = [pooled[pooled["true_hs"] == hs]["abs_error"].values for hs in hs_sorted]
    ax.boxplot(data, labels=[f"{hs:.1f}" for hs in hs_sorted])
    ax.set_xlabel("True Hs (cm)")
    ax.set_ylabel("Absolute Error (cm)")
    ax.set_title("Per-wave-train Error Distribution")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(fig_dir / "box_per_wavetrain.png", dpi=200)
    plt.close(fig)

    # ---- 4. RMSE vs Hs ----
    fig, ax = plt.subplots(figsize=(8, 5))
    per_wt_sorted = per_wt.sort_values("test_hs")
    ax.plot(per_wt_sorted["test_hs"], per_wt_sorted["rmse"],
            "o-", label="Per-fold RMSE")
    ax.axhline(summary["per_fold_mean"]["rmse"], color="g", linestyle="--",
               label=f"Mean = {summary['per_fold_mean']['rmse']:.4f}")
    ax.set_xlabel("Test Hs (cm)")
    ax.set_ylabel("Test RMSE (cm)")
    ax.set_title("RMSE vs. Test Wave-Train Hs")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / "rmse_vs_hs.png", dpi=200)
    plt.close(fig)

    print(f"Figures saved to: {fig_dir}")
    print(f"Summary:")
    print(f"  Pooled RMSE:  {summary['pooled']['rmse']:.4f} cm")
    print(f"  Pooled R²:    {summary['pooled']['r2']:.4f}")
    print(f"  Mean ± std:   {summary['per_fold_mean']['rmse']:.4f} "
          f"± {summary['per_fold_std']['rmse']:.4f} cm")


if __name__ == "__main__":
    main()