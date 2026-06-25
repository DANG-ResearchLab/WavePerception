#!/usr/bin/env python3
"""
Aggregate single-fold sensitivity runs into a summary CSV and a Markdown
table suitable for inclusion in the manuscript.

Reads result.json from each run directory under outputs/sensitivity/
and produces sensitivity_summary.csv at the top of that directory.
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    runs_dir = Path(args.runs_dir)

    rows = []
    for result_file in sorted(runs_dir.glob("*/result.json")):
        with open(result_file) as f:
            res = json.load(f)
        rows.append({
            "run_name": res["run_name"],
            "fold": res["fold"],
            "embed_dim": res["hyperparameters"]["embed_dim"],
            "n_blocks": res["hyperparameters"]["n_blocks"],
            "lr": res["hyperparameters"]["lr"],
            "n_params": res["n_params"],
            "best_epoch": res["best_epoch"],
            "test_rmse": res["test_metrics"]["rmse"],
            "test_mae": res["test_metrics"]["mae"],
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["run_name"]).reset_index(drop=True)

    csv_path = runs_dir / "sensitivity_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    print(df.to_string(index=False))

    # IEEE-style markdown table for manuscript
    md_path = runs_dir / "sensitivity_summary_for_manuscript.md"
    with open(md_path, "w") as f:
        f.write("# Sensitivity Analysis Results\n\n")
        f.write("Single-fold (fold k=5, Hs=1.5 cm) sensitivity on the "
                "full PAMS-ViT model.\n\n")
        f.write("| Run | embed_dim | n_blocks | lr | params | "
                "test RMSE (cm) | test MAE (cm) |\n")
        f.write("|-----|-----------|----------|-----|--------|"
                "----------------|---------------|\n")
        for _, row in df.iterrows():
            f.write(f"| {row['run_name']} | {row['embed_dim']} | "
                    f"{row['n_blocks']} | {row['lr']:.0e} | "
                    f"{row['n_params']:,} | {row['test_rmse']:.4f} | "
                    f"{row['test_mae']:.4f} |\n")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()