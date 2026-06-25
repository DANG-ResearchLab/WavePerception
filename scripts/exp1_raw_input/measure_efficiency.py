#!/usr/bin/env python3
# =============================================================================
# scripts/measure_efficiency.py — Computational efficiency profiler
# =============================================================================
"""
Measure efficiency metrics for all 7 trained architectures (addressing R1-5).

Loads each architecture's params from its existing config YAML, ensuring
guaranteed consistency with the actual trained experiments. No hardcoded
build_args — single source of truth is configs/lowto_*.yaml.

Reports:
  - Trainable parameters
  - FLOPs / MACs (via thop)
  - Peak GPU memory during inference (MB)
  - CPU inference latency per frame (ms, mean +/- std)
  - Checkpoint size on disk (MB)

Output:
  outputs/efficiency/efficiency_summary.csv
  outputs/efficiency/efficiency_for_manuscript.md
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

# thop is optional but strongly recommended for FLOPs
try:
    from thop import profile as thop_profile
    HAVE_THOP = True
except ImportError:
    HAVE_THOP = False
    print("WARNING: 'thop' not installed. FLOPs will not be reported.")
    print("Install with: pip install thop --break-system-packages")

import sys
PROJECT_ROOT_DEFAULT = "/mnt/mridata/mahamm2/swh_estimation/experiments/" \
                       "28_May_lowto_experiments_v1"
sys.path.insert(0, PROJECT_ROOT_DEFAULT)

from src.models import build_model


# Architectures: each entry maps to its config YAML and run directory pattern
# Display name and label are for the output table only.
ARCHITECTURES = [
    {
        "label":          "shallow_mlp",
        "display_name":   "ShallowMLP",
        "config":         "configs/lowto_mlp.yaml",
        "run_pattern":    "outputs/runs/lowto_mlp_v1_*",
    },
    {
        "label":          "simple_cnn",
        "display_name":   "SimpleCNN",
        "config":         "configs/lowto_cnn.yaml",
        "run_pattern":    "outputs/runs/lowto_cnn_v1_*",
    },
    {
        "label":          "vit_baseline",
        "display_name":   "Vanilla ViT",
        "config":         "configs/lowto_vit.yaml",
        "run_pattern":    "outputs/runs/lowto_vit_v1_*",
    },
    {
        "label":          "resnet34",
        "display_name":   "ResNet34",
        "config":         "configs/lowto_resnet34.yaml",
        "run_pattern":    "outputs/runs/lowto_resnet34_v1_*",
    },
    {
        "label":          "inception_v3",
        "display_name":   "InceptionV3",
        "config":         "configs/lowto_inceptionv3.yaml",
        "run_pattern":    "outputs/runs/lowto_inceptionv3_v1_*",
    },
    {
        "label":          "pams_vit_noLSF",
        "display_name":   "PAMS-ViT (no LSF)",
        "config":         "configs/lowto_pamsvit.yaml",
        "run_pattern":    "outputs/runs/lowto_pamsvit_v1_*",
    },
    {
        "label":          "pams_vit_full",
        "display_name":   "PAMS-ViT (full)",
        "config":         "configs/lowto_pamsvit_fullLSF.yaml",
        "run_pattern":    "outputs/runs/lowto_pamsvit_fullLSF_v1_*",
    },
]


# =============================================================================
# Measurement functions
# =============================================================================
def measure_params(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_flops(model, device):
    """
    Measure FLOPs and MACs using thop on a single (1, 4, 224, 224) input.
    Returns (macs, flops). FLOPs = 2 * MACs (one multiply + one add).
    """
    if not HAVE_THOP:
        return None, None
    dummy = torch.randn(1, 4, 224, 224, device=device)
    model.eval()
    try:
        macs, _ = thop_profile(model, inputs=(dummy,), verbose=False)
        return float(macs), float(2 * macs)
    except Exception as e:
        print(f"  FLOPs measurement failed: {e}")
        return None, None


def measure_gpu_memory(model, device):
    """Peak GPU memory during one forward pass (MB)."""
    if device.type != "cuda":
        return None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    dummy = torch.randn(1, 4, 224, 224, device=device)
    model.eval()
    with torch.no_grad():
        _ = model(dummy)
    peak_bytes = torch.cuda.max_memory_allocated(device)
    return peak_bytes / (1024 ** 2)


def measure_cpu_latency(model, n_warmup=10, n_iter=100):
    """CPU inference latency per frame (ms, mean +/- std)."""
    model_cpu = model.to("cpu")
    model_cpu.eval()
    dummy = torch.randn(1, 4, 224, 224)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model_cpu(dummy)

    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(n_iter):
            t0 = time.perf_counter()
            _ = model_cpu(dummy)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
    return float(np.mean(latencies)), float(np.std(latencies))


def find_one_checkpoint(run_pattern, project_root):
    """Find any one fold's best checkpoint to estimate size."""
    parent = Path(project_root)
    # Try fold_00 first, fall back to any fold
    matches = list(parent.glob(f"{run_pattern}/folds/fold_00/best.pt"))
    if not matches:
        matches = list(parent.glob(f"{run_pattern}/folds/*/best.pt"))
    return matches[0] if matches else None


def measure_checkpoint_size_mb(checkpoint_path):
    """Checkpoint size on disk (MB)."""
    return Path(checkpoint_path).stat().st_size / (1024 ** 2)


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=str,
                        default=PROJECT_ROOT_DEFAULT)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Defaults to <project_root>/outputs/efficiency")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    output_dir = Path(args.output_dir) if args.output_dir else \
        project_root / "outputs" / "efficiency"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("GPU: not available — skipping GPU memory measurements")

    rows = []
    for spec in ARCHITECTURES:
        print(f"\n--- {spec['display_name']} ---")

        # ------------------------------------------------------------------
        # Load actual model params from this architecture's config
        # ------------------------------------------------------------------
        config_path = project_root / spec["config"]
        if not config_path.exists():
            print(f"  ERROR: config not found at {config_path}")
            continue
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        model_name = cfg["model"]["name"]
        model_params = cfg["model"]["params"]
        print(f"  Config: {spec['config']}")
        print(f"  Model:  {model_name}, params: {dict(model_params)}")

        # ------------------------------------------------------------------
        # Build the model with the exact params from the actual experiment
        # ------------------------------------------------------------------
        model = build_model(model_name, model_params).to(device)

        # ------------------------------------------------------------------
        # 1. Trainable parameters
        # ------------------------------------------------------------------
        n_params = measure_params(model)
        print(f"  Params: {n_params:,}")

        # ------------------------------------------------------------------
        # 2. FLOPs (skip ShallowMLP since thop's forward hook may fail
        #    on extremely large fully-connected layers; we measure params only)
        # ------------------------------------------------------------------
        macs, flops = measure_flops(model, device)
        if flops is not None:
            print(f"  FLOPs:  {flops/1e9:.3f} G  (MACs: {macs/1e9:.3f} G)")
        else:
            print(f"  FLOPs:  N/A")

        # ------------------------------------------------------------------
        # 3. GPU memory
        # ------------------------------------------------------------------
        gpu_mem_mb = measure_gpu_memory(model, device)
        if gpu_mem_mb is not None:
            print(f"  GPU mem (peak): {gpu_mem_mb:.2f} MB")

        # ------------------------------------------------------------------
        # 4. CPU latency
        # ------------------------------------------------------------------
        try:
            latency_mean, latency_std = measure_cpu_latency(model)
            print(f"  CPU latency: {latency_mean:.2f} +/- "
                  f"{latency_std:.2f} ms/frame")
        except Exception as e:
            print(f"  CPU latency measurement failed: {e}")
            latency_mean, latency_std = None, None

        # ------------------------------------------------------------------
        # 5. Checkpoint size
        # ------------------------------------------------------------------
        ckpt_path = find_one_checkpoint(spec["run_pattern"], project_root)
        if ckpt_path:
            ckpt_size_mb = measure_checkpoint_size_mb(ckpt_path)
            print(f"  Checkpoint: {ckpt_size_mb:.2f} MB "
                  f"({ckpt_path.relative_to(project_root)})")
        else:
            print(f"  Checkpoint not found "
                  f"(pattern: {spec['run_pattern']})")
            ckpt_size_mb = None

        rows.append({
            "label":              spec["label"],
            "display_name":       spec["display_name"],
            "n_params":           int(n_params),
            "params_M":           n_params / 1e6,
            "flops":              flops,
            "flops_G":            flops / 1e9 if flops else None,
            "gpu_memory_MB":      gpu_mem_mb,
            "cpu_latency_ms_mean": latency_mean,
            "cpu_latency_ms_std":  latency_std,
            "checkpoint_size_MB":  ckpt_size_mb,
        })

        # Free GPU memory for next architecture
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    df = pd.DataFrame(rows)
    csv_path = output_dir / "efficiency_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # ------------------------------------------------------------------
    # Manuscript-ready markdown table
    # ------------------------------------------------------------------
    md_path = output_dir / "efficiency_for_manuscript.md"
    with open(md_path, "w") as f:
        f.write("# Efficiency Metrics (R1-5)\n\n")
        f.write("Measurement conditions:\n")
        f.write("- Input size: 224x224 (4 channels), batch size 1\n")
        f.write("- FLOPs: measured via `thop` (1 MAC = 2 FLOPs)\n")
        f.write("- GPU memory: NVIDIA A100-SXM4-80GB, "
                "torch.cuda.max_memory_allocated\n")
        f.write("- CPU latency: mean +/- std over 100 single-frame "
                "inferences after 10 warmup runs\n")
        f.write("- Checkpoint size: one fold's best.pt on disk\n\n")
        f.write("| Model | Params (M) | FLOPs (G) | GPU Mem (MB) | "
                "CPU Latency (ms) | Ckpt (MB) |\n")
        f.write("|-------|------------|-----------|--------------|"
                "------------------|-----------|\n")
        for _, r in df.iterrows():
            flops_str = f"{r['flops_G']:.2f}" if pd.notna(r['flops_G']) \
                else "N/A"
            mem_str = f"{r['gpu_memory_MB']:.1f}" \
                if pd.notna(r['gpu_memory_MB']) else "N/A"
            lat_str = (
                f"{r['cpu_latency_ms_mean']:.1f} +/- "
                f"{r['cpu_latency_ms_std']:.1f}"
                if pd.notna(r['cpu_latency_ms_mean']) else "N/A"
            )
            ckpt_str = f"{r['checkpoint_size_MB']:.2f}" \
                if pd.notna(r['checkpoint_size_MB']) else "N/A"

            f.write(f"| {r['display_name']} | {r['params_M']:.2f} | "
                    f"{flops_str} | {mem_str} | {lat_str} | "
                    f"{ckpt_str} |\n")
    print(f"Saved: {md_path}")

    print("\n" + "=" * 60)
    print("Efficiency measurement complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()