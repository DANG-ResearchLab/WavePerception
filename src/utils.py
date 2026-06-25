# =============================================================================
# src/utils.py — Shared utility functions
# =============================================================================
"""
Stateless helpers used across the codebase: seeding, device handling, config
loading, run-directory creation, logging setup, and optional git/env capture
for reproducibility.

Import as:
    from src.utils import set_seed, get_device, load_config, create_run_dir, \
                          setup_logging, save_config_snapshot
"""

from __future__ import annotations

import os
import random
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Union

import numpy as np
import torch
import yaml


# =============================================================================
# Seeding
# =============================================================================
def set_seed(seed: int) -> None:
    """
    Set seeds for Python, NumPy, and PyTorch (CPU and CUDA) for reproducibility.

    Note: Even with all seeds set, exact CUDA determinism requires additional
    flags that hurt performance. For LOWTO we accept the residual CUDA
    nondeterminism since the headline result is fold-level mean ± std, which
    is not sensitive to per-batch ordering.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Modest determinism flags — fast paths still allowed.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Device
# =============================================================================
def get_device(config_value: str) -> torch.device:
    """
    Resolve config's device string ("auto" / "cuda" / "cpu") into a torch.device.

    "auto" picks cuda if available, else cpu.
    Explicit "cuda" raises if no GPU is found (so cluster jobs fail loud, not
    silently fall back to CPU and take 20x longer than expected).
    """
    val = (config_value or "auto").lower()
    if val == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if val == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "config.device='cuda' but no CUDA device is available. "
                "Either request GPU resources in SLURM, set device to 'auto', "
                "or set to 'cpu' explicitly."
            )
        return torch.device("cuda")
    if val == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device value: {config_value!r} (use auto/cuda/cpu)")


# =============================================================================
# Config loading
# =============================================================================
def load_config(yaml_path: Union[str, Path]) -> dict:
    """
    Load a YAML config file into a Python dict. PyYAML handles scientific
    notation (1e-4) as floats automatically, so no extra coercion is needed.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be a mapping, got {type(cfg)}")
    return cfg


def save_config_snapshot(config: dict, output_dir: Union[str, Path],
                         filename: str = "config_snapshot.yaml") -> Path:
    """
    Save the active config to the run folder. Critical for reproducibility:
    six months from now, you can look at this file and know exactly what
    hyperparameters produced these results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / filename
    with open(snapshot_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)
    return snapshot_path


# =============================================================================
# Run directory
# =============================================================================
def create_run_dir(
    base_dir: Union[str, Path],
    experiment_name: str,
    timestamp_format: str = "%Y%m%d_%H%M%S",
) -> Path:
    """
    Create a fresh, timestamped run directory:
        {base_dir}/{experiment_name}_{YYYYMMDD_HHMMSS}/

    Returns the Path. Folder always unique (timestamp to the second).
    """
    base_dir = Path(base_dir)
    timestamp = datetime.now().strftime(timestamp_format)
    run_dir = base_dir / f"{experiment_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# =============================================================================
# Logging
# =============================================================================
def setup_logging(
    log_file: Union[str, Path] = None,
    level: int = logging.INFO,
    log_format: str = "%(asctime)s | %(levelname)-7s | %(message)s",
    date_format: str = "%H:%M:%S",
) -> logging.Logger:
    """
    Configure root logger to write to console (always) and optionally to file.

    Use:
        logger = setup_logging(run_dir / "run_log.txt")
        logger.info("Starting fold 0 ...")

    On SLURM, console output goes to the .out file automatically. The
    additional file handler gives a clean log inside the run folder itself.
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    # Clear any pre-existing handlers (e.g., if Jupyter/notebook re-runs)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    formatter = logging.Formatter(log_format, datefmt=date_format)

    # Console handler — always on
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler — if log_file path provided
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# =============================================================================
# Environment / Git capture for reproducibility
# =============================================================================
def get_git_commit() -> str:
    """
    Return the current git commit hash if running inside a git repo,
    otherwise return 'no_git'. Used in run metadata for traceability.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "no_git"


def collect_environment_info() -> dict:
    """
    Snapshot of runtime environment for the run metadata file.
    Helps debug "why does this give different numbers six months later".
    """
    return {
        "python_version":   ".".join(map(str, __import__("sys").version_info[:3])),
        "torch_version":    torch.__version__,
        "numpy_version":    np.__version__,
        "cuda_available":   torch.cuda.is_available(),
        "cuda_device_name": (torch.cuda.get_device_name(0)
                             if torch.cuda.is_available() else None),
        "git_commit":       get_git_commit(),
    }