# =============================================================================
# src/stats.py — Per-fold channel statistics (Welford streaming)
# =============================================================================
"""
Compute leakage-free per-channel mean and std over a training partition.

Called once per LOWTO fold by the experiment runner: each fold's stats are
computed only from THAT fold's train wave trains, then applied identically
to train, val, and test datasets (val and test never contribute to stats).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from src.dataset import PolarimetricWaveDataset

logger = logging.getLogger(__name__)


def compute_channel_stats(
    train_df: pd.DataFrame,
    dataset_dir: Union[str, Path],
    image_size: int = 224,
    progress_every: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Streaming per-channel mean/std over a fold's training frames.

    Uses the parallel/chunked Welford algorithm so memory stays O(1) regardless
    of the number of frames (important on 8 GB machines, harmless on the
    cluster). Pixel statistics are computed on [0, 1]-scaled images — the
    same domain in which downstream normalization will apply them.

    Parameters
    ----------
    train_df : pd.DataFrame
        This fold's training partition (~1000 frames over 10 wave trains).
    dataset_dir : Path
        Root directory for image paths.
    image_size : int
        Square resize target (must match dataset's image_size).
    progress_every : int
        Log progress every N frames.

    Returns
    -------
    (means, stds) : tuple of np.ndarray, shape (4,), dtype float32
    """
    # Build dataset WITHOUT normalization -> images in [0, 1]
    ds = PolarimetricWaveDataset(
        df=train_df, dataset_dir=dataset_dir,
        means=None, stds=None, image_size=image_size,
    )
    n_frames = len(ds)
    logger.info(f"Computing channel stats over {n_frames} training frames...")

    # Per-channel Welford accumulators (float64 for numerical stability)
    n_total = 0
    mean    = np.zeros(4, dtype=np.float64)
    M2      = np.zeros(4, dtype=np.float64)

    t_start = time.time()
    for i in range(n_frames):
        img, _ = ds[i]                                  # (4, 224, 224), float32
        img_np = img.numpy().astype(np.float64)

        n_new      = img_np.shape[1] * img_np.shape[2]  # 224*224
        frame_mean = img_np.mean(axis=(1, 2))           # (4,)
        frame_M2   = ((img_np - frame_mean[:, None, None]) ** 2).sum(axis=(1, 2))

        if n_total == 0:
            n_total = n_new
            mean    = frame_mean.copy()
            M2      = frame_M2.copy()
        else:
            n_combined = n_total + n_new
            delta      = frame_mean - mean
            mean       = mean + delta * (n_new / n_combined)
            M2         = M2 + frame_M2 + (delta ** 2) * (n_total * n_new / n_combined)
            n_total    = n_combined

        if (i + 1) % progress_every == 0 or (i + 1) == n_frames:
            rate = (i + 1) / max(time.time() - t_start, 1e-9)
            logger.info(f"  stats: {i+1:4d}/{n_frames} frames ({rate:.1f} f/s)")

    variance  = M2 / (n_total - 1)
    stds_f32  = np.sqrt(variance).astype(np.float32)
    means_f32 = mean.astype(np.float32)

    if np.any(stds_f32 <= 1e-4):
        raise ValueError(f"Near-zero std detected: {stds_f32}. "
                         f"Check for constant-valued channels.")

    elapsed = time.time() - t_start
    logger.info(f"  stats done in {elapsed:.1f}s. "
                f"means={np.round(means_f32, 4)}, stds={np.round(stds_f32, 4)}")
    return means_f32, stds_f32