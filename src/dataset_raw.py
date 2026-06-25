# =============================================================================
# src/dataset.py — PyTorch Dataset for polarimetric SWH estimation
# =============================================================================
"""
Loads one frame at a time from disk: TIFF -> demosaic -> resize -> normalize.

Responsibility limited to per-sample I/O and preprocessing. It does NOT know
about splits, folds, or training (those live in splits.py and experiments/).

The 2x2 polarimetric supercell convention (must match the sensor):
    I0   = arr[0::2, 0::2]
    I45  = arr[0::2, 1::2]
    I90  = arr[1::2, 0::2]
    I135 = arr[1::2, 1::2]
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd


class PolarimetricWaveDataset(Dataset):
    """
    PyTorch Dataset yielding (image_tensor, hs_cm_target) pairs.

    Parameters
    ----------
    df : pd.DataFrame
        Subset of splits.csv with at least: image_path, hs_cm, wave_train,
        frame_index. Indices will be reset internally.
    dataset_dir : Path
        Root directory; image paths are relative to this.
    means, stds : np.ndarray of shape (4,) or None
        Per-channel z-score statistics computed on this fold's TRAIN set.
        If None, the dataset returns images in [0, 1] without normalization
        (used only when computing the stats themselves, in stats.py).
    image_size : int
        Square resize target (default 224).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        dataset_dir: Union[str, Path],
        means: np.ndarray | None = None,
        stds:  np.ndarray | None = None,
        image_size: int = 224,
    ):
        self.df = df.reset_index(drop=True)
        self.dataset_dir = Path(dataset_dir)
        self.image_size = int(image_size)

        # Validate means/stds together
        if (means is None) != (stds is None):
            raise ValueError("`means` and `stds` must both be provided or both None.")
        if means is not None:
            means = np.asarray(means, dtype=np.float32).reshape(4)
            stds  = np.asarray(stds,  dtype=np.float32).reshape(4)
            if np.any(stds <= 0):
                raise ValueError(f"All per-channel stds must be positive; got {stds}")
        self.means = means
        self.stds  = stds

        # Cache columns as plain Python/numpy structures for fast indexing
        self._image_paths = self.df["image_path"].tolist()
        self._hs_cm       = self.df["hs_cm"].astype(np.float32).to_numpy()
        self._wave_trains = self.df["wave_train"].tolist()
        self._frame_idx   = self.df["frame_index"].astype(np.int64).to_numpy()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # ---- 1. Load TIFF ----
        full_path = self.dataset_dir / self._image_paths[idx]
        with Image.open(full_path) as im:
            arr = np.asarray(im)

        # ---- 2. Validate ----
        if arr.ndim != 2:
            raise ValueError(f"Expected 2-D grayscale TIFF, got shape "
                             f"{arr.shape} for {full_path.name}")
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        H, W = arr.shape
        if H % 2 or W % 2:
            raise ValueError(f"Polarimetric demosaic requires even dims; "
                             f"got {(H, W)} for {full_path.name}")

        # ---- 3. Demosaic 2x2 supercell -> 4 channels (H/2, W/2) ----
        I0   = arr[0::2, 0::2]
        I45  = arr[0::2, 1::2]
        I90  = arr[1::2, 0::2]
        I135 = arr[1::2, 1::2]
        pol = np.stack([I0, I45, I90, I135], axis=0)   # (4, H/2, W/2) uint8

        # ---- 4. Cast to float32 in [0, 1] ----
        pol = pol.astype(np.float32) / 255.0

        # ---- 5. Resize to (image_size, image_size) bilinear ----
        pol_t = torch.from_numpy(pol).unsqueeze(0)     # (1, 4, H/2, W/2)
        pol_t = F.interpolate(
            pol_t, size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False,
        ).squeeze(0)                                    # (4, 224, 224)

        # ---- 6. Z-score normalize (if stats provided) ----
        if self.means is not None:
            mean_t = torch.from_numpy(self.means).view(4, 1, 1)
            std_t  = torch.from_numpy(self.stds ).view(4, 1, 1)
            pol_t = (pol_t - mean_t) / std_t

        target = torch.tensor(self._hs_cm[idx], dtype=torch.float32)
        return pol_t, target

    def get_metadata(self, idx: int) -> dict:
        """Non-tensor side info for downstream analysis."""
        return {
            "wave_train":  self._wave_trains[idx],
            "frame_index": int(self._frame_idx[idx]),
            "image_path":  self._image_paths[idx],
            "hs_cm":       float(self._hs_cm[idx]),
        }