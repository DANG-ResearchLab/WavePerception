# =============================================================================
# src/dataset.py — PyTorch Dataset for polarimetric SWH estimation
# Experiment 2: Manual Stokes input [I, DoLP, sin(AoP), cos(AoP)]
# =============================================================================
"""
Loads one frame at a time from disk: TIFF -> demosaic -> derive Stokes
features -> resize -> normalize.

Input representation (Exp 2):
    Channel 0 : I        = (I0 + I45 + I90 + I135) / 2  — total intensity
    Channel 1 : DoLP     = sqrt(S1² + S2²) / (S0 + ε)   — degree of linear pol.
    Channel 2 : sin(AoP) = sin(0.5 * arctan2(S2, S1))   — AoP sine component
    Channel 3 : cos(AoP) = cos(0.5 * arctan2(S2, S1))   — AoP cosine component

where S0 = I0+I90, S1 = I0-I90, S2 = I45-I135.

Physically correct pipeline order:
    demosaic (full res) -> derive Stokes features (full res) -> resize each
    derived channel individually -> normalize to [0,1] -> z-score

Resizing derived channels (DoLP, AoP) individually at full resolution before
any normalization avoids interpolation artifacts that would arise from resizing
raw channels and recomputing nonlinear quantities at reduced resolution.

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
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd


# Small constant for numerical stability in DoLP denominator
_EPS = 1e-8


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
        If None, the dataset returns images in [0, 1] without z-score
        normalization (used only when computing stats in stats.py).
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

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resize_channel_pil(ch: np.ndarray, size: int) -> np.ndarray:
        """
        Resize a single 2-D float32 array to (size, size) using PIL bilinear.
        PIL requires uint8 or float mode; we use mode 'F' (32-bit float) to
        avoid precision loss during interpolation of derived quantities such
        as DoLP (range [0,1]) and AoP components (range [-1,1]).
        """
        pil_img = Image.fromarray(ch, mode='F')
        pil_img = pil_img.resize((size, size), resample=Image.BILINEAR)
        return np.array(pil_img, dtype=np.float32)

    @staticmethod
    def _normalize_to_01(arr: np.ndarray) -> np.ndarray:
        """Min-max normalize a float32 array to [0, 1]."""
        lo, hi = arr.min(), arr.max()
        if hi - lo < _EPS:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)

    def _derive_stokes_features(
        self,
        I0: np.ndarray,
        I45: np.ndarray,
        I90: np.ndarray,
        I135: np.ndarray,
        size: int,
    ) -> np.ndarray:
        """
        Derive [I, DoLP, sin(AoP), cos(AoP)] from full-resolution raw channels,
        resize each derived map to (size, size), then normalize to [0, 1].

        Returns
        -------
        np.ndarray of shape (4, size, size), dtype float32, values in [0, 1].
        """
        # --- Stokes parameters (full resolution, float32) ---
        S0 = I0 + I90                           # total horizontal+vertical
        S1 = I0 - I90                           # horizontal-vertical contrast
        S2 = I45 - I135                         # diagonal contrast

        # --- Derived polarimetric maps (full resolution) ---
        # Total intensity: standard formula for 4-channel polarimetric camera
        I_map   = (I0 + I45 + I90 + I135) / 2.0
        DoLP    = np.sqrt(S1**2 + S2**2) / (S0 + _EPS)
        AoP_rad = 0.5 * np.arctan2(S2, S1)     # range [-π/2, π/2]
        sin_AoP = np.sin(AoP_rad)               # range [-1, 1]
        cos_AoP = np.cos(AoP_rad)               # range [-1, 1]

        # --- Resize each derived channel individually (PIL bilinear, mode F) ---
        I_map   = self._resize_channel_pil(I_map,   size)
        DoLP    = self._resize_channel_pil(DoLP,    size)
        sin_AoP = self._resize_channel_pil(sin_AoP, size)
        cos_AoP = self._resize_channel_pil(cos_AoP, size)

        # --- Per-channel normalization to [0, 1] ---
        # I and DoLP: min-max (both are naturally non-negative)
        I_map = self._normalize_to_01(I_map)
        DoLP  = self._normalize_to_01(DoLP)
        # AoP components: shift from [-1, 1] to [0, 1] by (x + 1) / 2
        # This preserves the linear spacing and avoids min-max collapse
        # on frames where angular variation is small
        sin_AoP = (sin_AoP + 1.0) / 2.0
        cos_AoP = (cos_AoP + 1.0) / 2.0

        # --- Stack -> (4, H, W) ---
        stacked = np.stack([I_map, DoLP, sin_AoP, cos_AoP], axis=0)
        return stacked.astype(np.float32)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # ---- 1. Load TIFF ----
        full_path = self.dataset_dir / self._image_paths[idx]
        with Image.open(full_path) as im:
            arr = np.asarray(im)

        # ---- 2. Validate raw array ----
        if arr.ndim != 2:
            raise ValueError(
                f"Expected 2-D grayscale TIFF, got shape {arr.shape} "
                f"for {full_path.name}"
            )
        H, W = arr.shape
        if H % 2 or W % 2:
            raise ValueError(
                f"Polarimetric demosaic requires even spatial dims; "
                f"got {(H, W)} for {full_path.name}"
            )
        arr = arr.astype(np.float32)    # keep as float for Stokes arithmetic

        # ---- 3. Demosaic 2x2 supercell -> 4 raw channels (full resolution) ----
        I0   = arr[0::2, 0::2]
        I45  = arr[0::2, 1::2]
        I90  = arr[1::2, 0::2]
        I135 = arr[1::2, 1::2]

        # ---- 4. Derive manual Stokes features, resize, normalize to [0,1] ----
        # Output: float32 ndarray of shape (4, image_size, image_size)
        stokes = self._derive_stokes_features(
            I0, I45, I90, I135, self.image_size
        )

        # ---- 5. Convert to tensor ----
        pol_t = torch.from_numpy(stokes)        # (4, 224, 224), float32

        # ---- 6. Z-score normalize per channel (if fold stats are provided) ----
        if self.means is not None:
            mean_t = torch.from_numpy(self.means).view(4, 1, 1)
            std_t  = torch.from_numpy(self.stds ).view(4, 1, 1)
            pol_t  = (pol_t - mean_t) / std_t

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