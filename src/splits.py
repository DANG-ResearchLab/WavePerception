# =============================================================================
# src/splits.py — LOWTO fold assignment & leakage-free partition logic
# =============================================================================
"""
This module owns ONE concern: deciding which frames go into train, validation,
and test for each LOWTO fold. It does NOT load any images (that's dataset.py).

Wave-train-grouped partitioning (resolves R1-2): every frame's partition is
determined by its 'wave_train' column, not by random row sampling. This
guarantees that no wave train appears in more than one partition within a
single fold.

Cyclic-next val rule: for fold k, val is the wave train of fold ((k+1) % N).
Deterministic, reproducible, and consistent with our leakage-free principle
(val is an UNSEEN wave train, just like test).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Union

import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Loading & validation
# =============================================================================
REQUIRED_COLUMNS = [
    "image_path", "hs_cm", "wave_train", "frame_index",
    "split_legacy", "split_wavetrain", "fold_lowto", "hs_normalized",
]


def load_splits(csv_path: Union[str, Path]) -> tuple[pd.DataFrame, str]:
    """
    Load and validate splits.csv.

    Returns
    -------
    (splits_df, sha256_hash) : tuple
        splits_df : DataFrame with all 8 required columns.
        sha256_hash : SHA-256 hash of the file (reproducibility anchor).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"splits.csv not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    # Schema validation — fail loud if the CSV is from a different generator
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"splits.csv missing required columns: {missing}. "
            f"Got columns: {list(df.columns)}"
        )

    # SHA-256 hash — reproducibility anchor. If this changes, the split changed.
    h = hashlib.sha256()
    with open(csv_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    sha = h.hexdigest()

    logger.info(f"Loaded splits.csv: {len(df)} rows, SHA-256 = {sha[:16]}...")
    return df, sha


# =============================================================================
# Fold assignment
# =============================================================================
def build_fold_assignment(
    splits_df: pd.DataFrame,
    fold_column: str = "fold_lowto",
    val_rule: str = "cyclic_next",
) -> dict[int, dict]:
    """
    Build per-fold (train, val, test) wave train assignments for LOWTO.

    For each fold k (0..N-1):
      test_wt   = the wave train where fold_lowto == k
      val_wt    = the wave train at fold ((k+1) % N)        [cyclic_next]
      train_wts = the remaining N-2 wave trains

    Parameters
    ----------
    splits_df : pd.DataFrame
        Must contain `fold_column` and 'wave_train'.
    fold_column : str
        Name of the fold-id column (default 'fold_lowto').
    val_rule : str
        Currently only 'cyclic_next' is supported.

    Returns
    -------
    dict[int, dict]
        Keys are fold indices 0..N-1. Each value is a dict:
            {'test_wt': str, 'val_wt': str, 'train_wts': list[str]}
    """
    if val_rule != "cyclic_next":
        raise NotImplementedError(f"val_rule={val_rule!r} not implemented")

    # Map: fold index -> wave train (one wave train per fold)
    fold_to_wt = (
        splits_df[[fold_column, "wave_train"]]
        .drop_duplicates()
        .sort_values(fold_column)
        .set_index(fold_column)["wave_train"]
        .to_dict()
    )
    n_folds = len(fold_to_wt)

    assignment: dict[int, dict] = {}
    for k in range(n_folds):
        test_wt = fold_to_wt[k]
        val_wt  = fold_to_wt[(k + 1) % n_folds]
        train_wts = [wt for fk, wt in fold_to_wt.items()
                     if wt != test_wt and wt != val_wt]

        assert len(train_wts) == n_folds - 2, \
            f"Fold {k}: expected {n_folds-2} train wave trains, got {len(train_wts)}"

        assignment[k] = {
            "test_wt":   test_wt,
            "val_wt":    val_wt,
            "train_wts": train_wts,
        }

    logger.info(f"Built LOWTO assignment: {n_folds} folds, "
                f"val_rule='{val_rule}'")
    return assignment


# =============================================================================
# Partitioning a single fold
# =============================================================================
def get_fold_dataframes(
    splits_df: pd.DataFrame, fold_assign: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Given one fold's wave-train assignment, slice splits_df into the three
    partition DataFrames.

    Parameters
    ----------
    splits_df : pd.DataFrame
        Full splits table.
    fold_assign : dict
        One entry from build_fold_assignment(), with keys
        {'test_wt', 'val_wt', 'train_wts'}.

    Returns
    -------
    (train_df, val_df, test_df) : tuple of DataFrames
        Each filtered by wave_train. Indices reset.
    """
    train_df = (
        splits_df[splits_df["wave_train"].isin(fold_assign["train_wts"])]
        .copy().reset_index(drop=True)
    )
    val_df = (
        splits_df[splits_df["wave_train"] == fold_assign["val_wt"]]
        .copy().reset_index(drop=True)
    )
    test_df = (
        splits_df[splits_df["wave_train"] == fold_assign["test_wt"]]
        .copy().reset_index(drop=True)
    )
    return train_df, val_df, test_df


# =============================================================================
# Safety: leakage check
# =============================================================================
def verify_leakage_free(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
) -> None:
    """
    Assert that no wave train appears in more than one partition. Fail loud
    if leakage is detected — this is the property R1-2 is built on.
    """
    train_wts = set(train_df["wave_train"])
    val_wts   = set(val_df["wave_train"])
    test_wts  = set(test_df["wave_train"])

    if train_wts & val_wts:
        raise AssertionError(f"LEAKAGE: train ∩ val = {train_wts & val_wts}")
    if train_wts & test_wts:
        raise AssertionError(f"LEAKAGE: train ∩ test = {train_wts & test_wts}")
    if val_wts & test_wts:
        raise AssertionError(f"LEAKAGE: val ∩ test = {val_wts & test_wts}")