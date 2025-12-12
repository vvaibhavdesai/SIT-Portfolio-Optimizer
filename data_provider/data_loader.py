"""
Data loaders for the clean SIT project.

Provides:
- `Dataset_Sig`: computes path signatures on-the-fly for rolling windows.
- `PrecomputedSigDataset`: loads precomputed signatures from disk.

Signature definitions (per asset, window length W):
    level-1: p[-1] - p[0]
    level-2: sum(p[:-1] * diff(p))                # Riemann approx of ∫p dp
Cross-signature (assets j, k):
    if j == k: 0.0
    else: sum(cumsum(diff(p_j))[:-1] * diff(p_k)[1:])

Shapes per sample (before batching):
    x_sigs: (H, D, 2)            # per-asset path signatures
    cross_sigs: (H, D, D, 1)     # pairwise cross-signatures
    future_return_unscaled: (H, D)
    date_feats: (H, 3)           # DayOfWeek, DayOfMonth, DayOfYear in [-0.5, 0.5]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


def _path_signature(prices: np.ndarray) -> np.ndarray:
    """Compute level-1 and level-2 signatures for a single asset window."""
    if prices.ndim != 1:
        raise ValueError("prices must be 1D for signature computation.")
    if len(prices) < 2:
        raise ValueError("prices window must contain at least 2 points.")
    diffs = np.diff(prices)
    level1 = prices[-1] - prices[0]
    level2 = np.sum(prices[:-1] * diffs)
    return np.array([level1, level2], dtype=np.float32)

def _compute_cross_signatures_vectorized(window: np.ndarray) -> np.ndarray:
    """
    Fully vectorized computation of all cross-signatures for a price window.
    
    Args:
        window: (W, D) array of prices for D assets over W timesteps
        
    Returns:
        cross_mat: (D, D, 1) array where cross_mat[j, k, 0] is cross-sig between assets j and k
    """
    D = window.shape[1]
    cross_mat = np.zeros((D, D, 1), dtype=np.float32)
    
    if len(window) < 3:
        return cross_mat
    
    # Compute all diffs and cumulatives at once
    diffs = np.diff(window, axis=0)  # (W-1, D)
    cumulatives = np.cumsum(diffs, axis=0)  # (W-1, D)
    
    if len(cumulatives) < 2 or len(diffs) < 2:
        return cross_mat
    
    # Fully vectorized: compute all cross-signatures at once
    # For pair (j,k): sum(cumulative_j[:-1] * diff_k[1:])
    # This is equivalent to: cumulative[:-1, :].T @ diff[1:, :]
    cross_result = cumulatives[:-1, :].T @ diffs[1:, :]  # (D, D)
    
    # Set diagonal to 0 (self-cross-signatures)
    np.fill_diagonal(cross_result, 0)
    
    # Add the extra dimension
    cross_mat[:, :, 0] = cross_result
    
    return cross_mat

def _compute_date_features(dates: Sequence[pd.Timestamp]) -> np.ndarray:
    """Compute scaled time features for a sequence of dates."""
    ts = pd.to_datetime(dates)
    dow = ts.dayofweek.to_numpy(dtype=np.float32)  # 0-6
    dom = ts.day.to_numpy(dtype=np.float32)        # 1-31
    doy = ts.dayofyear.to_numpy(dtype=np.float32)  # 1-365/366

    # Scale each feature to approximately [-0.5, 0.5]
    dow_scaled = (dow / 6.0) - 0.5
    dom_scaled = ((dom - 1) / 30.0) - 0.5
    doy_scaled = ((doy - 1) / 365.0) - 0.5

    feats = np.stack([dow_scaled, dom_scaled, doy_scaled], axis=1)
    return feats.astype(np.float32)


@dataclass
class Sample:
    """Container for a single dataset sample."""

    x_sigs: np.ndarray                 # (H, D, 2)
    cross_sigs: np.ndarray             # (H, D, D, 1)
    future_return_unscaled: np.ndarray # (H, D)
    date_feats: np.ndarray             # (H, 3)
    anchor_date: pd.Timestamp


class Dataset_Sig(Dataset):
    """Compute signatures on-the-fly from price data."""

    def __init__(
        self,
        prices_df: pd.DataFrame,
        window_size: int,
        horizon: int,
        tickers: Optional[List[str]] = None,
        scaler: Optional[StandardScaler] = None,
    ):
        """
        Args:
            prices_df: DataFrame with columns ['Date', <tickers...>], sorted by Date.
            window_size: lookback length (W).
            horizon: prediction horizon length (H).
            tickers: optional explicit ticker order; defaults to dataframe columns order (excluding Date).
            scaler: optional StandardScaler to apply to signatures.
        """
        if "Date" not in prices_df.columns:
            raise ValueError("prices_df must contain a 'Date' column.")
        self.window_size = window_size
        self.horizon = horizon

        df = prices_df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        all_tickers = [c for c in df.columns if c != "Date"]
        self.tickers = tickers or all_tickers
        missing = [t for t in self.tickers if t not in df.columns]
        if missing:
            raise ValueError(f"Missing tickers in price data: {missing}")

        # Reindex to the desired ticker order and drop rows with any missing prices
        df = df[["Date"] + self.tickers].dropna(how="any")
        self.dates = df["Date"].to_numpy()
        self.prices = df[self.tickers].to_numpy(dtype=np.float32)
        if len(self.prices) < window_size + horizon:
            raise ValueError("Not enough rows to build any samples with given window and horizon.")

        self.samples: List[Sample] = []
        self._build_samples()

        # Shapes for reconstruction after scaling
        if len(self.samples) > 0:
            self._x_shape = self.samples[0].x_sigs.shape
            self._cross_shape = self.samples[0].cross_sigs.shape
            self._x_flat_dim = int(np.prod(self._x_shape))
            self._cross_flat_dim = int(np.prod(self._cross_shape))
        else:
            raise ValueError("No samples created - check data and parameters")

        if scaler is not None:
            self.apply_scaler(scaler)

    def _build_samples(self):
        """Generate samples with rolling windows, keeping only first prediction per date."""
        n = len(self.prices)
        seen_dates = set()
        max_start = n - (self.window_size + self.horizon)
        
        print(f"Building dataset: {max_start + 1} potential samples...")
        
        for i in range(max_start + 1):
            target_dates = self.dates[i + self.window_size : i + self.window_size + self.horizon]
            anchor_date = target_dates[0]
            if anchor_date in seen_dates:
                continue
            seen_dates.add(anchor_date)

            x_steps = []
            cross_steps = []
            future_steps = []

            for s in range(self.horizon):
                window = self.prices[i + s : i + self.window_size + s]
                
                # Path signatures per asset (vectorized)
                asset_sigs = np.stack([_path_signature(window[:, j]) for j in range(window.shape[1])], axis=0)
                x_steps.append(asset_sigs)  # (D, 2)

                # Cross signatures (vectorized)
                cross_mat = _compute_cross_signatures_vectorized(window)
                cross_steps.append(cross_mat)  # (D, D, 1)
                
                # Future returns
                prev_price = window[-1]
                next_price = self.prices[i + self.window_size + s]
                future_steps.append((next_price - prev_price) / prev_price)

            x_sigs = np.stack(x_steps, axis=0).astype(np.float32)                # (H, D, 2)
            cross_sigs = np.stack(cross_steps, axis=0).astype(np.float32)        # (H, D, D, 1)
            future_ret = np.stack(future_steps, axis=0).astype(np.float32)       # (H, D)
            date_feats = _compute_date_features(target_dates)                    # (H, 3)

            self.samples.append(
                Sample(
                    x_sigs=x_sigs,
                    cross_sigs=cross_sigs,
                    future_return_unscaled=future_ret,
                    date_feats=date_feats,
                    anchor_date=anchor_date,
                )
            )
        
        print(f"Created {len(self.samples)} samples")

    def signature_matrix(self) -> np.ndarray:
        """Return flattened signature vectors for scaler fitting/transform."""
        vectors = []
        for s in self.samples:
            vec = np.concatenate([s.x_sigs.reshape(-1), s.cross_sigs.reshape(-1)], axis=0)
            vectors.append(vec)
        return np.stack(vectors, axis=0)

    def apply_scaler(self, scaler: StandardScaler):
        """Apply a fitted scaler to x_sigs and cross_sigs in-place."""
        matrix = self.signature_matrix()
        scaled = scaler.transform(matrix)
        for idx, vec in enumerate(scaled):
            x_flat = vec[: self._x_flat_dim]
            cross_flat = vec[self._x_flat_dim :]
            self.samples[idx].x_sigs = x_flat.reshape(self._x_shape).astype(np.float32)
            self.samples[idx].cross_sigs = cross_flat.reshape(self._cross_shape).astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        sample = self.samples[idx]
        return {
            "x_sigs": sample.x_sigs,
            "cross_sigs": sample.cross_sigs,
            "future_return_unscaled": sample.future_return_unscaled,
            "date_feats": sample.date_feats,
            "date": sample.anchor_date,
        }


class PrecomputedSigDataset(Dataset):
    """Dataset wrapper for precomputed signatures stored on disk (npz)."""

    def __init__(self, npz_path: str):
        """
        Expect npz with keys:
            - x_sigs: (N, H, D, 2)
            - cross_sigs: (N, H, D, D, 1)
            - future_return_unscaled: (N, H, D)
            - date_feats: (N, H, 3)
            - dates (optional): (N,) anchor dates
        """
        data = np.load(npz_path, allow_pickle=True)
        required = ["x_sigs", "cross_sigs", "future_return_unscaled", "date_feats"]
        for key in required:
            if key not in data:
                raise ValueError(f"Missing required key '{key}' in {npz_path}")

        self.x_sigs = data["x_sigs"]
        self.cross_sigs = data["cross_sigs"]
        self.future_return_unscaled = data["future_return_unscaled"]
        self.date_feats = data["date_feats"]
        self.dates = data["dates"] if "dates" in data else None

        n = len(self.x_sigs)
        if not (len(self.cross_sigs) == len(self.future_return_unscaled) == len(self.date_feats) == n):
            raise ValueError("All arrays in precomputed dataset must have the same length.")

    def __len__(self) -> int:
        return len(self.x_sigs)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        return {
            "x_sigs": self.x_sigs[idx],
            "cross_sigs": self.cross_sigs[idx],
            "future_return_unscaled": self.future_return_unscaled[idx],
            "date_feats": self.date_feats[idx],
            "date": self.dates[idx] if self.dates is not None else None,
        }