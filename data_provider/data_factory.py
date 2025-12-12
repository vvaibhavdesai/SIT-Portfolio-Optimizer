"""
Data preparation utilities for SIT.

Responsibilities:
- Load raw price data from CSV.
- Build rolling-window datasets with on-the-fly signature computation.
- Apply StandardScaler to signatures using only the training split.
- Provide chronological train/val/test splits.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .data_loader import Dataset_Sig

TRAIN_END = pd.Timestamp("2016-12-31")
VAL_START = pd.Timestamp("2017-01-01")
VAL_END = pd.Timestamp("2019-12-31")
TEST_REF_DATE = pd.Timestamp("2020-01-01")
TEST_END = pd.Timestamp("2024-12-31")


def load_prices_csv(path: str, tickers: Optional[List[str]] = None) -> pd.DataFrame:
    """Load price data CSV with a 'Date' column and optional ticker filtering."""
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError("CSV must contain a 'Date' column.")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    if tickers is not None:
        missing = [t for t in tickers if t not in df.columns]
        if missing:
            raise ValueError(f"Missing requested tickers in CSV: {missing}")
        df = df[["Date"] + tickers]
    return df


def _slice(df: pd.DataFrame, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= df["Date"] >= start
    if end is not None:
        mask &= df["Date"] <= end
    return df.loc[mask].copy()


def _compute_test_start(window_size: int) -> pd.Timestamp:
    """Start test window window_size+1 business days before TEST_REF_DATE."""
    bdays = pd.bdate_range(end=TEST_REF_DATE, periods=window_size + 1)
    return bdays[0]


def _fit_scaler(train_ds: Dataset_Sig) -> StandardScaler:
    """Fit StandardScaler on flattened signatures from the training split."""
    scaler = StandardScaler()
    sig_matrix = train_ds.signature_matrix()
    scaler.fit(sig_matrix)
    return scaler


def prepare_datasets(
    prices_df: pd.DataFrame,
    window_size: int,
    horizon: int,
    tickers: Optional[List[str]] = None,
) -> Tuple[Dataset_Sig, Dataset_Sig, Dataset_Sig, StandardScaler]:
    """Create train/val/test Dataset_Sig instances and a fitted scaler."""
    df = prices_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # Chronological splits
    train_df = _slice(df, start=None, end=TRAIN_END)
    val_df = _slice(df, start=VAL_START, end=VAL_END)
    test_start = _compute_test_start(window_size)
    test_df = _slice(df, start=test_start, end=TEST_END)

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("One or more splits are empty; check date ranges and data coverage.")

    train_ds = Dataset_Sig(train_df, window_size=window_size, horizon=horizon, tickers=tickers, scaler=None)
    scaler = _fit_scaler(train_ds)
    train_ds.apply_scaler(scaler)

    val_ds = Dataset_Sig(val_df, window_size=window_size, horizon=horizon, tickers=tickers, scaler=scaler)
    test_ds = Dataset_Sig(test_df, window_size=window_size, horizon=horizon, tickers=tickers, scaler=scaler)

    return train_ds, val_ds, test_ds, scaler
