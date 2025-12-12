"""
Time feature encodings for SIT.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def time_features(dates: Iterable, freq: str = "B") -> np.ndarray:
    """
    Extract scaled time features for business-day frequency.

    Args:
        dates: iterable of datetime-like values.
        freq: frequency string (default 'B' for business days).

    Returns:
        np.ndarray of shape (len(dates), 3) with features:
            [DayOfWeek, DayOfMonth, DayOfYear], each scaled to [-0.5, 0.5].
    """
    ts = pd.to_datetime(list(dates))
    if freq != "B":
        # For extensibility; currently only business days are handled.
        pass

    dow = ts.dayofweek.to_numpy(dtype=np.float32)  # 0-6
    dom = ts.day.to_numpy(dtype=np.float32)        # 1-31
    doy = ts.dayofyear.to_numpy(dtype=np.float32)  # 1-365/366

    dow_scaled = (dow / 6.0) - 0.5
    dom_scaled = ((dom - 1) / 30.0) - 0.5
    doy_scaled = ((doy - 1) / 365.0) - 0.5

    feats = np.stack([dow_scaled, dom_scaled, doy_scaled], axis=1)
    return feats.astype(np.float32)
