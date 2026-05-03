"""
Load Databento MBP-10 parquet data into per-day event arrays.

Data is stored as columnar numpy arrays (not row objects) for speed.
One DayData = one regular-trading-hours session = one RL episode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from market_maker import config


# ---------------------------------------------------------------------------
# Per-day columnar storage
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class DayData:
    """All RTH events for one trading day, stored as numpy arrays."""
    date:    str
    times:   np.ndarray   # (N,) seconds since midnight UTC
    actions: np.ndarray   # (N,) str: 'A','C','T','F'
    sides:   np.ndarray   # (N,) str: 'B','A','N'
    prices:  np.ndarray   # (N,) float64 dollars
    sizes:   np.ndarray   # (N,) int64 shares
    bid_px:  np.ndarray   # (N, 5) float64
    bid_sz:  np.ndarray   # (N, 5) float64
    ask_px:  np.ndarray   # (N, 5) float64
    ask_sz:  np.ndarray   # (N, 5) float64

    def __len__(self) -> int:
        return len(self.times)

    # Convenience: mid-price and spread at index i
    def mid(self, i: int) -> float:
        b, a = self.bid_px[i, 0], self.ask_px[i, 0]
        if b > 0 and a > 0:
            return (b + a) / 2.0
        return b if b > 0 else a

    def spread(self, i: int) -> float:
        b, a = self.bid_px[i, 0], self.ask_px[i, 0]
        return max(0.0, a - b)

    def best_bid(self, i: int) -> float:
        return float(self.bid_px[i, 0])

    def best_ask(self, i: int) -> float:
        return float(self.ask_px[i, 0])


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_BID_PX = [f"bid_px_{i:02d}" for i in range(5)]
_BID_SZ = [f"bid_sz_{i:02d}" for i in range(5)]
_ASK_PX = [f"ask_px_{i:02d}" for i in range(5)]
_ASK_SZ = [f"ask_sz_{i:02d}" for i in range(5)]


def load_parquet(path: str) -> List[DayData]:
    """Read parquet, filter to RTH, return sorted list of DayData."""
    df = pd.read_parquet(path)

    ts = df["ts_event"].dt.tz_convert("UTC")
    total_minutes = ts.dt.hour * 60 + ts.dt.minute
    rth_start = config.RTH_START_HOUR * 60 + config.RTH_START_MINUTE
    rth_end   = config.RTH_END_HOUR   * 60 + config.RTH_END_MINUTE
    mask = (total_minutes >= rth_start) & (total_minutes < rth_end)
    df = df[mask].copy()
    ts = ts[df.index]

    # Fill NaN book values with 0
    for col in _BID_PX + _BID_SZ + _ASK_PX + _ASK_SZ:
        df[col] = df[col].fillna(0.0)

    df["_date"] = ts.dt.date
    df["_sec"]  = (ts.dt.hour * 3600.0
                   + ts.dt.minute * 60.0
                   + ts.dt.second
                   + ts.dt.microsecond * 1e-6)

    days: List[DayData] = []
    for date, grp in df.groupby("_date"):
        n = len(grp)
        days.append(DayData(
            date    = str(date),
            times   = grp["_sec"].to_numpy(dtype=np.float64),
            actions = grp["action"].to_numpy(),
            sides   = grp["side"].to_numpy(),
            prices  = grp["price"].to_numpy(dtype=np.float64),
            sizes   = grp["size"].to_numpy(dtype=np.int64),
            bid_px  = grp[_BID_PX].to_numpy(dtype=np.float64).reshape(n, 5),
            bid_sz  = grp[_BID_SZ].to_numpy(dtype=np.float64).reshape(n, 5),
            ask_px  = grp[_ASK_PX].to_numpy(dtype=np.float64).reshape(n, 5),
            ask_sz  = grp[_ASK_SZ].to_numpy(dtype=np.float64).reshape(n, 5),
        ))

    return sorted(days, key=lambda d: d.date)


def train_val_test_split(
    days: List[DayData],
) -> Tuple[List[DayData], List[DayData], List[DayData]]:
    """Chronological split: ~100 train / 28 val / 40 test."""
    n_test = config.TEST_SAMPLE_DAYS
    n_val  = config.VAL_SAMPLE_DAYS
    test  = days[-(n_test):]
    val   = days[-(n_test + n_val):-n_test]
    train = days[:-(n_test + n_val)]
    return train, val, test
