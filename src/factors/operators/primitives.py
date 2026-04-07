"""
Factor Operator Primitives — the building blocks AI uses to compose factors.

These are the ONLY operations AI is allowed to use. Each operator:
1. Takes standardized inputs (series of floats or scalars)
2. Returns a single float normalized to [-1, 1] where applicable
3. Is stateless — all state comes from the tick history passed in

This constraint prevents AI from generating unsafe or buggy factor code.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


# ============================================================
# Time Series Operators (ts_*)
# ============================================================


def ts_mean(series: Sequence[float], window: int) -> float:
    """Rolling mean over the last `window` values."""
    data = series[-window:]
    if not data:
        return 0.0
    return float(np.mean(data))


def ts_std(series: Sequence[float], window: int) -> float:
    """Rolling standard deviation over the last `window` values."""
    data = series[-window:]
    if len(data) < 2:
        return 0.0
    return float(np.std(data, ddof=1))


def ts_rank(series: Sequence[float], window: int) -> float:
    """Percentile rank of the latest value within the last `window` values.
    Returns value in [0, 1].
    """
    data = list(series[-window:])
    if len(data) < 2:
        return 0.5
    current = data[-1]
    rank = sum(1 for v in data if v <= current) / len(data)
    return rank


def ts_corr(series_a: Sequence[float], series_b: Sequence[float], window: int) -> float:
    """Rolling Pearson correlation between two series over `window`.
    Returns value in [-1, 1].
    """
    a = np.array(series_a[-window:], dtype=np.float64)
    b = np.array(series_b[-window:], dtype=np.float64)
    min_len = min(len(a), len(b))
    if min_len < 3:
        return 0.0
    a, b = a[-min_len:], b[-min_len:]
    std_a, std_b = np.std(a), np.std(b)
    if std_a < 1e-10 or std_b < 1e-10:
        return 0.0
    corr = float(np.corrcoef(a, b)[0, 1])
    if math.isnan(corr):
        return 0.0
    return max(-1.0, min(1.0, corr))


def ts_delta(series: Sequence[float], window: int) -> float:
    """Difference between current value and value `window` steps ago."""
    if len(series) <= window:
        return 0.0
    return series[-1] - series[-1 - window]


def ts_decay_linear(series: Sequence[float], window: int) -> float:
    """Linearly weighted moving average. Recent values get higher weight."""
    data = list(series[-window:])
    n = len(data)
    if n == 0:
        return 0.0
    weights = np.arange(1, n + 1, dtype=np.float64)
    return float(np.dot(data, weights) / weights.sum())


def ts_max(series: Sequence[float], window: int) -> float:
    data = series[-window:]
    return float(max(data)) if data else 0.0


def ts_min(series: Sequence[float], window: int) -> float:
    data = series[-window:]
    return float(min(data)) if data else 0.0


def ts_zscore(series: Sequence[float], window: int) -> float:
    """Z-score of the latest value relative to rolling window."""
    data = list(series[-window:])
    if len(data) < 2:
        return 0.0
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    if std < 1e-10:
        return 0.0
    return float((data[-1] - mean) / std)


# ============================================================
# Cross-Sectional Operators
# ============================================================


def rank(value: float, all_values: Sequence[float]) -> float:
    """Rank of a single value among all values. Returns [0, 1]."""
    if not all_values:
        return 0.5
    n = len(all_values)
    r = sum(1 for v in all_values if v <= value) / n
    return r


def sign(value: float) -> float:
    """Sign function: -1, 0, or 1."""
    if value > 0:
        return 1.0
    elif value < 0:
        return -1.0
    return 0.0


def clip(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    """Clip value to [lo, hi]."""
    return max(lo, min(hi, value))


# ============================================================
# Market Microstructure Operators
# ============================================================


def log_return(series: Sequence[float], window: int = 1) -> float:
    """Log return over `window` periods."""
    if len(series) <= window:
        return 0.0
    current = series[-1]
    past = series[-1 - window]
    if past <= 0 or current <= 0:
        return 0.0
    return math.log(current / past)


def order_book_imbalance(bid_depths: Sequence[float], ask_depths: Sequence[float]) -> float:
    """Order book imbalance from depth series. Returns [-1, 1]."""
    if not bid_depths or not ask_depths:
        return 0.0
    bid = bid_depths[-1]
    ask = ask_depths[-1]
    total = bid + ask
    if total < 1e-10:
        return 0.0
    return (bid - ask) / total


def spread(spreads: Sequence[float], window: int) -> float:
    """Normalized spread relative to rolling mean spread."""
    data = list(spreads[-window:])
    if len(data) < 2:
        return 0.0
    mean_spread = np.mean(data)
    if mean_spread < 1e-10:
        return 0.0
    return float((data[-1] - mean_spread) / mean_spread)


def volume_ratio(volumes: Sequence[float], short_window: int, long_window: int) -> float:
    """Ratio of short-term volume to long-term volume. > 1 means surge."""
    if len(volumes) < long_window:
        return 1.0
    short_mean = np.mean(volumes[-short_window:])
    long_mean = np.mean(volumes[-long_window:])
    if long_mean < 1e-10:
        return 1.0
    return float(short_mean / long_mean)


# ============================================================
# Operator Registry — AI can only use these
# ============================================================

OPERATOR_REGISTRY: dict[str, callable] = {
    "ts_mean": ts_mean,
    "ts_std": ts_std,
    "ts_rank": ts_rank,
    "ts_corr": ts_corr,
    "ts_delta": ts_delta,
    "ts_decay_linear": ts_decay_linear,
    "ts_max": ts_max,
    "ts_min": ts_min,
    "ts_zscore": ts_zscore,
    "rank": rank,
    "sign": sign,
    "clip": clip,
    "log_return": log_return,
    "order_book_imbalance": order_book_imbalance,
    "spread": spread,
    "volume_ratio": volume_ratio,
}
