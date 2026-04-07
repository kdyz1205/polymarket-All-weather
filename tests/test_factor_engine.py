"""
Tests for the factor engine — validates operator primitives,
expression parsing, and sandbox backtesting.
"""

from __future__ import annotations

import time

from src.core.models import Tick
from src.factors.engine import FactorEngine, FactorSandbox
from src.factors.operators.primitives import (
    clip,
    log_return,
    order_book_imbalance,
    sign,
    spread,
    ts_corr,
    ts_decay_linear,
    ts_delta,
    ts_mean,
    ts_rank,
    ts_std,
    ts_zscore,
    volume_ratio,
)


# ============================================================
# Operator Tests
# ============================================================


def test_ts_mean():
    series = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert ts_mean(series, 3) == 4.0  # mean of [3, 4, 5]
    assert ts_mean(series, 5) == 3.0
    assert ts_mean([], 5) == 0.0


def test_ts_std():
    series = [1.0, 1.0, 1.0]
    assert ts_std(series, 3) == 0.0
    series = [1.0, 2.0, 3.0]
    assert ts_std(series, 3) > 0


def test_ts_rank():
    series = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert ts_rank(series, 5) == 1.0  # 5 is the max -> rank 1.0
    series = [5.0, 4.0, 3.0, 2.0, 1.0]
    assert ts_rank(series, 5) == 0.2  # 1 is the min -> rank 0.2


def test_ts_delta():
    series = [10.0, 12.0, 15.0]
    assert ts_delta(series, 1) == 3.0  # 15 - 12
    assert ts_delta(series, 2) == 5.0  # 15 - 10


def test_ts_zscore():
    series = [10.0, 10.0, 10.0, 10.0, 20.0]
    z = ts_zscore(series, 5)
    assert z > 0  # 20 is above mean


def test_ts_decay_linear():
    series = [1.0, 2.0, 3.0]
    result = ts_decay_linear(series, 3)
    # weights: [1, 2, 3], values: [1, 2, 3]
    # = (1*1 + 2*2 + 3*3) / (1+2+3) = 14/6 ≈ 2.333
    assert abs(result - 14 / 6) < 0.001


def test_sign():
    assert sign(5.0) == 1.0
    assert sign(-3.0) == -1.0
    assert sign(0.0) == 0.0


def test_clip():
    assert clip(1.5) == 1.0
    assert clip(-1.5) == -1.0
    assert clip(0.5) == 0.5


def test_log_return():
    series = [100.0, 110.0]
    lr = log_return(series, 1)
    assert abs(lr - 0.09531) < 0.001  # ln(110/100)


def test_order_book_imbalance():
    assert order_book_imbalance([100.0], [100.0]) == 0.0
    assert order_book_imbalance([200.0], [100.0]) > 0
    assert order_book_imbalance([100.0], [200.0]) < 0


def test_volume_ratio():
    volumes = [100.0] * 50 + [200.0] * 5
    vr = volume_ratio(volumes, 5, 50)
    assert vr > 1.0  # short-term volume is higher


def test_spread_operator():
    spreads = [0.01] * 49 + [0.02]
    s = spread(spreads, 50)
    assert s > 0  # current spread above average


# ============================================================
# Expression Validation Tests
# ============================================================


def test_validate_allowed_expression():
    assert FactorEngine.validate_expression("clip(ts_zscore(book_imbalance, 50))")
    assert FactorEngine.validate_expression("ts_mean(mid_price, 20)")
    assert FactorEngine.validate_expression("sign(ts_delta(spread, 5))")


def test_validate_rejects_disallowed():
    assert not FactorEngine.validate_expression("__import__('os').system('rm -rf /')")
    assert not FactorEngine.validate_expression("open('/etc/passwd').read()")


# ============================================================
# Factor Computation Tests
# ============================================================


def _make_ticks(n: int = 100) -> list[Tick]:
    """Generate synthetic ticks for testing."""
    import numpy as np

    rng = np.random.default_rng(42)
    ticks = []
    price = 0.5
    for i in range(n):
        price += rng.normal(0, 0.002)
        price = max(0.01, min(0.99, price))
        bid_depth = rng.uniform(100, 5000)
        ask_depth = rng.uniform(100, 5000)
        ticks.append(
            Tick(
                market_id="test_market",
                timestamp_ms=int(time.time() * 1000) + i * 1000,
                mid_price=price,
                last_trade_price=price,
                volume_24h=rng.uniform(10000, 100000),
                spread=rng.uniform(0.005, 0.02),
                bid_depth=bid_depth,
                ask_depth=ask_depth,
                book_imbalance=(bid_depth - ask_depth) / (bid_depth + ask_depth),
            )
        )
    return ticks


def test_compute_factor():
    ticks = _make_ticks(200)
    value = FactorEngine.compute(
        expression="clip(ts_zscore(book_imbalance, 50))",
        tick=ticks[-1],
        tick_history=ticks,
    )
    assert -1.0 <= value <= 1.0


def test_compute_complex_expression():
    ticks = _make_ticks(200)
    value = FactorEngine.compute(
        expression="sign(ts_delta(ts_mean(mid_price, 10), 5))",
        tick=ticks[-1],
        tick_history=ticks,
    )
    assert value in (-1.0, 0.0, 1.0)


# ============================================================
# Sandbox Tests
# ============================================================


def test_sandbox_backtest():
    ticks = _make_ticks(500)
    sandbox = FactorSandbox(ticks)
    result = sandbox.backtest("clip(ts_zscore(book_imbalance, 50))")
    assert result.is_valid
    assert result.n_observations > 0
    assert -1.0 <= result.ic <= 1.0
    print(f"Sandbox result: {result.summary()}")


def test_sandbox_rejects_invalid():
    ticks = _make_ticks(500)
    sandbox = FactorSandbox(ticks)
    result = sandbox.backtest("invalid_function(x)")
    assert not result.is_valid
