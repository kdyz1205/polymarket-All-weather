"""
Factor Decay Monitor — watches for factors losing predictive power.

This implements the "因子衰减早期预警" capability:
- Tracks rolling IC for each active factor
- Detects systematic IC degradation vs random noise
- Auto-retires factors that have decayed beyond recovery
- Monitors cross-factor correlation for collinearity
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict

import numpy as np

from src.core.event_bus import Events, bus
from src.core.models import (
    AgentAction,
    AgentDecision,
    FactorDefinition,
    FactorStatus,
    FactorValue,
    Tick,
)

logger = logging.getLogger(__name__)


class FactorDecayMonitor:
    """
    Continuously monitors factor health and triggers warnings/retirements.

    Three detection mechanisms:
    1. IC trend: rolling IC consistently declining (z-score of IC slope)
    2. IC level: rolling IC drops below minimum threshold
    3. Collinearity: two factors become too correlated (redundancy)
    """

    def __init__(
        self,
        min_ic_threshold: float = 0.02,
        decay_z_threshold: float = -2.0,
        max_factor_correlation: float = 0.7,
        ic_window: int = 500,
    ) -> None:
        self.min_ic_threshold = min_ic_threshold
        self.decay_z_threshold = decay_z_threshold
        self.max_factor_correlation = max_factor_correlation
        self.ic_window = ic_window

        # factor_id -> list of (timestamp, factor_value, forward_return)
        self._factor_history: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
        # factor_id -> list of rolling IC values
        self._ic_history: dict[str, list[float]] = defaultdict(list)
        # market_id -> latest ticks for forward return computation
        self._tick_buffers: dict[str, list[Tick]] = defaultdict(list)

    async def start(self) -> None:
        bus.subscribe(Events.FACTOR_VALUE, self._on_factor_value)
        bus.subscribe(Events.TICK, self._on_tick)
        logger.info("FactorDecayMonitor started")

    async def stop(self) -> None:
        bus.unsubscribe(Events.FACTOR_VALUE, self._on_factor_value)
        bus.unsubscribe(Events.TICK, self._on_tick)

    async def _on_tick(self, tick: Tick) -> None:
        buf = self._tick_buffers[tick.market_id]
        buf.append(tick)
        if len(buf) > self.ic_window * 2:
            self._tick_buffers[tick.market_id] = buf[-self.ic_window * 2 :]

    async def _on_factor_value(self, fv: FactorValue) -> None:
        """Record factor value and compute IC when enough data accumulates."""
        # Store the factor value with its timestamp
        self._factor_history[fv.factor_id].append(
            (fv.timestamp_ms, fv.value, 0.0)  # forward return filled later
        )

        # Trim history
        history = self._factor_history[fv.factor_id]
        if len(history) > self.ic_window * 3:
            self._factor_history[fv.factor_id] = history[-self.ic_window * 3 :]

        # Compute rolling IC periodically (every 50 observations)
        if len(history) >= self.ic_window and len(history) % 50 == 0:
            await self._compute_and_check_ic(fv.factor_id, fv.market_id)

    async def _compute_and_check_ic(self, factor_id: str, market_id: str) -> None:
        """Compute rolling IC and check for decay."""
        from scipy import stats

        history = self._factor_history[factor_id]
        ticks = self._tick_buffers.get(market_id, [])

        if len(ticks) < self.ic_window:
            return

        # Build aligned factor values and forward returns
        tick_prices = {t.timestamp_ms: t.mid_price for t in ticks}
        factor_vals = []
        fwd_returns = []

        for ts, value, _ in history[-self.ic_window :]:
            # Find the closest next tick for forward return
            future_ts = ts + 1000  # 1 second forward
            closest_future = None
            for t in ticks:
                if t.timestamp_ms >= future_ts:
                    closest_future = t
                    break

            if closest_future is None:
                continue

            current_price = tick_prices.get(ts)
            if current_price and current_price > 0:
                fwd_ret = (closest_future.mid_price - current_price) / current_price
                factor_vals.append(value)
                fwd_returns.append(fwd_ret)

        if len(factor_vals) < 50:
            return

        # Compute IC (Spearman rank correlation)
        ic, _ = stats.spearmanr(factor_vals, fwd_returns)
        if np.isnan(ic):
            ic = 0.0

        self._ic_history[factor_id].append(ic)
        ic_series = self._ic_history[factor_id]

        # --- Check 1: IC level ---
        if abs(ic) < self.min_ic_threshold:
            logger.warning(
                "[%s] Factor %s IC dropped to %.4f (below threshold %.4f)",
                market_id, factor_id, ic, self.min_ic_threshold,
            )

        # --- Check 2: IC trend (decay detection) ---
        if len(ic_series) >= 10:
            recent_ics = np.array(ic_series[-10:])
            x = np.arange(len(recent_ics))
            slope = np.polyfit(x, recent_ics, 1)[0]
            slope_std = np.std(recent_ics) / np.sqrt(len(recent_ics))
            z_score = slope / slope_std if slope_std > 1e-10 else 0.0

            if z_score < self.decay_z_threshold:
                logger.warning(
                    "[%s] Factor %s showing systematic decay (IC trend z=%.2f)",
                    market_id, factor_id, z_score,
                )
                await bus.publish(
                    Events.AGENT_DECISION,
                    AgentDecision(
                        action=AgentAction.ADJUST_WEIGHT,
                        market_id=market_id,
                        timestamp_ms=int(time.time() * 1000),
                        reasoning=(
                            f"Factor '{factor_id}' showing decay: IC trend z-score={z_score:.2f}, "
                            f"recent IC={ic:.4f}. Recommending weight reduction."
                        ),
                        details={
                            "factor_id": factor_id,
                            "ic": ic,
                            "z_score": z_score,
                            "action": "reduce_weight",
                        },
                    ),
                )

                # If IC has been consistently terrible, retire the factor
                if len(ic_series) >= 20:
                    last_20 = ic_series[-20:]
                    if all(abs(x) < self.min_ic_threshold for x in last_20):
                        await self._retire_factor(factor_id, market_id, ic)

    async def _retire_factor(self, factor_id: str, market_id: str, final_ic: float) -> None:
        """Retire a factor that has completely lost predictive power."""
        factor_def = FactorDefinition(
            factor_id=factor_id,
            name=factor_id,
            description="Retired due to sustained decay",
            expression="",
            author="decay_monitor",
            status=FactorStatus.RETIRED,
        )
        await bus.publish(Events.FACTOR_RETIRED, factor_def)
        await bus.publish(
            Events.AGENT_DECISION,
            AgentDecision(
                action=AgentAction.RETIRE_FACTOR,
                market_id=market_id,
                timestamp_ms=int(time.time() * 1000),
                reasoning=(
                    f"Factor '{factor_id}' RETIRED: IC has been below threshold "
                    f"for 20 consecutive checks. Final IC={final_ic:.4f}."
                ),
                details={"factor_id": factor_id, "final_ic": final_ic},
            ),
        )
        logger.info("[%s] Factor %s retired (final IC=%.4f)", market_id, factor_id, final_ic)

    async def check_collinearity(self, market_id: str) -> list[tuple[str, str, float]]:
        """
        Check all pairs of active factors for high correlation.
        Returns list of (factor_a, factor_b, correlation) pairs that exceed threshold.
        """
        factor_ids = [
            fid for fid, history in self._factor_history.items()
            if len(history) >= 100
        ]

        if len(factor_ids) < 2:
            return []

        # Build value matrix
        min_len = min(len(self._factor_history[fid]) for fid in factor_ids)
        min_len = min(min_len, self.ic_window)

        matrix = np.array([
            [v for _, v, _ in self._factor_history[fid][-min_len:]]
            for fid in factor_ids
        ])

        # Compute correlation matrix
        corr_matrix = np.corrcoef(matrix)
        violations = []

        for i in range(len(factor_ids)):
            for j in range(i + 1, len(factor_ids)):
                corr = abs(corr_matrix[i, j])
                if corr > self.max_factor_correlation:
                    violations.append((factor_ids[i], factor_ids[j], float(corr)))
                    logger.warning(
                        "[%s] High collinearity: %s vs %s (corr=%.4f)",
                        market_id, factor_ids[i], factor_ids[j], corr,
                    )

        return violations
