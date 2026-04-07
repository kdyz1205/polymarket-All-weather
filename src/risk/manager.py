"""
Risk Management & Circuit Breaker System.

This module implements the "生存本能" — the AI-supervised risk layer
that can override all trading decisions when market conditions become
dangerous.

Two levels of risk management:
1. Statistical: volatility z-scores, drawdown limits (fast, automatic)
2. AI-supervised: regime detection, cross-market contagion analysis (slower, smarter)
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
    MarketRegime,
    MarketRisk,
    RegimeState,
    RiskSnapshot,
    Tick,
)

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Monitors all markets and enforces risk limits.
    Acts as the system's survival mechanism.
    """

    def __init__(
        self,
        max_drawdown_pct: float = 5.0,
        per_market_max_drawdown_pct: float = 3.0,
        volatility_circuit_breaker_z: float = 3.0,
        regime_window: int = 200,
    ) -> None:
        self.max_drawdown_pct = max_drawdown_pct
        self.per_market_max_drawdown_pct = per_market_max_drawdown_pct
        self.volatility_circuit_breaker_z = volatility_circuit_breaker_z
        self.regime_window = regime_window

        # Per-market state
        self._tick_buffers: dict[str, list[Tick]] = defaultdict(list)
        self._peak_equity: dict[str, float] = {}
        self._current_equity: dict[str, float] = {}
        self._circuit_broken: set[str] = set()
        self._market_risks: dict[str, MarketRisk] = {}

        # Global state
        self._total_equity = 100000.0  # starting equity
        self._peak_total_equity = 100000.0

    async def start(self) -> None:
        bus.subscribe(Events.TICK, self._on_tick)
        logger.info("RiskManager started")

    async def stop(self) -> None:
        bus.unsubscribe(Events.TICK, self._on_tick)

    async def _on_tick(self, tick: Tick) -> None:
        market_id = tick.market_id

        # Buffer ticks for regime detection
        buf = self._tick_buffers[market_id]
        buf.append(tick)
        if len(buf) > self.regime_window * 2:
            self._tick_buffers[market_id] = buf[-self.regime_window * 2 :]

        # --- Statistical Risk Checks ---

        # 1. Volatility circuit breaker
        regime_state = self._detect_regime(market_id)
        if regime_state:
            await bus.publish(Events.REGIME_CHANGE, regime_state)

            if (
                regime_state.volatility_z > self.volatility_circuit_breaker_z
                and market_id not in self._circuit_broken
            ):
                await self._trigger_circuit_break(
                    market_id,
                    f"Volatility z-score {regime_state.volatility_z:.2f} exceeds "
                    f"threshold {self.volatility_circuit_breaker_z}",
                )

            # Auto-resume if volatility normalizes
            if (
                market_id in self._circuit_broken
                and regime_state.volatility_z < self.volatility_circuit_breaker_z * 0.5
            ):
                await self._release_circuit_break(
                    market_id,
                    f"Volatility normalized to z={regime_state.volatility_z:.2f}",
                )

        # 2. Drawdown check
        drawdown = self._compute_drawdown(market_id)
        if drawdown > self.per_market_max_drawdown_pct and market_id not in self._circuit_broken:
            await self._trigger_circuit_break(
                market_id,
                f"Drawdown {drawdown:.2f}% exceeds limit {self.per_market_max_drawdown_pct}%",
            )

        # 3. Liquidity check (spread explosion)
        if len(buf) >= 50:
            recent_spreads = [t.spread for t in buf[-50:]]
            mean_spread = np.mean(recent_spreads[:-1])
            if mean_spread > 0 and tick.spread > mean_spread * 5:
                if market_id not in self._circuit_broken:
                    await self._trigger_circuit_break(
                        market_id,
                        f"Spread exploded to {tick.spread:.6f} "
                        f"(5x historical mean {mean_spread:.6f})",
                    )

        # Update risk snapshot
        self._update_risk(market_id, tick, drawdown, regime_state)

    def _detect_regime(self, market_id: str) -> RegimeState | None:
        """Detect current market regime using statistical methods."""
        buf = self._tick_buffers[market_id]
        if len(buf) < self.regime_window:
            return None

        window = buf[-self.regime_window :]
        prices = np.array([t.mid_price for t in window])
        spreads = np.array([t.spread for t in window])
        volumes = np.array([t.volume_24h for t in window])

        # Compute returns
        returns = np.diff(np.log(np.maximum(prices, 1e-10)))
        if len(returns) < 10:
            return None

        # Volatility z-score (compare recent vol to full window vol)
        short_vol = np.std(returns[-20:])
        long_vol = np.std(returns)
        vol_z = (short_vol - long_vol) / long_vol if long_vol > 1e-10 else 0.0

        # Liquidity z-score
        mean_spread = np.mean(spreads)
        std_spread = np.std(spreads)
        liq_z = (spreads[-1] - mean_spread) / std_spread if std_spread > 1e-10 else 0.0

        # Classify regime
        if vol_z > 2.0 and liq_z > 2.0:
            regime = MarketRegime.CRISIS
        elif vol_z > 2.0:
            regime = MarketRegime.HIGH_VOLATILITY
        elif liq_z > 2.0:
            regime = MarketRegime.LOW_LIQUIDITY
        else:
            # Check for trending vs mean-reverting
            # Simple Hurst exponent approximation
            half = len(returns) // 2
            var_full = np.var(returns)
            var_half = (np.var(returns[:half]) + np.var(returns[half:])) / 2
            hurst_proxy = np.log2(var_full / var_half) / 2 if var_half > 1e-20 else 0.5

            if hurst_proxy > 0.6:
                regime = MarketRegime.TRENDING
            elif hurst_proxy < 0.4:
                regime = MarketRegime.MEAN_REVERTING
            else:
                regime = MarketRegime.NORMAL

        return RegimeState(
            market_id=market_id,
            timestamp_ms=int(time.time() * 1000),
            regime=regime,
            confidence=min(1.0, len(buf) / self.regime_window),
            volatility_z=float(vol_z),
            liquidity_z=float(liq_z),
        )

    def _compute_drawdown(self, market_id: str) -> float:
        """Compute current drawdown percentage for a market."""
        current = self._current_equity.get(market_id, 0)
        peak = self._peak_equity.get(market_id, current)
        if peak <= 0:
            return 0.0
        return (peak - current) / peak * 100

    async def _trigger_circuit_break(self, market_id: str, reason: str) -> None:
        self._circuit_broken.add(market_id)
        await bus.publish(Events.CIRCUIT_BREAK, market_id)
        await bus.publish(
            Events.AGENT_DECISION,
            AgentDecision(
                action=AgentAction.CIRCUIT_BREAK,
                market_id=market_id,
                timestamp_ms=int(time.time() * 1000),
                reasoning=f"CIRCUIT BREAKER ACTIVATED: {reason}",
                details={"market_id": market_id},
            ),
        )
        logger.warning("[RISK] Circuit break on %s: %s", market_id, reason)

    async def _release_circuit_break(self, market_id: str, reason: str) -> None:
        self._circuit_broken.discard(market_id)
        await bus.publish(Events.CIRCUIT_RESUME, market_id)
        await bus.publish(
            Events.AGENT_DECISION,
            AgentDecision(
                action=AgentAction.RESUME_MARKET,
                market_id=market_id,
                timestamp_ms=int(time.time() * 1000),
                reasoning=f"Circuit breaker released: {reason}",
                details={"market_id": market_id},
            ),
        )
        logger.info("[RISK] Circuit break released on %s: %s", market_id, reason)

    def _update_risk(
        self,
        market_id: str,
        tick: Tick,
        drawdown: float,
        regime_state: RegimeState | None,
    ) -> None:
        regime = regime_state.regime if regime_state else MarketRegime.NORMAL
        self._market_risks[market_id] = MarketRisk(
            market_id=market_id,
            drawdown_pct=drawdown,
            regime=regime,
            position_value=0.0,
            slippage_estimate_bps=tick.spread * 10000 if tick.mid_price > 0 else 0,
            is_circuit_broken=market_id in self._circuit_broken,
        )

    def get_risk_snapshot(self) -> RiskSnapshot:
        total_drawdown = 0.0
        if self._peak_total_equity > 0:
            total_drawdown = (
                (self._peak_total_equity - self._total_equity)
                / self._peak_total_equity
                * 100
            )
        return RiskSnapshot(
            timestamp_ms=int(time.time() * 1000),
            total_equity=self._total_equity,
            total_pnl=self._total_equity - 100000.0,
            max_drawdown_pct=total_drawdown,
            market_risks=dict(self._market_risks),
            circuit_breakers_active=list(self._circuit_broken),
        )
