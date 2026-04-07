"""
MarketProcessor: the base class that encapsulates one market's full pipeline.

Each market (Polymarket election, Polymarket sports, etc.) gets its own
MarketProcessor instance. This is the "举一反三" — instantiate once per market,
all share the same architecture.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from src.core.event_bus import Events, bus
from src.core.models import (
    CompositeSignal,
    FactorDefinition,
    FactorStatus,
    FactorValue,
    MarketRegime,
    MarketRisk,
    RegimeState,
    Tick,
)

logger = logging.getLogger(__name__)


@dataclass
class MarketConfig:
    market_id: str
    market_type: str
    name: str
    endpoint: str
    max_position_usd: float
    factor_ids: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


class MarketProcessor:
    """
    One instance per market. Manages:
    - Tick ingestion
    - Factor computation
    - Signal aggregation
    - Local risk state
    """

    def __init__(self, config: MarketConfig) -> None:
        self.config = config
        self.market_id = config.market_id

        # Factor state
        self._active_factors: dict[str, FactorDefinition] = {}
        self._factor_weights: dict[str, float] = {}
        self._latest_factor_values: dict[str, FactorValue] = {}

        # Market state
        self._tick_buffer: list[Tick] = []
        self._tick_buffer_limit = 2000
        self._regime = MarketRegime.NORMAL
        self._is_circuit_broken = False

        # Risk
        self._risk = MarketRisk(
            market_id=self.market_id,
            drawdown_pct=0.0,
            regime=MarketRegime.NORMAL,
            position_value=0.0,
            slippage_estimate_bps=0.0,
        )

    async def start(self) -> None:
        """Subscribe to relevant events and begin processing."""
        bus.subscribe(Events.TICK, self._on_tick)
        bus.subscribe(Events.FACTOR_CREATED, self._on_factor_created)
        bus.subscribe(Events.FACTOR_RETIRED, self._on_factor_retired)
        bus.subscribe(Events.REGIME_CHANGE, self._on_regime_change)
        bus.subscribe(Events.CIRCUIT_BREAK, self._on_circuit_break)
        bus.subscribe(Events.CIRCUIT_RESUME, self._on_circuit_resume)
        logger.info("MarketProcessor started for %s (%s)", self.config.name, self.market_id)

    async def stop(self) -> None:
        bus.unsubscribe(Events.TICK, self._on_tick)
        logger.info("MarketProcessor stopped for %s", self.market_id)

    # --------------------------------------------------
    # Event Handlers
    # --------------------------------------------------

    async def _on_tick(self, tick: Tick) -> None:
        if tick.market_id != self.market_id:
            return
        if self._is_circuit_broken:
            return

        # Buffer tick
        self._tick_buffer.append(tick)
        if len(self._tick_buffer) > self._tick_buffer_limit:
            self._tick_buffer = self._tick_buffer[-self._tick_buffer_limit :]

        # Compute all active factors
        factor_values = await self._compute_factors(tick)

        # Aggregate into composite signal
        signal = self._aggregate_signal(tick, factor_values)
        await bus.publish(Events.COMPOSITE_SIGNAL, signal)

    async def _on_factor_created(self, factor_def: FactorDefinition) -> None:
        """AI agent created a new factor — register it."""
        self._active_factors[factor_def.factor_id] = factor_def
        self._factor_weights[factor_def.factor_id] = factor_def.initial_weight
        logger.info(
            "[%s] New factor registered: %s (weight=%.4f)",
            self.market_id,
            factor_def.name,
            factor_def.initial_weight,
        )

    async def _on_factor_retired(self, factor_def: FactorDefinition) -> None:
        self._active_factors.pop(factor_def.factor_id, None)
        self._factor_weights.pop(factor_def.factor_id, None)
        self._latest_factor_values.pop(factor_def.factor_id, None)
        logger.info("[%s] Factor retired: %s", self.market_id, factor_def.name)

    async def _on_regime_change(self, state: RegimeState) -> None:
        if state.market_id != self.market_id:
            return
        old = self._regime
        self._regime = state.regime
        self._risk.regime = state.regime
        logger.info("[%s] Regime changed: %s -> %s", self.market_id, old, state.regime)

    async def _on_circuit_break(self, market_id: str) -> None:
        if market_id != self.market_id:
            return
        self._is_circuit_broken = True
        self._risk.is_circuit_broken = True
        logger.warning("[%s] CIRCUIT BREAKER ACTIVATED", self.market_id)

    async def _on_circuit_resume(self, market_id: str) -> None:
        if market_id != self.market_id:
            return
        self._is_circuit_broken = False
        self._risk.is_circuit_broken = False
        logger.info("[%s] Circuit breaker released", self.market_id)

    # --------------------------------------------------
    # Factor Computation
    # --------------------------------------------------

    async def _compute_factors(self, tick: Tick) -> list[FactorValue]:
        """Compute all active factors for the current tick."""
        from src.factors.engine import FactorEngine

        values = []
        for factor_id, factor_def in self._active_factors.items():
            if factor_def.status not in (FactorStatus.ACTIVE, FactorStatus.GREY):
                continue
            value = FactorEngine.compute(
                expression=factor_def.expression,
                tick=tick,
                tick_history=self._tick_buffer,
            )
            fv = FactorValue(
                factor_id=factor_id,
                market_id=self.market_id,
                timestamp_ms=tick.timestamp_ms,
                value=value,
                weight=self._factor_weights.get(factor_id, 0.0),
                ic_rolling=0.0,  # computed asynchronously by IC monitor
                status=factor_def.status,
            )
            values.append(fv)
            self._latest_factor_values[factor_id] = fv
            await bus.publish(Events.FACTOR_VALUE, fv)
        return values

    # --------------------------------------------------
    # Signal Aggregation
    # --------------------------------------------------

    def _aggregate_signal(self, tick: Tick, factor_values: list[FactorValue]) -> CompositeSignal:
        """Weighted sum of all factor values -> composite signal."""
        if not factor_values:
            return CompositeSignal(
                market_id=self.market_id,
                timestamp_ms=tick.timestamp_ms,
                raw_score=0.0,
                normalized_score=0.0,
                confidence=0.0,
                active_factor_count=0,
                factor_contributions={},
            )

        contributions = {}
        raw_score = 0.0
        total_weight = 0.0

        for fv in factor_values:
            contribution = fv.value * fv.weight
            contributions[fv.factor_id] = contribution
            raw_score += contribution
            total_weight += abs(fv.weight)

        # Normalize to [-1, 1]
        normalized = raw_score / total_weight if total_weight > 0 else 0.0
        normalized = max(-1.0, min(1.0, normalized))

        # Confidence = how much the factors agree (low dispersion = high confidence)
        if len(factor_values) > 1:
            signs = [1 if fv.value > 0 else -1 for fv in factor_values]
            agreement = abs(sum(signs)) / len(signs)
        else:
            agreement = 1.0

        return CompositeSignal(
            market_id=self.market_id,
            timestamp_ms=tick.timestamp_ms,
            raw_score=raw_score,
            normalized_score=normalized,
            confidence=agreement,
            active_factor_count=len(factor_values),
            factor_contributions=contributions,
        )

    @property
    def risk_state(self) -> MarketRisk:
        return self._risk
