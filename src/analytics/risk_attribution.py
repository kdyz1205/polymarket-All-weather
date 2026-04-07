"""
Sub-module D: Risk attribution analyzer.

Answers: "When we lost money, WHY did we lose money?"

Decomposes total PnL into additive components:
  PnL = Signal Edge - Slippage - Delay Damage - Bad Fills - Hedging Cost - Fees

Each loss is tagged with its root cause so you know exactly
where to focus engineering effort.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src.analytics.session_metrics import FillRecord


@dataclass
class PnLComponent:
    """A single component of the PnL decomposition."""

    name: str
    value: float
    pct_of_gross: float = 0.0
    description: str = ""


@dataclass
class LossAttribution:
    """Individual loss event with root cause tag."""

    timestamp_ms: int
    amount: float
    root_cause: str  # "model_error", "execution_slip", "delay_damage", "gap_risk", "hedge_fail", "position_size"
    detail: str = ""
    runner_id: str = ""
    order_id: int = 0


@dataclass
class RiskAttributionReport:
    """Complete risk attribution report."""

    # PnL decomposition
    signal_edge: float = 0.0
    slippage_cost: float = 0.0
    delay_damage: float = 0.0
    bad_fills: float = 0.0
    hedging_cost: float = 0.0
    fees: float = 0.0
    net_pnl: float = 0.0

    # As percentages of theoretical edge
    slippage_pct: float = 0.0
    delay_pct: float = 0.0
    bad_fills_pct: float = 0.0
    hedging_pct: float = 0.0
    fees_pct: float = 0.0
    edge_retained_pct: float = 0.0

    # Loss breakdown by root cause
    losses_by_cause: dict[str, float] = field(default_factory=dict)
    loss_count_by_cause: dict[str, int] = field(default_factory=dict)

    # Worst losses
    worst_losses: list[LossAttribution] = field(default_factory=list)

    # Model accuracy
    model_correct_direction: int = 0
    model_wrong_direction: int = 0
    model_accuracy: float = 0.0

    # Position sizing
    avg_position_size: float = 0.0
    max_position_size: float = 0.0
    position_sizes_at_loss: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pnl_decomposition": {
                "signal_edge": round(self.signal_edge, 4),
                "slippage_cost": round(-self.slippage_cost, 4),
                "delay_damage": round(-self.delay_damage, 4),
                "bad_fills": round(-self.bad_fills, 4),
                "hedging_cost": round(-self.hedging_cost, 4),
                "fees": round(-self.fees, 4),
                "net_pnl": round(self.net_pnl, 4),
            },
            "edge_erosion_pct": {
                "slippage": round(self.slippage_pct, 2),
                "delay": round(self.delay_pct, 2),
                "bad_fills": round(self.bad_fills_pct, 2),
                "hedging": round(self.hedging_pct, 2),
                "fees": round(self.fees_pct, 2),
                "retained": round(self.edge_retained_pct, 2),
            },
            "losses_by_root_cause": {
                cause: {
                    "total_loss": round(amount, 4),
                    "count": self.loss_count_by_cause.get(cause, 0),
                }
                for cause, amount in self.losses_by_cause.items()
            },
            "model_accuracy": {
                "correct_direction": self.model_correct_direction,
                "wrong_direction": self.model_wrong_direction,
                "accuracy": round(self.model_accuracy, 4),
            },
            "position_sizing": {
                "avg_size": round(self.avg_position_size, 2),
                "max_size": round(self.max_position_size, 2),
            },
            "worst_losses": [
                {
                    "amount": round(l.amount, 4),
                    "cause": l.root_cause,
                    "detail": l.detail,
                }
                for l in self.worst_losses[:10]
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class RiskAttributionAnalyzer:
    """
    Decomposes PnL into root causes.

    For every fill, computes:
      theoretical_edge = fair_price - market_price (at signal time)
      execution_slip = market_price - fill_price
      delay_damage = market_move_during_delay
      Then nets them to get actual captured value.
    """

    def __init__(self, commission_rate: float = 0.02) -> None:
        self.commission_rate = commission_rate
        self._fills: list[FillRecord] = []
        self._losses: list[LossAttribution] = []

        # Running totals
        self._total_signal_edge = 0.0
        self._total_slippage = 0.0
        self._total_delay_damage = 0.0
        self._total_bad_fills = 0.0
        self._total_hedging_cost = 0.0
        self._total_fees = 0.0
        self._position_sizes: list[float] = []

        # Model tracking
        self._model_correct = 0
        self._model_wrong = 0

    def record_fill(
        self,
        fill: FillRecord,
        market_price_at_signal: float = 0.0,
        market_price_at_fill: float = 0.0,
        market_price_5s_later: float = 0.0,
        delay_ms: int = 0,
        was_hedge: bool = False,
    ) -> None:
        """Record a fill with full context for attribution."""
        self._fills.append(fill)
        self._position_sizes.append(fill.size)

        # 1. Signal edge: theoretical value at the time we decided to trade
        if fill.fair_price_at_fill > 0 and market_price_at_signal > 0:
            if fill.side == "back":
                edge_prob = (1.0 / fill.fair_price_at_fill) - (1.0 / market_price_at_signal)
            else:
                edge_prob = (1.0 / market_price_at_signal) - (1.0 / fill.fair_price_at_fill)

            signal_edge = edge_prob * fill.size * market_price_at_signal
            self._total_signal_edge += signal_edge

            # Track model direction accuracy
            if edge_prob > 0:
                self._model_correct += 1
            else:
                self._model_wrong += 1

        # 2. Slippage: difference between expected fill and actual fill
        if market_price_at_signal > 0 and fill.price > 0:
            if fill.side == "back":
                slip = (market_price_at_signal - fill.price) / market_price_at_signal
            else:
                slip = (fill.price - market_price_at_signal) / market_price_at_signal

            slip_cost = abs(slip) * fill.size if slip > 0 else 0
            self._total_slippage += slip_cost

        # 3. Delay damage
        if delay_ms > 0 and market_price_at_signal > 0 and market_price_at_fill > 0:
            if fill.side == "back":
                dd = (market_price_at_signal - market_price_at_fill) / market_price_at_signal
            else:
                dd = (market_price_at_fill - market_price_at_signal) / market_price_at_signal

            if dd > 0:  # price moved against us
                dd_cost = dd * fill.size
                self._total_delay_damage += dd_cost
                self._losses.append(LossAttribution(
                    timestamp_ms=fill.timestamp_ms,
                    amount=dd_cost,
                    root_cause="delay_damage",
                    detail=f"Price moved {dd*10000:.1f}bps during {delay_ms}ms delay",
                    runner_id=fill.runner_id,
                ))

        # 4. Bad fills (adverse selection: market moved against after fill)
        if market_price_5s_later > 0 and fill.price > 0:
            if fill.side == "back":
                adverse = (fill.price - market_price_5s_later) / fill.price
            else:
                adverse = (market_price_5s_later - fill.price) / fill.price

            if adverse > 0.001:  # >1bps adverse
                bad_fill_cost = adverse * fill.size
                self._total_bad_fills += bad_fill_cost
                self._losses.append(LossAttribution(
                    timestamp_ms=fill.timestamp_ms,
                    amount=bad_fill_cost,
                    root_cause="execution_slip",
                    detail=f"Adverse selection: {adverse*10000:.1f}bps move after fill",
                    runner_id=fill.runner_id,
                ))

        # 5. Hedging cost
        if was_hedge:
            # Hedge trades typically execute at a cost (crossing the spread)
            hedge_cost = fill.size * 0.005  # estimate: half the typical spread
            self._total_hedging_cost += hedge_cost

        # 6. Fees
        fee = fill.size * self.commission_rate
        self._total_fees += fee

    def record_gap_loss(
        self,
        timestamp_ms: int,
        runner_id: str,
        amount: float,
        detail: str = "",
    ) -> None:
        """Record a loss from suspend/reopen gap."""
        self._losses.append(LossAttribution(
            timestamp_ms=timestamp_ms,
            amount=amount,
            root_cause="gap_risk",
            detail=detail or "Price gap on market reopen",
            runner_id=runner_id,
        ))

    def record_model_error_loss(
        self,
        timestamp_ms: int,
        runner_id: str,
        amount: float,
        detail: str = "",
    ) -> None:
        """Record a loss attributed to model being wrong."""
        self._losses.append(LossAttribution(
            timestamp_ms=timestamp_ms,
            amount=amount,
            root_cause="model_error",
            detail=detail or "Model probability was incorrect",
            runner_id=runner_id,
        ))

    def record_position_size_loss(
        self,
        timestamp_ms: int,
        runner_id: str,
        amount: float,
        detail: str = "",
    ) -> None:
        """Record a loss attributed to position being too large."""
        self._losses.append(LossAttribution(
            timestamp_ms=timestamp_ms,
            amount=amount,
            root_cause="position_size",
            detail=detail or "Position size too large for available liquidity",
            runner_id=runner_id,
        ))

    def compute(self) -> RiskAttributionReport:
        report = RiskAttributionReport()

        report.signal_edge = self._total_signal_edge
        report.slippage_cost = self._total_slippage
        report.delay_damage = self._total_delay_damage
        report.bad_fills = self._total_bad_fills
        report.hedging_cost = self._total_hedging_cost
        report.fees = self._total_fees
        report.net_pnl = (
            self._total_signal_edge
            - self._total_slippage
            - self._total_delay_damage
            - self._total_bad_fills
            - self._total_hedging_cost
            - self._total_fees
        )

        # Percentages of signal edge
        abs_edge = abs(self._total_signal_edge) if self._total_signal_edge != 0 else 1.0
        report.slippage_pct = self._total_slippage / abs_edge * 100
        report.delay_pct = self._total_delay_damage / abs_edge * 100
        report.bad_fills_pct = self._total_bad_fills / abs_edge * 100
        report.hedging_pct = self._total_hedging_cost / abs_edge * 100
        report.fees_pct = self._total_fees / abs_edge * 100
        report.edge_retained_pct = report.net_pnl / abs_edge * 100 if self._total_signal_edge != 0 else 0

        # Loss breakdown by cause
        for loss in self._losses:
            report.losses_by_cause[loss.root_cause] = (
                report.losses_by_cause.get(loss.root_cause, 0) + loss.amount
            )
            report.loss_count_by_cause[loss.root_cause] = (
                report.loss_count_by_cause.get(loss.root_cause, 0) + 1
            )

        # Worst losses
        report.worst_losses = sorted(self._losses, key=lambda l: -l.amount)[:10]

        # Model accuracy
        total_dir = self._model_correct + self._model_wrong
        report.model_correct_direction = self._model_correct
        report.model_wrong_direction = self._model_wrong
        report.model_accuracy = self._model_correct / total_dir if total_dir > 0 else 0

        # Position sizing
        if self._position_sizes:
            report.avg_position_size = sum(self._position_sizes) / len(self._position_sizes)
            report.max_position_size = max(self._position_sizes)

        return report
