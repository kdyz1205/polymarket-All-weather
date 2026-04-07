"""
Sub-module A: Session-level metrics aggregator.

After every replay session, this module answers:
  "Did this strategy make money? How much? How consistently?"

Inputs:
  - All fills from the session
  - All fair values at each decision point
  - Market outcome (who won)
  - Session metadata (sport, market, duration)

Outputs:
  SessionSummary with:
  - ROI, turnover, gross/net PnL, max drawdown
  - Win rate, average hold time, Sharpe proxy
  - Per-runner breakdown
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FillRecord:
    """A single fill with all context needed for analysis."""

    fill_id: int
    order_id: int
    runner_id: str
    side: str  # "back" or "lay"
    price: float
    size: float
    timestamp_ms: int
    # Context at time of fill
    fair_price_at_fill: float  # model's fair odds when this fill happened
    market_price_at_fill: float  # best available market price
    elapsed_game_sec: float  # game clock at fill time
    strategy_tag: str = ""


@dataclass
class MarketOutcome:
    """How the market settled."""

    winning_runner_id: str  # "" if void
    settled: bool = True


@dataclass
class RunnerPnL:
    """PnL breakdown for a single runner."""

    runner_id: str
    back_stake: float = 0.0
    back_avg_odds: float = 0.0
    lay_stake: float = 0.0
    lay_avg_odds: float = 0.0
    pnl_if_wins: float = 0.0
    pnl_if_loses: float = 0.0
    actual_pnl: float = 0.0
    fill_count: int = 0
    turnover: float = 0.0


@dataclass
class EquityCurvePoint:
    timestamp_ms: int
    unrealized_pnl: float
    realized_pnl: float
    total_equity: float


@dataclass
class SessionSummary:
    """Complete session-level metrics."""

    # Identity
    session_id: str = ""
    sport: str = ""
    market_id: str = ""
    duration_sec: float = 0.0

    # Core PnL
    gross_pnl: float = 0.0
    net_pnl: float = 0.0  # after fees
    roi: float = 0.0  # net_pnl / turnover
    turnover: float = 0.0
    total_stake: float = 0.0

    # Drawdown
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    peak_equity: float = 0.0

    # Trade stats
    total_orders: int = 0
    total_fills: int = 0
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0  # gross_wins / gross_losses

    # Time stats
    avg_hold_time_sec: float = 0.0

    # Per-runner
    runner_pnls: dict[str, RunnerPnL] = field(default_factory=dict)

    # Equity curve
    equity_curve: list[EquityCurvePoint] = field(default_factory=list)

    # Sharpe proxy (annualized if enough data)
    sharpe_proxy: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = {
            "session_id": self.session_id,
            "sport": self.sport,
            "market_id": self.market_id,
            "duration_sec": self.duration_sec,
            "gross_pnl": round(self.gross_pnl, 4),
            "net_pnl": round(self.net_pnl, 4),
            "roi": round(self.roi, 6),
            "turnover": round(self.turnover, 2),
            "total_stake": round(self.total_stake, 2),
            "max_drawdown": round(self.max_drawdown, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "total_orders": self.total_orders,
            "total_fills": self.total_fills,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "win_rate": round(self.win_rate, 4),
            "avg_win": round(self.avg_win, 4),
            "avg_loss": round(self.avg_loss, 4),
            "profit_factor": round(self.profit_factor, 4),
            "sharpe_proxy": round(self.sharpe_proxy, 4),
            "runner_pnls": {
                rid: {
                    "back_stake": round(rp.back_stake, 2),
                    "lay_stake": round(rp.lay_stake, 2),
                    "actual_pnl": round(rp.actual_pnl, 4),
                    "fill_count": rp.fill_count,
                    "turnover": round(rp.turnover, 2),
                }
                for rid, rp in self.runner_pnls.items()
            },
        }
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class SessionAggregator:
    """
    Aggregates all fills and outcomes into a SessionSummary.

    Usage:
        agg = SessionAggregator("session_1", "basketball", "nba_ml_1")
        for fill in fills:
            agg.record_fill(fill)
        agg.set_outcome(MarketOutcome("home"))
        summary = agg.compute()
    """

    def __init__(
        self,
        session_id: str,
        sport: str,
        market_id: str,
        commission_rate: float = 0.02,
    ) -> None:
        self.session_id = session_id
        self.sport = sport
        self.market_id = market_id
        self.commission_rate = commission_rate

        self._fills: list[FillRecord] = []
        self._outcome: MarketOutcome | None = None
        self._runner_positions: dict[str, RunnerPnL] = {}
        self._equity_points: list[EquityCurvePoint] = []

    def record_fill(self, fill: FillRecord) -> None:
        self._fills.append(fill)

        # Update runner position
        if fill.runner_id not in self._runner_positions:
            self._runner_positions[fill.runner_id] = RunnerPnL(runner_id=fill.runner_id)

        rp = self._runner_positions[fill.runner_id]
        rp.fill_count += 1
        rp.turnover += fill.size

        if fill.side == "back":
            prev_stake = rp.back_stake
            rp.back_stake += fill.size
            if rp.back_stake > 0:
                rp.back_avg_odds = (
                    (prev_stake * rp.back_avg_odds + fill.size * fill.price) / rp.back_stake
                )
        else:  # lay
            prev_stake = rp.lay_stake
            rp.lay_stake += fill.size
            if rp.lay_stake > 0:
                rp.lay_avg_odds = (
                    (prev_stake * rp.lay_avg_odds + fill.size * fill.price) / rp.lay_stake
                )

        # Recompute if-wins / if-loses
        rp.pnl_if_wins = rp.back_stake * (rp.back_avg_odds - 1.0) - rp.lay_stake * (rp.lay_avg_odds - 1.0)
        rp.pnl_if_loses = -rp.back_stake + rp.lay_stake

    def record_equity_point(self, ts_ms: int, unrealized: float, realized: float) -> None:
        self._equity_points.append(EquityCurvePoint(
            timestamp_ms=ts_ms,
            unrealized_pnl=unrealized,
            realized_pnl=realized,
            total_equity=unrealized + realized,
        ))

    def set_outcome(self, outcome: MarketOutcome) -> None:
        self._outcome = outcome

    def compute(self, total_orders: int = 0) -> SessionSummary:
        """Compute the full session summary."""
        summary = SessionSummary(
            session_id=self.session_id,
            sport=self.sport,
            market_id=self.market_id,
            total_orders=total_orders,
            total_fills=len(self._fills),
        )

        if not self._fills:
            return summary

        # Duration
        first_ts = self._fills[0].timestamp_ms
        last_ts = self._fills[-1].timestamp_ms
        summary.duration_sec = (last_ts - first_ts) / 1000.0

        # Resolve actual PnL per runner
        if self._outcome and self._outcome.settled:
            winner = self._outcome.winning_runner_id
            for rid, rp in self._runner_positions.items():
                if rid == winner:
                    rp.actual_pnl = rp.pnl_if_wins
                else:
                    rp.actual_pnl = rp.pnl_if_loses
                # Apply commission on winnings
                if rp.actual_pnl > 0:
                    rp.actual_pnl *= (1.0 - self.commission_rate)

        summary.runner_pnls = dict(self._runner_positions)

        # Aggregate PnL
        summary.gross_pnl = sum(rp.actual_pnl for rp in self._runner_positions.values())
        summary.net_pnl = summary.gross_pnl  # commission already applied above
        summary.turnover = sum(rp.turnover for rp in self._runner_positions.values())
        summary.total_stake = sum(rp.back_stake + rp.lay_stake for rp in self._runner_positions.values())
        summary.roi = summary.net_pnl / summary.turnover if summary.turnover > 0 else 0.0

        # Win/loss counting (per-fill basis using fair price comparison)
        wins = []
        losses = []
        for fill in self._fills:
            if fill.side == "back":
                # Profitable if we backed below fair (got value)
                edge = (1.0 / fill.fair_price_at_fill - 1.0 / fill.price) if fill.fair_price_at_fill > 0 else 0
                theoretical_pnl = fill.size * edge * fill.price
            else:
                edge = (1.0 / fill.price - 1.0 / fill.fair_price_at_fill) if fill.fair_price_at_fill > 0 else 0
                theoretical_pnl = fill.size * edge * fill.price

            if theoretical_pnl >= 0:
                wins.append(theoretical_pnl)
            else:
                losses.append(theoretical_pnl)

        summary.win_count = len(wins)
        summary.loss_count = len(losses)
        total_trades = summary.win_count + summary.loss_count
        summary.win_rate = summary.win_count / total_trades if total_trades > 0 else 0.0
        summary.avg_win = sum(wins) / len(wins) if wins else 0.0
        summary.avg_loss = sum(losses) / len(losses) if losses else 0.0
        gross_wins = sum(wins)
        gross_losses = abs(sum(losses))
        summary.profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

        # Equity curve & drawdown
        if self._equity_points:
            summary.equity_curve = self._equity_points
            peak = 0.0
            max_dd = 0.0
            for pt in self._equity_points:
                if pt.total_equity > peak:
                    peak = pt.total_equity
                dd = peak - pt.total_equity
                if dd > max_dd:
                    max_dd = dd
            summary.max_drawdown = max_dd
            summary.peak_equity = peak
            summary.max_drawdown_pct = max_dd / peak if peak > 0 else 0.0

        # Sharpe proxy (using per-fill returns)
        if len(self._fills) > 1:
            returns = []
            for fill in self._fills:
                if fill.fair_price_at_fill > 0 and fill.price > 0:
                    ret = (1.0 / fill.fair_price_at_fill) / (1.0 / fill.price) - 1.0
                    returns.append(ret)
            if returns:
                mean_r = sum(returns) / len(returns)
                var_r = sum((r - mean_r) ** 2 for r in returns) / len(returns)
                std_r = math.sqrt(var_r) if var_r > 0 else 1e-10
                summary.sharpe_proxy = mean_r / std_r

        return summary
