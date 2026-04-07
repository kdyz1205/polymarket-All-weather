"""
Sub-module B: Execution quality analyzer.

Answers: "Is the execution layer eating my edge?"

For every order/fill, measures:
  - Did I get filled? (fill ratio)
  - How long did it take? (quote-to-fill latency)
  - Did the market move against me during delay? (delay damage)
  - Did I capture the theoretical edge? (edge capture ratio)
  - Was there adverse selection? (market moved against after fill)

Key metric: Edge Capture Ratio
  = actual PnL from fills / theoretical edge at time of signal
  If this is < 0.5, the execution layer is destroying value.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from src.analytics.session_metrics import FillRecord


@dataclass
class OrderRecord:
    """An order with its full lifecycle for execution analysis."""

    order_id: int
    runner_id: str
    side: str
    price: float
    size: float
    submit_ts_ms: int
    # Outcome
    status: str  # "filled", "partial", "cancelled", "rejected", "lapsed"
    filled_size: float = 0.0
    avg_fill_price: float = 0.0
    fill_ts_ms: int = 0
    cancel_ts_ms: int = 0
    # Context
    fair_at_submit: float = 0.0
    market_at_submit: float = 0.0
    fair_at_fill: float = 0.0
    market_at_fill: float = 0.0
    market_5s_after_fill: float = 0.0  # price 5 sec after fill (adverse selection check)
    was_suspended_during: bool = False
    delay_ms: int = 0
    strategy_tag: str = ""


@dataclass
class ExecutionMetrics:
    """Aggregate execution quality metrics."""

    # Fill stats
    total_orders: int = 0
    filled_orders: int = 0
    partial_fills: int = 0
    cancelled_orders: int = 0
    rejected_orders: int = 0
    lapsed_orders: int = 0

    fill_ratio: float = 0.0
    partial_fill_ratio: float = 0.0
    cancel_ratio: float = 0.0

    # Latency
    avg_quote_to_fill_ms: float = 0.0
    median_quote_to_fill_ms: float = 0.0
    p95_quote_to_fill_ms: float = 0.0

    # Edge capture — THE key metric
    total_theoretical_edge: float = 0.0
    total_captured_edge: float = 0.0
    edge_capture_ratio: float = 0.0

    # Slippage
    avg_slippage_bps: float = 0.0
    total_slippage_cost: float = 0.0

    # Delay damage
    avg_delay_damage_bps: float = 0.0
    total_delay_damage: float = 0.0
    orders_damaged_by_delay: int = 0

    # Adverse selection (market moved against us after fill)
    avg_adverse_move_bps: float = 0.0
    avg_favorable_move_bps: float = 0.0
    adverse_selection_ratio: float = 0.0  # fraction of fills where market moved against

    # Suspend impact
    orders_hit_by_suspend: int = 0
    suspend_hit_ratio: float = 0.0

    # Per-strategy breakdown
    per_strategy: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fill_stats": {
                "total_orders": self.total_orders,
                "filled_orders": self.filled_orders,
                "partial_fills": self.partial_fills,
                "cancelled_orders": self.cancelled_orders,
                "rejected_orders": self.rejected_orders,
                "fill_ratio": round(self.fill_ratio, 4),
                "cancel_ratio": round(self.cancel_ratio, 4),
            },
            "latency": {
                "avg_quote_to_fill_ms": round(self.avg_quote_to_fill_ms, 1),
                "median_quote_to_fill_ms": round(self.median_quote_to_fill_ms, 1),
                "p95_quote_to_fill_ms": round(self.p95_quote_to_fill_ms, 1),
            },
            "edge_capture": {
                "total_theoretical_edge": round(self.total_theoretical_edge, 4),
                "total_captured_edge": round(self.total_captured_edge, 4),
                "edge_capture_ratio": round(self.edge_capture_ratio, 4),
            },
            "slippage": {
                "avg_slippage_bps": round(self.avg_slippage_bps, 2),
                "total_slippage_cost": round(self.total_slippage_cost, 4),
            },
            "delay_damage": {
                "avg_delay_damage_bps": round(self.avg_delay_damage_bps, 2),
                "total_delay_damage": round(self.total_delay_damage, 4),
                "orders_damaged_by_delay": self.orders_damaged_by_delay,
            },
            "adverse_selection": {
                "avg_adverse_move_bps": round(self.avg_adverse_move_bps, 2),
                "avg_favorable_move_bps": round(self.avg_favorable_move_bps, 2),
                "adverse_selection_ratio": round(self.adverse_selection_ratio, 4),
            },
            "suspend_impact": {
                "orders_hit_by_suspend": self.orders_hit_by_suspend,
                "suspend_hit_ratio": round(self.suspend_hit_ratio, 4),
            },
            "per_strategy": self.per_strategy,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class ExecutionAnalyzer:
    """
    Analyzes execution quality from order records.

    Usage:
        analyzer = ExecutionAnalyzer()
        for order in orders:
            analyzer.record_order(order)
        metrics = analyzer.compute()
    """

    def __init__(self) -> None:
        self._orders: list[OrderRecord] = []

    def record_order(self, order: OrderRecord) -> None:
        self._orders.append(order)

    def compute(self) -> ExecutionMetrics:
        m = ExecutionMetrics()
        m.total_orders = len(self._orders)

        if not self._orders:
            return m

        # --- Fill stats ---
        m.filled_orders = sum(1 for o in self._orders if o.status == "filled")
        m.partial_fills = sum(1 for o in self._orders if o.status == "partial")
        m.cancelled_orders = sum(1 for o in self._orders if o.status == "cancelled")
        m.rejected_orders = sum(1 for o in self._orders if o.status == "rejected")
        m.lapsed_orders = sum(1 for o in self._orders if o.status == "lapsed")

        m.fill_ratio = (m.filled_orders + m.partial_fills) / m.total_orders if m.total_orders > 0 else 0
        m.partial_fill_ratio = m.partial_fills / m.total_orders if m.total_orders > 0 else 0
        m.cancel_ratio = m.cancelled_orders / m.total_orders if m.total_orders > 0 else 0

        # --- Latency ---
        latencies = []
        for o in self._orders:
            if o.fill_ts_ms > 0 and o.submit_ts_ms > 0:
                latencies.append(o.fill_ts_ms - o.submit_ts_ms)

        if latencies:
            latencies.sort()
            m.avg_quote_to_fill_ms = sum(latencies) / len(latencies)
            m.median_quote_to_fill_ms = latencies[len(latencies) // 2]
            m.p95_quote_to_fill_ms = latencies[int(len(latencies) * 0.95)]

        # --- Edge capture ---
        filled_orders = [o for o in self._orders if o.filled_size > 0]

        theoretical_edges = []
        captured_edges = []
        slippages = []
        delay_damages = []
        adverse_moves = []
        favorable_moves = []

        for o in filled_orders:
            # Theoretical edge at submission = difference between fair and market
            if o.fair_at_submit > 0 and o.market_at_submit > 0:
                if o.side == "back":
                    # We back: edge = 1/fair - 1/market (in probability terms)
                    # In currency: edge * stake
                    theo_edge_prob = (1.0 / o.fair_at_submit) - (1.0 / o.market_at_submit)
                else:
                    theo_edge_prob = (1.0 / o.market_at_submit) - (1.0 / o.fair_at_submit)

                theo_edge_value = theo_edge_prob * o.filled_size * o.market_at_submit
                theoretical_edges.append(theo_edge_value)

                # Captured edge = what we actually got vs fair
                if o.side == "back":
                    captured_prob = (1.0 / o.fair_at_fill) - (1.0 / o.avg_fill_price)
                else:
                    captured_prob = (1.0 / o.avg_fill_price) - (1.0 / o.fair_at_fill)

                captured_value = captured_prob * o.filled_size * o.avg_fill_price
                captured_edges.append(captured_value)

            # Slippage = difference between expected price and actual fill price
            if o.market_at_submit > 0 and o.avg_fill_price > 0:
                if o.side == "back":
                    slip_bps = (o.market_at_submit - o.avg_fill_price) / o.market_at_submit * 10000
                else:
                    slip_bps = (o.avg_fill_price - o.market_at_submit) / o.market_at_submit * 10000

                slippages.append(slip_bps)
                slip_cost = abs(slip_bps) / 10000 * o.filled_size
                m.total_slippage_cost += slip_cost

            # Delay damage = how much price moved against during bet delay
            if o.delay_ms > 0 and o.market_at_submit > 0 and o.market_at_fill > 0:
                if o.side == "back":
                    dd_bps = (o.market_at_submit - o.market_at_fill) / o.market_at_submit * 10000
                else:
                    dd_bps = (o.market_at_fill - o.market_at_submit) / o.market_at_submit * 10000

                delay_damages.append(dd_bps)
                if dd_bps > 0:  # damaged
                    m.orders_damaged_by_delay += 1
                    m.total_delay_damage += dd_bps / 10000 * o.filled_size

            # Adverse selection = market move 5s after fill
            if o.market_5s_after_fill > 0 and o.avg_fill_price > 0:
                if o.side == "back":
                    # Adverse for back = price dropped after we bought
                    move_bps = (o.market_5s_after_fill - o.avg_fill_price) / o.avg_fill_price * 10000
                else:
                    move_bps = (o.avg_fill_price - o.market_5s_after_fill) / o.avg_fill_price * 10000

                if move_bps < 0:
                    adverse_moves.append(abs(move_bps))
                else:
                    favorable_moves.append(move_bps)

            # Suspend hit
            if o.was_suspended_during:
                m.orders_hit_by_suspend += 1

        # Aggregate
        if theoretical_edges:
            m.total_theoretical_edge = sum(theoretical_edges)
        if captured_edges:
            m.total_captured_edge = sum(captured_edges)
        if m.total_theoretical_edge != 0:
            m.edge_capture_ratio = m.total_captured_edge / abs(m.total_theoretical_edge)

        if slippages:
            m.avg_slippage_bps = sum(slippages) / len(slippages)

        if delay_damages:
            m.avg_delay_damage_bps = sum(delay_damages) / len(delay_damages)

        if adverse_moves:
            m.avg_adverse_move_bps = sum(adverse_moves) / len(adverse_moves)
        if favorable_moves:
            m.avg_favorable_move_bps = sum(favorable_moves) / len(favorable_moves)

        total_post_fill = len(adverse_moves) + len(favorable_moves)
        m.adverse_selection_ratio = len(adverse_moves) / total_post_fill if total_post_fill > 0 else 0

        m.suspend_hit_ratio = m.orders_hit_by_suspend / m.total_orders if m.total_orders > 0 else 0

        # --- Per-strategy breakdown ---
        strategy_groups: dict[str, list[OrderRecord]] = {}
        for o in self._orders:
            tag = o.strategy_tag or "default"
            strategy_groups.setdefault(tag, []).append(o)

        for tag, orders in strategy_groups.items():
            filled = sum(1 for o in orders if o.filled_size > 0)
            total_filled_size = sum(o.filled_size for o in orders)
            m.per_strategy[tag] = {
                "orders": len(orders),
                "filled": filled,
                "fill_ratio": round(filled / len(orders), 4) if orders else 0,
                "total_size": round(total_filled_size, 2),
            }

        return m
