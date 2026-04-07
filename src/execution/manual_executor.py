"""
Manual executor — end-to-end pipeline from signal to confirmed order.

Orchestrates: signal → market lookup → live quote → pre-trade check →
order build → display for human confirmation.

This is the "last mile" before real money moves. Every order must be
explicitly confirmed by the operator.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any

from src.execution.market_mapper import MarketMapper, MarketMapping
from src.execution.quote_fetcher import LiveQuoteFetcher, LiveQuote
from src.execution.pre_trade_check import PreTradeComparator, DeviationResult, Signal
from src.execution.order_builder import OrderBuilder, ExecutionOrder


EXECUTION_LOG_PATH = "micro_live_logs/execution_log.jsonl"


@dataclass
class ExecutionDecision:
    """Complete record of an execution decision."""
    order: ExecutionOrder | None
    deviation: DeviationResult | None
    quote: LiveQuote | None
    mapping: MarketMapping | None

    # Decision
    action: str = ""           # "confirmed" | "rejected" | "blocked"
    reason: str = ""
    operator_notes: str = ""
    decided_at: str = ""

    def to_dict(self) -> dict:
        d: dict[str, Any] = {}
        d["action"] = self.action
        d["reason"] = self.reason
        d["operator_notes"] = self.operator_notes
        d["decided_at"] = self.decided_at
        if self.order:
            d["order"] = self.order.to_dict()
        if self.deviation:
            d["deviation"] = {
                "signal": self.deviation.signal.value,
                "system_price": self.deviation.system_price,
                "market_price": self.deviation.market_price,
                "deviation_pct": self.deviation.deviation_pct,
                "spread_bps": self.deviation.spread_bps,
                "reason": self.deviation.reason,
            }
        if self.mapping:
            d["mapping"] = {
                "internal_id": self.mapping.internal_id,
                "home_team": self.mapping.home_team,
                "away_team": self.mapping.away_team,
            }
        return d


class ManualExecutor:
    """End-to-end execution pipeline with manual confirmation.

    Usage:
        executor = ManualExecutor()
        decision = executor.evaluate_signal(
            internal_market_id="nba_lal_bos_ml",
            runner_id="home",
            side="buy",
            system_decimal_odds=1.80,
            edge_bps=297,
            net_edge_bps=67,
            size=1.0,
        )
        # Display decision to operator, then:
        executor.confirm(decision)   # or executor.reject(decision, "...")
    """

    def __init__(
        self,
        mapper: MarketMapper | None = None,
        fetcher: LiveQuoteFetcher | None = None,
        comparator: PreTradeComparator | None = None,
        builder: OrderBuilder | None = None,
    ):
        self.mapper = mapper or MarketMapper()
        self.fetcher = fetcher or LiveQuoteFetcher()
        self.comparator = comparator or PreTradeComparator()
        self.builder = builder or OrderBuilder()

    def evaluate_signal(
        self,
        internal_market_id: str,
        runner_id: str,
        side: str,
        system_decimal_odds: float,
        edge_bps: float,
        net_edge_bps: float,
        size: float = 1.0,
    ) -> ExecutionDecision:
        """Run the full pre-trade pipeline for a single signal.

        Returns an ExecutionDecision with all context needed for
        the operator to make a confirm/reject decision.
        """
        # Step 1: Market lookup
        mapping = self.mapper.lookup(internal_market_id)
        if not mapping:
            return ExecutionDecision(
                order=None, deviation=None, quote=None, mapping=None,
                action="blocked",
                reason=f"no market mapping for '{internal_market_id}'",
            )

        # Step 2: Fetch real-time quote
        token_id = mapping.yes_token_id if runner_id == "home" else mapping.no_token_id
        if not token_id:
            return ExecutionDecision(
                order=None, deviation=None, quote=None, mapping=mapping,
                action="blocked",
                reason=f"no token_id for runner '{runner_id}' in mapping",
            )

        quote = self.fetcher.fetch(token_id)

        # Step 3: Pre-trade deviation check
        deviation = self.comparator.compare(system_decimal_odds, quote)

        # Step 4: Build order (if allowed)
        order = self.builder.build(
            mapping=mapping,
            deviation=deviation,
            runner_id=runner_id,
            side=side,
            size=size,
            edge_bps=edge_bps,
            net_edge_bps=net_edge_bps,
        )

        if order is None:
            return ExecutionDecision(
                order=None, deviation=deviation, quote=quote, mapping=mapping,
                action="blocked",
                reason=f"pre-trade check: {deviation.reason}",
            )

        return ExecutionDecision(
            order=order, deviation=deviation, quote=quote, mapping=mapping,
        )

    def confirm(self, decision: ExecutionDecision, notes: str = "") -> None:
        """Operator confirms the order."""
        decision.action = "confirmed"
        decision.operator_notes = notes
        decision.decided_at = datetime.now().isoformat()
        self._log(decision)

    def reject(self, decision: ExecutionDecision, reason: str = "") -> None:
        """Operator rejects the order."""
        decision.action = "rejected"
        decision.reason = reason
        decision.decided_at = datetime.now().isoformat()
        self._log(decision)

    def _log(self, decision: ExecutionDecision) -> None:
        """Append decision to the execution log."""
        os.makedirs(os.path.dirname(EXECUTION_LOG_PATH) or ".", exist_ok=True)
        with open(EXECUTION_LOG_PATH, "a") as f:
            f.write(json.dumps(decision.to_dict()) + "\n")

    def display_for_review(self, decision: ExecutionDecision) -> str:
        """Format a decision for terminal display."""
        lines = []
        lines.append(f"{'='*60}")
        lines.append(f"  EXECUTION REVIEW")
        lines.append(f"{'='*60}")

        if decision.action == "blocked":
            lines.append(f"  STATUS: BLOCKED")
            lines.append(f"  Reason: {decision.reason}")
            lines.append(f"{'='*60}")
            return "\n".join(lines)

        o = decision.order
        d = decision.deviation
        m = decision.mapping

        if m:
            lines.append(f"  Game:     {m.home_team} vs {m.away_team} ({m.game_date})")
            lines.append(f"  Market:   {m.internal_id}")
        if o:
            lines.append(f"  Signal:   {o.runner_id} {o.side} (back {o.description})")
            lines.append(f"  Amount:   ${o.size:.2f}")
            lines.append(f"")
        if d:
            signal_icon = {"green": "[OK]", "yellow": "[!!]", "red": "[XX]"}
            lines.append(f"  Pre-trade: {signal_icon.get(d.signal.value, '?')} {d.signal.value.upper()}")
            lines.append(f"  System price:  {d.system_price:.3f} (implied {d.system_implied_prob:.1%})")
            lines.append(f"  Market price:  {d.market_price:.3f} (implied {d.market_implied_prob:.1%})")
            lines.append(f"  Deviation:     {d.deviation_pct:.1f}%")
            lines.append(f"  Spread:        {d.spread_bps:.0f}bps")
            lines.append(f"  Depth:         bid={d.bid_depth:.0f} ask={d.ask_depth:.0f}")
        if o:
            lines.append(f"")
            lines.append(f"  Edge:      {o.edge_bps:.0f}bps")
            lines.append(f"  Net edge:  {o.net_edge_bps:.0f}bps (after fees)")
            lines.append(f"  Token:     {o.token_id}")
            lines.append(f"  Price:     {o.price:.4f}")
            lines.append(f"  TTL:       {o.ttl_sec}s")

        lines.append(f"")
        lines.append(f"  (c)onfirm  |  (r)eject  |  (s)kip")
        lines.append(f"{'='*60}")
        return "\n".join(lines)
