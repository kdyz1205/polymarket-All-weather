"""
Sub-module C: Event attribution analyzer.

Answers: "Which game events help us and which hurt us?"

For each event type (goal, 3-pointer, strikeout, etc.), measures:
  - How did the model's fair prob change?
  - How did the market prob change?
  - Did we have open orders/positions when it happened?
  - What was the PnL impact?
  - Was there a suspend? How long? What was the gap?

This is critical for understanding:
  - Which events create tradeable edge
  - Which events destroy our positions
  - Whether the model reacts correctly to events
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EventSnapshot:
    """Snapshot of system state at the time of a match event."""

    # Event identity
    event_type: str
    team: str
    game_minute: float
    timestamp_ms: int

    # Score state
    home_score_before: int = 0
    away_score_before: int = 0
    home_score_after: int = 0
    away_score_after: int = 0

    # Model state
    fair_p_home_before: float = 0.0
    fair_p_home_after: float = 0.0
    model_prob_jump: float = 0.0  # after - before

    # Market state
    market_p_home_before: float = 0.0
    market_p_home_after: float = 0.0
    market_prob_jump: float = 0.0

    # Model vs market
    model_led_market: bool = False  # did model move before market?
    convergence_direction: str = ""  # "to_model", "to_market", "neither"

    # Position impact
    had_position: bool = False
    position_pnl_impact: float = 0.0  # MTM change due to this event
    exposure_at_event: float = 0.0

    # Suspend/reopen
    was_suspended: bool = False
    suspend_duration_ms: int = 0
    gap_on_reopen_bps: float = 0.0  # price jump on reopen vs pre-suspend


@dataclass
class EventTypeStats:
    """Aggregate stats for one type of event."""

    event_type: str
    count: int = 0

    # Probability impact
    avg_model_prob_jump: float = 0.0
    avg_market_prob_jump: float = 0.0
    max_model_prob_jump: float = 0.0
    model_jump_std: float = 0.0

    # PnL impact
    total_pnl_impact: float = 0.0
    avg_pnl_impact: float = 0.0
    events_with_position: int = 0
    events_positive_pnl: int = 0
    events_negative_pnl: int = 0

    # Model accuracy after event
    model_led_count: int = 0
    model_led_ratio: float = 0.0  # how often model moved in correct direction first
    converged_to_model_count: int = 0
    convergence_ratio: float = 0.0  # how often market converged to model

    # Suspend stats
    suspend_count: int = 0
    avg_suspend_duration_ms: float = 0.0
    avg_gap_on_reopen_bps: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "count": self.count,
            "probability_impact": {
                "avg_model_jump": round(self.avg_model_prob_jump, 4),
                "avg_market_jump": round(self.avg_market_prob_jump, 4),
                "max_model_jump": round(self.max_model_prob_jump, 4),
                "model_jump_std": round(self.model_jump_std, 4),
            },
            "pnl_impact": {
                "total": round(self.total_pnl_impact, 4),
                "avg": round(self.avg_pnl_impact, 4),
                "events_with_position": self.events_with_position,
                "positive_count": self.events_positive_pnl,
                "negative_count": self.events_negative_pnl,
            },
            "model_quality": {
                "model_led_ratio": round(self.model_led_ratio, 4),
                "convergence_ratio": round(self.convergence_ratio, 4),
            },
            "suspend": {
                "count": self.suspend_count,
                "avg_duration_ms": round(self.avg_suspend_duration_ms, 1),
                "avg_gap_bps": round(self.avg_gap_on_reopen_bps, 2),
            },
        }


@dataclass
class EventAttribution:
    """Full event attribution report."""

    by_event_type: dict[str, EventTypeStats] = field(default_factory=dict)
    by_game_phase: dict[str, dict[str, Any]] = field(default_factory=dict)
    total_events: int = 0
    total_score_changes: int = 0
    total_suspends: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "total_events": self.total_events,
                "total_score_changes": self.total_score_changes,
                "total_suspends": self.total_suspends,
            },
            "by_event_type": {k: v.to_dict() for k, v in self.by_event_type.items()},
            "by_game_phase": self.by_game_phase,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class EventAttributionAnalyzer:
    """
    Analyzes how different match events affected the strategy.

    Usage:
        analyzer = EventAttributionAnalyzer()
        for event in match_events:
            analyzer.record_event(event_snapshot)
        attribution = analyzer.compute()
    """

    def __init__(self) -> None:
        self._events: list[EventSnapshot] = []

    def record_event(self, snapshot: EventSnapshot) -> None:
        self._events.append(snapshot)

    def compute(self) -> EventAttribution:
        attr = EventAttribution()
        attr.total_events = len(self._events)

        if not self._events:
            return attr

        # Group by event type
        groups: dict[str, list[EventSnapshot]] = {}
        for ev in self._events:
            groups.setdefault(ev.event_type, []).append(ev)

        for etype, events in groups.items():
            stats = EventTypeStats(event_type=etype, count=len(events))

            model_jumps = [e.model_prob_jump for e in events]
            market_jumps = [e.market_prob_jump for e in events]
            pnl_impacts = [e.position_pnl_impact for e in events if e.had_position]

            # Probability impact
            stats.avg_model_prob_jump = sum(model_jumps) / len(model_jumps) if model_jumps else 0
            stats.avg_market_prob_jump = sum(market_jumps) / len(market_jumps) if market_jumps else 0
            stats.max_model_prob_jump = max((abs(j) for j in model_jumps), default=0)

            if len(model_jumps) > 1:
                mean = stats.avg_model_prob_jump
                variance = sum((j - mean) ** 2 for j in model_jumps) / len(model_jumps)
                stats.model_jump_std = variance ** 0.5

            # PnL impact
            stats.events_with_position = sum(1 for e in events if e.had_position)
            if pnl_impacts:
                stats.total_pnl_impact = sum(pnl_impacts)
                stats.avg_pnl_impact = stats.total_pnl_impact / len(pnl_impacts)
                stats.events_positive_pnl = sum(1 for p in pnl_impacts if p > 0)
                stats.events_negative_pnl = sum(1 for p in pnl_impacts if p < 0)

            # Model quality
            stats.model_led_count = sum(1 for e in events if e.model_led_market)
            stats.model_led_ratio = stats.model_led_count / len(events)
            stats.converged_to_model_count = sum(
                1 for e in events if e.convergence_direction == "to_model"
            )
            stats.convergence_ratio = stats.converged_to_model_count / len(events)

            # Suspend
            suspended = [e for e in events if e.was_suspended]
            stats.suspend_count = len(suspended)
            if suspended:
                stats.avg_suspend_duration_ms = (
                    sum(e.suspend_duration_ms for e in suspended) / len(suspended)
                )
                stats.avg_gap_on_reopen_bps = (
                    sum(e.gap_on_reopen_bps for e in suspended) / len(suspended)
                )

            # Score changes
            score_changes = sum(
                1 for e in events
                if (e.home_score_after != e.home_score_before
                    or e.away_score_after != e.away_score_before)
            )
            attr.total_score_changes += score_changes
            attr.total_suspends += stats.suspend_count

            attr.by_event_type[etype] = stats

        # Group by game phase (early/mid/late)
        phase_groups: dict[str, list[EventSnapshot]] = {"early": [], "mid": [], "late": [], "clutch": []}
        for ev in self._events:
            if ev.game_minute <= 20:
                phase_groups["early"].append(ev)
            elif ev.game_minute <= 60:
                phase_groups["mid"].append(ev)
            elif ev.game_minute <= 80:
                phase_groups["late"].append(ev)
            else:
                phase_groups["clutch"].append(ev)

        for phase, events in phase_groups.items():
            if not events:
                continue
            positioned = [e for e in events if e.had_position]
            attr.by_game_phase[phase] = {
                "event_count": len(events),
                "avg_model_jump": round(
                    sum(e.model_prob_jump for e in events) / len(events), 4
                ),
                "total_pnl_impact": round(
                    sum(e.position_pnl_impact for e in positioned), 4
                ) if positioned else 0,
                "suspend_count": sum(1 for e in events if e.was_suspended),
            }

        return attr
