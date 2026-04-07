"""
Strategy registry — manages lifecycle of strategy candidates.

Each strategy has a status:
  research  → being tested in replay/sweep/ablation
  paper     → running shadow orders on live data
  live      → trading with real capital (small size)
  retired   → no longer active (underperformed or replaced)

Promotion rules:
  research → paper:  replay PnL > baseline, edge_retained > 20%, 50+ games
  paper → live:      paper PnL > 0 for 5+ consecutive days, no anomalies
  any → retired:     drawdown > threshold, or replaced by better candidate
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any


VALID_STATUSES = {"research", "paper", "live", "retired"}

PROMOTION_RULES = {
    "research_to_paper": {
        "min_games": 30,
        "min_mean_pnl": 0.0,           # must beat zero
        "min_edge_retained_pct": 20.0,
        "min_direction_accuracy": 0.55,
        "min_promotion_score": 20.0,
    },
    "paper_to_live": {
        "min_paper_days": 5,
        "min_paper_pnl": 0.0,
        "max_drawdown_pct": 15.0,
        "max_anomaly_rate": 0.05,
    },
}


@dataclass
class StrategyEntry:
    """A single strategy in the registry."""

    strategy_id: str                   # spec_id or human-readable ID
    name: str
    sport: str
    status: str = "research"           # research / paper / live / retired

    # Config snapshot
    feature_flags: dict[str, bool] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)

    # Performance tracking
    research_pnl: float = 0.0
    research_accuracy: float = 0.0
    research_games: int = 0
    paper_pnl: float = 0.0
    paper_days: int = 0
    live_pnl: float = 0.0
    live_days: int = 0

    # Lifecycle
    promotion_score: float = 0.0
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    retired_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class StrategyRegistry:
    """Manages the strategy lifecycle registry."""

    def __init__(self, registry_path: str = "registry/strategies.json"):
        self.registry_path = registry_path
        self.strategies: dict[str, StrategyEntry] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.registry_path):
            with open(self.registry_path) as f:
                data = json.load(f)
            for entry_data in data:
                entry = StrategyEntry(**entry_data)
                self.strategies[entry.strategy_id] = entry

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.registry_path) or ".", exist_ok=True)
        entries = [e.to_dict() for e in self.strategies.values()]
        with open(self.registry_path, "w") as f:
            json.dump(entries, f, indent=2)

    def register(self, entry: StrategyEntry) -> None:
        self.strategies[entry.strategy_id] = entry
        self.save()

    def get(self, strategy_id: str) -> StrategyEntry | None:
        return self.strategies.get(strategy_id)

    def list_by_status(self, status: str) -> list[StrategyEntry]:
        return [e for e in self.strategies.values() if e.status == status]

    def promote(self, strategy_id: str, new_status: str, reason: str = "") -> bool:
        """Attempt to promote a strategy to a new status.

        Returns True if promotion succeeded, False if blocked by rules.
        """
        entry = self.strategies.get(strategy_id)
        if not entry:
            return False
        if new_status not in VALID_STATUSES:
            return False

        old_status = entry.status

        # Check promotion rules
        if old_status == "research" and new_status == "paper":
            rules = PROMOTION_RULES["research_to_paper"]
            if entry.research_games < rules["min_games"]:
                return False
            if entry.research_pnl < rules["min_mean_pnl"]:
                return False
            if entry.research_accuracy < rules["min_direction_accuracy"]:
                return False
            if entry.promotion_score < rules["min_promotion_score"]:
                return False

        elif old_status == "paper" and new_status == "live":
            rules = PROMOTION_RULES["paper_to_live"]
            if entry.paper_days < rules["min_paper_days"]:
                return False
            if entry.paper_pnl < rules["min_paper_pnl"]:
                return False

        entry.status = new_status
        entry.last_updated = time.time()
        self.save()
        return True

    def retire(self, strategy_id: str, reason: str = "") -> None:
        entry = self.strategies.get(strategy_id)
        if entry:
            entry.status = "retired"
            entry.retired_reason = reason
            entry.last_updated = time.time()
            self.save()

    def update_research_metrics(
        self, strategy_id: str, pnl: float, accuracy: float,
        games: int, score: float,
    ) -> None:
        entry = self.strategies.get(strategy_id)
        if entry:
            entry.research_pnl = pnl
            entry.research_accuracy = accuracy
            entry.research_games = games
            entry.promotion_score = score
            entry.last_updated = time.time()
            self.save()

    def update_paper_metrics(self, strategy_id: str, pnl: float, days: int) -> None:
        entry = self.strategies.get(strategy_id)
        if entry:
            entry.paper_pnl = pnl
            entry.paper_days = days
            entry.last_updated = time.time()
            self.save()

    def print_summary(self) -> None:
        """Print a formatted summary of all strategies."""
        print(f"\n{'='*90}")
        print(f"  STRATEGY REGISTRY — {len(self.strategies)} strategies")
        print(f"{'='*90}")

        for status in ["live", "paper", "research", "retired"]:
            entries = self.list_by_status(status)
            if not entries:
                continue
            print(f"\n  [{status.upper()}] ({len(entries)})")
            print(f"  {'ID':<14} {'Name':<20} {'Sport':<10} {'PnL':>10} {'Acc':>7} {'Score':>7}")
            print(f"  {'─'*70}")
            for e in entries:
                pnl = e.live_pnl if status == "live" else e.paper_pnl if status == "paper" else e.research_pnl
                print(f"  {e.strategy_id:<14} {e.name:<20} {e.sport:<10} "
                      f"{pnl:>+10.2f} {e.research_accuracy:>6.1%} {e.promotion_score:>7.1f}")

        print(f"\n{'='*90}\n")
