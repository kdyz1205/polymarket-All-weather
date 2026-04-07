"""
Market mapper — resolves internal signal identifiers to real Polymarket
market IDs and token IDs.

Maps (sport, home_team, away_team, market_type) → MarketMapping with
real condition_id, yes/no token_ids, and human-readable description.

Mappings are loaded from a JSON file that the operator maintains.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any


MAPPINGS_PATH = "config/market_mappings.json"


@dataclass
class MarketMapping:
    """One real-world market on Polymarket."""
    internal_id: str              # e.g. "nba_lal_bos_ml"
    sport: str                    # "basketball" | "baseball"
    home_team: str                # "LAL"
    away_team: str                # "BOS"
    market_type: str              # "moneyline" | "spread" | "total"
    game_date: str                # "2026-04-07"

    # Polymarket identifiers
    condition_id: str = ""        # 0x... condition hash
    yes_token_id: str = ""        # token for "Yes" / home win
    no_token_id: str = ""         # token for "No" / away win
    polymarket_slug: str = ""     # URL slug for human reference
    description: str = ""         # "Will LAL beat BOS on Apr 7?"

    # Status
    active: bool = True
    last_verified_ts: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> MarketMapping:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class MarketMapper:
    """Loads and queries market mappings.

    The operator manually adds mappings for each game they want to trade.
    This keeps the system honest: no auto-discovery of markets, explicit
    human verification of every market identity.
    """

    def __init__(self, mappings_path: str = MAPPINGS_PATH):
        self.mappings_path = mappings_path
        self.mappings: dict[str, MarketMapping] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.mappings_path):
            with open(self.mappings_path) as f:
                data = json.load(f)
            for entry in data:
                m = MarketMapping.from_dict(entry)
                self.mappings[m.internal_id] = m

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.mappings_path) or ".", exist_ok=True)
        entries = [m.to_dict() for m in self.mappings.values()]
        with open(self.mappings_path, "w") as f:
            json.dump(entries, f, indent=2)

    def register(self, mapping: MarketMapping) -> None:
        """Add or update a market mapping."""
        self.mappings[mapping.internal_id] = mapping
        self.save()

    def lookup(self, internal_id: str) -> MarketMapping | None:
        return self.mappings.get(internal_id)

    def find(self, sport: str, home_team: str, away_team: str,
             game_date: str, market_type: str = "moneyline") -> MarketMapping | None:
        """Find a mapping by game attributes."""
        for m in self.mappings.values():
            if (m.sport == sport and m.home_team == home_team
                    and m.away_team == away_team and m.game_date == game_date
                    and m.market_type == market_type and m.active):
                return m
        return None

    def list_active(self, sport: str | None = None) -> list[MarketMapping]:
        """List all active mappings, optionally filtered by sport."""
        results = [m for m in self.mappings.values() if m.active]
        if sport:
            results = [m for m in results if m.sport == sport]
        return results

    def list_today(self, game_date: str) -> list[MarketMapping]:
        return [m for m in self.mappings.values()
                if m.game_date == game_date and m.active]
