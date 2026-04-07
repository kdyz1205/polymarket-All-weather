"""
Persistent data cache — 5-layer storage for the trading system.

Layer 1: markets.json      — today's tradeable markets (condition_id, tokens, status)
Layer 2: teams.json        — team records, ratings, recent form
Layer 3: players.json      — player impact scores, injury status
Layer 4: orderbooks/       — historical bid/ask snapshots per market
Layer 5: features/         — per-game feature vectors for learning

All caches are JSON-backed, auto-expiring, and self-updating.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from typing import Any


CACHE_DIR = "data/cache"


def _now_iso() -> str:
    return datetime.now().isoformat()


def _load_json(path: str) -> list | dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def _save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ─── Layer 1: Market Cache ───

@dataclass
class CachedMarket:
    slug: str
    sport: str
    league: str                    # "NBA", "MLB"
    home_team: str
    away_team: str
    game_date: str
    game_time: str = ""
    condition_id: str = ""
    home_token_id: str = ""        # token for home/first outcome
    away_token_id: str = ""
    outcomes: list[str] = field(default_factory=list)
    home_price: float = 0.0
    away_price: float = 0.0
    volume: float = 0.0
    liquidity: float = 0.0
    status: str = "active"         # active | closed | settled
    best_bid: float = 0.0
    best_ask: float = 0.0
    spread: float = 0.0
    updated_at: str = ""
    context: str = ""              # event metadata context

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> CachedMarket:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @property
    def is_stale(self) -> bool:
        if not self.updated_at:
            return True
        try:
            updated = datetime.fromisoformat(self.updated_at)
            age_min = (datetime.now() - updated).total_seconds() / 60
            return age_min > 10  # stale after 10 minutes
        except (ValueError, TypeError):
            return True

    @property
    def age_minutes(self) -> float:
        if not self.updated_at:
            return 9999
        try:
            updated = datetime.fromisoformat(self.updated_at)
            return (datetime.now() - updated).total_seconds() / 60
        except (ValueError, TypeError):
            return 9999


class MarketCache:
    def __init__(self, path: str = f"{CACHE_DIR}/markets.json"):
        self.path = path
        self.markets: dict[str, CachedMarket] = {}
        self._load()

    def _load(self) -> None:
        data = _load_json(self.path)
        if isinstance(data, list):
            for d in data:
                m = CachedMarket.from_dict(d)
                self.markets[m.slug] = m

    def save(self) -> None:
        _save_json(self.path, [m.to_dict() for m in self.markets.values()])

    def upsert(self, market: CachedMarket) -> None:
        market.updated_at = _now_iso()
        self.markets[market.slug] = market
        self.save()

    def upsert_batch(self, markets: list[CachedMarket]) -> None:
        for m in markets:
            m.updated_at = _now_iso()
            self.markets[m.slug] = m
        self.save()

    def get(self, slug: str) -> CachedMarket | None:
        return self.markets.get(slug)

    def get_today(self, game_date: str | None = None) -> list[CachedMarket]:
        today = game_date or date.today().isoformat()
        return [m for m in self.markets.values()
                if m.game_date == today and m.status == "active"]

    def get_stale(self) -> list[CachedMarket]:
        return [m for m in self.markets.values() if m.is_stale and m.status == "active"]

    def count(self) -> int:
        return len(self.markets)


# ─── Layer 2: Team Cache ───

@dataclass
class CachedTeam:
    abbreviation: str
    full_name: str
    league: str = "NBA"
    wins: int = 0
    losses: int = 0
    elo: float = 1500.0
    offensive_rating: float = 0.0
    defensive_rating: float = 0.0
    pace: float = 100.0
    home_wins: int = 0
    home_losses: int = 0
    away_wins: int = 0
    away_losses: int = 0
    last5: str = ""                # "3-2"
    last10: str = ""               # "7-3"
    streak: int = 0                # positive=wins, negative=losses
    conference: str = ""
    division: str = ""
    playoff_seed: int = 0
    updated_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> CachedTeam:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @property
    def win_pct(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.5

    @property
    def is_stale(self) -> bool:
        if not self.updated_at:
            return True
        try:
            updated = datetime.fromisoformat(self.updated_at)
            age_hr = (datetime.now() - updated).total_seconds() / 3600
            return age_hr > 12  # stale after 12 hours
        except (ValueError, TypeError):
            return True


class TeamCache:
    def __init__(self, path: str = f"{CACHE_DIR}/teams.json"):
        self.path = path
        self.teams: dict[str, CachedTeam] = {}
        self._load()

    def _load(self) -> None:
        data = _load_json(self.path)
        if isinstance(data, list):
            for d in data:
                t = CachedTeam.from_dict(d)
                self.teams[t.abbreviation] = t

    def save(self) -> None:
        _save_json(self.path, [t.to_dict() for t in self.teams.values()])

    def upsert(self, team: CachedTeam) -> None:
        team.updated_at = _now_iso()
        self.teams[team.abbreviation] = team
        self.save()

    def get(self, abbr: str) -> CachedTeam | None:
        return self.teams.get(abbr.upper())

    def get_or_default(self, abbr: str, full_name: str = "") -> CachedTeam:
        t = self.get(abbr)
        if t:
            return t
        return CachedTeam(abbreviation=abbr.upper(), full_name=full_name or abbr)

    def get_stale(self) -> list[CachedTeam]:
        return [t for t in self.teams.values() if t.is_stale]


# ─── Layer 3: Player Cache ───

@dataclass
class CachedPlayer:
    name: str
    team: str
    impact_rating: float = 3.0
    position: str = ""
    injury_status: str = "healthy"   # healthy | out | questionable | doubtful
    injury_detail: str = ""
    games_played: int = 0
    ppg: float = 0.0
    rpg: float = 0.0
    apg: float = 0.0
    updated_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> CachedPlayer:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @property
    def is_available(self) -> bool:
        return self.injury_status in ("healthy", "probable")


class PlayerCache:
    def __init__(self, path: str = f"{CACHE_DIR}/players.json"):
        self.path = path
        self.players: dict[str, CachedPlayer] = {}
        self._load()

    def _load(self) -> None:
        data = _load_json(self.path)
        if isinstance(data, list):
            for d in data:
                p = CachedPlayer.from_dict(d)
                key = f"{p.team}:{p.name}"
                self.players[key] = p

    def save(self) -> None:
        _save_json(self.path, [p.to_dict() for p in self.players.values()])

    def upsert(self, player: CachedPlayer) -> None:
        player.updated_at = _now_iso()
        key = f"{player.team}:{player.name}"
        self.players[key] = player
        self.save()

    def upsert_batch(self, players: list[CachedPlayer]) -> None:
        for p in players:
            p.updated_at = _now_iso()
            key = f"{p.team}:{p.name}"
            self.players[key] = p
        self.save()

    def get(self, team: str, name: str) -> CachedPlayer | None:
        return self.players.get(f"{team}:{name}")

    def get_team_injuries(self, team: str) -> list[CachedPlayer]:
        return [p for p in self.players.values()
                if p.team == team and not p.is_available]

    def get_team_roster(self, team: str) -> list[CachedPlayer]:
        return [p for p in self.players.values() if p.team == team]


# ─── Layer 4: Orderbook History ───

@dataclass
class OrderbookSnap:
    slug: str
    timestamp: str
    best_bid: float
    best_ask: float
    mid: float
    spread: float
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    home_price: float = 0.0
    away_price: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class OrderbookHistory:
    def __init__(self, base_dir: str = f"{CACHE_DIR}/orderbooks"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def append(self, snap: OrderbookSnap) -> None:
        path = os.path.join(self.base_dir, f"{snap.slug}.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(snap.to_dict()) + "\n")

    def get_history(self, slug: str, limit: int = 100) -> list[OrderbookSnap]:
        path = os.path.join(self.base_dir, f"{slug}.jsonl")
        if not os.path.exists(path):
            return []
        snaps = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    snaps.append(OrderbookSnap(**json.loads(line)))
        return snaps[-limit:]

    def get_latest(self, slug: str) -> OrderbookSnap | None:
        history = self.get_history(slug, limit=1)
        return history[-1] if history else None


# ─── Layer 5: Feature Store ───

@dataclass
class GameFeatures:
    slug: str
    game_date: str
    timestamp: str
    home_team: str
    away_team: str
    # Inputs
    rating_diff: float = 0.0       # home_elo - away_elo
    win_pct_diff: float = 0.0      # home_wp - away_wp
    home_injury_impact: float = 0.0
    away_injury_impact: float = 0.0
    injury_diff: float = 0.0       # away_impact - home_impact (positive = home advantage)
    rest_diff: int = 0             # home_rest - away_rest
    home_court: float = 0.035
    momentum_diff: float = 0.0    # home_streak - away_streak
    market_skew: float = 0.0      # fair - market price
    # Model outputs
    fair_home_prob: float = 0.0
    market_home_prob: float = 0.0
    edge_home_bps: float = 0.0
    edge_away_bps: float = 0.0
    net_edge_bps: float = 0.0
    signal_strength: str = "none"
    # Outcomes (filled after game)
    traded: bool = False
    confirmed: bool = False
    filled: bool = False
    actual_winner: str = ""        # "home" | "away"
    pnl: float = 0.0
    was_correct: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> GameFeatures:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class FeatureStore:
    def __init__(self, base_dir: str = "data/features"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def save_features(self, features: GameFeatures) -> None:
        date_str = features.game_date.replace("-", "")
        path = os.path.join(self.base_dir, f"features_{date_str}.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(features.to_dict()) + "\n")

    def load_date(self, game_date: str) -> list[GameFeatures]:
        date_str = game_date.replace("-", "")
        path = os.path.join(self.base_dir, f"features_{date_str}.jsonl")
        if not os.path.exists(path):
            return []
        results = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    results.append(GameFeatures.from_dict(json.loads(line)))
        return results

    def load_recent(self, n_days: int = 30) -> list[GameFeatures]:
        """Load features from the last N days."""
        from datetime import timedelta
        results = []
        for i in range(n_days):
            d = (date.today() - timedelta(days=i)).isoformat()
            results.extend(self.load_date(d))
        return results

    def update_outcome(self, slug: str, game_date: str,
                       winner: str, pnl: float) -> None:
        """Update a game's features with actual outcome."""
        features = self.load_date(game_date)
        date_str = game_date.replace("-", "")
        path = os.path.join(self.base_dir, f"features_{date_str}.jsonl")
        updated = []
        for f in features:
            if f.slug == slug:
                f.actual_winner = winner
                f.pnl = pnl
                f.was_correct = (
                    (winner == "home" and f.edge_home_bps > 0) or
                    (winner == "away" and f.edge_away_bps > 0)
                )
            updated.append(f)
        with open(path, "w") as fp:
            for f in updated:
                fp.write(json.dumps(f.to_dict()) + "\n")
