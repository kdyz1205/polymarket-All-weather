"""
Data sync — auto-update team records, player injuries, and ratings.

Reads the MarketCache to know which teams play today, then updates:
  - TeamCache: win/loss records, streaks, conference standings
  - PlayerCache: injury status, impact ratings
  - OrderbookHistory: latest price snapshots

Sources (in priority order):
  1. Polymarket gamma API event metadata (always available)
  2. Manual overrides in config/team_overrides.json
  3. Future: balldontlie.io, ESPN, NBA.com APIs

Usage:
  python -m src.data.data_sync              # one-shot sync
  python -m src.data.data_sync --loop 3600  # every hour
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import date, datetime
from typing import Any

import requests

from src.data.cache import (
    MarketCache, CachedMarket,
    TeamCache, CachedTeam,
    PlayerCache, CachedPlayer,
)
from src.data.nba_client import (
    NBADataClient, GameData, TeamGameData, PlayerInjury,
    estimate_player_impact, STAR_IMPACT,
)

logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"
OVERRIDES_PATH = "config/team_overrides.json"


class DataSyncer:
    """Syncs team and player data from available sources."""

    def __init__(self, timeout: float = 15.0):
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})
        self._timeout = timeout
        self._market_cache = MarketCache()
        self._team_cache = TeamCache()
        self._player_cache = PlayerCache()
        self._nba_client = NBADataClient(timeout=timeout)

    def sync_all(self) -> dict[str, int]:
        """Full data sync: teams + players for today's markets.

        Returns counts of updated items.
        """
        stats = {"teams": 0, "players": 0, "markets_processed": 0}

        today_markets = self._market_cache.get_today()
        if not today_markets:
            logger.info("No active markets today, nothing to sync")
            return stats

        # Collect unique teams from today's games
        teams_needed: set[str] = set()
        for m in today_markets:
            teams_needed.add(m.home_team)
            teams_needed.add(m.away_team)

        # Sync each market's team/player data
        for m in today_markets:
            try:
                t_count, p_count = self._sync_market_data(m)
                stats["teams"] += t_count
                stats["players"] += p_count
                stats["markets_processed"] += 1
            except Exception as e:
                logger.error("Failed to sync data for %s: %s", m.slug, e)

        # Apply manual overrides if present
        self._apply_overrides()

        return stats

    def sync_teams_only(self) -> int:
        """Sync only team records (lighter operation)."""
        today_markets = self._market_cache.get_today()
        count = 0
        for m in today_markets:
            try:
                game = self._fetch_game_data(m)
                if game:
                    self._update_team_from_game(game.home_team)
                    self._update_team_from_game(game.away_team)
                    count += 2
            except Exception as e:
                logger.warning("Team sync failed for %s: %s", m.slug, e)
        return count

    def sync_injuries(self) -> int:
        """Sync injury reports for today's teams."""
        today_markets = self._market_cache.get_today()
        count = 0
        for m in today_markets:
            try:
                game = self._fetch_game_data(m)
                if game:
                    count += self._update_players_from_game(game)
            except Exception as e:
                logger.warning("Injury sync failed for %s: %s", m.slug, e)
        return count

    def _sync_market_data(self, market: CachedMarket) -> tuple[int, int]:
        """Sync all data for a single market. Returns (teams_updated, players_updated)."""
        game = self._fetch_game_data(market)
        if not game:
            return 0, 0

        team_count = 0
        player_count = 0

        # Update teams
        self._update_team_from_game(game.home_team)
        self._update_team_from_game(game.away_team)
        team_count = 2

        # Update players/injuries
        player_count = self._update_players_from_game(game)

        return team_count, player_count

    def _fetch_game_data(self, market: CachedMarket) -> GameData | None:
        """Fetch full game data for a market from gamma API."""
        try:
            game = self._nba_client.fetch_from_polymarket(market.slug)
            if game:
                return game
        except Exception as e:
            logger.warning("API fetch failed for %s: %s", market.slug, e)

        # Build from cache data as fallback
        return self._build_game_from_cache(market)

    def _build_game_from_cache(self, market: CachedMarket) -> GameData:
        """Build a minimal GameData from cached market info."""
        home_team_cache = self._team_cache.get_or_default(market.home_team)
        away_team_cache = self._team_cache.get_or_default(market.away_team)

        home = TeamGameData(
            abbreviation=market.home_team,
            full_name=home_team_cache.full_name or market.home_team,
            wins=home_team_cache.wins,
            losses=home_team_cache.losses,
            is_home=True,
        )
        away = TeamGameData(
            abbreviation=market.away_team,
            full_name=away_team_cache.full_name or market.away_team,
            wins=away_team_cache.wins,
            losses=away_team_cache.losses,
            is_home=False,
        )

        return GameData(
            game_id=market.slug,
            game_date=market.game_date,
            home_team=home,
            away_team=away,
            polymarket_slug=market.slug,
            polymarket_condition_id=market.condition_id,
            polymarket_home_token=market.home_token_id,
            polymarket_away_token=market.away_token_id,
            polymarket_home_price=market.home_price,
            polymarket_away_price=market.away_price,
        )

    def _update_team_from_game(self, team: TeamGameData) -> None:
        """Update TeamCache from game data."""
        existing = self._team_cache.get_or_default(team.abbreviation, team.full_name)

        existing.full_name = team.full_name or existing.full_name
        if team.wins > 0 or team.losses > 0:
            existing.wins = team.wins
            existing.losses = team.losses
        existing.offensive_rating = team.offensive_rating or existing.offensive_rating
        existing.defensive_rating = team.defensive_rating or existing.defensive_rating
        existing.pace = team.pace or existing.pace
        if team.streak != 0:
            existing.streak = team.streak
        if team.last5_record:
            existing.last5 = team.last5_record

        self._team_cache.upsert(existing)

    def _update_players_from_game(self, game: GameData) -> int:
        """Update PlayerCache with injury data from game. Returns count updated."""
        count = 0

        for team_data in [game.home_team, game.away_team]:
            for injury in team_data.injuries:
                player = CachedPlayer(
                    name=injury.name,
                    team=team_data.abbreviation,
                    impact_rating=injury.impact_rating,
                    injury_status=injury.status,
                    injury_detail=injury.injury,
                )
                self._player_cache.upsert(player)
                count += 1

            # Mark players NOT on injury list as healthy
            # (only if we have injury data for this team)
            if team_data.injuries:
                injured_names = {inj.name for inj in team_data.injuries}
                roster = self._player_cache.get_team_roster(team_data.abbreviation)
                for p in roster:
                    if p.name not in injured_names and p.injury_status != "healthy":
                        p.injury_status = "healthy"
                        p.injury_detail = ""
                        self._player_cache.upsert(p)
                        count += 1

        return count

    def _apply_overrides(self) -> None:
        """Apply manual team/player overrides from config file."""
        if not os.path.exists(OVERRIDES_PATH):
            return

        try:
            with open(OVERRIDES_PATH) as f:
                overrides = json.load(f)
        except (json.JSONDecodeError, OSError):
            return

        # Team overrides
        for team_data in overrides.get("teams", []):
            abbr = team_data.get("abbreviation", "")
            if not abbr:
                continue
            existing = self._team_cache.get_or_default(abbr)
            for k, v in team_data.items():
                if hasattr(existing, k) and k != "abbreviation":
                    setattr(existing, k, v)
            self._team_cache.upsert(existing)

        # Player overrides
        for player_data in overrides.get("players", []):
            name = player_data.get("name", "")
            team = player_data.get("team", "")
            if not name or not team:
                continue
            existing = self._player_cache.get(team, name)
            if existing:
                for k, v in player_data.items():
                    if hasattr(existing, k):
                        setattr(existing, k, v)
                self._player_cache.upsert(existing)
            else:
                self._player_cache.upsert(CachedPlayer(
                    name=name,
                    team=team,
                    impact_rating=player_data.get("impact_rating", estimate_player_impact(name)),
                    injury_status=player_data.get("injury_status", "healthy"),
                    injury_detail=player_data.get("injury_detail", ""),
                ))

    def get_team_cache(self) -> TeamCache:
        return self._team_cache

    def get_player_cache(self) -> PlayerCache:
        return self._player_cache

    def close(self) -> None:
        self._session.close()
        self._nba_client.close()


def run_sync(loop_sec: int = 0) -> None:
    """Run data sync (one-shot or loop)."""
    syncer = DataSyncer()

    while True:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n  [{ts}] Data sync starting...")

        try:
            stats = syncer.sync_all()
            print(f"    Markets processed: {stats['markets_processed']}")
            print(f"    Teams updated:     {stats['teams']}")
            print(f"    Players updated:   {stats['players']}")

            # Show team cache summary
            team_cache = syncer.get_team_cache()
            stale_teams = team_cache.get_stale()
            print(f"    Team cache: {len(team_cache.teams)} teams "
                  f"({len(stale_teams)} stale)")

            # Show injury summary
            player_cache = syncer.get_player_cache()
            market_cache = MarketCache()
            for m in market_cache.get_today():
                h_inj = player_cache.get_team_injuries(m.home_team)
                a_inj = player_cache.get_team_injuries(m.away_team)
                if h_inj or a_inj:
                    print(f"    {m.away_team}@{m.home_team}: "
                          f"{len(h_inj)} home injuries, {len(a_inj)} away injuries")
                    for p in h_inj + a_inj:
                        print(f"      {p.name} ({p.team}) — {p.injury_status} "
                              f"impact={p.impact_rating:.1f}")

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()

        if loop_sec <= 0:
            break

        print(f"    Next sync in {loop_sec}s...")
        time.sleep(loop_sec)

    syncer.close()


if __name__ == "__main__":
    import sys
    loop = 0
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--loop" and i < len(sys.argv) - 1:
            loop = int(sys.argv[i + 1])

    run_sync(loop_sec=loop)
