"""
Market sync — auto-discover and cache today's Polymarket sports markets.

Fetches NBA (and optionally MLB) markets from the Polymarket gamma API,
parses slug/teams/tokens/prices, writes to MarketCache.

Usage:
  python -m src.data.market_sync              # one-shot sync
  python -m src.data.market_sync --loop 300   # every 5 minutes
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import date, datetime
from typing import Any

import requests

from src.data.cache import (
    MarketCache, CachedMarket,
    OrderbookHistory, OrderbookSnap,
)

logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"

# Slug patterns for supported sports
SPORT_PATTERNS = {
    "basketball": re.compile(r"^nba-(\w+)-(\w+)-(\d{4}-\d{2}-\d{2})$"),
    "baseball":   re.compile(r"^mlb-(\w+)-(\w+)-(\d{4}-\d{2}-\d{2})$"),
}

# Common NBA team abbreviations for validation
NBA_TEAMS = {
    "atl", "bos", "bkn", "cha", "chi", "cle", "dal", "den", "det", "gsw",
    "hou", "ind", "lac", "lal", "mem", "mia", "mil", "min", "nop", "nyk",
    "okc", "orl", "phi", "phx", "por", "sac", "sas", "tor", "uta", "was",
}


class MarketSyncer:
    """Discovers and syncs Polymarket sports markets to local cache."""

    def __init__(self, timeout: float = 15.0):
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})
        self._timeout = timeout
        self._cache = MarketCache()
        self._ob_history = OrderbookHistory()

    def sync_today(self, sports: list[str] | None = None) -> list[CachedMarket]:
        """Discover and cache all markets for today.

        Returns list of new/updated CachedMarket objects.
        """
        sports = sports or ["basketball"]
        today = date.today().isoformat()
        all_markets: list[CachedMarket] = []

        for sport in sports:
            try:
                markets = self._fetch_sport_markets(sport, today)
                all_markets.extend(markets)
                logger.info("Synced %d %s markets for %s", len(markets), sport, today)
            except Exception as e:
                logger.error("Failed to sync %s markets: %s", sport, e)

        if all_markets:
            self._cache.upsert_batch(all_markets)

        return all_markets

    def sync_stale(self) -> list[CachedMarket]:
        """Re-fetch prices for stale markets (>10 minutes old)."""
        stale = self._cache.get_stale()
        if not stale:
            return []

        updated: list[CachedMarket] = []
        for m in stale:
            try:
                refreshed = self._refresh_market(m)
                if refreshed:
                    updated.append(refreshed)
            except Exception as e:
                logger.warning("Failed to refresh %s: %s", m.slug, e)

        if updated:
            self._cache.upsert_batch(updated)

        return updated

    def _fetch_sport_markets(self, sport: str, game_date: str) -> list[CachedMarket]:
        """Fetch all markets for a sport on a given date from gamma API."""
        markets: list[CachedMarket] = []

        # Search for active markets
        tag = "nba" if sport == "basketball" else "mlb" if sport == "baseball" else sport
        league = "NBA" if sport == "basketball" else "MLB" if sport == "baseball" else sport.upper()

        try:
            resp = self._session.get(
                f"{GAMMA_API}/events",
                params={
                    "tag": tag,
                    "active": "true",
                    "closed": "false",
                },
                timeout=self._timeout,
            )
            resp.raise_for_status()
            events = resp.json()
        except Exception as e:
            logger.error("gamma API events fetch failed: %s", e)
            # Fallback: try markets endpoint
            return self._fetch_markets_fallback(sport, game_date)

        if not isinstance(events, list):
            return markets

        for event in events:
            parsed = self._parse_event(event, sport, league, game_date)
            if parsed:
                markets.append(parsed)

        return markets

    def _fetch_markets_fallback(self, sport: str, game_date: str) -> list[CachedMarket]:
        """Fallback: search markets endpoint directly."""
        markets: list[CachedMarket] = []
        tag = "nba" if sport == "basketball" else "mlb"
        league = "NBA" if sport == "basketball" else "MLB"

        try:
            resp = self._session.get(
                f"{GAMMA_API}/markets",
                params={
                    "tag": tag,
                    "active": "true",
                    "closed": "false",
                },
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error("gamma markets fallback failed: %s", e)
            return markets

        if not isinstance(data, list):
            return markets

        for mkt in data:
            slug = mkt.get("slug", "")
            parsed = self._parse_market_data(mkt, slug, sport, league, game_date)
            if parsed:
                markets.append(parsed)

        return markets

    def _parse_event(self, event: dict, sport: str, league: str,
                     game_date: str) -> CachedMarket | None:
        """Parse a gamma API event into a CachedMarket."""
        slug = event.get("slug", "")
        pattern = SPORT_PATTERNS.get(sport)

        if pattern:
            m = pattern.match(slug)
            if not m:
                return None
            away_abbr = m.group(1).upper()
            home_abbr = m.group(2).upper()
            mkt_date = m.group(3)
        else:
            return None

        # Filter to target date
        if mkt_date != game_date:
            return None

        # Get the moneyline market from nested markets
        event_markets = event.get("markets", [])
        if not event_markets:
            return None

        mkt = event_markets[0] if isinstance(event_markets[0], dict) else {}

        return self._build_cached_market(
            mkt, slug, sport, league, away_abbr, home_abbr, mkt_date,
            context=event.get("description", ""),
        )

    def _parse_market_data(self, mkt: dict, slug: str, sport: str,
                           league: str, game_date: str) -> CachedMarket | None:
        """Parse a flat gamma market dict into a CachedMarket."""
        pattern = SPORT_PATTERNS.get(sport)
        if not pattern:
            return None

        m = pattern.match(slug)
        if not m:
            return None

        away_abbr = m.group(1).upper()
        home_abbr = m.group(2).upper()
        mkt_date = m.group(3)

        if mkt_date != game_date:
            return None

        return self._build_cached_market(
            mkt, slug, sport, league, away_abbr, home_abbr, mkt_date,
        )

    def _build_cached_market(
        self, mkt: dict, slug: str, sport: str, league: str,
        away_abbr: str, home_abbr: str, game_date: str,
        context: str = "",
    ) -> CachedMarket:
        """Build a CachedMarket from parsed market data."""
        # Parse JSON-encoded fields
        outcomes = _parse_json_field(mkt.get("outcomes", "[]"))
        prices = _parse_json_field(mkt.get("outcomePrices", "[]"))
        tokens = _parse_json_field(mkt.get("clobTokenIds", "[]"))

        # First outcome = away, second = home (Polymarket convention)
        away_price = float(prices[0]) if len(prices) > 0 else 0.0
        home_price = float(prices[1]) if len(prices) > 1 else 0.0
        away_token = tokens[0] if len(tokens) > 0 else ""
        home_token = tokens[1] if len(tokens) > 1 else ""

        condition_id = mkt.get("conditionId", "")
        volume = float(mkt.get("volume", 0) or 0)
        liquidity = float(mkt.get("liquidity", 0) or 0)

        return CachedMarket(
            slug=slug,
            sport=sport,
            league=league,
            home_team=home_abbr,
            away_team=away_abbr,
            game_date=game_date,
            condition_id=condition_id,
            home_token_id=home_token,
            away_token_id=away_token,
            outcomes=outcomes if isinstance(outcomes, list) else [],
            home_price=home_price,
            away_price=away_price,
            volume=volume,
            liquidity=liquidity,
            status="active",
            context=context[:500] if context else "",
        )

    def _refresh_market(self, market: CachedMarket) -> CachedMarket | None:
        """Refresh a single market's prices from the gamma API."""
        try:
            resp = self._session.get(
                f"{GAMMA_API}/markets",
                params={"slug": market.slug},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("refresh failed for %s: %s", market.slug, e)
            return None

        if not data or not isinstance(data, list) or len(data) == 0:
            return None

        mkt = data[0]
        prices = _parse_json_field(mkt.get("outcomePrices", "[]"))

        market.away_price = float(prices[0]) if len(prices) > 0 else market.away_price
        market.home_price = float(prices[1]) if len(prices) > 1 else market.home_price
        market.volume = float(mkt.get("volume", 0) or 0)
        market.liquidity = float(mkt.get("liquidity", 0) or 0)

        # Snapshot orderbook prices
        self._ob_history.append(OrderbookSnap(
            slug=market.slug,
            timestamp=datetime.now().isoformat(),
            best_bid=market.home_price - 0.005,
            best_ask=market.home_price + 0.005,
            mid=market.home_price,
            spread=0.01,
            home_price=market.home_price,
            away_price=market.away_price,
        ))

        return market

    def get_cache(self) -> MarketCache:
        return self._cache

    def close(self) -> None:
        self._session.close()


def _parse_json_field(val: Any) -> list:
    """Parse a possibly JSON-encoded string into a list."""
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, ValueError):
            return []
    if isinstance(val, list):
        return val
    return []


def run_sync(sports: list[str] | None = None, loop_sec: int = 0) -> None:
    """Run market sync (one-shot or loop)."""
    syncer = MarketSyncer()
    sports = sports or ["basketball"]

    while True:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n  [{ts}] Market sync starting...")

        try:
            new = syncer.sync_today(sports=sports)
            print(f"    Discovered {len(new)} markets for today")

            stale = syncer.sync_stale()
            if stale:
                print(f"    Refreshed {len(stale)} stale markets")

            cache = syncer.get_cache()
            today_markets = cache.get_today()
            print(f"    Total active today: {len(today_markets)}")

            for m in today_markets:
                age = f"{m.age_minutes:.0f}m" if m.age_minutes < 9999 else "new"
                print(f"      {m.away_team}@{m.home_team} "
                      f"H={m.home_price:.3f} A={m.away_price:.3f} "
                      f"vol=${m.volume:.0f} age={age}")

        except Exception as e:
            print(f"    ERROR: {e}")

        if loop_sec <= 0:
            break

        print(f"    Next sync in {loop_sec}s...")
        time.sleep(loop_sec)

    syncer.close()


if __name__ == "__main__":
    import sys
    loop = 0
    sports = ["basketball"]

    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--loop" and i < len(sys.argv) - 1:
            loop = int(sys.argv[i + 1])
        elif arg == "--mlb":
            sports.append("baseball")

    run_sync(sports=sports, loop_sec=loop)
