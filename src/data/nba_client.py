"""
NBA data client — fetches real game data from multiple sources.

Sources:
  1. Polymarket gamma API (game context, injuries, records from event metadata)
  2. Manual game data input (for when APIs are unavailable)
  3. NBA schedule/scores (extensible to balldontlie.io, nba_api, ESPN)

This module provides the raw data that RealGamePricer needs to
generate fair probabilities for real NBA games.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Any

import requests

logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"


@dataclass
class PlayerInjury:
    """A single player injury entry."""
    name: str
    team: str               # team abbreviation
    status: str              # "out" | "questionable" | "doubtful" | "probable"
    injury: str = ""         # "torn ACL", "right calf strain", etc.
    impact_rating: float = 0.0  # estimated impact 0-10 (star=8-10, starter=4-7, bench=1-3)


@dataclass
class TeamGameData:
    """Everything we know about one team going into a game."""
    abbreviation: str        # "MIN", "IND"
    full_name: str           # "Timberwolves", "Pacers"
    wins: int = 0
    losses: int = 0
    is_home: bool = False
    injuries: list[PlayerInjury] = field(default_factory=list)

    # Advanced (filled if available)
    offensive_rating: float = 0.0    # points per 100 possessions
    defensive_rating: float = 0.0
    pace: float = 0.0               # possessions per game
    rest_days: int = 1               # days since last game
    streak: int = 0                  # positive = win streak, negative = loss streak
    last5_record: str = ""           # "3-2"

    @property
    def win_pct(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.5

    @property
    def total_injury_impact(self) -> float:
        """Sum of impact ratings for players who are OUT."""
        return sum(p.impact_rating for p in self.injuries if p.status == "out")


@dataclass
class GameData:
    """Complete data for a single NBA game."""
    game_id: str
    game_date: str           # "2026-04-07"
    home_team: TeamGameData
    away_team: TeamGameData
    game_time: str = ""      # "7:00PM ET"
    venue: str = ""
    polymarket_slug: str = ""
    polymarket_condition_id: str = ""
    polymarket_home_token: str = ""  # token for home team win
    polymarket_away_token: str = ""  # token for away team win
    polymarket_home_price: float = 0.0  # current market price for home
    polymarket_away_price: float = 0.0

    # Context
    notes: str = ""          # free-form analyst notes

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Star player impact ratings ───
# These are rough estimates of how much a player's absence
# affects their team's win probability. Scale 0-10.

STAR_IMPACT = {
    # MVP-caliber
    "anthony edwards": 9.0, "luka doncic": 9.5, "nikola jokic": 9.5,
    "giannis antetokounmpo": 9.5, "shai gilgeous-alexander": 9.5,
    "jayson tatum": 9.0, "stephen curry": 9.0, "lebron james": 8.5,
    "kevin durant": 9.0, "joel embiid": 9.0, "damian lillard": 8.0,
    "donovan mitchell": 8.0, "devin booker": 8.5, "ja morant": 8.0,
    "jimmy butler": 8.0, "trae young": 8.0, "lamelo ball": 7.5,
    "paolo banchero": 7.5, "tyrese haliburton": 8.0, "de'aaron fox": 8.0,
    "cade cunningham": 7.5, "victor wembanyama": 9.0, "ant man": 9.0,

    # All-star level
    "pascal siakam": 7.5, "domantas sabonis": 7.5, "bam adebayo": 7.5,
    "karl-anthony towns": 7.5, "rudy gobert": 6.5, "jalen brunson": 8.0,
    "tyrese maxey": 7.0, "scottie barnes": 7.5, "evan mobley": 7.0,
    "franz wagner": 7.5, "alperen sengun": 7.0, "mikal bridges": 6.5,
    "jaden mcdaniels": 5.5, "andrew nembhard": 5.0,
    "t.j. mcconnell": 4.5, "johnny furphy": 3.0,
    "benedict mathurin": 5.5, "myles turner": 6.5,
    "aaron nesmith": 4.0, "obi toppin": 4.5,
}


def estimate_player_impact(name: str) -> float:
    """Estimate a player's impact rating from the lookup table."""
    key = name.lower().strip()
    # Try exact match first
    if key in STAR_IMPACT:
        return STAR_IMPACT[key]
    # Try partial match
    for k, v in STAR_IMPACT.items():
        if key in k or k in key:
            return v
    # Unknown player: assume role player
    return 3.0


class NBADataClient:
    """Fetches and assembles game data from available sources."""

    def __init__(self, timeout: float = 10.0):
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})
        self._timeout = timeout

    def fetch_from_polymarket(self, slug: str) -> GameData | None:
        """Fetch game data from Polymarket gamma API.

        The gamma API event metadata contains team records, injuries,
        and contextual analysis that we can parse.
        """
        # Try events endpoint first (richer data)
        data = self._fetch_gamma_events(slug)
        if not data:
            data = self._fetch_gamma_markets(slug)
        if not data:
            return None

        return self._parse_polymarket_data(data, slug)

    def _fetch_gamma_events(self, slug: str) -> dict | None:
        try:
            resp = self._session.get(
                f"{GAMMA_API}/events",
                params={"slug": slug},
                timeout=self._timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data and isinstance(data, list) and len(data) > 0:
                    return data[0]
        except Exception as e:
            logger.warning("gamma events fetch failed: %s", e)
        return None

    def _fetch_gamma_markets(self, slug: str) -> dict | None:
        try:
            resp = self._session.get(
                f"{GAMMA_API}/markets",
                params={"slug": slug},
                timeout=self._timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data and isinstance(data, list) and len(data) > 0:
                    return data[0]
        except Exception as e:
            logger.warning("gamma markets fetch failed: %s", e)
        return None

    def _parse_polymarket_data(self, data: dict, slug: str) -> GameData:
        """Parse Polymarket API response into structured GameData."""
        # Extract team names from slug: nba-min-ind-2026-04-07
        parts = slug.split("-")
        away_abbr = parts[1].upper() if len(parts) > 1 else ""
        home_abbr = parts[2].upper() if len(parts) > 2 else ""
        game_date = "-".join(parts[3:6]) if len(parts) >= 6 else ""

        # Get market data (may be nested in events or direct)
        markets = data.get("markets", [data])
        if isinstance(markets, list) and len(markets) > 0:
            market = markets[0] if isinstance(markets[0], dict) else data
        else:
            market = data

        # Outcomes and prices
        outcomes = json.loads(market.get("outcomes", "[]")) if isinstance(market.get("outcomes"), str) else market.get("outcomes", [])
        prices = json.loads(market.get("outcomePrices", "[]")) if isinstance(market.get("outcomePrices"), str) else market.get("outcomePrices", [])
        tokens = json.loads(market.get("clobTokenIds", "[]")) if isinstance(market.get("clobTokenIds"), str) else market.get("clobTokenIds", [])

        # Determine which outcome is which team
        # outcomes = ["Timberwolves", "Pacers"] — first is usually away, second home
        away_name = outcomes[0] if len(outcomes) > 0 else away_abbr
        home_name = outcomes[1] if len(outcomes) > 1 else home_abbr
        away_price = float(prices[0]) if len(prices) > 0 else 0.5
        home_price = float(prices[1]) if len(prices) > 1 else 0.5
        away_token = tokens[0] if len(tokens) > 0 else ""
        home_token = tokens[1] if len(tokens) > 1 else ""

        # Parse event metadata for context
        context = ""
        events = data.get("events", [])
        if events and isinstance(events, list):
            evt = events[0] if isinstance(events[0], dict) else {}
            meta = evt.get("eventMetadata", {})
            context = meta.get("context_description", "")
        elif "eventMetadata" in data:
            context = data["eventMetadata"].get("context_description", "")

        # Extract records from context
        home_wins, home_losses = self._extract_record(context, home_name)
        away_wins, away_losses = self._extract_record(context, away_name)

        # Extract injuries from context
        home_injuries = self._extract_injuries(context, home_name, home_abbr)
        away_injuries = self._extract_injuries(context, away_name, away_abbr)

        home_team = TeamGameData(
            abbreviation=home_abbr,
            full_name=home_name,
            wins=home_wins, losses=home_losses,
            is_home=True,
            injuries=home_injuries,
        )
        away_team = TeamGameData(
            abbreviation=away_abbr,
            full_name=away_name,
            wins=away_wins, losses=away_losses,
            is_home=False,
            injuries=away_injuries,
        )

        return GameData(
            game_id=slug,
            game_date=game_date,
            home_team=home_team,
            away_team=away_team,
            polymarket_slug=slug,
            polymarket_condition_id=market.get("conditionId", ""),
            polymarket_home_token=home_token,
            polymarket_away_token=away_token,
            polymarket_home_price=home_price,
            polymarket_away_price=away_price,
            notes=context[:500] if context else "",
        )

    def _extract_record(self, context: str, team_name: str) -> tuple[int, int]:
        """Extract W-L record from context text."""
        if not context:
            return 0, 0
        # Look for patterns like "their 46-30 record" or "Indiana's 18-60"
        # or "46-30 record" near team name
        patterns = [
            rf"{re.escape(team_name)}.*?(\d{{1,2}})-(\d{{1,2}})\s*record",
            rf"(\d{{1,2}})-(\d{{1,2}})\s*record.*?{re.escape(team_name)}",
            rf"{re.escape(team_name)}.*?(\d{{1,2}})-(\d{{1,2}})\s*mark",
            rf"Indiana.*?(\d{{1,2}})-(\d{{1,2}})\s*mark" if "Pacer" in team_name else r"$^",
            rf"their\s+(\d{{1,2}})-(\d{{1,2}})\s*record",
        ]
        for pat in patterns:
            m = re.search(pat, context, re.IGNORECASE)
            if m:
                return int(m.group(1)), int(m.group(2))
        return 0, 0

    def _extract_injuries(self, context: str, team_name: str,
                          team_abbr: str) -> list[PlayerInjury]:
        """Extract injury information from context text."""
        injuries = []
        if not context:
            return injuries

        # Common patterns in Polymarket event metadata:
        # "Indiana listing nine players on the injury report—including
        #  questionable Tyrese Haliburton (right calf strain), out T.J. McConnell,
        #  Andrew Nembhard, and Johnny Furphy sidelined by a torn ACL"
        # "Minnesota counters without star Anthony Edwards (left knee, ruled out)"

        # Extract player names near injury keywords
        # This is a best-effort heuristic parser
        injury_section = ""
        lower = context.lower()

        # Try to find the section about this team's injuries
        team_refs = [team_name.lower(), team_abbr.lower()]
        for ref in team_refs:
            idx = lower.find(ref)
            if idx >= 0:
                # Take a window around the team mention
                start = max(0, idx - 50)
                end = min(len(context), idx + 500)
                injury_section += context[start:end] + " "

        if not injury_section:
            return injuries

        # Look for "out" players
        out_patterns = [
            r"(?:ruled out|out|sidelined)\s+(\w+\s+\w+(?:\s+\w+)?)",
            r"(\w+\s+\w+)\s+(?:\(.*?(?:ruled out|out|sidelined))",
            r"without\s+(?:star\s+)?(\w+\s+\w+)",
            r"missing\s+(?:star\s+)?(\w+\s+\w+)",
        ]

        found_names = set()
        for pat in out_patterns:
            for m in re.finditer(pat, injury_section, re.IGNORECASE):
                name = m.group(1).strip()
                # Filter out non-name words
                skip_words = {"the", "and", "their", "with", "from", "this", "that",
                              "injury", "report", "season", "game", "team"}
                words = name.split()
                if len(words) >= 2 and words[0].lower() not in skip_words:
                    if name not in found_names:
                        found_names.add(name)
                        injuries.append(PlayerInjury(
                            name=name,
                            team=team_abbr,
                            status="out",
                            impact_rating=estimate_player_impact(name),
                        ))

        # Look for "questionable" players
        q_patterns = [
            r"questionable\s+(\w+\s+\w+)",
            r"(\w+\s+\w+)\s*\(.*?questionable",
        ]
        for pat in q_patterns:
            for m in re.finditer(pat, injury_section, re.IGNORECASE):
                name = m.group(1).strip()
                words = name.split()
                if len(words) >= 2 and name not in found_names:
                    found_names.add(name)
                    injuries.append(PlayerInjury(
                        name=name,
                        team=team_abbr,
                        status="questionable",
                        impact_rating=estimate_player_impact(name) * 0.5,  # 50% weight for questionable
                    ))

        return injuries

    def build_game_manual(
        self,
        home_abbr: str, away_abbr: str,
        home_name: str, away_name: str,
        home_wins: int, home_losses: int,
        away_wins: int, away_losses: int,
        game_date: str,
        home_injuries: list[dict] | None = None,
        away_injuries: list[dict] | None = None,
        polymarket_slug: str = "",
        condition_id: str = "",
        home_token: str = "",
        away_token: str = "",
        home_price: float = 0.0,
        away_price: float = 0.0,
    ) -> GameData:
        """Build GameData from manually provided inputs."""
        h_inj = []
        if home_injuries:
            for inj in home_injuries:
                h_inj.append(PlayerInjury(
                    name=inj["name"], team=home_abbr,
                    status=inj.get("status", "out"),
                    injury=inj.get("injury", ""),
                    impact_rating=estimate_player_impact(inj["name"]),
                ))

        a_inj = []
        if away_injuries:
            for inj in away_injuries:
                a_inj.append(PlayerInjury(
                    name=inj["name"], team=away_abbr,
                    status=inj.get("status", "out"),
                    injury=inj.get("injury", ""),
                    impact_rating=estimate_player_impact(inj["name"]),
                ))

        return GameData(
            game_id=f"nba_{away_abbr.lower()}_{home_abbr.lower()}_ml",
            game_date=game_date,
            home_team=TeamGameData(
                abbreviation=home_abbr, full_name=home_name,
                wins=home_wins, losses=home_losses, is_home=True,
                injuries=h_inj,
            ),
            away_team=TeamGameData(
                abbreviation=away_abbr, full_name=away_name,
                wins=away_wins, losses=away_losses, is_home=False,
                injuries=a_inj,
            ),
            polymarket_slug=polymarket_slug,
            polymarket_condition_id=condition_id,
            polymarket_home_token=home_token,
            polymarket_away_token=away_token,
            polymarket_home_price=home_price,
            polymarket_away_price=away_price,
        )

    def close(self) -> None:
        self._session.close()
