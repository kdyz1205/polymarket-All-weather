"""
Real game pricer — generates fair win probabilities for actual NBA games
using team records, injuries, home court advantage, and contextual factors.

Model layers (applied sequentially):
  1. Base probability from Log5 method using season win percentages
  2. Home court adjustment (+3.0 to +4.5 percentage points)
  3. Injury impact adjustment (star player absence shifts probability)
  4. Rest day adjustment (back-to-back = slight penalty)
  5. Late-season / motivation adjustment (tanking, playoff seeding)

The output is a fair probability that can be compared to the Polymarket
market price to identify edge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from src.data.nba_client import GameData, TeamGameData


@dataclass
class PricingResult:
    """Full breakdown of a real game pricing."""
    game_id: str
    home_team: str
    away_team: str

    # Base model
    home_win_pct: float = 0.0
    away_win_pct: float = 0.0
    log5_home_prob: float = 0.0

    # Adjustments (each is a delta to home probability)
    home_court_adj: float = 0.0
    home_injury_adj: float = 0.0
    away_injury_adj: float = 0.0
    rest_adj: float = 0.0
    motivation_adj: float = 0.0

    # Final
    fair_home_prob: float = 0.0
    fair_away_prob: float = 0.0
    fair_home_odds: float = 0.0    # decimal odds
    fair_away_odds: float = 0.0

    # Market comparison
    market_home_price: float = 0.0
    market_away_price: float = 0.0
    edge_home_pct: float = 0.0     # fair - market (positive = we think home underpriced)
    edge_away_pct: float = 0.0
    edge_home_bps: float = 0.0
    edge_away_bps: float = 0.0

    # Confidence
    confidence: float = 0.0        # 0-1, how much we trust this estimate
    model_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            d[k] = v
        return d

    @property
    def has_edge(self) -> bool:
        """True if either side has meaningful edge (>1%)."""
        return abs(self.edge_home_pct) > 0.01 or abs(self.edge_away_pct) > 0.01

    @property
    def best_side(self) -> str:
        """Which side has the most edge."""
        if self.edge_home_pct > self.edge_away_pct:
            return "home"
        return "away"

    @property
    def best_edge_bps(self) -> float:
        return max(self.edge_home_bps, self.edge_away_bps)


# ─── Constants ───

# Home court advantage in NBA (percentage points added to home win prob)
# Historical average is ~3.5%, but varies by team/venue
HOME_COURT_BASE = 0.035

# How much a single impact point shifts win probability
# A 10-impact player (MVP) being out shifts ~5% points
INJURY_IMPACT_MULTIPLIER = 0.005

# Maximum total injury adjustment (cap to prevent absurd values)
MAX_INJURY_ADJ = 0.15

# Back-to-back penalty (reduce win prob by this amount)
B2B_PENALTY = 0.015

# Confidence base and adjustments
BASE_CONFIDENCE = 0.60
RECORD_CONFIDENCE_BONUS = 0.15  # if both teams have 50+ games played
INJURY_CONFIDENCE_PENALTY = 0.10  # if major injuries create uncertainty


class RealGamePricer:
    """Prices real NBA games using fundamental analysis.

    Usage:
        pricer = RealGamePricer()
        result = pricer.price(game_data)
        print(f"Fair: home {result.fair_home_prob:.1%}, edge: {result.edge_home_bps:.0f}bps")
    """

    def __init__(
        self,
        home_court_base: float = HOME_COURT_BASE,
        injury_multiplier: float = INJURY_IMPACT_MULTIPLIER,
        max_injury_adj: float = MAX_INJURY_ADJ,
    ):
        self.home_court_base = home_court_base
        self.injury_multiplier = injury_multiplier
        self.max_injury_adj = max_injury_adj

    def price(self, game: GameData) -> PricingResult:
        """Generate fair probability for a real NBA game."""
        home = game.home_team
        away = game.away_team

        result = PricingResult(
            game_id=game.game_id,
            home_team=f"{home.full_name} ({home.abbreviation})",
            away_team=f"{away.full_name} ({away.abbreviation})",
            home_win_pct=home.win_pct,
            away_win_pct=away.win_pct,
            market_home_price=game.polymarket_home_price,
            market_away_price=game.polymarket_away_price,
        )

        # ─── Layer 1: Log5 base probability ───
        base = self._log5(home.win_pct, away.win_pct)
        result.log5_home_prob = base
        result.model_notes.append(
            f"Log5 base: {base:.3f} (home {home.win_pct:.3f} vs away {away.win_pct:.3f})"
        )

        # ─── Layer 2: Home court advantage ───
        hca = self._home_court_adjustment(home, away)
        result.home_court_adj = hca
        result.model_notes.append(f"Home court: {hca:+.3f}")

        # ─── Layer 3: Injury adjustments ───
        # Home team injuries hurt home, away team injuries help home
        h_inj = self._injury_adjustment(home)
        a_inj = self._injury_adjustment(away)
        result.home_injury_adj = -h_inj  # home injuries reduce home prob
        result.away_injury_adj = a_inj   # away injuries increase home prob
        result.model_notes.append(
            f"Home injuries: {-h_inj:+.3f} ({len(home.injuries)} players)"
        )
        result.model_notes.append(
            f"Away injuries: {a_inj:+.3f} ({len(away.injuries)} players)"
        )

        # ─── Layer 4: Rest adjustment ───
        rest = self._rest_adjustment(home, away)
        result.rest_adj = rest

        # ─── Layer 5: Motivation / tanking ───
        motivation = self._motivation_adjustment(home, away)
        result.motivation_adj = motivation
        if abs(motivation) > 0.001:
            result.model_notes.append(f"Motivation adj: {motivation:+.3f}")

        # ─── Combine ───
        fair_home = base + hca + (-h_inj) + a_inj + rest + motivation
        # Clamp to [0.02, 0.98]
        fair_home = max(0.02, min(0.98, fair_home))
        fair_away = 1.0 - fair_home

        result.fair_home_prob = fair_home
        result.fair_away_prob = fair_away
        result.fair_home_odds = 1.0 / fair_home if fair_home > 0 else 99.0
        result.fair_away_odds = 1.0 / fair_away if fair_away > 0 else 99.0

        # ─── Market comparison ───
        if game.polymarket_home_price > 0:
            result.edge_home_pct = fair_home - game.polymarket_home_price
            result.edge_home_bps = result.edge_home_pct * 10000
        if game.polymarket_away_price > 0:
            result.edge_away_pct = fair_away - game.polymarket_away_price
            result.edge_away_bps = result.edge_away_pct * 10000

        # ─── Confidence ───
        result.confidence = self._compute_confidence(home, away)

        return result

    def _log5(self, home_wp: float, away_wp: float) -> float:
        """Log5 method: fair probability of team A beating team B.

        P(A) = (pA - pA*pB) / (pA + pB - 2*pA*pB)

        This is the standard method for head-to-head probability
        from two teams' overall win percentages.
        """
        pa = max(0.01, min(0.99, home_wp))
        pb = max(0.01, min(0.99, away_wp))

        numerator = pa - pa * pb
        denominator = pa + pb - 2 * pa * pb

        if abs(denominator) < 0.001:
            return 0.5

        return numerator / denominator

    def _home_court_adjustment(self, home: TeamGameData,
                                away: TeamGameData) -> float:
        """Home court advantage adjustment.

        Base is ~3.5% for average teams. Adjusted for:
        - Strong home teams get slightly more
        - Bad home teams (tanking) get less
        """
        if home.win_pct < 0.25:
            # Very bad team: home court means less (empty arena, no crowd energy)
            return self.home_court_base * 0.6
        elif home.win_pct > 0.65:
            # Elite team at home: slight boost
            return self.home_court_base * 1.2
        return self.home_court_base

    def _injury_adjustment(self, team: TeamGameData) -> float:
        """Calculate win probability shift from injuries.

        Each player has an impact_rating (0-10). The adjustment is:
        sum(impact * multiplier) capped at MAX_INJURY_ADJ.

        Diminishing returns: the 3rd missing starter hurts less than the 1st
        because the team has already adjusted.
        """
        if not team.injuries:
            return 0.0

        total_impact = 0.0
        sorted_injuries = sorted(
            team.injuries, key=lambda p: p.impact_rating, reverse=True
        )

        for i, player in enumerate(sorted_injuries):
            # Diminishing returns: each subsequent injury has less marginal impact
            diminishing = 1.0 / (1.0 + 0.15 * i)
            total_impact += player.impact_rating * diminishing

        adj = total_impact * self.injury_multiplier
        return min(adj, self.max_injury_adj)

    def _rest_adjustment(self, home: TeamGameData,
                          away: TeamGameData) -> float:
        """Rest day advantage/disadvantage."""
        adj = 0.0
        if home.rest_days == 0:  # back-to-back
            adj -= B2B_PENALTY
        if away.rest_days == 0:
            adj += B2B_PENALTY
        return adj

    def _motivation_adjustment(self, home: TeamGameData,
                                away: TeamGameData) -> float:
        """Late-season motivation adjustment.

        Teams that are clearly tanking (very bad record late in season)
        may perform worse than their record suggests. Playoff-bound teams
        may rest starters in meaningless games.
        """
        adj = 0.0

        # Tanking detection: very bad record
        if home.win_pct < 0.25 and (home.wins + home.losses) > 60:
            adj -= 0.02  # likely tanking, perform worse
        if away.win_pct < 0.25 and (away.wins + away.losses) > 60:
            adj += 0.02

        return adj

    def _compute_confidence(self, home: TeamGameData,
                             away: TeamGameData) -> float:
        """Estimate confidence in our probability estimate."""
        conf = BASE_CONFIDENCE

        # More games = more reliable win percentages
        home_games = home.wins + home.losses
        away_games = away.wins + away.losses
        if home_games >= 50 and away_games >= 50:
            conf += RECORD_CONFIDENCE_BONUS
        elif home_games < 20 or away_games < 20:
            conf -= 0.10

        # Major injuries reduce confidence (harder to predict)
        total_impact = home.total_injury_impact + away.total_injury_impact
        if total_impact > 15:
            conf -= INJURY_CONFIDENCE_PENALTY

        # Very lopsided matchups: higher confidence
        prob_diff = abs(home.win_pct - away.win_pct)
        if prob_diff > 0.3:
            conf += 0.05

        return max(0.2, min(0.95, conf))

    def display_pricing(self, result: PricingResult) -> str:
        """Format pricing result for terminal display."""
        lines = []
        lines.append(f"\n{'='*70}")
        lines.append(f"  REAL GAME PRICING")
        lines.append(f"  {result.away_team} @ {result.home_team}")
        lines.append(f"{'='*70}")

        lines.append(f"\n  MODEL BREAKDOWN:")
        for note in result.model_notes:
            lines.append(f"    {note}")

        lines.append(f"\n  FAIR PROBABILITY:")
        lines.append(f"    Home ({result.home_team[:15]}): {result.fair_home_prob:.1%}  "
                      f"(odds {result.fair_home_odds:.2f})")
        lines.append(f"    Away ({result.away_team[:15]}): {result.fair_away_prob:.1%}  "
                      f"(odds {result.fair_away_odds:.2f})")

        if result.market_home_price > 0:
            lines.append(f"\n  MARKET PRICE (Polymarket):")
            lines.append(f"    Home: {result.market_home_price:.1%}  "
                          f"(odds {1/result.market_home_price:.2f})")
            lines.append(f"    Away: {result.market_away_price:.1%}  "
                          f"(odds {1/result.market_away_price:.2f})")

            lines.append(f"\n  EDGE:")
            h_icon = "+" if result.edge_home_bps > 0 else ""
            a_icon = "+" if result.edge_away_bps > 0 else ""
            lines.append(f"    Home: {h_icon}{result.edge_home_bps:.0f}bps "
                          f"({h_icon}{result.edge_home_pct:.1%})")
            lines.append(f"    Away: {a_icon}{result.edge_away_bps:.0f}bps "
                          f"({a_icon}{result.edge_away_pct:.1%})")

            if result.has_edge:
                best = result.best_side
                edge = result.edge_home_bps if best == "home" else result.edge_away_bps
                lines.append(f"\n    >>> EDGE DETECTED: {best.upper()} {edge:+.0f}bps")
            else:
                lines.append(f"\n    No actionable edge (all <100bps)")

        lines.append(f"\n  Confidence: {result.confidence:.0%}")
        lines.append(f"{'='*70}")

        return "\n".join(lines)
