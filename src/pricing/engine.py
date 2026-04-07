"""
Pricing engine — layered probability model for sports trading.

Section 6-7 of the blueprint:
  Layer 1: Pre-match prior (Elo/rating based)
  Layer 2: Time decay (clock drift)
  Layer 3: Event shock (score changes, fouls, ejections)
  Layer 4: Market calibration (anchor to exchange-implied probs)

Supports basketball, baseball, and football.
The model outputs a FairState (p_home, p_away, p_draw) at every tick.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence


# ============================================================
# Layer 1: Pre-match prior
# ============================================================


@dataclass
class TeamRating:
    """Pre-match team strength. Can come from Elo, SPI, or market-implied."""

    name: str
    rating: float  # Elo-like rating (1500 = average)
    home_advantage: float = 60.0  # Elo points for home court/field
    offense_rating: float = 0.0  # sport-specific
    defense_rating: float = 0.0


def elo_win_probability(rating_a: float, rating_b: float) -> float:
    """Standard Elo expected score formula."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


class PreMatchPrior:
    """Compute pre-match win probabilities from team ratings."""

    @staticmethod
    def basketball(home: TeamRating, away: TeamRating) -> tuple[float, float]:
        """Basketball: no draw, just P(home) and P(away)."""
        home_elo = home.rating + home.home_advantage
        p_home = elo_win_probability(home_elo, away.rating)
        return p_home, 1.0 - p_home

    @staticmethod
    def baseball(home: TeamRating, away: TeamRating) -> tuple[float, float]:
        """Baseball: no draw. Slightly lower home advantage than basketball."""
        home_elo = home.rating + home.home_advantage * 0.6  # smaller HFA in baseball
        p_home = elo_win_probability(home_elo, away.rating)
        return p_home, 1.0 - p_home

    @staticmethod
    def football(home: TeamRating, away: TeamRating) -> tuple[float, float, float]:
        """Football: three-way (home, draw, away)."""
        home_elo = home.rating + home.home_advantage
        raw_home = elo_win_probability(home_elo, away.rating)
        # Draw probability from empirical calibration (~25% base)
        draw_base = 0.25
        draw_adjustment = 1.0 - abs(raw_home - 0.5) * 0.4
        p_draw = draw_base * draw_adjustment
        p_home = raw_home * (1.0 - p_draw)
        p_away = (1.0 - raw_home) * (1.0 - p_draw)
        return p_home, p_away, p_draw


# ============================================================
# Layer 2: Time decay
# ============================================================


class TimeDecay:
    """
    As the game progresses, remaining uncertainty shrinks.
    The score becomes more decisive as time runs out.
    """

    @staticmethod
    def basketball_factor(
        elapsed_sec: float,
        home_score: int,
        away_score: int,
        prior_p_home: float,
    ) -> tuple[float, float]:
        """
        NBA: 4x12min = 2880 sec total.
        Score difference matters more as time decreases.
        Uses a simple logistic model based on score diff and remaining time.
        """
        total = 2880.0
        remaining = max(1.0, total - elapsed_sec)
        remaining_frac = remaining / total

        diff = home_score - away_score

        # Points per second rate in NBA ~ 0.075
        expected_remaining_std = math.sqrt(remaining) * 0.075 * 2

        if expected_remaining_std < 0.1:
            # Game essentially over
            return (1.0, 0.0) if diff > 0 else (0.0, 1.0) if diff < 0 else (0.5, 0.5)

        # Z-score of current lead relative to expected remaining variation
        z = diff / expected_remaining_std

        # Blend prior with score-based estimate
        score_based = 1.0 / (1.0 + math.exp(-z * 1.5))

        # Weight: early game = trust prior more; late game = trust score more
        score_weight = 1.0 - remaining_frac ** 0.5
        p_home = prior_p_home * (1.0 - score_weight) + score_based * score_weight

        return p_home, 1.0 - p_home

    @staticmethod
    def baseball_factor(
        inning: int,
        is_top: bool,
        outs: int,
        home_score: int,
        away_score: int,
        prior_p_home: float,
        # Enhanced inputs (Issue 6)
        runners_on_base: int = 0,      # 0-3 runners currently on
        batting_team_is_home: bool = False,
        pitch_count: int = 0,          # starter's pitch count
        bullpen_era: float = 4.00,     # relievers' ERA
    ) -> tuple[float, float]:
        """
        Baseball: 9 innings, 54 total outs.
        Enhanced model with leverage index, base-out state, pitcher
        fatigue, walk-off endgame, and close-game asymmetry.
        """
        # --- Progress ---
        half_innings_done = (inning - 1) * 2 + (0 if is_top else 1)
        outs_done = half_innings_done * 3 + outs
        total_outs = 54.0
        remaining_frac = max(0.01, (total_outs - outs_done) / total_outs)

        diff = home_score - away_score

        # --- Expected remaining runs (refined) ---
        # Average MLB run rate ~0.11 per out, but varies by game phase
        # Late innings with fresh bullpen arms → lower run environment
        if inning <= 5:
            run_rate = 0.12  # starters tire, more offence
        elif inning <= 7:
            run_rate = 0.10  # setup relievers, tighter
        else:
            run_rate = 0.085  # closer territory, lowest run rate

        # Pitcher fatigue: pitch count > 90 increases run rate
        if pitch_count > 90:
            fatigue_boost = (pitch_count - 90) * 0.001
            run_rate += fatigue_boost

        remaining_outs = total_outs - outs_done
        expected_remaining_runs = remaining_outs * run_rate
        std_remaining = math.sqrt(max(expected_remaining_runs, 0.1))

        # --- Base-out state: leverage index approximation ---
        # Runners on base increase run expectancy of the batting team
        # RE24 simplified: 0 runners = baseline, each runner ≈ +0.3 expected runs
        base_run_boost = runners_on_base * 0.3

        # Leverage index: higher in late close games
        leverage = 1.0
        abs_diff = abs(diff)
        if inning >= 7:
            if abs_diff <= 1:
                leverage = 2.5  # very high leverage
            elif abs_diff <= 2:
                leverage = 1.8
            elif abs_diff <= 3:
                leverage = 1.3
        elif inning >= 5:
            if abs_diff <= 1:
                leverage = 1.5
            elif abs_diff <= 2:
                leverage = 1.2

        # Outs matter: 2 outs in a close game is maximum leverage
        if outs == 2 and inning >= 7 and abs_diff <= 2:
            leverage *= 1.2

        # --- Home bats last advantage ---
        home_advantage = 0.0
        if diff < 0 and not is_top:
            # Home is trailing but gets to bat — structural advantage
            # Gets stronger in late innings (fewer outs left means
            # each remaining at-bat is worth more)
            if inning >= 9:
                home_advantage = 0.04  # bottom 9th trailing = big boost
            elif inning >= 7:
                home_advantage = 0.03
            else:
                home_advantage = 0.02
        elif diff < 0 and is_top:
            # Home trailing while away bats — home still has the last licks
            home_advantage = 0.015

        # --- Walk-off endgame (bottom 9th+, home trailing or tied) ---
        walkoff_boost = 0.0
        if inning >= 9 and not is_top:
            if diff == 0:
                # Tied in bottom 9th: home only needs one run, away needs to hold
                walkoff_boost = 0.06
            elif diff == -1:
                # Home down 1 in bottom 9th: one swing can tie or walk off
                walkoff_boost = 0.03
                # Runners amplify walk-off probability
                walkoff_boost += runners_on_base * 0.01
            elif diff > 0:
                # Home already winning in bottom 9th: near certainty
                # (handled by z-score being large, but add small boost)
                walkoff_boost = 0.02

        # --- Close-game asymmetry ---
        # In blowouts (>5 run diff), the trailing team's win probability
        # collapses faster than a linear model would suggest — comebacks
        # require sustained multi-inning rallies which are multiplicatively unlikely
        blowout_penalty = 0.0
        if abs_diff >= 5 and remaining_frac < 0.5:
            blowout_penalty = (abs_diff - 4) * 0.02 * (1.0 - remaining_frac)

        # --- Compute z-score with adjustments ---
        # Adjust effective differential for base-out state
        if batting_team_is_home:
            effective_diff = diff + base_run_boost * 0.15  # runners help home
        else:
            effective_diff = diff - base_run_boost * 0.15  # runners help away

        z = effective_diff / std_remaining if std_remaining > 0.1 else (
            10.0 if diff > 0 else -10.0
        )

        # Score-based probability with leverage-scaled sensitivity
        score_based = 1.0 / (1.0 + math.exp(-z * (1.2 + 0.3 * (leverage - 1.0))))

        # Apply all adjustments
        score_based += home_advantage + walkoff_boost
        if diff > 0:
            score_based += blowout_penalty  # helps leading team
        elif diff < 0:
            score_based -= blowout_penalty  # hurts trailing team

        # --- Blend prior with score-based ---
        # Prior matters more early; score-based dominates late
        # Baseball converges faster than basketball because discrete outs
        score_weight = 1.0 - remaining_frac ** 0.5
        p_home = prior_p_home * (1.0 - score_weight) + score_based * score_weight
        p_home = max(0.001, min(0.999, p_home))

        return p_home, 1.0 - p_home


# ============================================================
# Layer 3: Event shock
# ============================================================


@dataclass
class EventShock:
    """Multiplicative shock to scoring rates after an event."""

    event_type: str
    team: str  # "home" or "away"
    magnitude: float  # multiplier on scoring rate (e.g., 1.05 = +5%)
    decay_sec: float  # how long the shock persists
    timestamp_sec: float  # when it occurred


class ShockAccumulator:
    """
    Tracks active shocks and computes the current scoring rate multiplier.
    Shocks decay exponentially.
    """

    def __init__(self) -> None:
        self._shocks: list[EventShock] = []

    def add_shock(self, shock: EventShock) -> None:
        self._shocks.append(shock)

    def current_multiplier(self, team: str, current_sec: float) -> float:
        """Compute total active multiplier for a team's scoring rate."""
        mult = 1.0
        for shock in self._shocks:
            if shock.team != team:
                continue
            elapsed = current_sec - shock.timestamp_sec
            if elapsed < 0 or elapsed > shock.decay_sec * 3:
                continue
            # Exponential decay
            decay = math.exp(-elapsed / shock.decay_sec)
            mult *= 1.0 + (shock.magnitude - 1.0) * decay
        return mult

    def prune(self, current_sec: float) -> None:
        """Remove fully decayed shocks."""
        self._shocks = [
            s for s in self._shocks if (current_sec - s.timestamp_sec) < s.decay_sec * 3
        ]

    # Predefined shock factories for basketball
    @staticmethod
    def basketball_three_pointer(team: str, at_sec: float) -> EventShock:
        return EventShock("three_pointer", team, 1.03, 30.0, at_sec)

    @staticmethod
    def basketball_turnover(team: str, at_sec: float) -> EventShock:
        """Turnover boosts the OTHER team's scoring rate."""
        other = "away" if team == "home" else "home"
        return EventShock("turnover", other, 1.05, 20.0, at_sec)

    @staticmethod
    def basketball_technical_foul(team: str, at_sec: float) -> EventShock:
        other = "away" if team == "home" else "home"
        return EventShock("technical_foul", other, 1.04, 60.0, at_sec)

    # Predefined for baseball
    @staticmethod
    def baseball_home_run(team: str, at_sec: float) -> EventShock:
        return EventShock("home_run", team, 1.15, 120.0, at_sec)

    @staticmethod
    def baseball_grand_slam(team: str, at_sec: float) -> EventShock:
        """Grand slam: massive momentum swing."""
        return EventShock("grand_slam", team, 1.30, 180.0, at_sec)

    @staticmethod
    def baseball_pitcher_change(team: str, at_sec: float, is_upgrade: bool = True) -> EventShock:
        """Pitcher change: bullpen arm is often fresher but sometimes worse."""
        other = "away" if team == "home" else "home"
        # Upgrade = good for the pitching team (reduces opponent scoring)
        mult = 0.92 if is_upgrade else 1.10
        return EventShock("pitcher_change", other, mult, 300.0, at_sec)

    @staticmethod
    def baseball_error(team: str, at_sec: float) -> EventShock:
        other = "away" if team == "home" else "home"
        return EventShock("error", other, 1.08, 60.0, at_sec)

    @staticmethod
    def baseball_double_play(team: str, at_sec: float) -> EventShock:
        """DP kills rally momentum for batting team."""
        return EventShock("double_play", team, 0.92, 45.0, at_sec)

    @staticmethod
    def baseball_leadoff_walk(team: str, at_sec: float) -> EventShock:
        """Leadoff walk: strong rally indicator."""
        return EventShock("leadoff_walk", team, 1.06, 30.0, at_sec)

    @staticmethod
    def baseball_stolen_base(team: str, at_sec: float) -> EventShock:
        """Stolen base: runner in scoring position, small boost."""
        return EventShock("stolen_base", team, 1.03, 20.0, at_sec)


# ============================================================
# Layer 4: Market calibration
# ============================================================


class BaseballLeverageIndex:
    """
    Computes leverage index for a baseball game state.

    Leverage index measures how much the current situation matters
    relative to an average plate appearance. LI=1.0 is average,
    LI=3.0+ is very high leverage (late close game).

    Used to:
      - Scale edge thresholds (trade more aggressively in high leverage)
      - Weight event attribution (high-LI events matter more)
      - Adjust position sizing
    """

    @staticmethod
    def compute(
        inning: int,
        is_top: bool,
        outs: int,
        run_diff: int,
        runners_on_base: int = 0,
    ) -> float:
        """Returns leverage index (1.0 = average)."""
        abs_diff = abs(run_diff)

        # Base leverage by inning
        if inning <= 3:
            base = 0.8
        elif inning <= 5:
            base = 1.0
        elif inning <= 7:
            base = 1.4
        elif inning == 8:
            base = 1.8
        else:
            base = 2.5  # 9th+

        # Score closeness multiplier
        if abs_diff == 0:
            close_mult = 1.5
        elif abs_diff == 1:
            close_mult = 1.3
        elif abs_diff == 2:
            close_mult = 1.0
        elif abs_diff == 3:
            close_mult = 0.7
        else:
            close_mult = 0.4

        # Outs multiplier: 2 outs = highest leverage in close games
        if abs_diff <= 2:
            outs_mult = {0: 0.9, 1: 1.0, 2: 1.3}.get(outs, 1.0)
        else:
            outs_mult = 1.0

        # Runners on base boost
        runner_mult = 1.0 + runners_on_base * 0.15

        # Walk-off situations
        walkoff_mult = 1.0
        if inning >= 9 and not is_top and run_diff <= 0:
            walkoff_mult = 1.4

        li = base * close_mult * outs_mult * runner_mult * walkoff_mult
        return round(max(0.1, li), 2)


class MarketCalibration:
    """
    Anchors model output to exchange-implied probabilities.
    Prevents the model from drifting too far from market reality.
    """

    @staticmethod
    def calibrate(
        model_p_home: float,
        model_p_away: float,
        market_p_home: float,
        market_p_away: float,
        model_weight: float = 0.6,
    ) -> tuple[float, float]:
        """
        Blend model and market probabilities.
        model_weight: how much to trust the model vs market (0=all market, 1=all model).
        """
        if market_p_home <= 0 or market_p_away <= 0:
            return model_p_home, model_p_away

        p_home = model_p_home * model_weight + market_p_home * (1.0 - model_weight)
        p_away = model_p_away * model_weight + market_p_away * (1.0 - model_weight)

        # Renormalize
        total = p_home + p_away
        if total > 0:
            p_home /= total
            p_away /= total

        return p_home, p_away

    @staticmethod
    def calibrate_three_way(
        model_p: tuple[float, float, float],
        market_p: tuple[float, float, float],
        model_weight: float = 0.6,
    ) -> tuple[float, float, float]:
        """Three-way calibration for football."""
        blended = [
            m * model_weight + mk * (1.0 - model_weight)
            for m, mk in zip(model_p, market_p)
        ]
        total = sum(blended)
        if total > 0:
            blended = [b / total for b in blended]
        return tuple(blended)


# ============================================================
# Unified pricing engine
# ============================================================


class PricingEngine:
    """
    Orchestrates all four layers to produce a fair probability estimate.
    Call `update()` on every tick or event to get the latest FairState.
    """

    def __init__(
        self,
        sport: str,
        home_rating: TeamRating,
        away_rating: TeamRating,
        model_weight: float = 0.6,
    ) -> None:
        self.sport = sport
        self.home_rating = home_rating
        self.away_rating = away_rating
        self.model_weight = model_weight
        self.shock_accumulator = ShockAccumulator()

        # Compute prior
        if sport == "basketball":
            ph, pa = PreMatchPrior.basketball(home_rating, away_rating)
            self.prior = (ph, pa, 0.0)
        elif sport == "baseball":
            ph, pa = PreMatchPrior.baseball(home_rating, away_rating)
            self.prior = (ph, pa, 0.0)
        else:
            self.prior = PreMatchPrior.football(home_rating, away_rating)

        self.latest_fair: tuple[float, ...] = self.prior

    def update(
        self,
        elapsed_sec: float,
        home_score: int,
        away_score: int,
        market_implied: tuple[float, ...] | None = None,
        # Baseball-specific
        inning: int = 1,
        is_top: bool = True,
        outs: int = 0,
    ) -> dict:
        """
        Recompute fair probabilities given current game state.
        Returns dict with p_home, p_away, p_draw, edge_home, edge_away.
        """
        # Layer 2: Time decay
        if self.sport == "basketball":
            p_home, p_away = TimeDecay.basketball_factor(
                elapsed_sec, home_score, away_score, self.prior[0]
            )
            p_draw = 0.0
        elif self.sport == "baseball":
            p_home, p_away = TimeDecay.baseball_factor(
                inning, is_top, outs, home_score, away_score, self.prior[0]
            )
            p_draw = 0.0
        else:
            # Football: simplified time decay
            total = 5400.0
            remaining_frac = max(0.01, (total - elapsed_sec) / total)
            diff = home_score - away_score
            if remaining_frac < 0.05:
                if diff > 0:
                    p_home, p_draw, p_away = 0.95, 0.04, 0.01
                elif diff < 0:
                    p_home, p_draw, p_away = 0.01, 0.04, 0.95
                else:
                    p_home, p_draw, p_away = 0.10, 0.80, 0.10
            else:
                p_home, p_away, p_draw = self.prior

        # Layer 3: Event shocks (modify effective scoring rates)
        home_mult = self.shock_accumulator.current_multiplier("home", elapsed_sec)
        away_mult = self.shock_accumulator.current_multiplier("away", elapsed_sec)

        # Apply shock as a probability adjustment
        shock_adj = (home_mult - away_mult) * 0.02
        p_home = max(0.001, min(0.999, p_home + shock_adj))
        p_away = max(0.001, min(0.999, p_away - shock_adj))

        # Renormalize
        total_p = p_home + p_away + p_draw
        p_home /= total_p
        p_away /= total_p
        p_draw /= total_p

        # Layer 4: Market calibration
        if market_implied and len(market_implied) >= 2:
            if self.sport in ("basketball", "baseball"):
                p_home, p_away = MarketCalibration.calibrate(
                    p_home, p_away, market_implied[0], market_implied[1], self.model_weight
                )
                p_draw = 0.0
            else:
                mk = market_implied if len(market_implied) >= 3 else (*market_implied, 0.0)
                p_home, p_away, p_draw = MarketCalibration.calibrate_three_way(
                    (p_home, p_away, p_draw), mk, self.model_weight
                )

        self.latest_fair = (p_home, p_away, p_draw)
        self.shock_accumulator.prune(elapsed_sec)

        # Compute edges vs market
        edge_home = 0.0
        edge_away = 0.0
        if market_implied and len(market_implied) >= 2:
            edge_home = p_home - market_implied[0]
            edge_away = p_away - market_implied[1]

        return {
            "p_home": p_home,
            "p_away": p_away,
            "p_draw": p_draw,
            "fair_odds_home": 1.0 / p_home if p_home > 0.001 else 999.0,
            "fair_odds_away": 1.0 / p_away if p_away > 0.001 else 999.0,
            "fair_odds_draw": 1.0 / p_draw if p_draw > 0.001 else 999.0,
            "edge_home": edge_home,
            "edge_away": edge_away,
            "home_shock_mult": home_mult,
            "away_shock_mult": away_mult,
        }
