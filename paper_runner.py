"""
Paper trading runner — shadow orders on simulated live games.

Bridges between research (replay on historical) and live (real money).
Runs a full game simulation but tracks shadow orders: what would have
been traded, at what price, with what outcome.

Usage:
  source .venv/bin/activate && python paper_runner.py [n_games] [sport]

Outputs: reports/paper_YYYYMMDD_HHMMSS.json
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime

import sports_engine as se
from src.pricing.engine import (
    PricingEngine,
    TeamRating,
    ShockAccumulator,
    BaseballLeverageIndex,
    BaseballPitchingContext,
    BaseballFeatureFlags,
)
from src.strategy import (
    BasketballStrategyConfig,
    BaseballStrategyConfig,
    StrategyGatekeeper,
)


@dataclass
class ShadowOrder:
    """An order that would have been placed."""
    game_id: int
    timestamp_ms: int
    runner_id: str
    side: str
    price: float
    size: float
    edge_bps: float
    net_edge_bps: float
    would_have_filled: bool
    fill_price: float = 0.0
    pnl_if_filled: float = 0.0
    game_sec: float = 0.0
    sport: str = ""
    strategy_tag: str = ""


@dataclass
class PaperGameResult:
    """Result of one paper-traded game."""
    game_id: int
    sport: str
    home_score: int = 0
    away_score: int = 0
    winner: str = ""
    shadow_orders: int = 0
    shadow_fills: int = 0
    shadow_pnl: float = 0.0
    direction_correct: bool = False
    mid_game_accuracy: float = 0.0
    total_signals: int = 0
    total_passed: int = 0


@dataclass
class PaperReport:
    """Aggregate paper trading report."""
    sport: str
    n_games: int
    strategy_name: str
    timestamp: str = ""
    games: list[PaperGameResult] = field(default_factory=list)

    # Aggregates
    total_shadow_orders: int = 0
    total_shadow_fills: int = 0
    total_shadow_pnl: float = 0.0
    mean_pnl_per_game: float = 0.0
    std_pnl: float = 0.0
    direction_accuracy: float = 0.0
    fill_rate: float = 0.0
    mean_edge_bps: float = 0.0

    def compute_aggregates(self) -> None:
        if not self.games:
            return
        self.n_games = len(self.games)
        self.total_shadow_orders = sum(g.shadow_orders for g in self.games)
        self.total_shadow_fills = sum(g.shadow_fills for g in self.games)
        self.total_shadow_pnl = sum(g.shadow_pnl for g in self.games)
        self.mean_pnl_per_game = self.total_shadow_pnl / self.n_games
        pnls = [g.shadow_pnl for g in self.games]
        if len(pnls) >= 2:
            m = self.mean_pnl_per_game
            self.std_pnl = (sum((x - m) ** 2 for x in pnls) / (len(pnls) - 1)) ** 0.5
        correct = sum(1 for g in self.games if g.direction_correct)
        self.direction_accuracy = correct / self.n_games
        self.fill_rate = self.total_shadow_fills / max(self.total_shadow_orders, 1)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def save(self, reports_dir: str = "reports") -> str:
        os.makedirs(reports_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamp = ts
        path = os.path.join(reports_dir, f"paper_{self.sport}_{ts}.json")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    def print_summary(self) -> None:
        print(f"\n{'='*80}")
        print(f"  PAPER TRADING REPORT — {self.sport.upper()}")
        print(f"  Strategy: {self.strategy_name}")
        print(f"  Games: {self.n_games}")
        print(f"{'='*80}")
        print(f"  Shadow orders:   {self.total_shadow_orders}")
        print(f"  Shadow fills:    {self.total_shadow_fills} ({self.fill_rate:.1%})")
        print(f"  Shadow PnL:      {self.total_shadow_pnl:+.2f}")
        print(f"  Mean PnL/game:   {self.mean_pnl_per_game:+.2f}")
        print(f"  Std PnL:         {self.std_pnl:.2f}")
        sharpe = self.mean_pnl_per_game / self.std_pnl if self.std_pnl > 0 else 0
        print(f"  Sharpe-like:     {sharpe:.2f}")
        print(f"  Direction acc:   {self.direction_accuracy:.1%}")
        print(f"{'='*80}\n")


def generate_book_levels(rng: random.Random, fair_odds: float,
                         noise: float = 0.03, market_bias: float = 0.0) -> tuple:
    center = max(1.02, fair_odds + market_bias)
    back_levels, lay_levels = [], []
    for i in range(5):
        bp = center * (1 - 0.005 * (i + 1)) + rng.gauss(0, noise)
        bp = max(1.01, round(bp, 2))
        bv = rng.uniform(50, 500) * (5 - i)
        back_levels.append((bp, bv))
        lp = center * (1 + 0.005 * (i + 1)) + rng.gauss(0, noise)
        lp = max(bp + 0.01, round(lp, 2))
        lv = rng.uniform(50, 500) * (5 - i)
        lay_levels.append((lp, lv))
    return back_levels, lay_levels


_ETYPE_TO_PA = {
    se.MatchEventType.Strikeout: "strikeout",
    se.MatchEventType.Walk: "walk",
    se.MatchEventType.Single: "single",
    se.MatchEventType.Double: "double",
    se.MatchEventType.Triple: "triple",
    se.MatchEventType.HomeRun: "home_run",
    se.MatchEventType.DoublePlay: "double_play",
}


def paper_trade_basketball(game_id: int, seed: int) -> PaperGameResult:
    """Paper trade one basketball game."""
    rng = random.Random(seed)
    config = BasketballStrategyConfig.fee_aware()
    gatekeeper = StrategyGatekeeper()

    market = se.Market("paper_nba", se.Sport.Basketball, f"g{game_id}", "ml")
    match_state = se.MatchState(f"g{game_id}", se.Sport.Basketball)
    book = se.MarketBook("paper_nba")
    book.add_runner("home", "H")
    book.add_runner("away", "A")
    exchange = se.MockExchange(3000, 0.4)
    risk = se.RiskEngine("paper_nba", 5000.0, 3000.0)
    risk.add_runner("home")
    risk.add_runner("away")

    h_rating = TeamRating("H", 1500 + rng.randint(-100, 100), home_advantage=70)
    a_rating = TeamRating("A", 1500 + rng.randint(-100, 100))
    pricer = PricingEngine("basketball", h_rating, a_rating, model_weight=0.65)

    se.MarketStateMachine.apply_to_market(market, se.MarketStatus.OpenPrematch)
    se.MarketStateMachine.apply_to_market(market, se.MarketStatus.InPlay)

    shadow_orders: list[ShadowOrder] = []
    predictions: list[float] = []
    game_pnl = 0.0

    for sec in range(0, 2880, 5):
        ts_ms = sec * 1000
        match_state.match_clock_sec = float(sec)
        quarter = sec // 720 + 1

        if sec % 720 == 0:
            event = se.MatchEvent(f"q{quarter}", ts_ms,
                                  se.MatchEventType.PeriodStart, sec / 60.0, "home")
            se.MatchStateMachine.apply_event(match_state, event)

        r = rng.random()
        if r < 0.015:
            team = "home" if rng.random() < 0.52 else "away"
            shot_type = rng.choices(
                [se.MatchEventType.FieldGoal2, se.MatchEventType.FieldGoal3, se.MatchEventType.FreeThrow],
                weights=[0.55, 0.30, 0.15]
            )[0]
            event = se.MatchEvent(f"s_{sec}", ts_ms, shot_type, sec / 60.0, team)
            se.MatchStateMachine.apply_event(match_state, event)

        raw = pricer.update(elapsed_sec=sec, home_score=match_state.home_score,
                            away_score=match_state.away_score)

        for rid in ["home", "away"]:
            odds_key = "fair_odds_home" if rid == "home" else "fair_odds_away"
            fo = raw[odds_key]
            bias = fo * 0.04 + rng.gauss(0, 0.01) if fo < 2.0 else -fo * 0.03 + rng.gauss(0, 0.01)
            bl, ll = generate_book_levels(rng, fo, noise=0.02, market_bias=bias)
            book.update_runner_back(rid, bl)
            book.update_runner_lay(rid, ll)

        hs = book.get_runner_snapshot("home")
        aws = book.get_runner_snapshot("away")
        mkt_h = 1.0 / ((hs.best_back_price + hs.best_lay_price) / 2) if hs.best_back_price > 0 and hs.best_lay_price > 0 else raw["p_home"]
        mkt_a = 1.0 / ((aws.best_back_price + aws.best_lay_price) / 2) if aws.best_back_price > 0 and aws.best_lay_price > 0 else raw["p_away"]
        fair = pricer.update(elapsed_sec=sec, home_score=match_state.home_score,
                             away_score=match_state.away_score, market_implied=(mkt_h, mkt_a))

        if sec >= 720:  # predictions from Q2 onward
            predictions.append(fair["p_home"])

        for rid, edge_key in [("home", "edge_home"), ("away", "edge_away")]:
            edge_bps = fair[edge_key] * 10000
            if edge_bps <= 0:
                continue

            snap = book.get_runner_snapshot(rid)
            passed = gatekeeper.check_basketball(
                config=config, runner_id=rid, edge_bps=edge_bps,
                best_back_price=snap.best_back_price,
                best_lay_price=snap.best_lay_price,
                best_back_volume=snap.best_back_size,
                current_sec=float(sec), delay_ms=3000,
                risk_allows=risk.check_limits(rid, se.Side.Back, snap.best_back_price, config.base_stake),
                kill_switch=risk.is_kill_switch_active(),
                quarter=quarter,
            )

            if passed:
                net_edge = edge_bps - config.fee_bps_roundtrip - config.delay_penalty_bps_per_ms * 3000
                exchange.submit_order(market.market_id, rid, se.Side.Back,
                                      snap.best_back_price, config.base_stake, f"edge_{rid}", ts_ms)
                shadow_orders.append(ShadowOrder(
                    game_id=game_id, timestamp_ms=ts_ms, runner_id=rid,
                    side="back", price=snap.best_back_price, size=config.base_stake,
                    edge_bps=edge_bps, net_edge_bps=net_edge,
                    would_have_filled=False, game_sec=float(sec),
                    sport="basketball", strategy_tag="fee_aware",
                ))

        fills = exchange.process_tick(ts_ms, rng.random())
        for fill in fills:
            risk.record_fill(fill)
            fair_key = "fair_odds_home" if fill.runner_id == "home" else "fair_odds_away"
            pnl = (fair[fair_key] - fill.price) * fill.size
            game_pnl += pnl
            # Mark corresponding shadow order as filled
            for so in reversed(shadow_orders):
                if so.runner_id == fill.runner_id and not so.would_have_filled:
                    so.would_have_filled = True
                    so.fill_price = fill.price
                    so.pnl_if_filled = pnl
                    break

    home_won = match_state.home_score > match_state.away_score
    correct = sum(1 for p in predictions if (p > 0.5) == home_won)
    acc = correct / len(predictions) if predictions else 0

    return PaperGameResult(
        game_id=game_id, sport="basketball",
        home_score=match_state.home_score, away_score=match_state.away_score,
        winner="home" if home_won else "away",
        shadow_orders=len(shadow_orders),
        shadow_fills=sum(1 for so in shadow_orders if so.would_have_filled),
        shadow_pnl=game_pnl,
        direction_correct=(predictions[-1] > 0.5) == home_won if predictions else False,
        mid_game_accuracy=acc,
        total_signals=gatekeeper.rejection_log.total_signals,
        total_passed=gatekeeper.rejection_log.total_passed,
    )


def paper_trade_baseball(game_id: int, seed: int) -> PaperGameResult:
    """Paper trade one baseball game."""
    rng = random.Random(seed)
    config = BaseballStrategyConfig.aggressive()
    gatekeeper = StrategyGatekeeper()

    market = se.Market("paper_mlb", se.Sport.Baseball, f"g{game_id}", "ml")
    match_state = se.MatchState(f"g{game_id}", se.Sport.Baseball)
    book = se.MarketBook("paper_mlb")
    book.add_runner("home", "H")
    book.add_runner("away", "A")
    exchange = se.MockExchange(5000, 0.25)
    risk = se.RiskEngine("paper_mlb", 5000.0, 3000.0)
    risk.add_runner("home")
    risk.add_runner("away")

    h_rating = TeamRating("H", 1500 + rng.randint(-100, 100))
    a_rating = TeamRating("A", 1500 + rng.randint(-100, 100))
    pricer = PricingEngine("baseball", h_rating, a_rating, model_weight=0.6,
                           baseball_flags=BaseballFeatureFlags.trading_default())

    se.MarketStateMachine.apply_to_market(market, se.MarketStatus.OpenPrematch)
    se.MarketStateMachine.apply_to_market(market, se.MarketStatus.InPlay)

    pitching_ctx = BaseballPitchingContext(
        home_bullpen_era=3.80 + rng.gauss(0, 0.4),
        away_bullpen_era=3.80 + rng.gauss(0, 0.4),
    )

    shadow_orders: list[ShadowOrder] = []
    predictions: list[float] = []
    game_pnl = 0.0
    game_sec = 0

    for inning in range(1, 10):
        for is_top in [True, False]:
            ts_ms = game_sec * 1000
            event = se.MatchEvent(f"inn_{inning}", ts_ms,
                                  se.MatchEventType.PeriodStart, float(inning),
                                  "away" if is_top else "home")
            se.MatchStateMachine.apply_event(match_state, event)

            if inning == 9 and not is_top and match_state.home_score > match_state.away_score:
                break

            batting_team = "away" if is_top else "home"
            pa_count = 0

            while match_state.outs < 3 and pa_count < 15:
                pa_count += 1
                game_sec += rng.randint(15, 45)
                ts_ms = game_sec * 1000

                r = rng.random()
                event_type = None
                if r < 0.22:
                    event_type = se.MatchEventType.Strikeout
                    match_state.outs += 1
                elif r < 0.30:
                    event_type = se.MatchEventType.Walk
                elif r < 0.50:
                    event_type = se.MatchEventType.Single
                elif r < 0.58:
                    event_type = se.MatchEventType.Double
                elif r < 0.60:
                    event_type = se.MatchEventType.Triple
                elif r < 0.63:
                    event_type = se.MatchEventType.HomeRun
                elif r < 0.70:
                    event_type = se.MatchEventType.DoublePlay
                else:
                    match_state.outs += 1

                if event_type:
                    event = se.MatchEvent(f"pa_{game_sec}", ts_ms,
                                          event_type, float(inning), batting_team)
                    se.MatchStateMachine.apply_event(match_state, event)

                pa_outcome = _ETYPE_TO_PA.get(event_type, "out") if event_type else "out"
                pitching_ctx.record_plate_appearance(batting_team, pa_outcome)

                if match_state.outs >= 3:
                    break

                defensive_team = "away" if batting_team == "home" else "home"
                if pitching_ctx.should_change_pitcher(defensive_team, inning):
                    pitching_ctx.pitcher_change(defensive_team, "reliever")

                runners = match_state.runners_on_base()
                def_pc, def_era = pitching_ctx.defensive_state(batting_team)

                raw = pricer.update(
                    elapsed_sec=game_sec, home_score=match_state.home_score,
                    away_score=match_state.away_score,
                    inning=inning, is_top=is_top, outs=match_state.outs,
                    runners_on_base=runners, batting_team_is_home=(batting_team == "home"),
                    defensive_pitch_count=def_pc, defensive_bullpen_era=def_era,
                )

                for rid in ["home", "away"]:
                    odds_key = "fair_odds_home" if rid == "home" else "fair_odds_away"
                    fo = raw[odds_key]
                    bias = fo * 0.04 + rng.gauss(0, 0.01) if fo < 2.0 else -fo * 0.03 + rng.gauss(0, 0.01)
                    bl, ll = generate_book_levels(rng, fo, noise=0.03, market_bias=bias)
                    book.update_runner_back(rid, bl)
                    book.update_runner_lay(rid, ll)

                hs = book.get_runner_snapshot("home")
                aws = book.get_runner_snapshot("away")
                mkt_h = 1.0 / ((hs.best_back_price + hs.best_lay_price) / 2) if hs.best_back_price > 0 and hs.best_lay_price > 0 else raw["p_home"]
                mkt_a = 1.0 / ((aws.best_back_price + aws.best_lay_price) / 2) if aws.best_back_price > 0 and aws.best_lay_price > 0 else raw["p_away"]

                fair = pricer.update(
                    elapsed_sec=game_sec, home_score=match_state.home_score,
                    away_score=match_state.away_score,
                    inning=inning, is_top=is_top, outs=match_state.outs,
                    market_implied=(mkt_h, mkt_a),
                    runners_on_base=runners, batting_team_is_home=(batting_team == "home"),
                    defensive_pitch_count=def_pc, defensive_bullpen_era=def_era,
                )

                if inning >= 3:
                    predictions.append(fair["p_home"])

                run_diff = match_state.home_score - match_state.away_score
                leverage_idx = BaseballLeverageIndex.compute(
                    inning=inning, is_top=is_top, outs=match_state.outs,
                    run_diff=run_diff, runners_on_base=runners,
                )

                for rid, edge_key in [("home", "edge_home"), ("away", "edge_away")]:
                    edge_bps = fair[edge_key] * 10000
                    if edge_bps <= 0:
                        continue

                    snap = book.get_runner_snapshot(rid)
                    passed = gatekeeper.check_baseball(
                        config=config, runner_id=rid, edge_bps=edge_bps,
                        best_back_price=snap.best_back_price,
                        best_lay_price=snap.best_lay_price,
                        best_back_volume=snap.best_back_size,
                        current_sec=float(game_sec), delay_ms=5000,
                        risk_allows=risk.check_limits(rid, se.Side.Back, snap.best_back_price, config.base_stake),
                        kill_switch=risk.is_kill_switch_active(),
                        inning=inning, outs=match_state.outs,
                        run_diff=run_diff, leverage_index=leverage_idx,
                    )

                    if passed:
                        exchange.submit_order(market.market_id, rid, se.Side.Back,
                                              snap.best_back_price, config.base_stake,
                                              f"edge_{rid}", ts_ms)
                        shadow_orders.append(ShadowOrder(
                            game_id=game_id, timestamp_ms=ts_ms, runner_id=rid,
                            side="back", price=snap.best_back_price, size=config.base_stake,
                            edge_bps=edge_bps, net_edge_bps=edge_bps,
                            would_have_filled=False, game_sec=float(game_sec),
                            sport="baseball", strategy_tag="trading_default",
                        ))

                fills = exchange.process_tick(ts_ms, rng.random())
                for fill in fills:
                    risk.record_fill(fill)
                    fair_key = "fair_odds_home" if fill.runner_id == "home" else "fair_odds_away"
                    pnl = (fair[fair_key] - fill.price) * fill.size
                    game_pnl += pnl
                    for so in reversed(shadow_orders):
                        if so.runner_id == fill.runner_id and not so.would_have_filled:
                            so.would_have_filled = True
                            so.fill_price = fill.price
                            so.pnl_if_filled = pnl
                            break

            match_state.outs = 0
            if inning >= 9 and not is_top and match_state.home_score > match_state.away_score:
                break

    home_won = match_state.home_score > match_state.away_score
    correct = sum(1 for p in predictions if (p > 0.5) == home_won)
    acc = correct / len(predictions) if predictions else 0

    return PaperGameResult(
        game_id=game_id, sport="baseball",
        home_score=match_state.home_score, away_score=match_state.away_score,
        winner="home" if home_won else "away",
        shadow_orders=len(shadow_orders),
        shadow_fills=sum(1 for so in shadow_orders if so.would_have_filled),
        shadow_pnl=game_pnl,
        direction_correct=(predictions[-1] > 0.5) == home_won if predictions else False,
        mid_game_accuracy=acc,
        total_signals=gatekeeper.rejection_log.total_signals,
        total_passed=gatekeeper.rejection_log.total_passed,
    )


def run_paper(sport: str = "basketball", n_games: int = 20,
              base_seed: int = 7777) -> PaperReport:
    """Run paper trading session."""
    strategy_name = "fee_aware" if sport == "basketball" else "trading_default"
    report = PaperReport(sport=sport, n_games=n_games, strategy_name=strategy_name)

    print(f"\n  Paper trading {sport} — {n_games} games...")
    for i in range(n_games):
        seed = base_seed + i
        if sport == "basketball":
            result = paper_trade_basketball(i, seed)
        else:
            result = paper_trade_baseball(i, seed)
        report.games.append(result)
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{n_games} done")

    report.compute_aggregates()
    return report


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    sport = sys.argv[2] if len(sys.argv) > 2 else "basketball"

    report = run_paper(sport=sport, n_games=n)
    report.print_summary()
    path = report.save()
    print(f"  Saved: {path}")
