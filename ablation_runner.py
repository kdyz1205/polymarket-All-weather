"""
Baseball model ablation sweep — measures marginal value of each feature layer.

Runs N simulated games per configuration, comparing:
  A: baseline (inning + score only)
  B: A + runners_on_base
  C: B + leverage_index (in gatekeeper)
  D: C + defensive_pitch_count
  E: D + defensive_bullpen_era

Metrics per config:
  - direction accuracy (did model predict correct winner?)
  - mean absolute edge (bps)
  - signal count / pass rate
  - total fills
  - net PnL

Run: source .venv/bin/activate && python ablation_runner.py
"""

import random
import time
from dataclasses import dataclass, field

import sports_engine as se
from src.pricing.engine import (
    PricingEngine,
    TeamRating,
    ShockAccumulator,
    BaseballLeverageIndex,
    BaseballPitchingContext,
)
from src.strategy import (
    BaseballStrategyConfig,
    StrategyGatekeeper,
)


@dataclass
class AblationConfig:
    """Which feature layers are enabled for this ablation run."""
    name: str
    use_runners: bool = False
    use_leverage: bool = False
    use_fatigue: bool = False
    use_bullpen: bool = False


@dataclass
class AblationResult:
    """Aggregated results for one ablation configuration."""
    name: str
    games: int = 0
    correct_predictions: int = 0
    total_predictions: int = 0
    total_signals: int = 0
    total_passed: int = 0
    total_fills: int = 0
    total_edge_bps: float = 0.0
    edge_samples: int = 0
    total_pnl: float = 0.0

    @property
    def accuracy(self) -> float:
        return self.correct_predictions / max(self.total_predictions, 1)

    @property
    def mean_edge_bps(self) -> float:
        return self.total_edge_bps / max(self.edge_samples, 1)

    @property
    def pass_rate(self) -> float:
        return self.total_passed / max(self.total_signals, 1)

    @property
    def fill_rate(self) -> float:
        return self.total_fills / max(self.total_passed, 1)


def generate_book_levels(fair_odds: float, noise: float = 0.03, market_bias: float = 0.0) -> tuple:
    center = max(1.02, fair_odds + market_bias)
    back_levels, lay_levels = [], []
    for i in range(5):
        bp = center * (1 - 0.005 * (i + 1)) + random.gauss(0, noise)
        bp = max(1.01, round(bp, 2))
        bv = random.uniform(50, 500) * (5 - i)
        back_levels.append((bp, bv))
        lp = center * (1 + 0.005 * (i + 1)) + random.gauss(0, noise)
        lp = max(bp + 0.01, round(lp, 2))
        lv = random.uniform(50, 500) * (5 - i)
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


def run_one_game(cfg: AblationConfig, seed: int) -> AblationResult:
    """Simulate one baseball game with the given feature configuration."""
    rng = random.Random(seed)
    result = AblationResult(name=cfg.name)

    market = se.Market("abl_game", se.Sport.Baseball, f"game_{seed}", "moneyline")
    match_state = se.MatchState(f"game_{seed}", se.Sport.Baseball)
    book = se.MarketBook("abl_game")
    book.add_runner("home", "TeamA")
    book.add_runner("away", "TeamB")
    exchange = se.MockExchange(5000, 0.25)
    risk = se.RiskEngine("abl_game", 5000.0, 3000.0)
    risk.add_runner("home")
    risk.add_runner("away")

    home_rating = TeamRating("TeamA", 1500 + rng.randint(-100, 100))
    away_rating = TeamRating("TeamB", 1500 + rng.randint(-100, 100))
    pricer = PricingEngine("baseball", home_rating, away_rating, model_weight=0.6)

    se.MarketStateMachine.apply_to_market(market, se.MarketStatus.OpenPrematch)
    se.MarketStateMachine.apply_to_market(market, se.MarketStatus.InPlay)

    strat_config = BaseballStrategyConfig.aggressive()
    gatekeeper = StrategyGatekeeper()

    pitching_ctx = BaseballPitchingContext(
        home_bullpen_era=3.80 + rng.gauss(0, 0.4),
        away_bullpen_era=3.80 + rng.gauss(0, 0.4),
    )

    game_sec = 0
    mid_game_predictions = []  # (inning, model_p_home) for accuracy tracking

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

                # Track pitching
                pa_outcome = _ETYPE_TO_PA.get(event_type, "out") if event_type else "out"
                pitching_ctx.record_plate_appearance(batting_team, pa_outcome)

                if match_state.outs >= 3:
                    break

                # Auto pitcher change
                defensive_team = "away" if batting_team == "home" else "home"
                if pitching_ctx.should_change_pitcher(defensive_team, inning):
                    pitching_ctx.pitcher_change(defensive_team, "reliever")

                # Build update kwargs based on ablation config
                runners = match_state.runners_on_base() if cfg.use_runners else 0
                def_pc, def_era = pitching_ctx.defensive_state(batting_team)

                update_kwargs = dict(
                    elapsed_sec=game_sec,
                    home_score=match_state.home_score,
                    away_score=match_state.away_score,
                    inning=inning, is_top=is_top, outs=match_state.outs,
                    runners_on_base=runners,
                    batting_team_is_home=(batting_team == "home"),
                    defensive_pitch_count=def_pc if cfg.use_fatigue else 0,
                    defensive_bullpen_era=def_era if cfg.use_bullpen else 4.0,
                )

                # Step 1: raw model
                raw = pricer.update(**update_kwargs)

                # Step 2: biased book
                for rid in ["home", "away"]:
                    odds_key = "fair_odds_home" if rid == "home" else "fair_odds_away"
                    fo = raw[odds_key]
                    bias = fo * 0.04 + rng.gauss(0, 0.01) if fo < 2.0 else -fo * 0.03 + rng.gauss(0, 0.01)
                    bl, ll = generate_book_levels(fo, noise=0.03, market_bias=bias)
                    book.update_runner_back(rid, bl)
                    book.update_runner_lay(rid, ll)

                # Step 3: market implied
                hs = book.get_runner_snapshot("home")
                aws = book.get_runner_snapshot("away")
                mkt_h = 1.0 / ((hs.best_back_price + hs.best_lay_price) / 2) if hs.best_back_price > 0 and hs.best_lay_price > 0 else raw["p_home"]
                mkt_a = 1.0 / ((aws.best_back_price + aws.best_lay_price) / 2) if aws.best_back_price > 0 and aws.best_lay_price > 0 else raw["p_away"]

                # Step 4: calibrated update
                update_kwargs["market_implied"] = (mkt_h, mkt_a)
                fair = pricer.update(**update_kwargs)

                # Record mid-game prediction at inning boundaries
                if pa_count == 1 and inning >= 3:
                    mid_game_predictions.append((inning, fair["p_home"]))

                # Edge signals + gatekeeper
                run_diff = match_state.home_score - match_state.away_score
                leverage_idx = BaseballLeverageIndex.compute(
                    inning=inning, is_top=is_top, outs=match_state.outs,
                    run_diff=run_diff, runners_on_base=runners,
                ) if cfg.use_leverage else 1.0

                for rid, edge_key in [("home", "edge_home"), ("away", "edge_away")]:
                    edge_bps = fair[edge_key] * 10000
                    if edge_bps <= 0:
                        continue

                    result.total_edge_bps += edge_bps
                    result.edge_samples += 1

                    snap = book.get_runner_snapshot(rid)
                    passed = gatekeeper.check_baseball(
                        config=strat_config,
                        runner_id=rid, edge_bps=edge_bps,
                        best_back_price=snap.best_back_price,
                        best_lay_price=snap.best_lay_price,
                        best_back_volume=snap.best_back_size,
                        current_sec=float(game_sec), delay_ms=5000,
                        risk_allows=risk.check_limits(rid, se.Side.Back, snap.best_back_price, strat_config.base_stake),
                        kill_switch=risk.is_kill_switch_active(),
                        inning=inning, outs=match_state.outs,
                        run_diff=run_diff,
                        leverage_index=leverage_idx,
                    )

                    if passed:
                        exchange.submit_order(market.market_id, rid, se.Side.Back,
                                              snap.best_back_price, strat_config.base_stake,
                                              f"edge_{rid}", ts_ms)

                fills = exchange.process_tick(ts_ms, rng.random())
                for fill in fills:
                    risk.record_fill(fill)
                    result.total_fills += 1
                    fair_key = "fair_odds_home" if fill.runner_id == "home" else "fair_odds_away"
                    edge_captured = (fair[fair_key] - fill.price) * fill.size
                    result.total_pnl += edge_captured

            match_state.outs = 0
            if inning >= 9 and not is_top and match_state.home_score > match_state.away_score:
                break

    # Determine winner
    home_won = match_state.home_score > match_state.away_score

    # Evaluate direction accuracy for mid-game predictions
    for inning_num, p_home in mid_game_predictions:
        predicted_home = p_home > 0.5
        if predicted_home == home_won:
            result.correct_predictions += 1
        result.total_predictions += 1

    result.games = 1
    result.total_signals = gatekeeper.rejection_log.total_signals
    result.total_passed = gatekeeper.rejection_log.total_passed

    return result


def merge_results(results: list[AblationResult]) -> AblationResult:
    """Combine results from multiple games."""
    if not results:
        raise ValueError("No results to merge")
    merged = AblationResult(name=results[0].name)
    for r in results:
        merged.games += r.games
        merged.correct_predictions += r.correct_predictions
        merged.total_predictions += r.total_predictions
        merged.total_signals += r.total_signals
        merged.total_passed += r.total_passed
        merged.total_fills += r.total_fills
        merged.total_edge_bps += r.total_edge_bps
        merged.edge_samples += r.edge_samples
        merged.total_pnl += r.total_pnl
    return merged


ABLATION_CONFIGS = [
    AblationConfig(name="A: baseline"),
    AblationConfig(name="B: +runners", use_runners=True),
    AblationConfig(name="C: +leverage", use_runners=True, use_leverage=True),
    AblationConfig(name="D: +fatigue", use_runners=True, use_leverage=True, use_fatigue=True),
    AblationConfig(name="E: +bullpen", use_runners=True, use_leverage=True, use_fatigue=True, use_bullpen=True),
]


def run_ablation(n_games: int = 50, base_seed: int = 42) -> None:
    """Run full ablation sweep and print comparison table."""
    print(f"\n{'='*80}")
    print(f"  BASEBALL MODEL ABLATION SWEEP — {n_games} games per configuration")
    print(f"{'='*80}\n")

    all_merged: list[AblationResult] = []

    for cfg in ABLATION_CONFIGS:
        results = []
        for i in range(n_games):
            r = run_one_game(cfg, seed=base_seed + i)
            results.append(r)
        merged = merge_results(results)
        all_merged.append(merged)

        print(f"  {cfg.name}: done ({merged.games} games)")

    # Print comparison table
    print(f"\n{'─'*80}")
    print(f"  {'Config':<20} {'Acc%':>6} {'AvgEdge':>8} {'Signals':>8} {'Passed':>8} "
          f"{'Fills':>7} {'PassRate':>9} {'PnL':>10}")
    print(f"{'─'*80}")

    baseline_acc = all_merged[0].accuracy if all_merged else 0

    for m in all_merged:
        delta = f"({'+'if m.accuracy > baseline_acc else ''}{(m.accuracy - baseline_acc)*100:+.1f}%)" if m.name != "A: baseline" else ""
        print(f"  {m.name:<20} {m.accuracy*100:5.1f}% {m.mean_edge_bps:7.1f} {m.total_signals:8d} "
              f"{m.total_passed:8d} {m.total_fills:7d} {m.pass_rate*100:8.1f}% {m.total_pnl:10.2f}")
        if delta:
            print(f"  {'':20} {delta}")

    print(f"{'─'*80}")
    print(f"\n  Legend: Acc% = mid-game direction accuracy (inning 3+)")
    print(f"  AvgEdge = mean positive edge in bps")
    print(f"  PassRate = signals that passed all gates")
    print(f"  PnL = theoretical edge captured (before fees)\n")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    run_ablation(n_games=n)
