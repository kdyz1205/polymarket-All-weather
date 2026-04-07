"""
Threshold sweep runner — finds the activation frontier.

For each sport, sweeps min_edge_bps across a range and reports:
  - How many signals triggered
  - How many passed all gates
  - How many got filled
  - Fill ratio, edge capture ratio
  - Binding constraint at each threshold

This answers: "At what parameter settings do we start getting trades?"

Run: source .venv/bin/activate && python sweep_runner.py
"""

import random
import time
import sys
import json
import os
from dataclasses import dataclass

import sports_engine as se
from src.pricing.engine import PricingEngine, TeamRating, ShockAccumulator, BaseballLeverageIndex, BaseballPitchingContext, BaseballFeatureFlags
from src.strategy import (
    BasketballStrategyConfig,
    BaseballStrategyConfig,
    StrategyGatekeeper,
    RejectionLog,
)
from src.analytics import (
    FillRecord,
    MarketOutcome,
    SessionAggregator,
    OrderRecord,
    ExecutionAnalyzer,
    EventAttributionAnalyzer,
    RiskAttributionAnalyzer,
    ReplayReport,
)


def generate_book_levels(fair_odds: float, noise: float = 0.03, market_bias: float = 0.0) -> tuple:
    center = max(1.02, fair_odds + market_bias)
    back_levels = []
    lay_levels = []
    for i in range(5):
        bp = center * (1 - 0.005 * (i + 1)) + random.gauss(0, noise)
        bp = max(1.01, round(bp, 2))
        bv = random.uniform(50, 500) * (5 - i)
        back_levels.append((bp, bv))

        lp = center * (1 + 0.005 * (i + 1)) + random.gauss(0, noise)
        lp = max(1.01, round(lp, 2))
        lv = random.uniform(50, 500) * (5 - i)
        lay_levels.append((lp, lv))

    back_levels.sort(key=lambda x: -x[0])
    lay_levels.sort(key=lambda x: x[0])
    return back_levels, lay_levels


def compute_market_bias(fair_odds: float) -> float:
    """Favourite-longshot bias: market over-estimates underdog."""
    if fair_odds < 2.0:
        return fair_odds * 0.04 + random.gauss(0, 0.01)
    else:
        return -fair_odds * 0.03 + random.gauss(0, 0.01)


@dataclass
class SweepResult:
    min_edge_bps: float
    total_signals: int
    total_passed: int
    pass_rate: float
    total_orders: int
    total_fills: int
    fill_ratio: float
    binding_constraint: str  # most common rejection reason
    rejection_breakdown: dict
    net_pnl: float
    edge_capture_ratio: float


def sweep_basketball(edge_bps_range: list[float], seed: int = 123) -> list[SweepResult]:
    """Run basketball sim at each edge threshold and collect results."""
    results = []

    for min_edge in edge_bps_range:
        random.seed(seed)

        config = BasketballStrategyConfig(
            min_edge_bps=min_edge,
            min_liquidity=10.0,
            max_spread_bps=2000.0,
            cooldown_sec=5.0,
            max_position_per_runner=2000.0,
            max_total_position=5000.0,
            max_orders_per_min=60,
            base_stake=30.0,
            delay_penalty_bps_per_ms=0.0,
        )

        market = se.Market("nba_sweep", se.Sport.Basketball, "game_sweep", "moneyline")
        match_state = se.MatchState("game_sweep", se.Sport.Basketball)
        book = se.MarketBook("nba_sweep")
        book.add_runner("home", "Lakers")
        book.add_runner("away", "Celtics")
        exchange = se.MockExchange(0, 0.3)
        risk = se.RiskEngine("nba_sweep", 5000.0, 3000.0)
        risk.add_runner("home")
        risk.add_runner("away")

        home_rating = TeamRating("Lakers", 1620, home_advantage=70)
        away_rating = TeamRating("Celtics", 1580)
        pricer = PricingEngine("basketball", home_rating, away_rating, model_weight=0.65)

        se.MarketStateMachine.apply_to_market(market, se.MarketStatus.OpenPrematch)

        gatekeeper = StrategyGatekeeper()
        session_agg = SessionAggregator("sweep", "basketball", "nba_sweep")
        exec_analyzer = ExecutionAnalyzer()
        fill_counter = 0
        total_trades = 0
        delay_ms = 0

        for sec in range(0, 2880, 5):
            ts_ms = sec * 1000  # simulated game time

            if sec == 0:
                se.MarketStateMachine.apply_to_market(market, se.MarketStatus.InPlay)
                exchange = se.MockExchange(3000, 0.4)
                delay_ms = 3000

            match_state.match_clock_sec = float(sec)
            quarter = sec // 720 + 1

            if sec % 720 == 0:
                event = se.MatchEvent(f"q{quarter}", ts_ms,
                                      se.MatchEventType.PeriodStart, sec / 60.0, "home")
                se.MatchStateMachine.apply_event(match_state, event)

            r = random.random()
            if r < 0.015:
                team = "home" if random.random() < 0.52 else "away"
                shot_type = random.choices(
                    [se.MatchEventType.FieldGoal2, se.MatchEventType.FieldGoal3, se.MatchEventType.FreeThrow],
                    weights=[0.55, 0.30, 0.15]
                )[0]
                event = se.MatchEvent(f"s_{sec}", ts_ms, shot_type, sec / 60.0, team)
                se.MatchStateMachine.apply_event(match_state, event)
                if shot_type == se.MatchEventType.FieldGoal3:
                    pricer.shock_accumulator.add_shock(
                        ShockAccumulator.basketball_three_pointer(team, sec))
            elif r < 0.02:
                team = "home" if random.random() < 0.5 else "away"
                event = se.MatchEvent(f"to_{sec}", ts_ms,
                                      se.MatchEventType.Turnover, sec / 60.0, team)
                se.MatchStateMachine.apply_event(match_state, event)
                pricer.shock_accumulator.add_shock(
                    ShockAccumulator.basketball_turnover(team, sec))

            raw = pricer.update(
                elapsed_sec=sec,
                home_score=match_state.home_score,
                away_score=match_state.away_score,
            )

            for rid in ["home", "away"]:
                odds_key = "fair_odds_home" if rid == "home" else "fair_odds_away"
                bl, ll = generate_book_levels(raw[odds_key], noise=0.02,
                                              market_bias=compute_market_bias(raw[odds_key]))
                book.update_runner_back(rid, bl)
                book.update_runner_lay(rid, ll)

            # Market-implied from biased book
            hs = book.get_runner_snapshot("home")
            aws = book.get_runner_snapshot("away")
            mkt_h = 1.0 / ((hs.best_back_price + hs.best_lay_price) / 2) if hs.best_back_price > 0 and hs.best_lay_price > 0 else raw["p_home"]
            mkt_a = 1.0 / ((aws.best_back_price + aws.best_lay_price) / 2) if aws.best_back_price > 0 and aws.best_lay_price > 0 else raw["p_away"]
            fair = pricer.update(
                elapsed_sec=sec,
                home_score=match_state.home_score,
                away_score=match_state.away_score,
                market_implied=(mkt_h, mkt_a),
            )

            # Try to trade each runner
            for rid, edge_key in [("home", "edge_home"), ("away", "edge_away")]:
                edge_bps = fair[edge_key] * 10000
                if edge_bps <= 0:
                    continue

                snap = book.get_runner_snapshot(rid)
                lay_snap = book.get_runner_snapshot(rid)

                passed = gatekeeper.check_basketball(
                    config=config,
                    runner_id=rid,
                    edge_bps=edge_bps,
                    best_back_price=snap.best_back_price,
                    best_lay_price=snap.best_lay_price,
                    best_back_volume=snap.best_back_size,
                    current_sec=float(sec),
                    delay_ms=delay_ms,
                    risk_allows=risk.check_limits(rid, se.Side.Back, snap.best_back_price, config.base_stake),
                    kill_switch=risk.is_kill_switch_active(),
                    quarter=quarter,
                )

                if passed:
                    exchange.submit_order(market.market_id, rid, se.Side.Back,
                                          snap.best_back_price, config.base_stake,
                                          "edge_back", ts_ms)
                    total_trades += 1

            fills = exchange.process_tick(ts_ms, random.random())
            for fill in fills:
                risk.record_fill(fill)
                fill_counter += 1
                fair_odds_key = "fair_odds_home" if fill.runner_id == "home" else "fair_odds_away"

                fill_rec = FillRecord(
                    fill_id=fill_counter, order_id=fill_counter,
                    runner_id=fill.runner_id,
                    side="back" if str(fill.side) == "Side.Back" else "lay",
                    price=fill.price, size=fill.size, timestamp_ms=ts_ms,
                    fair_price_at_fill=fair[fair_odds_key],
                    market_price_at_fill=fill.price,
                    elapsed_game_sec=float(sec), strategy_tag="edge_back",
                )
                session_agg.record_fill(fill_rec)
                exec_analyzer.record_order(OrderRecord(
                    order_id=fill_counter, runner_id=fill.runner_id, side="back",
                    price=fill.price, size=fill.size, submit_ts_ms=ts_ms - delay_ms,
                    status="filled", filled_size=fill.size, avg_fill_price=fill.price,
                    fill_ts_ms=ts_ms,
                    fair_at_submit=fair[fair_odds_key], market_at_submit=fill.price,
                    fair_at_fill=fair[fair_odds_key], market_at_fill=fill.price,
                    market_5s_after_fill=fill.price * (1 + random.gauss(0, 0.003)),
                    delay_ms=delay_ms, strategy_tag="edge_back",
                ))

        # Settle
        if match_state.home_score > match_state.away_score:
            session_agg.set_outcome(MarketOutcome(winning_runner_id="home"))
        else:
            session_agg.set_outcome(MarketOutcome(winning_runner_id="away"))

        summary = session_agg.compute(total_orders=total_trades)
        exec_metrics = exec_analyzer.compute()

        # Find binding constraint
        rej = gatekeeper.rejection_log
        binding = ""
        if rej.counts:
            binding = max(rej.counts, key=rej.counts.get)

        results.append(SweepResult(
            min_edge_bps=min_edge,
            total_signals=rej.total_signals,
            total_passed=rej.total_passed,
            pass_rate=rej.pass_rate,
            total_orders=total_trades,
            total_fills=fill_counter,
            fill_ratio=fill_counter / total_trades if total_trades > 0 else 0,
            binding_constraint=binding,
            rejection_breakdown=dict(rej.counts),
            net_pnl=summary.net_pnl,
            edge_capture_ratio=exec_metrics.edge_capture_ratio,
        ))

    return results


def sweep_baseball(edge_bps_range: list[float], seed: int = 456) -> list[SweepResult]:
    """Run baseball sim at each edge threshold and collect results."""
    results = []

    for min_edge in edge_bps_range:
        random.seed(seed)

        config = BaseballStrategyConfig(
            min_edge_bps=min_edge,
            min_liquidity=5.0,
            max_spread_bps=3000.0,
            cooldown_sec=3.0,
            max_position_per_runner=2000.0,
            max_total_position=5000.0,
            max_orders_per_min=60,
            base_stake=20.0,
            delay_penalty_bps_per_ms=0.0,
            close_game_only=False,
        )

        market = se.Market("mlb_sweep", se.Sport.Baseball, "game_sweep", "moneyline")
        match_state = se.MatchState("game_sweep", se.Sport.Baseball)
        book = se.MarketBook("mlb_sweep")
        book.add_runner("home", "Yankees")
        book.add_runner("away", "Dodgers")
        exchange = se.MockExchange(5000, 0.25)
        risk = se.RiskEngine("mlb_sweep", 5000.0, 3000.0)
        risk.add_runner("home")
        risk.add_runner("away")

        home_rating = TeamRating("Yankees", 1550)
        away_rating = TeamRating("Dodgers", 1600)
        pricer = PricingEngine("baseball", home_rating, away_rating, model_weight=0.6,
                               baseball_flags=BaseballFeatureFlags.trading_default())

        se.MarketStateMachine.apply_to_market(market, se.MarketStatus.OpenPrematch)
        se.MarketStateMachine.apply_to_market(market, se.MarketStatus.InPlay)

        gatekeeper = StrategyGatekeeper()
        session_agg = SessionAggregator("sweep", "baseball", "mlb_sweep")
        exec_analyzer = ExecutionAnalyzer()
        fill_counter = 0
        total_trades = 0
        game_sec = 0
        delay_ms = 5000

        pitching_ctx = BaseballPitchingContext(
            home_bullpen_era=3.80 + random.gauss(0, 0.4),
            away_bullpen_era=3.80 + random.gauss(0, 0.4),
        )

        _EVENT_TO_PA_SWEEP = {
            "strikeout": "strikeout", "walk": "walk", "single": "single",
            "double": "double", "triple": "triple", "home_run": "home_run",
            "double_play": "double_play", "out": "out",
        }

        for inning in range(1, 10):
            for is_top in [True, False]:
                event = se.MatchEvent(f"inn_{inning}", game_sec * 1000,
                                      se.MatchEventType.PeriodStart, float(inning),
                                      "away" if is_top else "home")
                se.MatchStateMachine.apply_event(match_state, event)

                if inning == 9 and not is_top and match_state.home_score > match_state.away_score:
                    break

                batting_team = "away" if is_top else "home"
                pa_count = 0

                while match_state.outs < 3 and pa_count < 15:
                    pa_count += 1
                    game_sec += random.randint(15, 45)
                    ts_ms = game_sec * 1000  # simulated game time

                    r = random.random()
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

                    # Map event type to PA outcome for pitch tracking
                    _etype_to_pa = {
                        se.MatchEventType.Strikeout: "strikeout",
                        se.MatchEventType.Walk: "walk",
                        se.MatchEventType.Single: "single",
                        se.MatchEventType.Double: "double",
                        se.MatchEventType.Triple: "triple",
                        se.MatchEventType.HomeRun: "home_run",
                        se.MatchEventType.DoublePlay: "double_play",
                    }

                    if event_type:
                        event = se.MatchEvent(f"pa_{game_sec}", ts_ms,
                                              event_type, float(inning), batting_team)
                        se.MatchStateMachine.apply_event(match_state, event)
                        if event_type == se.MatchEventType.HomeRun:
                            pricer.shock_accumulator.add_shock(
                                ShockAccumulator.baseball_home_run(batting_team, game_sec))
                        pa_outcome = _etype_to_pa.get(event_type, "out")
                    else:
                        pa_outcome = "out"

                    pitching_ctx.record_plate_appearance(batting_team, pa_outcome)

                    if match_state.outs >= 3:
                        break

                    # Auto pitcher change if fatigued
                    defensive_team = "away" if batting_team == "home" else "home"
                    if pitching_ctx.should_change_pitcher(defensive_team, inning):
                        pitching_ctx.pitcher_change(defensive_team, "reliever")

                    runners_on_base = match_state.runners_on_base()
                    def_pitch_count, def_bullpen_era = pitching_ctx.defensive_state(batting_team)
                    raw = pricer.update(
                        elapsed_sec=game_sec,
                        home_score=match_state.home_score,
                        away_score=match_state.away_score,
                        inning=inning, is_top=is_top, outs=match_state.outs,
                        runners_on_base=runners_on_base,
                        batting_team_is_home=(batting_team == "home"),
                        defensive_pitch_count=def_pitch_count,
                        defensive_bullpen_era=def_bullpen_era,
                    )

                    for rid in ["home", "away"]:
                        odds_key = "fair_odds_home" if rid == "home" else "fair_odds_away"
                        bl, ll = generate_book_levels(raw[odds_key], noise=0.03,
                                                      market_bias=compute_market_bias(raw[odds_key]))
                        book.update_runner_back(rid, bl)
                        book.update_runner_lay(rid, ll)

                    hs = book.get_runner_snapshot("home")
                    aws = book.get_runner_snapshot("away")
                    mkt_h = 1.0 / ((hs.best_back_price + hs.best_lay_price) / 2) if hs.best_back_price > 0 and hs.best_lay_price > 0 else raw["p_home"]
                    mkt_a = 1.0 / ((aws.best_back_price + aws.best_lay_price) / 2) if aws.best_back_price > 0 and aws.best_lay_price > 0 else raw["p_away"]
                    fair = pricer.update(
                        elapsed_sec=game_sec,
                        home_score=match_state.home_score,
                        away_score=match_state.away_score,
                        inning=inning, is_top=is_top, outs=match_state.outs,
                        market_implied=(mkt_h, mkt_a),
                        runners_on_base=runners_on_base,
                        batting_team_is_home=(batting_team == "home"),
                        defensive_pitch_count=def_pitch_count,
                        defensive_bullpen_era=def_bullpen_era,
                    )

                    run_diff = match_state.home_score - match_state.away_score
                    leverage_idx = BaseballLeverageIndex.compute(
                        inning=inning, is_top=is_top, outs=match_state.outs,
                        run_diff=run_diff, runners_on_base=runners_on_base,
                    )

                    for rid, edge_key in [("home", "edge_home"), ("away", "edge_away")]:
                        edge_bps = fair[edge_key] * 10000
                        if edge_bps <= 0:
                            continue

                        snap = book.get_runner_snapshot(rid)
                        passed = gatekeeper.check_baseball(
                            config=config,
                            runner_id=rid, edge_bps=edge_bps,
                            best_back_price=snap.best_back_price,
                            best_lay_price=snap.best_lay_price,
                            best_back_volume=snap.best_back_size,
                            current_sec=float(game_sec), delay_ms=delay_ms,
                            risk_allows=risk.check_limits(rid, se.Side.Back, snap.best_back_price, config.base_stake),
                            kill_switch=risk.is_kill_switch_active(),
                            inning=inning, outs=match_state.outs,
                            run_diff=run_diff,
                            leverage_index=leverage_idx,
                        )

                        if passed:
                            exchange.submit_order(market.market_id, rid, se.Side.Back,
                                                  snap.best_back_price, config.base_stake,
                                                  "edge_back", ts_ms)
                            total_trades += 1

                    fills = exchange.process_tick(ts_ms, random.random())
                    for fill in fills:
                        risk.record_fill(fill)
                        fill_counter += 1
                        fair_odds_key = "fair_odds_home" if fill.runner_id == "home" else "fair_odds_away"

                        fill_rec = FillRecord(
                            fill_id=fill_counter, order_id=fill_counter,
                            runner_id=fill.runner_id,
                            side="back" if str(fill.side) == "Side.Back" else "lay",
                            price=fill.price, size=fill.size, timestamp_ms=ts_ms,
                            fair_price_at_fill=fair[fair_odds_key],
                            market_price_at_fill=fill.price,
                            elapsed_game_sec=float(game_sec), strategy_tag="edge_back",
                        )
                        session_agg.record_fill(fill_rec)
                        exec_analyzer.record_order(OrderRecord(
                            order_id=fill_counter, runner_id=fill.runner_id, side="back",
                            price=fill.price, size=fill.size, submit_ts_ms=ts_ms - delay_ms,
                            status="filled", filled_size=fill.size, avg_fill_price=fill.price,
                            fill_ts_ms=ts_ms,
                            fair_at_submit=fair[fair_odds_key], market_at_submit=fill.price,
                            fair_at_fill=fair[fair_odds_key], market_at_fill=fill.price,
                            market_5s_after_fill=fill.price * (1 + random.gauss(0, 0.003)),
                            delay_ms=delay_ms, strategy_tag="edge_back",
                        ))

                match_state.outs = 0
                if inning >= 9 and not is_top and match_state.home_score > match_state.away_score:
                    break

        if match_state.home_score > match_state.away_score:
            session_agg.set_outcome(MarketOutcome(winning_runner_id="home"))
        else:
            session_agg.set_outcome(MarketOutcome(winning_runner_id="away"))

        summary = session_agg.compute(total_orders=total_trades)
        exec_metrics = exec_analyzer.compute()

        rej = gatekeeper.rejection_log
        binding = max(rej.counts, key=rej.counts.get) if rej.counts else ""

        results.append(SweepResult(
            min_edge_bps=min_edge,
            total_signals=rej.total_signals,
            total_passed=rej.total_passed,
            pass_rate=rej.pass_rate,
            total_orders=total_trades,
            total_fills=fill_counter,
            fill_ratio=fill_counter / total_trades if total_trades > 0 else 0,
            binding_constraint=binding,
            rejection_breakdown=dict(rej.counts),
            net_pnl=summary.net_pnl,
            edge_capture_ratio=exec_metrics.edge_capture_ratio,
        ))

    return results


def print_sweep_table(sport: str, results: list[SweepResult]) -> None:
    print()
    print("=" * 100)
    print(f"  ACTIVATION FRONTIER: {sport.upper()}")
    print("=" * 100)
    print(f"  {'Edge(bps)':>10} {'Signals':>8} {'Passed':>8} {'Pass%':>7} "
          f"{'Orders':>7} {'Fills':>6} {'FillR':>6} {'NetPnL':>10} {'ECR':>7} {'Binding Constraint'}")
    print("-" * 100)

    for r in results:
        print(f"  {r.min_edge_bps:>10.0f} {r.total_signals:>8} {r.total_passed:>8} "
              f"{r.pass_rate:>6.1%} {r.total_orders:>7} {r.total_fills:>6} "
              f"{r.fill_ratio:>5.1%} {r.net_pnl:>+10.4f} {r.edge_capture_ratio:>7.3f} "
              f"{r.binding_constraint}")

    print("-" * 100)

    # Find activation point
    first_trade = next((r for r in results if r.total_fills > 0), None)
    if first_trade:
        print(f"  >>> First trades appear at min_edge_bps = {first_trade.min_edge_bps:.0f}")
    else:
        print(f"  >>> NO trades at any threshold! Check book generation / pricing / exchange params.")

    # Find sweet spot (highest fill ratio with non-zero trades)
    with_trades = [r for r in results if r.total_fills > 0]
    if with_trades:
        best = max(with_trades, key=lambda r: r.total_fills)
        print(f"  >>> Most fills ({best.total_fills}) at min_edge_bps = {best.min_edge_bps:.0f} "
              f"(pass_rate={best.pass_rate:.1%}, PnL={best.net_pnl:+.4f})")
    print()


if __name__ == "__main__":
    # Sweep from very aggressive (10bps) to conservative (500bps)
    edge_range = [10, 25, 50, 75, 100, 150, 200, 300, 400, 500]

    print("Running basketball sweep...")
    bball_results = sweep_basketball(edge_range)
    print_sweep_table("basketball", bball_results)

    print("Running baseball sweep...")
    baseball_results = sweep_baseball(edge_range)
    print_sweep_table("baseball", baseball_results)

    # Save full results
    os.makedirs("output", exist_ok=True)
    all_results = {
        "basketball": [
            {
                "min_edge_bps": r.min_edge_bps,
                "signals": r.total_signals,
                "passed": r.total_passed,
                "pass_rate": round(r.pass_rate, 4),
                "orders": r.total_orders,
                "fills": r.total_fills,
                "fill_ratio": round(r.fill_ratio, 4),
                "net_pnl": round(r.net_pnl, 4),
                "ecr": round(r.edge_capture_ratio, 4),
                "binding": r.binding_constraint,
                "rejections": r.rejection_breakdown,
            }
            for r in bball_results
        ],
        "baseball": [
            {
                "min_edge_bps": r.min_edge_bps,
                "signals": r.total_signals,
                "passed": r.total_passed,
                "pass_rate": round(r.pass_rate, 4),
                "orders": r.total_orders,
                "fills": r.total_fills,
                "fill_ratio": round(r.fill_ratio, 4),
                "net_pnl": round(r.net_pnl, 4),
                "ecr": round(r.edge_capture_ratio, 4),
                "binding": r.binding_constraint,
                "rejections": r.rejection_breakdown,
            }
            for r in baseball_results
        ],
    }

    with open("output/sweep_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"  [Saved] output/sweep_results.json")
    print()
    print("=" * 100)
    print("  SUMMARY")
    print("=" * 100)
    print("  The sweep reveals the activation frontier — the edge threshold")
    print("  below which trades start appearing. Use this to calibrate")
    print("  strategy parameters before attempting to optimize for PnL.")
    print("=" * 100)
