"""
Baseball model ablation sweep — measures marginal value of each feature layer.

Each configuration uses the SAME game path (same seed → same PA outcomes,
same score changes, same book noise). Only BaseballFeatureFlags differ.
This isolates the effect of each enhancement layer.

7 staged configurations:
  baseline      → all flags off
  +base_out     → enable_base_out_state
  +leverage     → + enable_leverage
  +walkoff      → + enable_walkoff
  +fatigue      → + enable_fatigue
  +bullpen      → + enable_bullpen
  full          → + enable_blowout_asymmetry (all on)

Run: source .venv/bin/activate && python ablation_runner.py [N_GAMES]
"""

import csv
import json
import os
import random
import sys
from dataclasses import dataclass, field, asdict

import sports_engine as se
from src.pricing.engine import (
    PricingEngine,
    TeamRating,
    ShockAccumulator,
    BaseballLeverageIndex,
    BaseballPitchingContext,
    BaseballFeatureFlags,
)
from src.strategy import BaseballStrategyConfig, StrategyGatekeeper


# ─── Ablation configurations (staged, cumulative) ───

ABLATIONS: dict[str, BaseballFeatureFlags] = {
    "baseline": BaseballFeatureFlags(
        enable_base_out_state=False,
        enable_leverage=False,
        enable_walkoff=False,
        enable_fatigue=False,
        enable_bullpen=False,
        enable_blowout_asymmetry=False,
    ),
    "+base_out": BaseballFeatureFlags(
        enable_base_out_state=True,
        enable_leverage=False,
        enable_walkoff=False,
        enable_fatigue=False,
        enable_bullpen=False,
        enable_blowout_asymmetry=False,
    ),
    "+leverage": BaseballFeatureFlags(
        enable_base_out_state=True,
        enable_leverage=True,
        enable_walkoff=False,
        enable_fatigue=False,
        enable_bullpen=False,
        enable_blowout_asymmetry=False,
    ),
    "+walkoff": BaseballFeatureFlags(
        enable_base_out_state=True,
        enable_leverage=True,
        enable_walkoff=True,
        enable_fatigue=False,
        enable_bullpen=False,
        enable_blowout_asymmetry=False,
    ),
    "+fatigue": BaseballFeatureFlags(
        enable_base_out_state=True,
        enable_leverage=True,
        enable_walkoff=True,
        enable_fatigue=True,
        enable_bullpen=False,
        enable_blowout_asymmetry=False,
    ),
    "+bullpen": BaseballFeatureFlags(
        enable_base_out_state=True,
        enable_leverage=True,
        enable_walkoff=True,
        enable_fatigue=True,
        enable_bullpen=True,
        enable_blowout_asymmetry=False,
    ),
    "full": BaseballFeatureFlags(
        enable_base_out_state=True,
        enable_leverage=True,
        enable_walkoff=True,
        enable_fatigue=True,
        enable_bullpen=True,
        enable_blowout_asymmetry=True,
    ),
}

# Quick 3-config comparison: the decision that matters
COMPARISON_CONFIGS: dict[str, BaseballFeatureFlags] = {
    "baseline": ABLATIONS["baseline"],
    "trading": BaseballFeatureFlags.trading_default(),
    "full": BaseballFeatureFlags.research_full(),
}


# ─── Result container ───

@dataclass
class AblationResult:
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
    pnl_per_game: list[float] = field(default_factory=list)

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

    @property
    def mean_pnl(self) -> float:
        return self.total_pnl / max(self.games, 1)

    @property
    def std_pnl(self) -> float:
        if len(self.pnl_per_game) < 2:
            return 0.0
        m = self.mean_pnl
        return (sum((x - m) ** 2 for x in self.pnl_per_game) / (len(self.pnl_per_game) - 1)) ** 0.5

    @property
    def edge_retained_pct(self) -> float:
        if self.total_edge_bps <= 0 or self.edge_samples == 0:
            return 0.0
        # Edge retained = realized PnL / theoretical edge
        theoretical = self.total_edge_bps / 10000.0 * 20.0  # approx: edge_bps * base_stake
        return (self.total_pnl / theoretical * 100) if theoretical > 0 else 0.0


# ─── Book generation (deterministic with rng) ───

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


# ─── PA outcome mapping ───

_ETYPE_TO_PA = {
    se.MatchEventType.Strikeout: "strikeout",
    se.MatchEventType.Walk: "walk",
    se.MatchEventType.Single: "single",
    se.MatchEventType.Double: "double",
    se.MatchEventType.Triple: "triple",
    se.MatchEventType.HomeRun: "home_run",
    se.MatchEventType.DoublePlay: "double_play",
}


# ─── Pre-generate game paths (seed → list of PA events) ───

@dataclass
class PAEvent:
    """A single plate appearance outcome."""
    game_sec: int
    inning: int
    is_top: bool
    batting_team: str
    event_type: object  # se.MatchEventType or None
    pa_outcome: str     # key into _PA_PITCH_ESTIMATES
    is_generic_out: bool


def generate_game_path(seed: int) -> tuple[list[PAEvent], int, int]:
    """Generate a deterministic sequence of PA events for a game.

    Returns (events, home_rating_offset, away_rating_offset) so all
    configs see the same team strengths too.
    """
    rng = random.Random(seed)
    home_off = rng.randint(-100, 100)
    away_off = rng.randint(-100, 100)
    events: list[PAEvent] = []
    game_sec = 0

    # We simulate the full 9-inning game structure to generate events,
    # but don't apply them to match state here — that happens per-config.
    for inning in range(1, 10):
        for is_top in [True, False]:
            batting_team = "away" if is_top else "home"
            outs = 0
            pa_count = 0
            while outs < 3 and pa_count < 15:
                pa_count += 1
                game_sec += rng.randint(15, 45)

                r = rng.random()
                event_type = None
                is_generic_out = False
                if r < 0.22:
                    event_type = se.MatchEventType.Strikeout
                    outs += 1
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
                    outs += 1
                    is_generic_out = True

                pa_outcome = _ETYPE_TO_PA.get(event_type, "out") if event_type else "out"
                events.append(PAEvent(
                    game_sec=game_sec, inning=inning, is_top=is_top,
                    batting_team=batting_team, event_type=event_type,
                    pa_outcome=pa_outcome, is_generic_out=is_generic_out,
                ))

    return events, home_off, away_off


def run_one_game(
    flags: BaseballFeatureFlags,
    config_name: str,
    game_path: list[PAEvent],
    home_off: int,
    away_off: int,
    book_seed: int,
) -> AblationResult:
    """Simulate one baseball game with the given feature flags.

    Uses pre-generated game_path for PA events and a separate book_seed
    for book noise — both identical across configs.
    """
    book_rng = random.Random(book_seed)
    result = AblationResult(name=config_name)

    market = se.Market("abl", se.Sport.Baseball, "g", "ml")
    match_state = se.MatchState("g", se.Sport.Baseball)
    book = se.MarketBook("abl")
    book.add_runner("home", "H")
    book.add_runner("away", "A")
    exchange = se.MockExchange(5000, 0.25)
    risk = se.RiskEngine("abl", 5000.0, 3000.0)
    risk.add_runner("home")
    risk.add_runner("away")

    pricer = PricingEngine(
        "baseball",
        TeamRating("H", 1500 + home_off),
        TeamRating("A", 1500 + away_off),
        model_weight=0.6,
        baseball_flags=flags,
    )

    se.MarketStateMachine.apply_to_market(market, se.MarketStatus.OpenPrematch)
    se.MarketStateMachine.apply_to_market(market, se.MarketStatus.InPlay)

    strat_config = BaseballStrategyConfig.aggressive()
    gatekeeper = StrategyGatekeeper()

    pitching_ctx = BaseballPitchingContext(
        home_bullpen_era=3.80 + random.Random(book_seed + 1000).gauss(0, 0.4),
        away_bullpen_era=3.80 + random.Random(book_seed + 2000).gauss(0, 0.4),
    )

    mid_game_predictions: list[tuple[int, float]] = []
    game_pnl = 0.0
    prev_inning = 0
    prev_is_top = True

    for pa in game_path:
        ts_ms = pa.game_sec * 1000

        # Start new half-inning if needed
        if pa.inning != prev_inning or pa.is_top != prev_is_top:
            event = se.MatchEvent(
                f"inn_{pa.inning}", ts_ms,
                se.MatchEventType.PeriodStart, float(pa.inning),
                "away" if pa.is_top else "home",
            )
            se.MatchStateMachine.apply_event(match_state, event)
            prev_inning = pa.inning
            prev_is_top = pa.is_top

        # Walk-off skip
        if pa.inning >= 9 and not pa.is_top and match_state.home_score > match_state.away_score:
            continue

        # Apply event
        if pa.is_generic_out:
            match_state.outs += 1
        elif pa.event_type:
            if pa.event_type == se.MatchEventType.Strikeout:
                match_state.outs += 1
            event = se.MatchEvent(
                f"pa_{pa.game_sec}", ts_ms,
                pa.event_type, float(pa.inning), pa.batting_team,
            )
            se.MatchStateMachine.apply_event(match_state, event)

        # Track pitching
        pitching_ctx.record_plate_appearance(pa.batting_team, pa.pa_outcome)

        if match_state.outs >= 3:
            continue

        # Auto pitcher change
        defensive_team = "away" if pa.batting_team == "home" else "home"
        if pitching_ctx.should_change_pitcher(defensive_team, pa.inning):
            pitching_ctx.pitcher_change(defensive_team, "reliever")

        # Get state
        runners = match_state.runners_on_base()
        def_pc, def_era = pitching_ctx.defensive_state(pa.batting_team)

        update_kwargs = dict(
            elapsed_sec=pa.game_sec,
            home_score=match_state.home_score,
            away_score=match_state.away_score,
            inning=pa.inning, is_top=pa.is_top, outs=match_state.outs,
            runners_on_base=runners,
            batting_team_is_home=(pa.batting_team == "home"),
            defensive_pitch_count=def_pc,
            defensive_bullpen_era=def_era,
        )

        # Step 1: raw model probs
        raw = pricer.update(**update_kwargs)

        # Step 2: biased book (same rng → same noise across configs)
        for rid in ["home", "away"]:
            odds_key = "fair_odds_home" if rid == "home" else "fair_odds_away"
            fo = raw[odds_key]
            bias = fo * 0.04 + book_rng.gauss(0, 0.01) if fo < 2.0 else -fo * 0.03 + book_rng.gauss(0, 0.01)
            bl, ll = generate_book_levels(book_rng, fo, noise=0.03, market_bias=bias)
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

        # Mid-game prediction (inning 3+, first PA of half-inning)
        if pa.inning >= 3:
            mid_game_predictions.append((pa.inning, fair["p_home"]))

        # Edge signals + gatekeeper
        run_diff = match_state.home_score - match_state.away_score
        leverage_idx = BaseballLeverageIndex.compute(
            inning=pa.inning, is_top=pa.is_top, outs=match_state.outs,
            run_diff=run_diff, runners_on_base=runners,
        )

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
                current_sec=float(pa.game_sec), delay_ms=5000,
                risk_allows=risk.check_limits(rid, se.Side.Back, snap.best_back_price, strat_config.base_stake),
                kill_switch=risk.is_kill_switch_active(),
                inning=pa.inning, outs=match_state.outs,
                run_diff=run_diff,
                leverage_index=leverage_idx,
            )

            if passed:
                exchange.submit_order(market.market_id, rid, se.Side.Back,
                                      snap.best_back_price, strat_config.base_stake,
                                      f"edge_{rid}", ts_ms)

        fills = exchange.process_tick(ts_ms, book_rng.random())
        for fill in fills:
            risk.record_fill(fill)
            result.total_fills += 1
            fair_key = "fair_odds_home" if fill.runner_id == "home" else "fair_odds_away"
            edge_captured = (fair[fair_key] - fill.price) * fill.size
            game_pnl += edge_captured

    # Determine winner
    home_won = match_state.home_score > match_state.away_score

    # Direction accuracy
    for _, p_home in mid_game_predictions:
        predicted_home = p_home > 0.5
        if predicted_home == home_won:
            result.correct_predictions += 1
        result.total_predictions += 1

    result.games = 1
    result.total_signals = gatekeeper.rejection_log.total_signals
    result.total_passed = gatekeeper.rejection_log.total_passed
    result.total_pnl = game_pnl
    result.pnl_per_game = [game_pnl]

    return result


def merge_results(results: list[AblationResult]) -> AblationResult:
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
        merged.pnl_per_game.extend(r.pnl_per_game)
    return merged


def run_ablation(n_games: int = 50, base_seed: int = 42,
                 mode: str = "full") -> list[AblationResult]:
    """Run ablation sweep.

    mode='full': 7 staged configs (detailed marginal analysis)
    mode='compare': 3 configs (baseline / trading_default / research_full)
    """
    configs = ABLATIONS if mode == "full" else COMPARISON_CONFIGS
    print(f"\n{'='*90}")
    print(f"  BASEBALL MODEL ABLATION — {mode.upper()} — {n_games} games × {len(configs)} configs")
    print(f"  Same game path per seed, only feature flags differ")
    print(f"{'='*90}\n")

    # Pre-generate all game paths
    game_paths = []
    for i in range(n_games):
        path, h_off, a_off = generate_game_path(base_seed + i)
        game_paths.append((path, h_off, a_off, base_seed + i + 10000))  # book_seed offset

    all_merged: list[AblationResult] = []

    for config_name, flags in configs.items():
        results = []
        for path, h_off, a_off, book_seed in game_paths:
            r = run_one_game(flags, config_name, path, h_off, a_off, book_seed)
            results.append(r)
        merged = merge_results(results)
        all_merged.append(merged)
        print(f"  {config_name:<14} done — acc={merged.accuracy:.1%} fills={merged.total_fills} pnl={merged.total_pnl:+.1f}")

    # ─── Absolute results table ───
    print(f"\n{'─'*90}")
    print(f"  {'Config':<14} {'Acc%':>7} {'AvgEdge':>8} {'Signals':>8} {'Pass%':>8} "
          f"{'Fills':>7} {'RetainPct':>10} {'MeanPnL':>10} {'StdPnL':>9}")
    print(f"{'─'*90}")

    for m in all_merged:
        print(f"  {m.name:<14} {m.accuracy*100:6.1f}% {m.mean_edge_bps:7.1f} {m.total_signals:8d} "
              f"{m.pass_rate*100:7.1f}% {m.total_fills:7d} {m.edge_retained_pct:9.1f}% "
              f"{m.mean_pnl:+10.2f} {m.std_pnl:9.2f}")

    # ─── Marginal delta table ───
    print(f"\n{'─'*90}")
    print(f"  MARGINAL DELTAS (each row = effect of adding that feature)")
    print(f"{'─'*90}")
    print(f"  {'Layer':<14} {'ΔAcc':>8} {'ΔEdge':>8} {'ΔPass%':>8} "
          f"{'ΔFills':>8} {'ΔRetain':>9} {'ΔPnL':>10}")
    print(f"{'─'*90}")

    for i in range(1, len(all_merged)):
        prev = all_merged[i - 1]
        curr = all_merged[i]
        d_acc = (curr.accuracy - prev.accuracy) * 100
        d_edge = curr.mean_edge_bps - prev.mean_edge_bps
        d_pass = (curr.pass_rate - prev.pass_rate) * 100
        d_fills = curr.total_fills - prev.total_fills
        d_retain = curr.edge_retained_pct - prev.edge_retained_pct
        d_pnl = curr.mean_pnl - prev.mean_pnl

        print(f"  {curr.name:<14} {d_acc:+7.1f}% {d_edge:+7.1f} {d_pass:+7.1f}% "
              f"{d_fills:+8d} {d_retain:+8.1f}% {d_pnl:+10.2f}")

    print(f"{'─'*90}")
    print(f"\n  Total: baseline→full  ΔAcc={((all_merged[-1].accuracy - all_merged[0].accuracy)*100):+.1f}%  "
          f"ΔPnL={all_merged[-1].mean_pnl - all_merged[0].mean_pnl:+.2f}\n")

    return all_merged


def export_results(results: list[AblationResult], output_dir: str = "output") -> None:
    """Export ablation results to JSON and CSV."""
    os.makedirs(output_dir, exist_ok=True)

    # JSON
    json_data = []
    for r in results:
        json_data.append({
            "name": r.name,
            "games": r.games,
            "accuracy": round(r.accuracy, 4),
            "mean_edge_bps": round(r.mean_edge_bps, 2),
            "total_signals": r.total_signals,
            "total_passed": r.total_passed,
            "pass_rate": round(r.pass_rate, 4),
            "total_fills": r.total_fills,
            "fill_rate": round(r.fill_rate, 4),
            "edge_retained_pct": round(r.edge_retained_pct, 2),
            "mean_pnl": round(r.mean_pnl, 2),
            "std_pnl": round(r.std_pnl, 2),
            "total_pnl": round(r.total_pnl, 2),
        })

    json_path = os.path.join(output_dir, "baseball_ablation.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  Exported: {json_path}")

    # CSV
    csv_path = os.path.join(output_dir, "baseball_ablation.csv")
    if json_data:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=json_data[0].keys())
            writer.writeheader()
            writer.writerows(json_data)
    print(f"  Exported: {csv_path}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    mode = sys.argv[2] if len(sys.argv) > 2 else "compare"
    results = run_ablation(n_games=n, mode=mode)
    export_results(results)
