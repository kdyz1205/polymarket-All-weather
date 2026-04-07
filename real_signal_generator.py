"""
Real signal generator — end-to-end pipeline from real NBA data to
actionable trading signals on Polymarket.

Pipeline:
  1. Fetch game data (from Polymarket gamma API or manual input)
  2. Price the game (Log5 + injuries + home court)
  3. Fetch live Polymarket quotes
  4. Pre-trade deviation check
  5. Generate execution-ready signals

Usage:
  # Price a game from Polymarket slug (needs API access):
  python real_signal_generator.py price nba-min-ind-2026-04-07

  # Price from manual data (works offline):
  python real_signal_generator.py manual

  # Full pipeline with execution bridge:
  python real_signal_generator.py trade nba-min-ind-2026-04-07
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from dataclasses import dataclass, asdict

from src.data.nba_client import (
    NBADataClient, GameData, TeamGameData, PlayerInjury,
    estimate_player_impact,
)
from src.pricing.real_game_pricer import RealGamePricer, PricingResult
from src.execution.market_mapper import MarketMapper, MarketMapping
from src.execution.quote_fetcher import LiveQuoteFetcher, LiveQuote
from src.execution.pre_trade_check import PreTradeComparator, DeviationResult, Signal
from src.execution.order_builder import OrderBuilder, ExecutionOrder
from src.execution.manual_executor import ManualExecutor, ExecutionDecision

from live_observer import (
    MicroLiveState, load_state, save_state, PendingOrder,
    queue_order, QUEUE_PATH, MAX_STAKE_PER_ORDER,
)


SIGNALS_LOG = "micro_live_logs/real_signals.jsonl"


@dataclass
class RealSignal:
    """A trading signal generated from real game analysis."""
    game_id: str
    timestamp: str
    side: str                  # "home" | "away"
    team: str                  # team name
    fair_prob: float
    market_prob: float
    edge_bps: float
    net_edge_bps: float        # after estimated fees
    confidence: float
    signal_strength: str       # "strong" | "moderate" | "weak" | "none"
    actionable: bool
    fee_bps: float = 80.0
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def price_from_polymarket(slug: str) -> PricingResult | None:
    """Fetch data from Polymarket and price the game."""
    client = NBADataClient()
    game = client.fetch_from_polymarket(slug)
    client.close()

    if not game:
        print(f"  Could not fetch game data for slug: {slug}")
        return None

    pricer = RealGamePricer()
    result = pricer.price(game)
    print(pricer.display_pricing(result))
    return result


def price_min_vs_ind() -> tuple[GameData, PricingResult]:
    """Price the MIN vs IND game using known data from the user's screenshots.

    This is the manually-verified data from the Polymarket gamma API response.
    """
    client = NBADataClient()

    game = client.build_game_manual(
        home_abbr="IND", away_abbr="MIN",
        home_name="Pacers", away_name="Timberwolves",
        home_wins=18, home_losses=60,
        away_wins=46, away_losses=30,
        game_date="2026-04-07",
        home_injuries=[
            {"name": "Tyrese Haliburton", "status": "questionable", "injury": "right calf strain"},
            {"name": "Pascal Siakam", "status": "out", "injury": "torn ACL"},
            {"name": "T.J. McConnell", "status": "out", "injury": "unknown"},
            {"name": "Andrew Nembhard", "status": "out", "injury": "unknown"},
            {"name": "Johnny Furphy", "status": "out", "injury": "unknown"},
        ],
        away_injuries=[
            {"name": "Anthony Edwards", "status": "out", "injury": "left knee, ruled out"},
            {"name": "Jaden McDaniels", "status": "out", "injury": "knee tendinitis"},
        ],
        polymarket_slug="nba-min-ind-2026-04-07",
        condition_id="0xb16fe5b3db23336a64193f4724af38294775bf17f79bf2d513a3093c77d9933a",
        home_token="102209872571136621994776319275684405848207233446672233571313042167300202637340",
        away_token="67001198723470928325024317997267428884145695729404141328875363513742499928133",
        home_price=0.135,
        away_price=0.865,
    )

    pricer = RealGamePricer()
    result = pricer.price(game)
    return game, result


def generate_real_signals(game: GameData, result: PricingResult,
                          fee_bps: float = 80.0) -> list[RealSignal]:
    """Generate trading signals from pricing result."""
    signals = []
    ts = datetime.now().isoformat()

    for side in ["home", "away"]:
        if side == "home":
            fair = result.fair_home_prob
            market = result.market_home_price
            edge = result.edge_home_bps
            team = result.home_team
        else:
            fair = result.fair_away_prob
            market = result.market_away_price
            edge = result.edge_away_bps
            team = result.away_team

        net_edge = edge - fee_bps

        if edge > 200:
            strength = "strong"
        elif edge > 100:
            strength = "moderate"
        elif edge > 50:
            strength = "weak"
        else:
            strength = "none"

        actionable = net_edge > 20 and result.confidence >= 0.5

        signal = RealSignal(
            game_id=game.game_id,
            timestamp=ts,
            side=side,
            team=team,
            fair_prob=fair,
            market_prob=market,
            edge_bps=edge,
            net_edge_bps=net_edge,
            confidence=result.confidence,
            signal_strength=strength,
            actionable=actionable,
            fee_bps=fee_bps,
            notes=f"Log5={result.log5_home_prob:.3f}, HCA={result.home_court_adj:+.3f}, "
                  f"H_inj={result.home_injury_adj:+.3f}, A_inj={result.away_injury_adj:+.3f}",
        )
        signals.append(signal)

    return signals


def run_trade_pipeline(slug: str) -> None:
    """Full pipeline: fetch → price → signal → pre-trade check → queue."""
    print(f"\n{'='*70}")
    print(f"  REAL SIGNAL GENERATOR — TRADE PIPELINE")
    print(f"  Slug: {slug}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    # Step 1: Get game data
    print("\n  [1] Fetching game data...")
    client = NBADataClient()
    game = client.fetch_from_polymarket(slug)
    client.close()

    if not game:
        print("  FAILED: Could not fetch game data.")
        print("  Falling back to manual mode...")
        if slug == "nba-min-ind-2026-04-07":
            game, result = price_min_vs_ind()
        else:
            print("  No manual data available for this game.")
            return
    else:
        pricer = RealGamePricer()
        result = pricer.price(game)

    # Step 2: Display pricing
    print("\n  [2] Pricing result:")
    pricer = RealGamePricer()
    print(pricer.display_pricing(result))

    # Step 3: Generate signals
    print("\n  [3] Generating signals...")
    signals = generate_real_signals(game, result)

    actionable = [s for s in signals if s.actionable]
    print(f"    Total signals: {len(signals)}")
    print(f"    Actionable: {len(actionable)}")

    for s in signals:
        icon = ">>>" if s.actionable else "   "
        print(f"    {icon} {s.side:>5} ({s.team[:20]:>20}): "
              f"edge={s.edge_bps:+.0f}bps  net={s.net_edge_bps:+.0f}bps  "
              f"strength={s.signal_strength}  {'ACTIONABLE' if s.actionable else 'skip'}")

    if not actionable:
        print(f"\n  NO ACTIONABLE SIGNALS — no trade today on this game.")
        _log_signals(signals)
        return

    # Step 4: Pre-trade check with real Polymarket quotes
    print(f"\n  [4] Fetching live Polymarket quotes...")
    fetcher = LiveQuoteFetcher()
    comparator = PreTradeComparator()
    builder = OrderBuilder()
    state = load_state()

    for signal in actionable:
        token_id = (game.polymarket_home_token if signal.side == "home"
                    else game.polymarket_away_token)

        if not token_id:
            print(f"    No token ID for {signal.side}, skipping.")
            continue

        print(f"    Fetching quote for {signal.side} ({token_id[:30]}...)...", end=" ")
        quote = fetcher.fetch(token_id)

        if not quote.is_valid:
            print(f"FAILED ({quote.error})")
            # Use market price from gamma API as fallback
            print(f"    Using gamma API price as fallback: {signal.market_prob:.3f}")
            quote = LiveQuote(
                token_id=token_id,
                timestamp_ms=int(time.time() * 1000),
                best_bid=signal.market_prob - 0.005,
                best_ask=signal.market_prob + 0.005,
                mid_price=signal.market_prob,
                spread=0.01,
                bid_depth=1000,
                ask_depth=1000,
            )

        # Convert fair probability to decimal odds for comparator
        fair_odds = 1.0 / signal.fair_prob if signal.fair_prob > 0 else 99.0
        deviation = comparator.compare(fair_odds, quote)

        signal_icon = {"green": "GREEN", "yellow": "YELLOW", "red": "RED"}
        print(f"{signal_icon.get(deviation.signal.value, '???')}")
        print(f"      Fair: {signal.fair_prob:.1%} ({fair_odds:.2f})  "
              f"Market: {quote.mid_price:.3f} ({1/quote.mid_price:.2f})  "
              f"Dev: {deviation.deviation_pct:.1f}%")

        if deviation.signal == Signal.RED:
            print(f"      BLOCKED by pre-trade check: {deviation.reason}")
            continue

        # Step 5: Queue for manual confirmation
        print(f"\n  [5] Queuing order for manual confirmation...")
        order = PendingOrder(
            order_id=f"real_{game.game_id}_{signal.side}_{int(time.time())}",
            timestamp=datetime.now().isoformat(),
            sport="basketball",
            market_id=game.polymarket_slug,
            runner_id=signal.side,
            side="buy",
            price=fair_odds,
            size=1.0,
            edge_bps=signal.edge_bps,
            net_edge_bps=signal.net_edge_bps,
            fee_bps=signal.fee_bps,
        )

        if queue_order(order, state):
            print(f"    Order queued for confirmation!")
            print(f"    Run: python live_observer.py review")
        else:
            print(f"    Order BLOCKED by safety limits.")

    fetcher.close()
    _log_signals(signals)
    save_state(state)

    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Review orders: python live_observer.py review")
    print(f"  Check status:  python live_observer.py status")
    print(f"{'='*70}")


def run_manual_pricing() -> None:
    """Interactive manual game pricing."""
    print(f"\n  MANUAL GAME PRICING")
    print(f"  {'─'*50}")

    try:
        away_abbr = input("  Away team abbrev (e.g. MIN): ").strip().upper()
        home_abbr = input("  Home team abbrev (e.g. IND): ").strip().upper()
        away_name = input(f"  Away team name [{away_abbr}]: ").strip() or away_abbr
        home_name = input(f"  Home team name [{home_abbr}]: ").strip() or home_abbr
        away_record = input("  Away record (W-L, e.g. 46-30): ").strip()
        home_record = input("  Home record (W-L, e.g. 18-60): ").strip()
        game_date = input(f"  Game date [2026-04-07]: ").strip() or "2026-04-07"

        aw, al = map(int, away_record.split("-"))
        hw, hl = map(int, home_record.split("-"))

        # Injuries
        away_injuries = []
        print(f"\n  Away injuries (enter player names, empty to stop):")
        while True:
            name = input("    Player name: ").strip()
            if not name:
                break
            status = input(f"    Status [out]: ").strip() or "out"
            away_injuries.append({"name": name, "status": status})

        home_injuries = []
        print(f"  Home injuries:")
        while True:
            name = input("    Player name: ").strip()
            if not name:
                break
            status = input(f"    Status [out]: ").strip() or "out"
            home_injuries.append({"name": name, "status": status})

        # Market prices
        away_price = float(input(f"\n  Polymarket {away_name} price (0-1, e.g. 0.865): ").strip() or "0")
        home_price = float(input(f"  Polymarket {home_name} price (0-1, e.g. 0.135): ").strip() or "0")

        client = NBADataClient()
        game = client.build_game_manual(
            home_abbr=home_abbr, away_abbr=away_abbr,
            home_name=home_name, away_name=away_name,
            home_wins=hw, home_losses=hl,
            away_wins=aw, away_losses=al,
            game_date=game_date,
            home_injuries=home_injuries,
            away_injuries=away_injuries,
            home_price=home_price,
            away_price=away_price,
        )

        pricer = RealGamePricer()
        result = pricer.price(game)
        print(pricer.display_pricing(result))

        signals = generate_real_signals(game, result)
        print(f"\n  SIGNALS:")
        for s in signals:
            icon = ">>>" if s.actionable else "   "
            print(f"    {icon} {s.side:>5}: edge={s.edge_bps:+.0f}bps  "
                  f"net={s.net_edge_bps:+.0f}bps  {s.signal_strength}")

        _log_signals(signals)

    except (EOFError, KeyboardInterrupt):
        print("\n  Cancelled.")


def _log_signals(signals: list[RealSignal]) -> None:
    """Append signals to the log."""
    os.makedirs(os.path.dirname(SIGNALS_LOG) or ".", exist_ok=True)
    with open(SIGNALS_LOG, "a") as f:
        for s in signals:
            f.write(json.dumps(s.to_dict()) + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python real_signal_generator.py price <slug>      — price from Polymarket API")
        print("  python real_signal_generator.py trade <slug>      — full trade pipeline")
        print("  python real_signal_generator.py manual            — manual data entry")
        print("  python real_signal_generator.py min-ind           — price MIN vs IND (hardcoded)")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "price" and len(sys.argv) > 2:
        price_from_polymarket(sys.argv[2])
    elif cmd == "trade" and len(sys.argv) > 2:
        run_trade_pipeline(sys.argv[2])
    elif cmd == "manual":
        run_manual_pricing()
    elif cmd == "min-ind":
        game, result = price_min_vs_ind()
        pricer = RealGamePricer()
        print(pricer.display_pricing(result))
        signals = generate_real_signals(game, result)
        print(f"\n  SIGNALS:")
        for s in signals:
            icon = ">>>" if s.actionable else "   "
            print(f"    {icon} {s.side:>5} ({s.team[:20]:>20}): "
                  f"edge={s.edge_bps:+.0f}bps  net={s.net_edge_bps:+.0f}bps  "
                  f"{s.signal_strength}  {'TRADE' if s.actionable else 'skip'}")
        _log_signals(signals)
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
