"""
Execution CLI — single entry point for the full micro-live pipeline.

Combines: signal generation → market lookup → live quote → pre-trade
deviation check → order build → interactive confirm/reject.

Usage:
  # Full pipeline: generate signals, check real prices, review
  python execution_cli.py run [n_games]

  # Just check real prices for existing queued orders
  python execution_cli.py check

  # Show today's status
  python execution_cli.py status

  # Register a market mapping interactively
  python execution_cli.py add-market

  # Run nightly paper + observe + status
  python execution_cli.py nightly
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import date, datetime

from src.execution.market_mapper import MarketMapper, MarketMapping
from src.execution.quote_fetcher import LiveQuoteFetcher, LiveQuote
from src.execution.pre_trade_check import PreTradeComparator, DeviationResult, Signal
from src.execution.order_builder import OrderBuilder, ExecutionOrder
from src.execution.manual_executor import ManualExecutor, ExecutionDecision
from live_observer import (
    load_state, save_state, show_status, PendingOrder,
    QUEUE_PATH, MAX_STAKE_PER_ORDER, MAX_TOTAL_EXPOSURE, MAX_DAILY_LOSS,
)


EXECUTION_QUEUE_PATH = "execution_queue.jsonl"


def run_full_pipeline(n_games: int = 1) -> None:
    """Full pipeline: generate signals → check prices → review."""
    print(f"\n{'='*70}")
    print(f"  EXECUTION PIPELINE — {date.today().isoformat()}")
    print(f"{'='*70}")

    # Step 1: Paper sanity check
    print("\n  [1/5] Paper sanity check...")
    from paper_runner import run_paper
    report = run_paper(sport="basketball", n_games=5, base_seed=int(time.time()) % 10000)
    report.compute_aggregates()
    print(f"    PnL: {report.total_shadow_pnl:+.2f} ({report.n_games} games)")
    print(f"    Fills: {report.total_shadow_fills}, Acc: {report.direction_accuracy:.0%}")

    if report.total_shadow_pnl < -500:
        print(f"    ABORT: paper PnL too negative ({report.total_shadow_pnl:+.2f})")
        return

    # Step 2: Generate candidate signals
    print(f"\n  [2/5] Generating signals ({n_games} game sim)...")
    from live_observer import run_observe
    run_observe(n_games=n_games, seed=int(time.time()) % 100000)

    # Step 3: Load queued orders and check against real markets
    print(f"\n  [3/5] Checking real market prices...")
    check_real_prices()

    # Step 4: Interactive review
    print(f"\n  [4/5] Review execution candidates...")
    review_execution_candidates()

    # Step 5: Status
    print(f"\n  [5/5] Final status:")
    show_status()


def check_real_prices() -> None:
    """Check queued orders against real Polymarket prices."""
    if not os.path.exists(QUEUE_PATH):
        print("    No queued orders.")
        return

    with open(QUEUE_PATH) as f:
        orders = [json.loads(l) for l in f if l.strip()]

    pending = [o for o in orders if o["status"] == "pending"]
    if not pending:
        print("    No pending orders.")
        return

    mapper = MarketMapper()
    fetcher = LiveQuoteFetcher()
    comparator = PreTradeComparator()
    builder = OrderBuilder()

    active_mappings = mapper.list_active(sport="basketball")

    if not active_mappings:
        print("    No active market mappings found.")
        print("    Add mappings with: python execution_cli.py add-market")
        print(f"    Or edit: config/market_mappings.json")
        print(f"\n    Candidate orders (system prices, no real comparison):")
        for i, o in enumerate(pending):
            print(f"    [{i+1}] {o['runner_id']} {o['side']} @{o['price']:.3f} "
                  f"x${o['size']:.2f} edge={o['edge_bps']:.0f}bps "
                  f"net={o['net_edge_bps']:.0f}bps")
        return

    print(f"    Found {len(active_mappings)} active mapping(s)")

    execution_candidates = []

    for o in pending:
        # Try to match order to a real market
        best_mapping = active_mappings[0] if active_mappings else None

        if not best_mapping:
            continue

        token_id = (best_mapping.yes_token_id if o["runner_id"] == "home"
                    else best_mapping.no_token_id)

        if not token_id or token_id.startswith("REPLACE"):
            print(f"    [{o['order_id']}] Skipped: no real token_id in mapping")
            continue

        # Fetch real quote
        print(f"    Fetching quote for {token_id[:20]}...", end=" ", flush=True)
        quote = fetcher.fetch(token_id)

        if not quote.is_valid:
            print(f"FAILED ({quote.error})")
            continue

        # Pre-trade deviation check
        system_odds = o["price"]  # decimal odds from the system
        deviation = comparator.compare(system_odds, quote)

        signal_icon = {"green": "GREEN", "yellow": "YELLOW", "red": "RED  "}
        print(f"{signal_icon[deviation.signal.value]}")

        print(f"      System: {deviation.system_price:.3f}  "
              f"Market: {deviation.market_price:.3f}  "
              f"Dev: {deviation.deviation_pct:.1f}%  "
              f"Spread: {deviation.spread_bps:.0f}bps")

        if deviation.signal != Signal.RED:
            exec_order = builder.build(
                mapping=best_mapping,
                deviation=deviation,
                runner_id=o["runner_id"],
                side="buy",
                size=o["size"],
                edge_bps=o["edge_bps"],
                net_edge_bps=o["net_edge_bps"],
            )
            if exec_order:
                execution_candidates.append((o, exec_order, deviation))

    if execution_candidates:
        # Write execution candidates
        with open(EXECUTION_QUEUE_PATH, "w") as f:
            for orig, order, dev in execution_candidates:
                entry = {
                    "original_order": orig,
                    "execution_order": order.to_dict(),
                    "deviation": {
                        "signal": dev.signal.value,
                        "deviation_pct": dev.deviation_pct,
                        "system_price": dev.system_price,
                        "market_price": dev.market_price,
                    },
                }
                f.write(json.dumps(entry) + "\n")
        print(f"\n    {len(execution_candidates)} orders passed pre-trade check → {EXECUTION_QUEUE_PATH}")
    else:
        print(f"\n    No orders passed pre-trade check.")

    fetcher.close()


def review_execution_candidates() -> None:
    """Interactive review of execution-ready candidates."""
    if not os.path.exists(EXECUTION_QUEUE_PATH):
        # Fall back to reviewing raw queued orders
        if os.path.exists(QUEUE_PATH):
            print("\n    No execution candidates (no real price data).")
            print("    Reviewing raw signal candidates instead:\n")
            _review_raw_orders()
        return

    with open(EXECUTION_QUEUE_PATH) as f:
        candidates = [json.loads(l) for l in f if l.strip()]

    if not candidates:
        print("    No execution candidates to review.")
        return

    executor = ManualExecutor()
    state = load_state()

    print(f"\n{'='*70}")
    print(f"  EXECUTION REVIEW — {len(candidates)} candidates")
    print(f"  Exposure: ${state.total_exposure:.2f}/${MAX_TOTAL_EXPOSURE:.2f}  "
          f"Daily PnL: {state.daily_pnl:+.2f}  "
          f"Kill: {'ON' if state.kill_switch else 'off'}")
    print(f"{'='*70}")

    for i, c in enumerate(candidates):
        o = c["execution_order"]
        d = c["deviation"]
        signal_icon = {"green": "[OK]", "yellow": "[!!]", "red": "[XX]"}

        print(f"\n  [{i+1}/{len(candidates)}] {o.get('description', o['runner_id'])}")
        print(f"    Signal:     {signal_icon.get(d['signal'], '?')} {d['signal'].upper()}")
        print(f"    Sys price:  {d['system_price']:.3f}")
        print(f"    Mkt price:  {d['market_price']:.3f}")
        print(f"    Deviation:  {d['deviation_pct']:.1f}%")
        print(f"    Edge:       {o['edge_bps']:.0f}bps  Net: {o['net_edge_bps']:.0f}bps")
        print(f"    Amount:     ${o['size']:.2f}")
        print(f"    Token:      {o['token_id']}")

        print(f"\n    (c)onfirm  (r)eject  (s)kip-all")
        try:
            cmd = input("    > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd in ("c", "confirm"):
            print(f"    CONFIRMED ✓")
            _log_execution(c, "confirmed")
        elif cmd in ("r", "reject"):
            print(f"    REJECTED")
            _log_execution(c, "rejected")
        elif cmd in ("s", "skip-all"):
            print(f"    Skipping remaining orders.")
            break
        else:
            print(f"    Skipped.")


def _review_raw_orders() -> None:
    """Review raw queued orders when no real market data available."""
    with open(QUEUE_PATH) as f:
        orders = [json.loads(l) for l in f if l.strip()]

    pending = [o for o in orders if o["status"] == "pending"]
    state = load_state()

    print(f"    Exposure: ${state.total_exposure:.2f}/${MAX_TOTAL_EXPOSURE:.2f}  "
          f"Kill: {'ON' if state.kill_switch else 'off'}")
    print()

    for i, o in enumerate(pending):
        print(f"    [{i+1}] {o['runner_id']} {o['side']} @{o['price']:.3f} "
              f"x${o['size']:.2f}")
        print(f"        edge={o['edge_bps']:.0f}bps  net={o['net_edge_bps']:.0f}bps  "
              f"time={o['timestamp']}")

    print(f"\n    To execute these on Polymarket:")
    print(f"    1. Find the matching market on polymarket.com")
    print(f"    2. Compare real price to system price above")
    print(f"    3. If deviation < 5%, buy $1 of the 'home' outcome")
    print(f"    4. Record fill in micro_live_logs/day1_20260407.md")


def _log_execution(candidate: dict, action: str) -> None:
    """Log an execution decision."""
    from src.execution.manual_executor import EXECUTION_LOG_PATH
    os.makedirs(os.path.dirname(EXECUTION_LOG_PATH) or ".", exist_ok=True)
    entry = {
        "action": action,
        "timestamp": datetime.now().isoformat(),
        "order": candidate["execution_order"],
        "deviation": candidate["deviation"],
    }
    with open(EXECUTION_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def add_market() -> None:
    """Interactively add a market mapping."""
    print(f"\n  ADD MARKET MAPPING")
    print(f"  {'─'*50}")
    print(f"  To find token IDs on Polymarket:")
    print(f"  1. Go to the market page on polymarket.com")
    print(f"  2. Open browser DevTools → Network tab")
    print(f"  3. Filter for 'book' or 'clob' requests")
    print(f"  4. Find the token_id parameter")
    print(f"  Or use the Polymarket API docs.\n")

    try:
        sport = input("  Sport [basketball]: ").strip() or "basketball"
        home = input("  Home team (abbrev, e.g. LAL): ").strip().upper()
        away = input("  Away team (abbrev, e.g. BOS): ").strip().upper()
        game_date = input(f"  Game date [{date.today().isoformat()}]: ").strip() or date.today().isoformat()
        condition_id = input("  Condition ID (0x...): ").strip()
        yes_token = input("  Yes token ID (home win): ").strip()
        no_token = input("  No token ID (away win): ").strip()
        slug = input("  Market slug (optional): ").strip()
        desc = input(f"  Description [Will {home} beat {away}?]: ").strip() or f"Will {home} beat {away}?"

        internal_id = f"nba_{home.lower()}_{away.lower()}_ml"

        mapping = MarketMapping(
            internal_id=internal_id,
            sport=sport,
            home_team=home,
            away_team=away,
            market_type="moneyline",
            game_date=game_date,
            condition_id=condition_id,
            yes_token_id=yes_token,
            no_token_id=no_token,
            polymarket_slug=slug,
            description=desc,
            active=True,
            last_verified_ts=datetime.now().isoformat(),
        )

        mapper = MarketMapper()
        mapper.register(mapping)
        print(f"\n  Registered: {internal_id}")
        print(f"  Saved to: config/market_mappings.json")

    except (EOFError, KeyboardInterrupt):
        print("\n  Cancelled.")


def run_nightly_micro() -> None:
    """Nightly routine: paper check → observe → status."""
    print(f"\n{'='*70}")
    print(f"  NIGHTLY MICRO-LIVE ROUTINE — {date.today().isoformat()}")
    print(f"{'='*70}")

    # Step 1: Orchestrator nightly
    print("\n  [1/4] Running orchestrator nightly...")
    from orchestrator import run_nightly
    run_nightly()

    # Step 2: Generate signals
    print("\n  [2/4] Generating micro-live signals...")
    from live_observer import run_observe
    run_observe(n_games=1, seed=int(time.time()) % 100000)

    # Step 3: Check real prices (if mappings exist)
    print("\n  [3/4] Checking real prices...")
    check_real_prices()

    # Step 4: Status
    print("\n  [4/4] Status:")
    show_status()

    print(f"\n  Next: python execution_cli.py check")
    print(f"  Then: review and confirm orders")


def run_auto_scan() -> None:
    """Auto-scan: market sync → data sync → signal scan → queue."""
    from orchestrator import run_live
    run_live()


def real_trade(slug: str) -> None:
    """Trade a real game using the real signal generator."""
    from real_signal_generator import run_trade_pipeline
    run_trade_pipeline(slug)


def real_price(slug: str) -> None:
    """Price a real game without trading."""
    if slug == "min-ind" or slug == "nba-min-ind-2026-04-07":
        from real_signal_generator import price_min_vs_ind, generate_real_signals
        from src.pricing.real_game_pricer import RealGamePricer
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
    else:
        from real_signal_generator import price_from_polymarket
        price_from_polymarket(slug)


def run_full_auto(dry_run: bool = False) -> None:
    """Full auto: sync → price → signal → execute on Polymarket CLOB."""
    from src.execution.auto_executor import run_full_auto as _run
    _run(dry_run=dry_run)


def run_full_auto_loop(interval: int = 300) -> None:
    """Continuous auto execution loop."""
    from src.execution.auto_executor import run_loop
    run_loop(interval_sec=interval, dry_run=False)


def portfolio_status() -> None:
    """Show portfolio and trading status."""
    from src.execution.auto_executor import AutoExecutor
    executor = AutoExecutor(dry_run=True)
    executor.show_status()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python execution_cli.py <command> [args]")
        print()
        print("  FULL AUTO (end-to-end, real orders):")
        print("  go                  One-shot: sync→price→signal→EXECUTE")
        print("  go-loop [sec]       Continuous auto (default 300s)")
        print("  go-dry              Dry run (no real orders)")
        print("  portfolio           Show portfolio & trade status")
        print()
        print("  SEMI-AUTO (queue + review):")
        print("  auto                Sync→price→signal→queue (no execution)")
        print("  live                Alias for auto")
        print("  check               Check real prices for queued orders")
        print("  status              Show today's micro-live status")
        print()
        print("  REAL TRADING (manual):")
        print("  real-price <slug>   Price a real NBA game")
        print("  real-trade <slug>   Full pipeline: price→signal→pre-trade→queue")
        print("  add-market          Register a Polymarket market mapping")
        print()
        print("  SIMULATION:")
        print("  run [n_games]       Simulated paper→signals→prices→review")
        print("  nightly             Full nightly routine with simulation")
        print()
        print("  Environment:")
        print("  POLYMARKET_PRIVATE_KEY=0x...  (required for go/go-loop)")
        print()
        print("  Examples:")
        print("  POLYMARKET_PRIVATE_KEY=0x... python execution_cli.py go")
        print("  python execution_cli.py go-dry")
        print("  python execution_cli.py auto")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "go":
        run_full_auto(dry_run=False)
    elif cmd == "go-dry":
        run_full_auto(dry_run=True)
    elif cmd == "go-loop":
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 300
        run_full_auto_loop(interval=interval)
    elif cmd == "portfolio":
        portfolio_status()
    elif cmd in ("auto", "live"):
        run_auto_scan()
    elif cmd == "real-price" and len(sys.argv) > 2:
        real_price(sys.argv[2])
    elif cmd == "real-trade" and len(sys.argv) > 2:
        real_trade(sys.argv[2])
    elif cmd == "run":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        run_full_pipeline(n_games=n)
    elif cmd == "check":
        check_real_prices()
    elif cmd == "status":
        show_status()
    elif cmd == "add-market":
        add_market()
    elif cmd == "nightly":
        run_nightly_micro()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
