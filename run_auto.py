#!/usr/bin/env python3
"""
ONE-COMMAND AUTO TRADER
=======================

Run this on your local machine with your wallet key.
It does EVERYTHING automatically:

  1. Discovers all NBA markets on Polymarket
  2. Fetches team records and injuries
  3. Prices every game (Log5 + injuries + HCA)
  4. Compares to market prices to find edge
  5. Places real limit orders where edge exists

Usage:
  # First time setup:
  pip install py-clob-client requests python-dotenv

  # Run once:
  POLYMARKET_PRIVATE_KEY=0x... python run_auto.py

  # Run continuously (every 5 minutes):
  POLYMARKET_PRIVATE_KEY=0x... python run_auto.py --loop

  # Dry run (see what it would do, no real orders):
  python run_auto.py --dry

Safety:
  - $20 total bankroll cap
  - $2 max per order
  - $5 max daily loss
  - Only basketball
  - Only buys (no shorts)
  - Minimum 20bps net edge to trade
"""

import json
import os
import sys
import time
import logging
from datetime import date, datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Try to load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def run_once(dry_run: bool = False) -> None:
    """Single execution cycle."""
    from src.execution.auto_executor import run_full_auto
    run_full_auto(dry_run=dry_run, sports=["basketball"])


def run_loop(interval: int = 300, dry_run: bool = False) -> None:
    """Continuous execution loop."""
    from src.execution.auto_executor import run_loop as _loop
    _loop(interval_sec=interval, dry_run=dry_run)


def show_portfolio() -> None:
    """Show current portfolio status."""
    from src.execution.auto_executor import AutoExecutor
    executor = AutoExecutor(dry_run=True)
    executor.show_status()


def scan_only() -> None:
    """Scan markets without executing — just show what's available."""
    from src.data.market_sync import MarketSyncer
    from src.data.data_sync import DataSyncer
    from src.data.auto_signal_runner import AutoSignalRunner

    print(f"\n{'='*70}")
    print(f"  MARKET SCANNER — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    # Step 1: Sync markets
    print("\n  [1] Discovering markets...")
    syncer = MarketSyncer()
    try:
        markets = syncer.sync_today(sports=["basketball"])
        cache = syncer.get_cache()
        today = cache.get_today()
        print(f"      Found {len(today)} NBA markets today")
        for m in today:
            print(f"        {m.away_team}@{m.home_team} "
                  f"H={m.home_price:.3f} A={m.away_price:.3f} "
                  f"vol=${m.volume:,.0f}")
    except Exception as e:
        print(f"      Error: {e}")
    finally:
        syncer.close()

    # Step 2: Sync team data
    print("\n  [2] Fetching team/player data...")
    data_syncer = DataSyncer()
    try:
        stats = data_syncer.sync_all()
        print(f"      Teams: {stats['teams']} | Players: {stats['players']}")
    except Exception as e:
        print(f"      Error: {e}")
    finally:
        data_syncer.close()

    # Step 3: Scan for edge
    print("\n  [3] Scanning for edge...")
    runner = AutoSignalRunner()
    results = runner.scan_all()

    if not results:
        print("      No markets to scan.")
        return

    print(f"\n  {'Away':>5}@{'Home':<5} {'Fair':>6} {'Market':>7} {'Edge':>7} "
          f"{'Net':>7} {'Signal':>8} {'Conf':>5} {'Action':>8}")
    print(f"  {'─'*65}")

    for r in results:
        action = ">>> BUY" if r.actionable else "skip"
        print(f"  {r.away_team:>5}@{r.home_team:<5} "
              f"{r.fair_home_prob:>5.1%} {r.market_home_price:>6.1%} "
              f"{r.best_edge_bps:>+6.0f}bp {r.net_edge_bps:>+6.0f}bp "
              f"{r.signal_strength:>8} {r.confidence:>4.0%} {action:>8}")

    actionable = [r for r in results if r.actionable]
    print(f"\n  Total: {len(results)} markets, {len(actionable)} actionable")

    if actionable:
        print(f"\n  To execute these trades, run:")
        print(f"  POLYMARKET_PRIVATE_KEY=0x... python run_auto.py")
    else:
        print(f"\n  No edge found. Markets are efficiently priced today.")


if __name__ == "__main__":
    args = sys.argv[1:]

    if "--help" in args or "-h" in args:
        print(__doc__)
        sys.exit(0)

    if "--portfolio" in args or "--status" in args:
        show_portfolio()
        sys.exit(0)

    if "--scan" in args:
        scan_only()
        sys.exit(0)

    dry_run = "--dry" in args or "--dry-run" in args
    loop = "--loop" in args

    # Parse loop interval
    interval = 300
    for i, a in enumerate(args):
        if a == "--interval" and i + 1 < len(args):
            interval = int(args[i + 1])

    if not dry_run:
        pk = os.environ.get("POLYMARKET_PRIVATE_KEY", "")
        if not pk:
            print("  ERROR: POLYMARKET_PRIVATE_KEY not set.")
            print()
            print("  Options:")
            print("  1. Export your wallet key:")
            print("     export POLYMARKET_PRIVATE_KEY=0x...")
            print()
            print("  2. Create a .env file:")
            print("     echo 'POLYMARKET_PRIVATE_KEY=0x...' > .env")
            print()
            print("  3. Run in dry-run mode (no real orders):")
            print("     python run_auto.py --dry")
            print()
            print("  4. Just scan for opportunities:")
            print("     python run_auto.py --scan")
            sys.exit(1)

    if loop:
        print(f"  Starting auto-trader loop (interval={interval}s)")
        print(f"  Mode: {'DRY RUN' if dry_run else 'LIVE'}")
        print(f"  Press Ctrl+C to stop.\n")
        run_loop(interval=interval, dry_run=dry_run)
    else:
        run_once(dry_run=dry_run)
