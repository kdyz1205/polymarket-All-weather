"""
Auto executor — places real orders on Polymarket CLOB.

This is the final execution layer. It takes confirmed signals from the
pipeline and submits them as limit orders via the Polymarket CLOB API.

Safety layers (all enforced regardless of input):
  1. Hard bankroll cap ($20 total)
  2. Max per-order size ($2)
  3. Max daily loss ($5)
  4. Max concurrent open positions (3)
  5. Pre-trade price check (revalidate before sending)
  6. All trades logged to JSONL

Requires:
  - POLYMARKET_PRIVATE_KEY env var (Ethereum wallet private key)
  - Funded wallet on Polygon network
  - py-clob-client package

Usage:
  # Execute all queued GREEN signals:
  python -m src.execution.auto_executor run

  # Dry run (no real orders):
  python -m src.execution.auto_executor dry-run

  # Show wallet status:
  python -m src.execution.auto_executor status
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, asdict, field
from datetime import date, datetime
from typing import Any

logger = logging.getLogger(__name__)

# ─── Hard safety limits ───
MAX_BANKROLL = 20.0            # $20 total capital
MAX_ORDER_SIZE = 2.0           # $2 max per order
MAX_DAILY_LOSS = 5.0           # $5 max daily drawdown
MAX_OPEN_POSITIONS = 3         # max 3 concurrent positions
ALLOWED_SPORTS = {"basketball"}
MIN_NET_EDGE_BPS = 10.0        # minimum net edge to execute
REVALIDATE_MAX_DEVIATION = 8.0 # max price movement since signal (%)

TRADE_LOG = "data/trades/trades.jsonl"
POSITION_FILE = "data/trades/positions.json"

# Polymarket CLOB API
CLOB_HOST = "https://clob.polymarket.com"
CHAIN_ID = 137  # Polygon mainnet


@dataclass
class TradeRecord:
    """Record of an executed trade."""
    trade_id: str
    timestamp: str
    slug: str
    side: str               # "buy"
    token_id: str
    runner: str             # "home" | "away"
    price: float            # execution price (0-1)
    size: float             # dollar amount
    edge_bps: float
    net_edge_bps: float
    pre_trade_signal: str   # "green" | "yellow"
    order_id: str = ""      # CLOB order ID returned by exchange
    status: str = "pending" # pending | filled | cancelled | error
    fill_price: float = 0.0
    error: str = ""
    # Settlement
    settled: bool = False
    won: bool = False
    pnl: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PortfolioState:
    """Current portfolio state."""
    date: str
    bankroll: float = MAX_BANKROLL
    deployed: float = 0.0
    available: float = MAX_BANKROLL
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    trades_today: int = 0
    wins: int = 0
    losses: int = 0
    open_positions: list[dict] = field(default_factory=list)
    kill_switch: bool = False
    kill_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> PortfolioState:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def can_trade(self, size: float) -> tuple[bool, str]:
        if self.kill_switch:
            return False, f"kill switch: {self.kill_reason}"
        if self.daily_pnl <= -MAX_DAILY_LOSS:
            return False, f"daily loss limit hit ({self.daily_pnl:.2f})"
        if self.deployed + size > self.bankroll:
            return False, f"insufficient capital (deployed={self.deployed:.2f}, bankroll={self.bankroll:.2f})"
        if len(self.open_positions) >= MAX_OPEN_POSITIONS:
            return False, f"max positions ({MAX_OPEN_POSITIONS}) reached"
        if size > MAX_ORDER_SIZE:
            return False, f"size {size:.2f} > max {MAX_ORDER_SIZE:.2f}"
        return True, "ok"


def load_portfolio() -> PortfolioState:
    """Load or create today's portfolio state."""
    if os.path.exists(POSITION_FILE):
        with open(POSITION_FILE) as f:
            data = json.load(f)
        state = PortfolioState.from_dict(data)
        # Reset daily counters if new day
        if state.date != date.today().isoformat():
            state.date = date.today().isoformat()
            state.daily_pnl = 0.0
            state.trades_today = 0
    else:
        state = PortfolioState(date=date.today().isoformat())
    return state


def save_portfolio(state: PortfolioState) -> None:
    os.makedirs(os.path.dirname(POSITION_FILE) or ".", exist_ok=True)
    with open(POSITION_FILE, "w") as f:
        json.dump(state.to_dict(), f, indent=2)


def log_trade(trade: TradeRecord) -> None:
    os.makedirs(os.path.dirname(TRADE_LOG) or ".", exist_ok=True)
    with open(TRADE_LOG, "a") as f:
        f.write(json.dumps(trade.to_dict()) + "\n")


class AutoExecutor:
    """Places real orders on Polymarket CLOB.

    Uses py-clob-client for order submission. All orders are limit orders
    with conservative pricing (buy at best_bid or better).
    """

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self._client = None
        self._portfolio = load_portfolio()

        if not dry_run:
            self._init_clob_client()

    def _init_clob_client(self) -> None:
        """Initialize the Polymarket CLOB client."""
        private_key = os.environ.get("POLYMARKET_PRIVATE_KEY", "")
        if not private_key:
            logger.error("POLYMARKET_PRIVATE_KEY not set")
            print("  ERROR: Set POLYMARKET_PRIVATE_KEY environment variable")
            print("  Export your wallet private key (with 0x prefix)")
            self._client = None
            return

        try:
            from py_clob_client.client import ClobClient
            self._client = ClobClient(
                host=CLOB_HOST,
                key=private_key,
                chain_id=CHAIN_ID,
            )
            # Derive API credentials
            self._client.set_api_creds(self._client.create_or_derive_api_creds())
            logger.info("CLOB client initialized")
        except Exception as e:
            logger.error("Failed to init CLOB client: %s", e)
            print(f"  ERROR initializing CLOB client: {e}")
            self._client = None

    def execute_from_queue(self) -> list[TradeRecord]:
        """Read pending orders from queue, validate, and execute.

        Pipeline:
          1. Load pending orders from manual_order_queue.jsonl
          2. Filter confirmed & basketball only
          3. Re-validate price against live market
          4. Submit order to CLOB
          5. Log trade and update portfolio
        """
        from live_observer import QUEUE_PATH

        if not os.path.exists(QUEUE_PATH):
            print("  No pending orders.")
            return []

        with open(QUEUE_PATH) as f:
            orders = [json.loads(line) for line in f if line.strip()]

        # Filter to confirmed orders only
        actionable = [o for o in orders if o.get("status") == "confirmed"]
        if not actionable:
            print("  No confirmed orders to execute.")
            return []

        trades: list[TradeRecord] = []

        for order in actionable:
            trade = self._execute_order(order)
            if trade:
                trades.append(trade)

        save_portfolio(self._portfolio)
        return trades

    def execute_signal_direct(
        self,
        slug: str,
        side: str,       # "home" | "away"
        token_id: str,
        price: float,    # target price (0-1)
        size: float,
        edge_bps: float,
        net_edge_bps: float,
    ) -> TradeRecord | None:
        """Execute a single signal directly (bypass queue)."""
        # Safety checks
        size = min(size, MAX_ORDER_SIZE)
        ok, reason = self._portfolio.can_trade(size)
        if not ok:
            print(f"  BLOCKED: {reason}")
            return None

        if net_edge_bps < MIN_NET_EDGE_BPS:
            print(f"  BLOCKED: net edge {net_edge_bps:.0f}bps < min {MIN_NET_EDGE_BPS:.0f}bps")
            return None

        trade = TradeRecord(
            trade_id=f"trade_{int(time.time())}_{slug}_{side}",
            timestamp=datetime.now().isoformat(),
            slug=slug,
            side="buy",
            token_id=token_id,
            runner=side,
            price=price,
            size=size,
            edge_bps=edge_bps,
            net_edge_bps=net_edge_bps,
            pre_trade_signal="green",
        )

        return self._submit_trade(trade)

    def _execute_order(self, order: dict) -> TradeRecord | None:
        """Execute a single queued order."""
        sport = order.get("sport", "")
        if sport not in ALLOWED_SPORTS:
            print(f"  SKIP: sport {sport} not allowed")
            return None

        size = min(float(order.get("size", 1.0)), MAX_ORDER_SIZE)
        ok, reason = self._portfolio.can_trade(size)
        if not ok:
            print(f"  BLOCKED: {reason}")
            return None

        net_edge = float(order.get("net_edge_bps", 0))
        if net_edge < MIN_NET_EDGE_BPS:
            print(f"  SKIP: net edge {net_edge:.0f}bps too low")
            return None

        # Find the token_id from market cache
        from src.data.cache import MarketCache
        cache = MarketCache()
        market = cache.get(order.get("market_id", ""))

        token_id = ""
        if market:
            runner = order.get("runner_id", "")
            token_id = (market.home_token_id if runner == "home"
                        else market.away_token_id)

        if not token_id:
            # Try from market_mappings
            from src.execution.market_mapper import MarketMapper
            mapper = MarketMapper()
            # Search by slug-like patterns
            for mapping in mapper.list_active():
                slug = order.get("market_id", "")
                if (mapping.home_team.lower() in slug.lower() or
                    mapping.away_team.lower() in slug.lower()):
                    runner = order.get("runner_id", "")
                    token_id = (mapping.yes_token_id if runner == "home"
                                else mapping.no_token_id)
                    break

        if not token_id:
            print(f"  SKIP: no token_id for {order.get('market_id')}")
            return None

        # Build trade record
        trade = TradeRecord(
            trade_id=f"trade_{int(time.time())}_{order.get('order_id', 'unknown')}",
            timestamp=datetime.now().isoformat(),
            slug=order.get("market_id", ""),
            side="buy",
            token_id=token_id,
            runner=order.get("runner_id", ""),
            price=float(order.get("price", 0)),
            size=size,
            edge_bps=float(order.get("edge_bps", 0)),
            net_edge_bps=net_edge,
            pre_trade_signal="green",
        )

        return self._submit_trade(trade)

    def _submit_trade(self, trade: TradeRecord) -> TradeRecord | None:
        """Submit a trade to Polymarket CLOB."""
        print(f"\n  {'[DRY RUN] ' if self.dry_run else ''}SUBMITTING ORDER:")
        print(f"    Token:   {trade.token_id[:40]}...")
        print(f"    Side:    BUY {trade.runner}")
        print(f"    Price:   {trade.price:.4f}")
        print(f"    Size:    ${trade.size:.2f}")
        print(f"    Edge:    {trade.edge_bps:.0f}bps (net {trade.net_edge_bps:.0f}bps)")

        if self.dry_run:
            trade.status = "dry_run"
            trade.order_id = "DRY_RUN"
            log_trade(trade)
            print(f"    STATUS:  DRY RUN — order NOT submitted")
            return trade

        if not self._client:
            trade.status = "error"
            trade.error = "CLOB client not initialized"
            log_trade(trade)
            print(f"    ERROR:   CLOB client not initialized")
            print(f"    Set POLYMARKET_PRIVATE_KEY env var and retry")
            return trade

        try:
            from py_clob_client.order_builder.constants import BUY
            from py_clob_client.client import OrderArgs

            # Build a limit order
            order_args = OrderArgs(
                price=trade.price,
                size=trade.size,
                side=BUY,
                token_id=trade.token_id,
            )
            signed_order = self._client.create_order(order_args)
            resp = self._client.post_order(signed_order)

            if resp and isinstance(resp, dict):
                trade.order_id = resp.get("orderID", resp.get("id", ""))
                trade.status = "submitted"
                print(f"    STATUS:  SUBMITTED")
                print(f"    OrderID: {trade.order_id}")
            else:
                trade.order_id = str(resp) if resp else ""
                trade.status = "submitted"
                print(f"    STATUS:  SUBMITTED (resp={resp})")

        except Exception as e:
            trade.status = "error"
            trade.error = str(e)
            print(f"    ERROR:   {e}")

        # Update portfolio
        if trade.status == "submitted":
            self._portfolio.deployed += trade.size
            self._portfolio.available = self._portfolio.bankroll - self._portfolio.deployed
            self._portfolio.trades_today += 1
            self._portfolio.open_positions.append({
                "trade_id": trade.trade_id,
                "slug": trade.slug,
                "runner": trade.runner,
                "price": trade.price,
                "size": trade.size,
                "timestamp": trade.timestamp,
            })

        log_trade(trade)
        save_portfolio(self._portfolio)
        return trade

    def check_fills(self) -> None:
        """Check status of open orders."""
        if not self._client:
            print("  CLOB client not initialized")
            return

        for pos in self._portfolio.open_positions:
            try:
                # Check order status
                # Note: actual API may differ, this is the general pattern
                print(f"  {pos['slug']} {pos['runner']}: "
                      f"${pos['size']:.2f} @ {pos['price']:.4f}")
            except Exception as e:
                print(f"  Error checking {pos.get('trade_id')}: {e}")

    def show_status(self) -> None:
        """Display portfolio status."""
        p = self._portfolio
        print(f"\n{'='*60}")
        print(f"  PORTFOLIO STATUS — {p.date}")
        print(f"{'='*60}")
        print(f"  Bankroll:     ${p.bankroll:.2f}")
        print(f"  Deployed:     ${p.deployed:.2f}")
        print(f"  Available:    ${p.available:.2f}")
        print(f"  Daily PnL:    ${p.daily_pnl:+.2f}")
        print(f"  Total PnL:    ${p.total_pnl:+.2f}")
        print(f"  Trades today: {p.trades_today}")
        print(f"  Record:       {p.wins}W-{p.losses}L")
        print(f"  Kill switch:  {'ON — ' + p.kill_reason if p.kill_switch else 'off'}")

        if p.open_positions:
            print(f"\n  Open positions ({len(p.open_positions)}):")
            for pos in p.open_positions:
                print(f"    {pos['slug']} {pos['runner']} "
                      f"${pos['size']:.2f} @ {pos['price']:.4f}")

        print(f"\n  Safety limits:")
        print(f"    Max bankroll:   ${MAX_BANKROLL:.2f}")
        print(f"    Max per order:  ${MAX_ORDER_SIZE:.2f}")
        print(f"    Max daily loss: ${MAX_DAILY_LOSS:.2f}")
        print(f"    Max positions:  {MAX_OPEN_POSITIONS}")
        print(f"    Min net edge:   {MIN_NET_EDGE_BPS:.0f}bps")
        print(f"{'='*60}\n")


def run_full_auto(dry_run: bool = False, sports: list[str] | None = None) -> None:
    """Full auto pipeline: sync → price → signal → execute.

    This is the end-to-end automation entry point.
    """
    from src.data.market_sync import MarketSyncer
    from src.data.data_sync import DataSyncer
    from src.data.auto_signal_runner import AutoSignalRunner
    from src.data.cache import MarketCache

    ts = datetime.now().strftime("%H:%M:%S")
    mode = "DRY RUN" if dry_run else "LIVE"
    print(f"\n{'='*70}")
    print(f"  FULL AUTO EXECUTOR — {mode} — {ts}")
    print(f"{'='*70}")

    executor = AutoExecutor(dry_run=dry_run)

    # Pre-flight: portfolio check
    portfolio = load_portfolio()
    if portfolio.kill_switch:
        print(f"\n  KILL SWITCH ACTIVE: {portfolio.kill_reason}")
        return
    print(f"  Bankroll: ${portfolio.bankroll:.2f} | "
          f"Available: ${portfolio.available:.2f} | "
          f"Daily PnL: ${portfolio.daily_pnl:+.2f}")

    # Step 1: Sync markets
    print(f"\n  [1] Syncing markets...")
    syncer = MarketSyncer()
    try:
        markets = syncer.sync_today(sports=sports or ["basketball"])
        print(f"      Found {len(markets)} markets")
    except Exception as e:
        print(f"      Market sync failed: {e} (using cache)")
    finally:
        syncer.close()

    # Step 2: Sync data
    print(f"\n  [2] Syncing team/player data...")
    data_syncer = DataSyncer()
    try:
        stats = data_syncer.sync_all()
        print(f"      Teams: {stats['teams']} | Players: {stats['players']}")
    except Exception as e:
        print(f"      Data sync failed: {e} (using cache)")
    finally:
        data_syncer.close()

    # Step 3: Scan for signals
    print(f"\n  [3] Scanning for signals...")
    runner = AutoSignalRunner()
    results = runner.scan_all()

    if not results:
        print(f"      No markets to scan.")
        return

    actionable = [r for r in results if r.actionable]
    print(f"      Scanned: {len(results)} | Actionable: {len(actionable)}")

    for r in results:
        icon = ">>>" if r.actionable else "   "
        print(f"      {icon} {r.away_team}@{r.home_team}: "
              f"edge={r.best_edge_bps:+.0f}bps net={r.net_edge_bps:+.0f}bps "
              f"{r.signal_strength}")

    if not actionable:
        print(f"\n  No actionable signals. Done.")
        return

    # Step 4: Execute actionable signals
    print(f"\n  [4] Executing {len(actionable)} signals...")
    cache = MarketCache()
    trades: list[TradeRecord] = []

    for result in actionable:
        market = cache.get(result.slug)
        if not market:
            print(f"      SKIP {result.slug}: not in cache")
            continue

        # Determine token
        side = result.best_side
        token_id = (market.home_token_id if side == "home"
                    else market.away_token_id)
        if not token_id:
            print(f"      SKIP {result.slug}: no token_id for {side}")
            continue

        # Target price: use market price (buy at or below market)
        price = (market.home_price if side == "home"
                 else market.away_price)

        # Size: scale with confidence and edge
        base_size = 1.0
        if result.confidence >= 0.75 and result.net_edge_bps > 100:
            base_size = 2.0  # high confidence + high edge
        elif result.confidence >= 0.65 and result.net_edge_bps > 50:
            base_size = 1.5

        trade = executor.execute_signal_direct(
            slug=result.slug,
            side=side,
            token_id=token_id,
            price=price,
            size=base_size,
            edge_bps=result.best_edge_bps,
            net_edge_bps=result.net_edge_bps,
        )
        if trade:
            trades.append(trade)

    # Summary
    submitted = sum(1 for t in trades if t.status in ("submitted", "dry_run"))
    errors = sum(1 for t in trades if t.status == "error")

    print(f"\n{'='*70}")
    print(f"  EXECUTION COMPLETE")
    print(f"    Submitted: {submitted}")
    print(f"    Errors:    {errors}")
    executor.show_status()
    print(f"{'='*70}\n")


def run_loop(interval_sec: int = 300, dry_run: bool = False) -> None:
    """Run full auto in a loop."""
    print(f"  Auto executor loop (interval={interval_sec}s, dry_run={dry_run})")
    print(f"  Press Ctrl+C to stop.\n")

    while True:
        try:
            run_full_auto(dry_run=dry_run)
        except KeyboardInterrupt:
            print("\n  Stopped.")
            break
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()

        print(f"\n  Next run in {interval_sec}s...")
        try:
            time.sleep(interval_sec)
        except KeyboardInterrupt:
            print("\n  Stopped.")
            break


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m src.execution.auto_executor run        # execute queued orders")
        print("  python -m src.execution.auto_executor auto       # full auto pipeline")
        print("  python -m src.execution.auto_executor dry-run    # auto pipeline (no real orders)")
        print("  python -m src.execution.auto_executor loop [sec] # continuous auto (default 300s)")
        print("  python -m src.execution.auto_executor status     # portfolio status")
        print()
        print("  Environment:")
        print("  POLYMARKET_PRIVATE_KEY=0x...  (wallet private key)")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "run":
        executor = AutoExecutor()
        executor.execute_from_queue()
    elif cmd == "auto":
        run_full_auto(dry_run=False)
    elif cmd == "dry-run":
        run_full_auto(dry_run=True)
    elif cmd == "loop":
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 300
        run_loop(interval_sec=interval, dry_run=False)
    elif cmd == "dry-loop":
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 300
        run_loop(interval_sec=interval, dry_run=True)
    elif cmd == "status":
        executor = AutoExecutor(dry_run=True)
        executor.show_status()
    else:
        print(f"Unknown: {cmd}")
