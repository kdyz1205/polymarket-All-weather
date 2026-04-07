"""
Live observer — micro-live with manual order confirmation.

Runs real-time signal generation but writes orders to a JSONL queue
instead of executing them. The operator manually confirms each order
before it goes to the exchange.

Safety limits are hardcoded and non-negotiable:
  - $1 per trade
  - $3 max total exposure
  - $2 max daily loss
  - Basketball only
  - 1 market at a time

Usage:
  # Generate signals (writes to manual_order_queue.jsonl):
  source .venv/bin/activate && python live_observer.py observe

  # Review and confirm pending orders:
  python live_observer.py review

  # Show daily P&L and exposure summary:
  python live_observer.py status
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, date
from typing import Any


# ─── Hard safety limits (non-negotiable) ───

MAX_STAKE_PER_ORDER = 1.0      # $1 per trade
MAX_TOTAL_EXPOSURE = 3.0       # $3 max total position
MAX_DAILY_LOSS = 2.0           # $2 max daily drawdown
ALLOWED_SPORTS = {"basketball"}
MAX_CONCURRENT_MARKETS = 1

QUEUE_PATH = "manual_order_queue.jsonl"
FILLS_PATH = "micro_live_fills.jsonl"
DAILY_LOG_DIR = "micro_live_logs"


@dataclass
class PendingOrder:
    """An order waiting for manual confirmation."""
    order_id: str
    timestamp: str
    sport: str
    market_id: str
    runner_id: str
    side: str
    price: float
    size: float
    edge_bps: float
    net_edge_bps: float
    fee_bps: float
    status: str = "pending"       # pending / confirmed / rejected / expired
    confirmed_at: str = ""
    rejected_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MicroLiveState:
    """Tracks current micro-live exposure and daily P&L."""
    date: str = ""
    total_exposure: float = 0.0
    daily_pnl: float = 0.0
    open_positions: dict[str, float] = field(default_factory=dict)  # runner_id -> size
    active_markets: list[str] = field(default_factory=list)
    orders_today: int = 0
    fills_today: int = 0
    kill_switch: bool = False
    kill_reason: str = ""

    def check_limits(self, size: float, market_id: str) -> tuple[bool, str]:
        """Check all safety limits before queuing an order."""
        if self.kill_switch:
            return False, f"kill switch active: {self.kill_reason}"

        if size > MAX_STAKE_PER_ORDER:
            return False, f"size {size} > max {MAX_STAKE_PER_ORDER}"

        if self.total_exposure + size > MAX_TOTAL_EXPOSURE:
            return False, f"exposure {self.total_exposure}+{size} > max {MAX_TOTAL_EXPOSURE}"

        if self.daily_pnl < -MAX_DAILY_LOSS:
            return False, f"daily loss {self.daily_pnl:.2f} exceeds max -{MAX_DAILY_LOSS}"

        if market_id not in self.active_markets:
            if len(self.active_markets) >= MAX_CONCURRENT_MARKETS:
                return False, f"already in {len(self.active_markets)} markets (max {MAX_CONCURRENT_MARKETS})"

        return True, "ok"

    def record_fill(self, runner_id: str, size: float, pnl: float) -> None:
        self.open_positions[runner_id] = self.open_positions.get(runner_id, 0) + size
        self.total_exposure = sum(self.open_positions.values())
        self.daily_pnl += pnl
        self.fills_today += 1

        # Auto kill switch on daily loss
        if self.daily_pnl < -MAX_DAILY_LOSS:
            self.kill_switch = True
            self.kill_reason = f"daily loss {self.daily_pnl:.2f} hit limit"

    def to_dict(self) -> dict:
        return asdict(self)


def load_state() -> MicroLiveState:
    """Load or create today's state."""
    today = date.today().isoformat()
    state_path = os.path.join(DAILY_LOG_DIR, f"state_{today}.json")

    if os.path.exists(state_path):
        with open(state_path) as f:
            data = json.load(f)
        state = MicroLiveState(**data)
    else:
        state = MicroLiveState(date=today)

    return state


def save_state(state: MicroLiveState) -> None:
    os.makedirs(DAILY_LOG_DIR, exist_ok=True)
    state_path = os.path.join(DAILY_LOG_DIR, f"state_{state.date}.json")
    with open(state_path, "w") as f:
        json.dump(state.to_dict(), f, indent=2)


def queue_order(order: PendingOrder, state: MicroLiveState) -> bool:
    """Add an order to the manual confirmation queue.

    Returns True if queued, False if blocked by safety limits.
    """
    # Enforce sport restriction
    if order.sport not in ALLOWED_SPORTS:
        print(f"  BLOCKED: sport {order.sport} not allowed (only {ALLOWED_SPORTS})")
        return False

    # Enforce size cap
    order.size = min(order.size, MAX_STAKE_PER_ORDER)

    # Check limits
    ok, reason = state.check_limits(order.size, order.market_id)
    if not ok:
        print(f"  BLOCKED: {reason}")
        return False

    # Write to queue
    with open(QUEUE_PATH, "a") as f:
        f.write(json.dumps(order.to_dict()) + "\n")

    state.orders_today += 1
    if order.market_id not in state.active_markets:
        state.active_markets.append(order.market_id)
    save_state(state)

    print(f"  QUEUED: {order.order_id} — {order.runner_id} {order.side} "
          f"@{order.price:.3f} x${order.size:.2f} (edge={order.edge_bps:.0f}bps)")
    return True


def review_orders() -> None:
    """Interactive review of pending orders."""
    if not os.path.exists(QUEUE_PATH):
        print("  No pending orders.")
        return

    orders = []
    with open(QUEUE_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                orders.append(json.loads(line))

    pending = [o for o in orders if o["status"] == "pending"]
    if not pending:
        print("  No pending orders to review.")
        return

    state = load_state()

    print(f"\n{'='*70}")
    print(f"  PENDING ORDERS — {len(pending)} waiting for confirmation")
    print(f"  Daily PnL: {state.daily_pnl:+.2f} | Exposure: ${state.total_exposure:.2f}/{MAX_TOTAL_EXPOSURE}")
    print(f"{'='*70}")

    for i, o in enumerate(pending):
        print(f"\n  [{i+1}] {o['order_id']}")
        print(f"      {o['runner_id']} {o['side']} @{o['price']:.3f} x${o['size']:.2f}")
        print(f"      edge={o['edge_bps']:.0f}bps  net={o['net_edge_bps']:.0f}bps  fee={o['fee_bps']:.0f}bps")
        print(f"      market={o['market_id']}  time={o['timestamp']}")

    print(f"\n  Commands: (c)onfirm <n>, (r)eject <n>, (a)ll-reject, (q)uit")

    while True:
        try:
            cmd = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd in ("q", "quit", ""):
            break

        if cmd == "a" or cmd == "all-reject":
            for o in pending:
                o["status"] = "rejected"
                o["rejected_reason"] = "manual bulk reject"
            _rewrite_queue(orders)
            print(f"  Rejected all {len(pending)} orders.")
            break

        parts = cmd.split()
        if len(parts) != 2:
            print("  Usage: c <n> or r <n>")
            continue

        action, idx_str = parts
        try:
            idx = int(idx_str) - 1
        except ValueError:
            print("  Invalid index.")
            continue

        if idx < 0 or idx >= len(pending):
            print(f"  Index out of range (1-{len(pending)}).")
            continue

        o = pending[idx]
        if action in ("c", "confirm"):
            o["status"] = "confirmed"
            o["confirmed_at"] = datetime.now().isoformat()
            print(f"  CONFIRMED: {o['order_id']}")
        elif action in ("r", "reject"):
            o["status"] = "rejected"
            o["rejected_reason"] = "manual reject"
            print(f"  REJECTED: {o['order_id']}")
        else:
            print("  Unknown action. Use c or r.")
            continue

        _rewrite_queue(orders)

    save_state(state)


def _rewrite_queue(orders: list[dict]) -> None:
    """Rewrite the full queue file."""
    with open(QUEUE_PATH, "w") as f:
        for o in orders:
            f.write(json.dumps(o) + "\n")


def show_status() -> None:
    """Print daily status summary."""
    state = load_state()

    print(f"\n{'='*60}")
    print(f"  MICRO-LIVE STATUS — {state.date}")
    print(f"{'='*60}")
    print(f"  Daily PnL:       {state.daily_pnl:+.2f}")
    print(f"  Total exposure:  ${state.total_exposure:.2f} / ${MAX_TOTAL_EXPOSURE:.2f}")
    print(f"  Orders today:    {state.orders_today}")
    print(f"  Fills today:     {state.fills_today}")
    print(f"  Kill switch:     {'ON — ' + state.kill_reason if state.kill_switch else 'off'}")
    print(f"  Active markets:  {state.active_markets or 'none'}")

    if state.open_positions:
        print(f"\n  Open positions:")
        for rid, size in state.open_positions.items():
            print(f"    {rid}: ${size:.2f}")

    # Queue stats
    if os.path.exists(QUEUE_PATH):
        with open(QUEUE_PATH) as f:
            orders = [json.loads(l) for l in f if l.strip()]
        pending = sum(1 for o in orders if o["status"] == "pending")
        confirmed = sum(1 for o in orders if o["status"] == "confirmed")
        rejected = sum(1 for o in orders if o["status"] == "rejected")
        print(f"\n  Queue: {pending} pending, {confirmed} confirmed, {rejected} rejected")

    print(f"\n  Limits:")
    print(f"    Max stake/order:  ${MAX_STAKE_PER_ORDER:.2f}")
    print(f"    Max exposure:     ${MAX_TOTAL_EXPOSURE:.2f}")
    print(f"    Max daily loss:   ${MAX_DAILY_LOSS:.2f}")
    print(f"    Allowed sports:   {', '.join(ALLOWED_SPORTS)}")
    print(f"    Max markets:      {MAX_CONCURRENT_MARKETS}")
    print(f"{'='*60}\n")


def run_observe(n_games: int = 5, seed: int = 9999) -> None:
    """Run live observation mode — generate signals, queue orders.

    Uses the same paper trading engine but writes to manual_order_queue
    instead of auto-executing.
    """
    import random
    import sports_engine as se
    from src.pricing.engine import PricingEngine, TeamRating
    from src.strategy import BasketballStrategyConfig, StrategyGatekeeper
    from paper_runner import generate_book_levels

    config = BasketballStrategyConfig.micro_live()
    state = load_state()

    if state.kill_switch:
        print(f"  KILL SWITCH ACTIVE: {state.kill_reason}")
        print("  Cannot observe. Reset kill switch to continue.")
        return

    gatekeeper = StrategyGatekeeper()
    rng = random.Random(seed)
    queued_count = 0

    print(f"\n  Live observer — {n_games} games, micro-live limits")
    print(f"  Config: fee={config.fee_bps_roundtrip}bps, net={config.min_net_edge_bps}bps, "
          f"stake=${config.base_stake:.2f}")

    for g in range(n_games):
        game_id = f"obs_{state.date}_{g}"
        market_id = f"nba_live_{g}"

        match_state = se.MatchState(game_id, se.Sport.Basketball)
        book = se.MarketBook("obs_nba")
        book.add_runner("home", "H")
        book.add_runner("away", "A")

        h_rating = TeamRating("H", 1500 + rng.randint(-100, 100), home_advantage=70)
        a_rating = TeamRating("A", 1500 + rng.randint(-100, 100))
        pricer = PricingEngine("basketball", h_rating, a_rating, model_weight=0.65)

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
                    [se.MatchEventType.FieldGoal2, se.MatchEventType.FieldGoal3,
                     se.MatchEventType.FreeThrow],
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
                                 away_score=match_state.away_score,
                                 market_implied=(mkt_h, mkt_a))

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
                    risk_allows=True, kill_switch=state.kill_switch,
                    quarter=quarter,
                )

                if passed:
                    net_edge = edge_bps - config.fee_bps_roundtrip - config.delay_penalty_bps_per_ms * 3000
                    order = PendingOrder(
                        order_id=f"{game_id}_{rid}_{sec}",
                        timestamp=datetime.now().isoformat(),
                        sport="basketball",
                        market_id=market_id,
                        runner_id=rid,
                        side="back",
                        price=snap.best_back_price,
                        size=config.base_stake,
                        edge_bps=edge_bps,
                        net_edge_bps=net_edge,
                        fee_bps=config.fee_bps_roundtrip,
                    )
                    if queue_order(order, state):
                        queued_count += 1

                    # Re-check kill switch after each queue
                    if state.kill_switch:
                        break

            if state.kill_switch:
                break

        if state.kill_switch:
            break

    save_state(state)

    print(f"\n  Observation complete: {queued_count} orders queued")
    print(f"  Review with: python live_observer.py review")
    print(f"  Status with: python live_observer.py status")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python live_observer.py [observe|review|status]")
        print("  observe [n_games]  — run signal generation, queue orders")
        print("  review             — interactively confirm/reject pending orders")
        print("  status             — show daily exposure and P&L summary")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "observe":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        run_observe(n_games=n)
    elif mode == "review":
        review_orders()
    elif mode == "status":
        show_status()
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
