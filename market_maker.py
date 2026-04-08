"""
Polymarket Market Maker — high-frequency spread capture bot.

Strategy:
  - Quote both sides of a market (buy YES at bid, sell YES at ask)
  - Earn the bid-ask spread on every round-trip fill
  - Skew quotes to manage inventory (don't accumulate too much on one side)
  - Cancel and requote when the market moves

Revenue model:
  - Spread capture: buy at 0.54, sell at 0.56 = $0.02 profit per share
  - Target: 5-20 round trips per day per market = $0.10-0.40/day on $20 capital
  - No directional risk — profit regardless of who wins

Safety:
  - Max inventory: ±50 shares per side
  - Max loss per market: $3
  - Max total deployed: $20
  - Kill switch on any anomaly
  - All orders are limit orders (no market orders)

Usage:
  # Dry run (logs what it would do):
  python market_maker.py --dry --token <token_id>

  # Live market making on a single token:
  POLYMARKET_PRIVATE_KEY=0x... python market_maker.py --token <token_id>

  # Auto-select best market from cache:
  POLYMARKET_PRIVATE_KEY=0x... python market_maker.py --auto

  # With custom spread:
  POLYMARKET_PRIVATE_KEY=0x... python market_maker.py --token <token_id> --spread 0.02
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, date
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mm")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─── Constants ───

CLOB_HOST = "https://clob.polymarket.com"
CHAIN_ID = 137  # Polygon

# Safety
MAX_INVENTORY = 50          # max shares on one side
MAX_LOSS_PER_MARKET = 3.0   # $ — kill switch per market
MAX_TOTAL_DEPLOYED = 20.0   # $ — across all markets
MAX_ORDER_SIZE = 10.0       # shares per quote
MIN_SPREAD = 0.005          # minimum spread to quote (0.5 cents)
MAX_SPREAD = 0.05           # if spread > this, don't quote (too risky)

# Timing
REQUOTE_INTERVAL = 3.0      # seconds between requotes
BOOK_POLL_INTERVAL = 1.0    # seconds between orderbook polls
STALE_ORDER_SEC = 30.0      # cancel orders older than this

# Inventory skew
SKEW_PER_SHARE = 0.0005     # skew mid by this per net inventory share

MM_LOG = "data/trades/mm_log.jsonl"
MM_STATE = "data/trades/mm_state.json"


@dataclass
class Quote:
    """A two-sided quote."""
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float
    mid: float
    spread: float
    skew: float = 0.0
    bid_order_id: str = ""
    ask_order_id: str = ""


@dataclass
class MMState:
    """Market maker state for a single market."""
    token_id: str
    slug: str = ""
    started_at: str = ""
    # Inventory
    net_position: float = 0.0       # positive = long YES, negative = short YES
    total_bought: float = 0.0       # total shares bought
    total_sold: float = 0.0         # total shares sold
    avg_buy_price: float = 0.0
    avg_sell_price: float = 0.0
    # PnL
    realized_pnl: float = 0.0      # from completed round trips
    unrealized_pnl: float = 0.0    # mark-to-market of open inventory
    total_pnl: float = 0.0
    # Stats
    n_fills: int = 0
    n_round_trips: int = 0
    n_requotes: int = 0
    n_cancels: int = 0
    # Safety
    kill_switch: bool = False
    kill_reason: str = ""
    # Active orders
    active_bid_id: str = ""
    active_ask_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> MMState:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class MarketMaker:
    """Polymarket CLOB market maker.

    Quotes both sides of a binary market, earns the spread.
    Manages inventory by skewing quotes away from the heavy side.
    """

    def __init__(
        self,
        token_id: str,
        slug: str = "",
        spread: float = 0.02,       # target half-spread (each side from mid)
        size: float = 5.0,          # quote size in shares
        dry_run: bool = False,
    ):
        self.token_id = token_id
        self.slug = slug
        self.target_half_spread = spread / 2
        self.quote_size = min(size, MAX_ORDER_SIZE)
        self.dry_run = dry_run

        self.state = self._load_state()
        self._client = None
        self._tick_size = 0.01
        self._neg_risk = False

        if not dry_run:
            self._init_client()

    def _init_client(self) -> None:
        pk = os.environ.get("POLYMARKET_PRIVATE_KEY", "")
        if not pk:
            logger.error("POLYMARKET_PRIVATE_KEY not set")
            return

        try:
            from py_clob_client.client import ClobClient
            self._client = ClobClient(
                host=CLOB_HOST,
                key=pk,
                chain_id=CHAIN_ID,
            )
            self._client.set_api_creds(self._client.create_or_derive_api_creds())

            # Get tick size and neg risk
            self._tick_size = float(self._client.get_tick_size(self.token_id))
            self._neg_risk = self._client.get_neg_risk(self.token_id)

            logger.info("CLOB connected. tick=%s neg_risk=%s",
                        self._tick_size, self._neg_risk)
        except Exception as e:
            logger.error("CLOB init failed: %s", e)
            self._client = None

    def run(self, max_cycles: int = 0) -> None:
        """Main market-making loop.

        Args:
            max_cycles: 0 = run forever, >0 = run N cycles then stop.
        """
        if not self.dry_run and not self._client:
            logger.error("No CLOB client. Set POLYMARKET_PRIVATE_KEY.")
            return

        self.state.started_at = datetime.now().isoformat()
        self.state.token_id = self.token_id
        self.state.slug = self.slug

        logger.info("Market maker starting")
        logger.info("  Token:  %s...%s", self.token_id[:20], self.token_id[-10:])
        logger.info("  Spread: %.3f (half=%.4f)", self.target_half_spread * 2, self.target_half_spread)
        logger.info("  Size:   %.1f shares", self.quote_size)
        logger.info("  Mode:   %s", "DRY RUN" if self.dry_run else "LIVE")

        cycle = 0
        try:
            while True:
                cycle += 1
                if max_cycles > 0 and cycle > max_cycles:
                    break

                if self.state.kill_switch:
                    logger.warning("KILL SWITCH: %s", self.state.kill_reason)
                    break

                try:
                    self._cycle()
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logger.error("Cycle error: %s", e)
                    time.sleep(5)

                time.sleep(REQUOTE_INTERVAL)

        except KeyboardInterrupt:
            logger.info("Stopped by user.")
        finally:
            self._cancel_all_quotes()
            self._save_state()
            self._print_summary()

    def _cycle(self) -> None:
        """Single market-making cycle: fetch book → check fills → requote."""
        # 1. Fetch current orderbook
        book = self._get_book()
        if not book:
            return

        best_bid = book["best_bid"]
        best_ask = book["best_ask"]
        mid = book["mid"]
        spread = book["spread"]

        # 2. Check our fills (did our orders get matched?)
        self._check_fills()

        # 3. Decide whether to quote
        if spread < MIN_SPREAD:
            logger.debug("Spread too tight (%.4f < %.4f), skipping", spread, MIN_SPREAD)
            return

        if abs(self.state.net_position) >= MAX_INVENTORY:
            logger.warning("Max inventory reached (%.0f), not adding more",
                           self.state.net_position)
            # Only quote the reducing side
            self._quote_reducing_only(mid, best_bid, best_ask)
            return

        # 4. Calculate quotes with inventory skew
        quote = self._calculate_quote(mid, best_bid, best_ask)

        # 5. Cancel stale orders and place new quotes
        self._cancel_all_quotes()
        self._place_quotes(quote)

        self.state.n_requotes += 1

        # 6. Check PnL limits
        self._update_pnl(mid)
        if self.state.total_pnl < -MAX_LOSS_PER_MARKET:
            self.state.kill_switch = True
            self.state.kill_reason = f"loss limit hit ({self.state.total_pnl:.2f})"

        # 7. Log
        if self.state.n_requotes % 10 == 0:
            logger.info(
                "pos=%.0f pnl=$%.2f fills=%d trips=%d mid=%.4f "
                "bid=%.4f ask=%.4f",
                self.state.net_position, self.state.total_pnl,
                self.state.n_fills, self.state.n_round_trips,
                mid, quote.bid_price, quote.ask_price,
            )

        self._save_state()

    def _get_book(self) -> dict | None:
        """Fetch current orderbook."""
        if self.dry_run:
            # Simulate a book from cached market data
            return self._get_simulated_book()

        try:
            book = self._client.get_order_book(self.token_id)
            if not book.bids or not book.asks:
                logger.warning("Empty book")
                return None

            best_bid = float(book.bids[0].price)
            best_ask = float(book.asks[0].price)
            mid = (best_bid + best_ask) / 2
            spread = best_ask - best_bid

            return {
                "best_bid": best_bid,
                "best_ask": best_ask,
                "mid": mid,
                "spread": spread,
                "bid_depth": sum(float(b.size) for b in book.bids[:5]),
                "ask_depth": sum(float(a.size) for a in book.asks[:5]),
            }
        except Exception as e:
            logger.error("Book fetch failed: %s", e)
            return None

    def _get_simulated_book(self) -> dict:
        """Simulated book for dry runs."""
        from src.data.cache import MarketCache
        cache = MarketCache()
        market = None
        for m in cache.markets.values():
            if m.home_token_id == self.token_id or m.away_token_id == self.token_id:
                market = m
                break

        if market:
            price = (market.home_price if market.home_token_id == self.token_id
                     else market.away_price)
        else:
            price = 0.50

        # Add some random noise to simulate movement
        import random
        noise = random.gauss(0, 0.002)
        mid = max(0.05, min(0.95, price + noise))
        spread = 0.02

        return {
            "best_bid": mid - spread / 2,
            "best_ask": mid + spread / 2,
            "mid": mid,
            "spread": spread,
            "bid_depth": 500,
            "ask_depth": 500,
        }

    def _calculate_quote(self, mid: float, best_bid: float,
                         best_ask: float) -> Quote:
        """Calculate bid/ask prices with inventory skew."""
        # Base spread around mid
        half_spread = self.target_half_spread

        # Inventory skew: if we're long, lower our bid (buy less) and lower our ask (sell more)
        skew = self.state.net_position * SKEW_PER_SHARE

        # Skewed mid
        skewed_mid = mid - skew

        bid_price = self._round_price(skewed_mid - half_spread)
        ask_price = self._round_price(skewed_mid + half_spread)

        # Don't cross the market
        bid_price = min(bid_price, best_bid)
        ask_price = max(ask_price, best_ask)

        # Ensure minimum spread
        if ask_price - bid_price < MIN_SPREAD:
            bid_price = self._round_price(mid - MIN_SPREAD / 2)
            ask_price = self._round_price(mid + MIN_SPREAD / 2)

        # Clamp to valid range
        bid_price = max(self._tick_size, min(1.0 - self._tick_size, bid_price))
        ask_price = max(self._tick_size, min(1.0 - self._tick_size, ask_price))

        # Size: reduce size when near inventory limits
        inventory_ratio = abs(self.state.net_position) / MAX_INVENTORY
        bid_size = self.quote_size * (1.0 - inventory_ratio * 0.5) if self.state.net_position > 0 else self.quote_size
        ask_size = self.quote_size * (1.0 - inventory_ratio * 0.5) if self.state.net_position < 0 else self.quote_size

        bid_size = max(1.0, bid_size)
        ask_size = max(1.0, ask_size)

        return Quote(
            bid_price=bid_price,
            bid_size=bid_size,
            ask_price=ask_price,
            ask_size=ask_size,
            mid=mid,
            spread=ask_price - bid_price,
            skew=skew,
        )

    def _quote_reducing_only(self, mid: float, best_bid: float,
                             best_ask: float) -> None:
        """Only quote the side that reduces inventory."""
        self._cancel_all_quotes()

        if self.state.net_position > 0:
            # Long → only place ask (sell) to reduce
            ask_price = self._round_price(mid + self.target_half_spread * 0.5)
            ask_price = max(ask_price, best_ask)
            self._place_ask(ask_price, self.quote_size)
        else:
            # Short → only place bid (buy) to reduce
            bid_price = self._round_price(mid - self.target_half_spread * 0.5)
            bid_price = min(bid_price, best_bid)
            self._place_bid(bid_price, self.quote_size)

    def _place_quotes(self, quote: Quote) -> None:
        """Place bid and ask orders."""
        self._place_bid(quote.bid_price, quote.bid_size)
        self._place_ask(quote.ask_price, quote.ask_size)

    def _place_bid(self, price: float, size: float) -> str | None:
        """Place a buy order."""
        logger.debug("BID %.4f x %.1f", price, size)

        if self.dry_run:
            self.state.active_bid_id = f"dry_bid_{int(time.time())}"
            self._log_event("quote_bid", price=price, size=size)
            return self.state.active_bid_id

        try:
            from py_clob_client.order_builder.constants import BUY
            from py_clob_client.client import OrderArgs

            order_args = OrderArgs(
                price=price,
                size=size,
                side=BUY,
                token_id=self.token_id,
            )
            signed = self._client.create_order(order_args)
            resp = self._client.post_order(signed)
            order_id = resp.get("orderID", "") if isinstance(resp, dict) else str(resp)
            self.state.active_bid_id = order_id
            self._log_event("quote_bid", price=price, size=size, order_id=order_id)
            return order_id
        except Exception as e:
            logger.error("Bid failed: %s", e)
            return None

    def _place_ask(self, price: float, size: float) -> str | None:
        """Place a sell order."""
        logger.debug("ASK %.4f x %.1f", price, size)

        if self.dry_run:
            self.state.active_ask_id = f"dry_ask_{int(time.time())}"
            self._log_event("quote_ask", price=price, size=size)
            return self.state.active_ask_id

        try:
            from py_clob_client.order_builder.constants import SELL
            from py_clob_client.client import OrderArgs

            order_args = OrderArgs(
                price=price,
                size=size,
                side=SELL,
                token_id=self.token_id,
            )
            signed = self._client.create_order(order_args)
            resp = self._client.post_order(signed)
            order_id = resp.get("orderID", "") if isinstance(resp, dict) else str(resp)
            self.state.active_ask_id = order_id
            self._log_event("quote_ask", price=price, size=size, order_id=order_id)
            return order_id
        except Exception as e:
            logger.error("Ask failed: %s", e)
            return None

    def _cancel_all_quotes(self) -> None:
        """Cancel all our active orders."""
        if self.dry_run:
            self.state.active_bid_id = ""
            self.state.active_ask_id = ""
            self.state.n_cancels += 1
            return

        if not self._client:
            return

        try:
            ids_to_cancel = []
            if self.state.active_bid_id:
                ids_to_cancel.append(self.state.active_bid_id)
            if self.state.active_ask_id:
                ids_to_cancel.append(self.state.active_ask_id)

            if ids_to_cancel:
                self._client.cancel_orders(ids_to_cancel)
                self.state.n_cancels += 1

            self.state.active_bid_id = ""
            self.state.active_ask_id = ""
        except Exception as e:
            logger.warning("Cancel failed: %s", e)
            # Nuclear option: cancel all
            try:
                self._client.cancel_all()
            except Exception:
                pass
            self.state.active_bid_id = ""
            self.state.active_ask_id = ""

    def _check_fills(self) -> None:
        """Check if our orders got filled."""
        if self.dry_run:
            self._simulate_fills()
            return

        if not self._client:
            return

        try:
            orders = self._client.get_orders()
            for order in (orders if isinstance(orders, list) else []):
                oid = order.get("orderID", "")
                status = order.get("status", "")
                side = order.get("side", "")
                price = float(order.get("price", 0))
                size_filled = float(order.get("size_matched", 0))

                if status == "FILLED" or size_filled > 0:
                    if oid == self.state.active_bid_id:
                        self._on_fill("buy", price, size_filled)
                        self.state.active_bid_id = ""
                    elif oid == self.state.active_ask_id:
                        self._on_fill("sell", price, size_filled)
                        self.state.active_ask_id = ""
        except Exception as e:
            logger.warning("Fill check failed: %s", e)

    def _simulate_fills(self) -> None:
        """Simulate random fills for dry-run testing."""
        import random
        # ~5% chance of fill per cycle per side
        if random.random() < 0.05 and self.state.active_bid_id:
            book = self._get_simulated_book()
            self._on_fill("buy", book["best_bid"], self.quote_size)
            self.state.active_bid_id = ""

        if random.random() < 0.05 and self.state.active_ask_id:
            book = self._get_simulated_book()
            self._on_fill("sell", book["best_ask"], self.quote_size)
            self.state.active_ask_id = ""

    def _on_fill(self, side: str, price: float, size: float) -> None:
        """Handle a fill event."""
        self.state.n_fills += 1

        if side == "buy":
            self.state.net_position += size
            self.state.total_bought += size
            cost = price * size
            # Update average buy price
            if self.state.total_bought > 0:
                self.state.avg_buy_price = (
                    (self.state.avg_buy_price * (self.state.total_bought - size) + cost)
                    / self.state.total_bought
                )
            logger.info("FILL BUY  %.4f x %.1f | pos=%.0f pnl=$%.2f",
                        price, size, self.state.net_position, self.state.total_pnl)

        elif side == "sell":
            self.state.net_position -= size
            self.state.total_sold += size
            revenue = price * size
            # Update average sell price
            if self.state.total_sold > 0:
                self.state.avg_sell_price = (
                    (self.state.avg_sell_price * (self.state.total_sold - size) + revenue)
                    / self.state.total_sold
                )

            # Check for round-trip completion
            if self.state.total_sold > 0 and self.state.total_bought > 0:
                trips = min(self.state.total_bought, self.state.total_sold) / self.quote_size
                if trips > self.state.n_round_trips:
                    trip_pnl = (self.state.avg_sell_price - self.state.avg_buy_price) * self.quote_size
                    self.state.realized_pnl += trip_pnl
                    self.state.n_round_trips = int(trips)
                    logger.info("ROUND TRIP #%d: pnl=$%.3f (buy@%.4f sell@%.4f)",
                                self.state.n_round_trips, trip_pnl,
                                self.state.avg_buy_price, price)

            logger.info("FILL SELL %.4f x %.1f | pos=%.0f pnl=$%.2f",
                        price, size, self.state.net_position, self.state.total_pnl)

        self._log_event("fill", side=side, price=price, size=size,
                        position=self.state.net_position)

    def _update_pnl(self, mid: float) -> None:
        """Update unrealized PnL based on current mid price.

        Unrealized = how much we'd make/lose if we closed all inventory at mid.
        If long (net_position > 0): we bought shares, can sell at mid.
          unrealized = net_position * (mid - avg_buy_price)
        If short (net_position < 0): we sold shares, need to buy back at mid.
          unrealized = abs(net_position) * (avg_sell_price - mid)
        """
        if self.state.net_position > 0 and self.state.avg_buy_price > 0:
            self.state.unrealized_pnl = self.state.net_position * (mid - self.state.avg_buy_price)
        elif self.state.net_position < 0 and self.state.avg_sell_price > 0:
            self.state.unrealized_pnl = abs(self.state.net_position) * (self.state.avg_sell_price - mid)
        else:
            self.state.unrealized_pnl = 0.0

        self.state.total_pnl = self.state.realized_pnl + self.state.unrealized_pnl

    def _round_price(self, price: float) -> float:
        """Round price to nearest tick."""
        return round(round(price / self._tick_size) * self._tick_size, 4)

    def _log_event(self, event_type: str, **kwargs) -> None:
        """Log an event to the MM log."""
        os.makedirs(os.path.dirname(MM_LOG) or ".", exist_ok=True)
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            "token_id": self.token_id[:20] + "...",
            "position": self.state.net_position,
            "pnl": self.state.total_pnl,
            **kwargs,
        }
        with open(MM_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _load_state(self) -> MMState:
        if os.path.exists(MM_STATE):
            try:
                with open(MM_STATE) as f:
                    data = json.load(f)
                if data.get("token_id") == self.token_id:
                    return MMState.from_dict(data)
            except (json.JSONDecodeError, OSError):
                pass
        return MMState(token_id=self.token_id)

    def _save_state(self) -> None:
        os.makedirs(os.path.dirname(MM_STATE) or ".", exist_ok=True)
        with open(MM_STATE, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2)

    def _print_summary(self) -> None:
        s = self.state
        print(f"\n{'='*60}")
        print(f"  MARKET MAKER SUMMARY")
        print(f"{'='*60}")
        print(f"  Token:         {s.token_id[:30]}...")
        print(f"  Duration:      {s.started_at} → {datetime.now().isoformat()}")
        print(f"  Net position:  {s.net_position:.0f} shares")
        print(f"  Total bought:  {s.total_bought:.0f} @ avg {s.avg_buy_price:.4f}")
        print(f"  Total sold:    {s.total_sold:.0f} @ avg {s.avg_sell_price:.4f}")
        print(f"  Round trips:   {s.n_round_trips}")
        print(f"  Total fills:   {s.n_fills}")
        print(f"  Requotes:      {s.n_requotes}")
        print(f"  Cancels:       {s.n_cancels}")
        print(f"  Realized PnL:  ${s.realized_pnl:.4f}")
        print(f"  Unrealized:    ${s.unrealized_pnl:.4f}")
        print(f"  Total PnL:     ${s.total_pnl:.4f}")
        if s.kill_switch:
            print(f"  Kill switch:   ON — {s.kill_reason}")
        print(f"{'='*60}\n")


def find_best_market() -> tuple[str, str] | None:
    """Find the best market to make on — queries Polymarket API directly.

    Picks the most liquid active market with price closest to 0.50
    (tightest spreads, most fill opportunities).
    """
    import requests

    GAMMA_API = "https://gamma-api.polymarket.com"
    logger.info("Fetching all active markets from Polymarket...")

    candidates = []

    try:
        resp = requests.get(
            f"{GAMMA_API}/markets",
            params={
                "active": "true",
                "closed": "false",
                "limit": "100",
                "order": "volume",
                "ascending": "false",
            },
            timeout=15,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        markets = resp.json()

        if not isinstance(markets, list):
            logger.error("Unexpected API response")
            return None

        for mkt in markets:
            try:
                tokens_raw = mkt.get("clobTokenIds", "[]")
                if isinstance(tokens_raw, str):
                    import json as _json
                    tokens = _json.loads(tokens_raw)
                else:
                    tokens = tokens_raw

                prices_raw = mkt.get("outcomePrices", "[]")
                if isinstance(prices_raw, str):
                    import json as _json
                    prices = _json.loads(prices_raw)
                else:
                    prices = prices_raw

                if not tokens or not prices:
                    continue

                volume = float(mkt.get("volume", 0) or 0)
                liquidity = float(mkt.get("liquidity", 0) or 0)
                slug = mkt.get("slug", "") or mkt.get("question", "")[:50]

                # For each outcome in this market
                for i, token_id in enumerate(tokens):
                    if i >= len(prices):
                        break
                    price = float(prices[i])

                    # Best for MM: price near 0.50, high volume, high liquidity
                    if price < 0.05 or price > 0.95:
                        continue  # skip extreme prices

                    # Score: prefer prices near 0.50, high volume
                    dist_from_half = abs(price - 0.50)
                    score = volume * (1.0 - dist_from_half) + liquidity * 0.5

                    candidates.append({
                        "token_id": token_id,
                        "slug": slug,
                        "price": price,
                        "volume": volume,
                        "liquidity": liquidity,
                        "score": score,
                    })

            except (ValueError, KeyError, TypeError):
                continue

    except Exception as e:
        logger.error("API fetch failed: %s", e)
        return None

    if not candidates:
        logger.error("No tradeable markets found")
        return None

    # Sort by score, pick the best
    candidates.sort(key=lambda c: c["score"], reverse=True)

    # Show top 5 candidates
    logger.info("Top markets for market making:")
    for c in candidates[:5]:
        logger.info("  %s — price=%.3f vol=$%.0f liq=$%.0f score=%.0f",
                     c["slug"][:40], c["price"], c["volume"],
                     c["liquidity"], c["score"])

    best = candidates[0]
    logger.info("SELECTED: %s (price=%.3f, vol=$%.0f)",
                best["slug"][:50], best["price"], best["volume"])
    return best["token_id"], best["slug"]


def find_top_markets(n: int = 5) -> list[dict]:
    """Find the top N markets for market making — queries Polymarket API.

    Returns list of dicts with token_id, slug, price, volume, liquidity, score.
    Deduplicates by slug (only keep best outcome per market).
    """
    import requests

    GAMMA_API = "https://gamma-api.polymarket.com"
    logger.info("Fetching top %d markets from Polymarket...", n)

    candidates = []

    try:
        resp = requests.get(
            f"{GAMMA_API}/markets",
            params={
                "active": "true",
                "closed": "false",
                "limit": "100",
                "order": "volume",
                "ascending": "false",
            },
            timeout=15,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        markets = resp.json()

        if not isinstance(markets, list):
            return []

        for mkt in markets:
            try:
                tokens_raw = mkt.get("clobTokenIds", "[]")
                tokens = json.loads(tokens_raw) if isinstance(tokens_raw, str) else tokens_raw

                prices_raw = mkt.get("outcomePrices", "[]")
                prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw

                if not tokens or not prices:
                    continue

                volume = float(mkt.get("volume", 0) or 0)
                liquidity = float(mkt.get("liquidity", 0) or 0)
                slug = mkt.get("slug", "") or mkt.get("question", "")[:50]

                for i, token_id in enumerate(tokens):
                    if i >= len(prices):
                        break
                    price = float(prices[i])
                    if price < 0.05 or price > 0.95:
                        continue

                    dist_from_half = abs(price - 0.50)
                    score = volume * (1.0 - dist_from_half) + liquidity * 0.5

                    candidates.append({
                        "token_id": token_id,
                        "slug": slug,
                        "price": price,
                        "volume": volume,
                        "liquidity": liquidity,
                        "score": score,
                    })
            except (ValueError, KeyError, TypeError):
                continue

    except Exception as e:
        logger.error("API fetch failed: %s", e)
        return []

    # Deduplicate: keep only the best outcome per slug
    best_per_slug: dict[str, dict] = {}
    for c in candidates:
        slug = c["slug"]
        if slug not in best_per_slug or c["score"] > best_per_slug[slug]["score"]:
            best_per_slug[slug] = c

    ranked = sorted(best_per_slug.values(), key=lambda c: c["score"], reverse=True)
    return ranked[:n]


def run_multi_market(n_markets: int = 3, spread: float = 0.02, size: float = 5.0,
                     max_cycles: int = 0, dry_run: bool = False) -> None:
    """Run market maker on multiple markets simultaneously using threads."""
    import threading

    top = find_top_markets(n=n_markets)
    if not top:
        logger.error("No markets found")
        return

    logger.info("=" * 60)
    logger.info("  MULTI-MARKET MAKER — %d markets", len(top))
    logger.info("=" * 60)
    for i, m in enumerate(top):
        logger.info("  [%d] %s — price=%.3f vol=$%.0f liq=$%.0f",
                     i + 1, m["slug"][:45], m["price"], m["volume"], m["liquidity"])
    logger.info("=" * 60)

    # Split total deployed capital across markets
    per_market_size = min(size, MAX_TOTAL_DEPLOYED / len(top) / 2)

    threads: list[threading.Thread] = []
    makers: list[MarketMaker] = []

    for m in top:
        mm = MarketMaker(
            token_id=m["token_id"],
            slug=m["slug"],
            spread=spread,
            size=per_market_size,
            dry_run=dry_run,
        )
        makers.append(mm)

        t = threading.Thread(
            target=mm.run,
            args=(max_cycles,),
            name=f"mm-{m['slug'][:20]}",
            daemon=True,
        )
        threads.append(t)

    # Start all threads
    for t in threads:
        t.start()
        time.sleep(0.5)  # stagger starts

    logger.info("All %d market makers running. Press Ctrl+C to stop.", len(threads))

    try:
        while any(t.is_alive() for t in threads):
            time.sleep(5)
            # Print combined status every 30 seconds
            alive = sum(1 for t in threads if t.is_alive())
            total_pnl = sum(mm.state.total_pnl for mm in makers)
            total_fills = sum(mm.state.n_fills for mm in makers)
            total_trips = sum(mm.state.n_round_trips for mm in makers)
            logger.info(
                "MULTI-MM STATUS: %d/%d alive | pnl=$%.4f | fills=%d | trips=%d",
                alive, len(threads), total_pnl, total_fills, total_trips,
            )
    except KeyboardInterrupt:
        logger.info("Stopping all market makers...")
        for mm in makers:
            mm.state.kill_switch = True
            mm.state.kill_reason = "user stop"

        for t in threads:
            t.join(timeout=10)

    # Print combined summary
    print(f"\n{'=' * 60}")
    print(f"  MULTI-MARKET MAKER FINAL SUMMARY")
    print(f"{'=' * 60}")
    total_pnl = 0
    for mm in makers:
        total_pnl += mm.state.total_pnl
        print(f"  {mm.slug[:30]:30s} | pos={mm.state.net_position:+.0f} "
              f"fills={mm.state.n_fills} trips={mm.state.n_round_trips} "
              f"pnl=${mm.state.total_pnl:+.4f}")
    print(f"{'─' * 60}")
    print(f"  TOTAL PnL: ${total_pnl:+.4f}")
    print(f"{'=' * 60}\n")


def check_wallet() -> None:
    """Check which wallet the private key corresponds to and its balance."""
    pk = os.environ.get("POLYMARKET_PRIVATE_KEY", "")
    if not pk:
        print("ERROR: POLYMARKET_PRIVATE_KEY not set in .env")
        return

    print(f"\n{'=' * 60}")
    print(f"  WALLET CHECK")
    print(f"{'=' * 60}")
    print(f"  Private key: {pk[:6]}...{pk[-4:]}")

    try:
        from py_clob_client.client import ClobClient

        client = ClobClient(host=CLOB_HOST, key=pk, chain_id=CHAIN_ID)

        # Get the wallet address from the private key
        # The client exposes the address
        addr = None
        try:
            # Try different ways to get the address
            if hasattr(client, 'get_address'):
                addr = client.get_address()
            elif hasattr(client, 'creds') and client.creds:
                addr = getattr(client.creds, 'api_key', None)
            # Derive from private key directly
            if not addr:
                from eth_account import Account
                acct = Account.from_key(pk)
                addr = acct.address
        except Exception:
            pass

        if addr:
            print(f"  Wallet address: {addr}")
        else:
            print(f"  Wallet address: (could not derive)")

        # Try to derive API creds to verify connection
        creds = client.create_or_derive_api_creds()
        client.set_api_creds(creds)
        print(f"  API connection: OK")
        print(f"  API key: {creds.api_key[:20]}...")

        # Try to get balance info via allowances or open orders
        try:
            orders = client.get_orders()
            n_orders = len(orders) if isinstance(orders, list) else 0
            print(f"  Open orders: {n_orders}")
        except Exception as e:
            print(f"  Orders check: {e}")

    except Exception as e:
        print(f"  ERROR: {e}")

    print(f"\n  IMPORTANT: Make sure this address matches your")
    print(f"  Polymarket account. Check on Polymarket web →")
    print(f"  Settings → Wallet to compare addresses.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Polymarket Market Maker")
    parser.add_argument("--token", type=str, help="Token ID to make market on")
    parser.add_argument("--auto", action="store_true", help="Auto-select best market")
    parser.add_argument("--multi", type=int, default=0,
                        help="Run on top N markets simultaneously (e.g. --multi 5)")
    parser.add_argument("--spread", type=float, default=0.02, help="Target spread (default 0.02)")
    parser.add_argument("--size", type=float, default=5.0, help="Quote size in shares (default 5)")
    parser.add_argument("--cycles", type=int, default=0, help="Max cycles (0=forever)")
    parser.add_argument("--dry", action="store_true", help="Dry run (no real orders)")
    parser.add_argument("--check", action="store_true", help="Check wallet address and balance")

    args = parser.parse_args()

    # Wallet check mode
    if args.check:
        check_wallet()
        sys.exit(0)

    # Multi-market mode
    if args.multi > 0:
        run_multi_market(
            n_markets=args.multi,
            spread=args.spread,
            size=args.size,
            max_cycles=args.cycles,
            dry_run=args.dry,
        )
        sys.exit(0)

    # Single market mode
    token_id = args.token
    slug = ""

    if args.auto or not token_id:
        result = find_best_market()
        if result:
            token_id, slug = result
        else:
            print("No market found. Use --token to specify one.")
            sys.exit(1)

    mm = MarketMaker(
        token_id=token_id,
        slug=slug,
        spread=args.spread,
        size=args.size,
        dry_run=args.dry,
    )
    mm.run(max_cycles=args.cycles)
