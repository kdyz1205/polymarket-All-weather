"""
Rust Bridge Feed — connects the Rust sports_engine.MarketBook to the
Python Event Bus, enabling the hybrid architecture:

  External API → Rust MarketBook (microsecond updates) → Python Event Bus (millisecond consumers)

This module also provides a real Polymarket WebSocket feed that pushes
data through the Rust engine before publishing to the event bus.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import aiohttp

from src.core.event_bus import Events, bus
from src.core.models import OrderBookLevel, OrderBookSnapshot, Tick

logger = logging.getLogger(__name__)

# Import the Rust engine
try:
    import sports_engine as se

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    logger.warning("sports_engine not available — falling back to pure Python")


class RustBridgeFeed:
    """
    Wraps a Rust MarketBook and publishes updates to the Python Event Bus.

    Data flow:
    1. External source (API, WebSocket, simulator) calls update_book()
    2. Rust MarketBook is updated in-place (O(1) operations)
    3. Python Tick is constructed from Rust snapshot
    4. Tick is published to Event Bus for Factor Engine, Dashboard, etc.
    """

    def __init__(self, market_id: str, runner_ids: list[str]) -> None:
        self.market_id = market_id
        self._running = False

        if RUST_AVAILABLE:
            self._book = se.MarketBook(market_id)
            for rid in runner_ids:
                self._book.add_runner(rid, rid)
            logger.info("[%s] Rust MarketBook initialized with %d runners", market_id, len(runner_ids))
        else:
            self._book = None

    async def update_book(
        self,
        runner_id: str,
        back_levels: list[tuple[float, float]],
        lay_levels: list[tuple[float, float]],
    ) -> None:
        """
        Push new book data into Rust, then publish a Tick to the event bus.
        This is the core bridge: Rust handles the raw data, Python consumes the result.
        """
        if not self._book:
            return

        # Update Rust (microseconds)
        self._book.update_runner_back(runner_id, back_levels)
        self._book.update_runner_lay(runner_id, lay_levels)

        # Extract snapshot from Rust -> publish as Python Tick
        snap = self._book.get_runner_snapshot(runner_id)
        now_ms = int(time.time() * 1000)

        tick = Tick(
            market_id=self.market_id,
            timestamp_ms=now_ms,
            mid_price=(snap.best_back_price + snap.best_lay_price) / 2
            if snap.best_back_price > 0 and snap.best_lay_price > 0
            else 0.0,
            last_trade_price=snap.last_traded_price,
            volume_24h=snap.traded_volume,
            spread=snap.spread if snap.spread < 1e10 else 0.0,
            bid_depth=snap.back_depth,
            ask_depth=snap.lay_depth,
            book_imbalance=snap.imbalance,
        )

        await bus.publish(Events.TICK, tick)

        # Also publish the raw book snapshot for detailed consumers
        book_snapshot = OrderBookSnapshot(
            market_id=self.market_id,
            timestamp_ms=now_ms,
            bids=[
                OrderBookLevel(price=snap.best_back_price, size=snap.best_back_size)
            ] if snap.best_back_price > 0 else [],
            asks=[
                OrderBookLevel(price=snap.best_lay_price, size=snap.best_lay_size)
            ] if snap.best_lay_price > 0 else [],
        )
        await bus.publish(Events.ORDER_BOOK, book_snapshot)

    def get_rust_snapshot(self):
        """Get full Rust MarketBookSnapshot for direct access."""
        if self._book:
            return self._book.snapshot()
        return None


class PolymarketWebSocketFeed:
    """
    Real-time Polymarket CLOB WebSocket feed that pushes data through
    the Rust MarketBook before publishing to the Python Event Bus.

    Connects to the Polymarket WebSocket API for live order book updates.
    """

    POLYMARKET_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

    def __init__(
        self,
        market_id: str,
        token_id: str,
        condition_id: str = "",
    ) -> None:
        self.market_id = market_id
        self.token_id = token_id
        self.condition_id = condition_id
        self._bridge = RustBridgeFeed(market_id, ["yes", "no"])
        self._running = False
        self._ws = None
        self._session: aiohttp.ClientSession | None = None
        self._reconnect_delay = 1.0

    async def start(self) -> None:
        self._running = True
        self._session = aiohttp.ClientSession()
        logger.info("[%s] Starting Polymarket WebSocket feed", self.market_id)
        asyncio.create_task(self._connect_loop())

    async def stop(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()
        if self._session:
            await self._session.close()
        logger.info("[%s] Polymarket WebSocket feed stopped", self.market_id)

    async def _connect_loop(self) -> None:
        """Reconnection loop with exponential backoff."""
        while self._running:
            try:
                await self._connect_and_stream()
            except Exception:
                logger.exception("[%s] WebSocket error, reconnecting in %.1fs",
                                 self.market_id, self._reconnect_delay)
            if self._running:
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 30.0)

    async def _connect_and_stream(self) -> None:
        """Connect to Polymarket WS and process messages."""
        async with self._session.ws_connect(self.POLYMARKET_WS_URL) as ws:
            self._ws = ws
            self._reconnect_delay = 1.0
            logger.info("[%s] WebSocket connected", self.market_id)

            # Subscribe to this market's book
            subscribe_msg = {
                "type": "subscribe",
                "channel": "book",
                "markets": [self.token_id],
            }
            await ws.send_json(subscribe_msg)
            logger.info("[%s] Subscribed to book channel for %s", self.market_id, self.token_id)

            async for msg in ws:
                if not self._running:
                    break

                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self._handle_message(data)
                    except json.JSONDecodeError:
                        logger.warning("[%s] Invalid JSON: %s", self.market_id, msg.data[:100])
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error("[%s] WebSocket error: %s", self.market_id, ws.exception())
                    break
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.info("[%s] WebSocket closed", self.market_id)
                    break

    async def _handle_message(self, data: dict) -> None:
        """Process a Polymarket WebSocket message."""
        msg_type = data.get("type", "")

        if msg_type == "book":
            # Full book snapshot or delta
            bids = data.get("bids", [])
            asks = data.get("asks", [])

            # Convert Polymarket format to (price, size) tuples
            # Polymarket: YES token price 0-1, size in USDC
            back_levels = [
                (float(b.get("price", 0)), float(b.get("size", 0)))
                for b in bids
                if float(b.get("size", 0)) > 0
            ]
            lay_levels = [
                (float(a.get("price", 0)), float(a.get("size", 0)))
                for a in asks
                if float(a.get("size", 0)) > 0
            ]

            # Sort: back descending, lay ascending
            back_levels.sort(key=lambda x: -x[0])
            lay_levels.sort(key=lambda x: x[0])

            # Push through Rust bridge -> Event Bus
            await self._bridge.update_book("yes", back_levels, lay_levels)

        elif msg_type == "price_change":
            # Price update message
            price = float(data.get("price", 0))
            if price > 0:
                logger.debug("[%s] Price change: %.4f", self.market_id, price)

    @property
    def rust_book(self):
        """Direct access to the underlying Rust MarketBook."""
        return self._bridge


class PolymarketRESTFeed:
    """
    REST-based Polymarket feed for when WebSocket is unavailable.
    Polls the CLOB API at regular intervals and pushes through Rust bridge.
    """

    CLOB_BASE = "https://clob.polymarket.com"

    def __init__(
        self,
        market_id: str,
        token_id: str,
        poll_interval_ms: int = 2000,
    ) -> None:
        self.market_id = market_id
        self.token_id = token_id
        self.poll_interval_ms = poll_interval_ms
        self._bridge = RustBridgeFeed(market_id, ["yes", "no"])
        self._running = False
        self._session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        self._running = True
        self._session = aiohttp.ClientSession()
        logger.info("[%s] Starting Polymarket REST feed (polling every %dms)",
                    self.market_id, self.poll_interval_ms)
        asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        self._running = False
        if self._session:
            await self._session.close()

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                await self._fetch_and_push()
            except Exception:
                logger.exception("[%s] REST poll error", self.market_id)
            await asyncio.sleep(self.poll_interval_ms / 1000)

    async def _fetch_and_push(self) -> None:
        """Fetch order book from REST API and push through Rust bridge."""
        url = f"{self.CLOB_BASE}/book"
        params = {"token_id": self.token_id}

        async with self._session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status != 200:
                logger.warning("[%s] REST fetch failed: HTTP %d", self.market_id, resp.status)
                return
            data = await resp.json()

        bids = [(float(b["price"]), float(b["size"])) for b in data.get("bids", [])]
        asks = [(float(a["price"]), float(a["size"])) for a in data.get("asks", [])]
        bids.sort(key=lambda x: -x[0])
        asks.sort(key=lambda x: x[0])

        await self._bridge.update_book("yes", bids, asks)

    @property
    def rust_book(self):
        return self._bridge
