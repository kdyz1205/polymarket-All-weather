"""
Polymarket data feed — connects to the CLOB API and streams
order book updates, trades, and market data.

This is the Data Ingestion Layer for Polymarket-type prediction markets.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import aiohttp

from src.core.event_bus import Events, bus
from src.core.models import (
    OrderBookLevel,
    OrderBookSnapshot,
    Side,
    Tick,
    Trade,
)

logger = logging.getLogger(__name__)


class PolymarketFeed:
    """
    Fetches order book and trade data from Polymarket's CLOB API.

    In production, this would use WebSocket streams for real-time data.
    For now, we poll the REST API at configurable intervals.
    """

    def __init__(
        self,
        market_id: str,
        endpoint: str,
        token_id: str,
        poll_interval_ms: int = 1000,
    ) -> None:
        self.market_id = market_id
        self.endpoint = endpoint.rstrip("/")
        self.token_id = token_id
        self.poll_interval_ms = poll_interval_ms
        self._running = False
        self._session: aiohttp.ClientSession | None = None
        self._cumulative_volume = 0.0

    async def start(self) -> None:
        self._running = True
        self._session = aiohttp.ClientSession()
        logger.info("[%s] Polymarket feed started (polling every %dms)", self.market_id, self.poll_interval_ms)
        asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        self._running = False
        if self._session:
            await self._session.close()
        logger.info("[%s] Polymarket feed stopped", self.market_id)

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                await self._fetch_and_publish()
            except Exception:
                logger.exception("[%s] Feed poll error", self.market_id)
            await asyncio.sleep(self.poll_interval_ms / 1000)

    async def _fetch_and_publish(self) -> None:
        if not self._session:
            return

        # Fetch order book
        book = await self._fetch_order_book()
        if book:
            await bus.publish(Events.ORDER_BOOK, book)

        # Fetch recent trades
        trades = await self._fetch_trades()
        for trade in trades:
            await bus.publish(Events.TRADE, trade)

        # Build and publish tick
        if book:
            tick = Tick.from_book_and_trades(book, trades, self._cumulative_volume)
            await bus.publish(Events.TICK, tick)

    async def _fetch_order_book(self) -> OrderBookSnapshot | None:
        """Fetch L2 order book from Polymarket CLOB."""
        url = f"{self.endpoint}/book"
        params = {"token_id": self.token_id}

        try:
            async with self._session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    logger.warning("[%s] Book fetch failed: HTTP %d", self.market_id, resp.status)
                    return None
                data = await resp.json()

            bids = [
                OrderBookLevel(price=float(b["price"]), size=float(b["size"]))
                for b in data.get("bids", [])
            ]
            asks = [
                OrderBookLevel(price=float(a["price"]), size=float(a["size"]))
                for a in data.get("asks", [])
            ]
            # Sort: bids descending, asks ascending
            bids.sort(key=lambda x: x.price, reverse=True)
            asks.sort(key=lambda x: x.price)

            return OrderBookSnapshot(
                market_id=self.market_id,
                timestamp_ms=int(time.time() * 1000),
                bids=bids,
                asks=asks,
            )
        except Exception as e:
            logger.error("[%s] Book fetch error: %s", self.market_id, e)
            return None

    async def _fetch_trades(self) -> list[Trade]:
        """Fetch recent trades. Returns empty list on failure."""
        # Polymarket doesn't have a direct trades endpoint in all versions;
        # this is a placeholder that can be adapted to the specific API version.
        return []


class SimulatedFeed:
    """
    Simulated data feed for testing and development.
    Generates realistic-looking prediction market data.
    """

    def __init__(
        self,
        market_id: str,
        initial_price: float = 0.5,
        volatility: float = 0.002,
        tick_interval_ms: int = 1000,
    ) -> None:
        self.market_id = market_id
        self.price = initial_price
        self.volatility = volatility
        self.tick_interval_ms = tick_interval_ms
        self._running = False
        self._tick_count = 0

    async def start(self) -> None:
        import numpy as np

        self._running = True
        self._rng = np.random.default_rng()
        logger.info("[%s] Simulated feed started (price=%.4f, vol=%.4f)", self.market_id, self.price, self.volatility)
        asyncio.create_task(self._generate_loop())

    async def stop(self) -> None:
        self._running = False

    async def _generate_loop(self) -> None:
        import numpy as np

        while self._running:
            # Random walk with mean reversion to 0.5
            shock = self._rng.normal(0, self.volatility)
            mean_revert = 0.001 * (0.5 - self.price)
            self.price += shock + mean_revert
            self.price = max(0.01, min(0.99, self.price))  # prediction market bounds

            spread = self._rng.uniform(0.005, 0.02)
            bid_depth = self._rng.uniform(100, 5000)
            ask_depth = self._rng.uniform(100, 5000)
            volume = self._rng.uniform(10000, 100000)

            now = int(time.time() * 1000)
            tick = Tick(
                market_id=self.market_id,
                timestamp_ms=now,
                mid_price=self.price,
                last_trade_price=self.price + self._rng.normal(0, spread / 4),
                volume_24h=volume,
                spread=spread,
                bid_depth=bid_depth,
                ask_depth=ask_depth,
                book_imbalance=(bid_depth - ask_depth) / (bid_depth + ask_depth),
            )

            await bus.publish(Events.TICK, tick)
            self._tick_count += 1

            if self._tick_count % 100 == 0:
                logger.debug(
                    "[%s] Tick #%d: price=%.4f spread=%.4f",
                    self.market_id, self._tick_count, self.price, spread,
                )

            await asyncio.sleep(self.tick_interval_ms / 1000)
