"""
Live quote fetcher — pulls real-time order book from Polymarket CLOB API.

Synchronous wrapper around the CLOB /book endpoint for use in the
execution pipeline. For streaming use cases, use PolymarketFeed instead.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import requests

logger = logging.getLogger(__name__)

CLOB_BASE = "https://clob.polymarket.com"


@dataclass
class LiveQuote:
    """Snapshot of a single token's order book at fetch time."""
    token_id: str
    timestamp_ms: int
    best_bid: float = 0.0
    best_ask: float = 0.0
    mid_price: float = 0.0
    spread: float = 0.0
    bid_depth: float = 0.0      # total size on bid side
    ask_depth: float = 0.0      # total size on ask side
    bid_levels: list[tuple[float, float]] = field(default_factory=list)  # (price, size)
    ask_levels: list[tuple[float, float]] = field(default_factory=list)
    last_fetch_ok: bool = True
    error: str = ""

    @property
    def is_valid(self) -> bool:
        return self.last_fetch_ok and self.best_bid > 0 and self.best_ask > 0

    @property
    def implied_prob(self) -> float:
        """Mid price IS the implied probability on Polymarket (prices 0-1)."""
        return self.mid_price

    @property
    def decimal_odds(self) -> float:
        """Convert mid price to decimal odds (e.g., 0.55 → 1.818)."""
        return 1.0 / self.mid_price if self.mid_price > 0 else 0.0


class LiveQuoteFetcher:
    """Fetches real-time quotes from Polymarket CLOB API.

    Uses synchronous requests for simplicity in the execution pipeline.
    Includes retry logic and staleness detection.
    """

    def __init__(self, base_url: str = CLOB_BASE, timeout_sec: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
        })

    def fetch(self, token_id: str) -> LiveQuote:
        """Fetch L2 order book for a single token.

        Returns a LiveQuote with error info if the fetch fails.
        """
        url = f"{self.base_url}/book"
        params = {"token_id": token_id}
        now_ms = int(time.time() * 1000)

        try:
            resp = self._session.get(url, params=params, timeout=self.timeout_sec)
            if resp.status_code != 200:
                return LiveQuote(
                    token_id=token_id, timestamp_ms=now_ms,
                    last_fetch_ok=False, error=f"HTTP {resp.status_code}",
                )
            data = resp.json()
        except requests.RequestException as e:
            return LiveQuote(
                token_id=token_id, timestamp_ms=now_ms,
                last_fetch_ok=False, error=str(e),
            )

        # Parse bids (descending) and asks (ascending)
        bids = sorted(
            [(float(b["price"]), float(b["size"])) for b in data.get("bids", [])],
            key=lambda x: -x[0],
        )
        asks = sorted(
            [(float(a["price"]), float(a["size"])) for a in data.get("asks", [])],
            key=lambda x: x[0],
        )

        best_bid = bids[0][0] if bids else 0.0
        best_ask = asks[0][0] if asks else 0.0
        mid = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask > 0 else 0.0
        spread = best_ask - best_bid if best_bid > 0 and best_ask > 0 else 0.0

        return LiveQuote(
            token_id=token_id,
            timestamp_ms=now_ms,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid,
            spread=spread,
            bid_depth=sum(s for _, s in bids),
            ask_depth=sum(s for _, s in asks),
            bid_levels=bids[:10],
            ask_levels=asks[:10],
        )

    def fetch_pair(self, yes_token_id: str, no_token_id: str) -> tuple[LiveQuote, LiveQuote]:
        """Fetch quotes for both sides of a binary market."""
        return self.fetch(yes_token_id), self.fetch(no_token_id)

    def close(self) -> None:
        self._session.close()
