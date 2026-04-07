"""
Order builder — constructs standardized order objects from signals.

Takes a validated signal (past pre-trade check) and builds an
ExecutionOrder with all parameters needed for Polymarket CLOB submission.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, asdict
from typing import Any

from src.execution.pre_trade_check import DeviationResult, Signal
from src.execution.market_mapper import MarketMapping


@dataclass
class ExecutionOrder:
    """A fully-specified order ready for submission or manual review."""
    order_id: str
    timestamp: str

    # Market identity
    market_id: str           # internal ID
    condition_id: str        # Polymarket condition
    token_id: str            # which token to buy/sell
    description: str         # human-readable "LAL to beat BOS"

    # Order parameters
    side: str                # "buy" | "sell"
    price: float             # limit price (0-1 scale for Polymarket)
    size: float              # dollar amount
    max_slippage_pct: float  # reject if fill price deviates more than this

    # Signal context
    system_price: float      # model's decimal odds
    market_price: float      # real market decimal odds
    deviation_pct: float     # pre-trade deviation
    edge_bps: float          # estimated edge in basis points
    net_edge_bps: float      # edge after fees
    pre_trade_signal: str    # "green" | "yellow"

    # Safety
    ttl_sec: int = 60        # order expires after this many seconds
    sport: str = ""
    runner_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_expired(self) -> bool:
        """Check if order has expired based on TTL."""
        created = time.mktime(time.strptime(self.timestamp, "%Y-%m-%dT%H:%M:%S"))
        return (time.time() - created) > self.ttl_sec


# Micro-live hard limits
MAX_ORDER_SIZE = 1.0
MAX_SLIPPAGE_PCT = 2.0


class OrderBuilder:
    """Builds execution orders from pre-trade check results.

    Enforces hard safety limits regardless of what the signal says.
    """

    def __init__(
        self,
        max_size: float = MAX_ORDER_SIZE,
        max_slippage_pct: float = MAX_SLIPPAGE_PCT,
        default_ttl_sec: int = 60,
    ):
        self.max_size = max_size
        self.max_slippage_pct = max_slippage_pct
        self.default_ttl_sec = default_ttl_sec

    def build(
        self,
        mapping: MarketMapping,
        deviation: DeviationResult,
        runner_id: str,
        side: str,
        size: float,
        edge_bps: float,
        net_edge_bps: float,
    ) -> ExecutionOrder | None:
        """Build an execution order if pre-trade check allows it.

        Returns None if the signal is RED or parameters violate limits.
        """
        if deviation.signal == Signal.RED:
            return None

        # Enforce hard size cap
        capped_size = min(size, self.max_size)

        # Determine which token to trade
        # If backing "home" (yes outcome), buy the yes token
        # If backing "away" (no outcome), buy the no token
        if runner_id == "home":
            token_id = mapping.yes_token_id
            desc = f"{mapping.home_team} to beat {mapping.away_team}"
        else:
            token_id = mapping.no_token_id
            desc = f"{mapping.away_team} to beat {mapping.home_team}"

        # Convert system decimal odds to Polymarket price (0-1)
        # Polymarket price = implied probability = 1/decimal_odds
        poly_price = 1.0 / deviation.system_price if deviation.system_price > 0 else 0

        return ExecutionOrder(
            order_id=f"exec_{uuid.uuid4().hex[:8]}",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            market_id=mapping.internal_id,
            condition_id=mapping.condition_id,
            token_id=token_id,
            description=desc,
            side=side,
            price=round(poly_price, 4),
            size=capped_size,
            max_slippage_pct=self.max_slippage_pct,
            system_price=deviation.system_price,
            market_price=deviation.market_price,
            deviation_pct=deviation.deviation_pct,
            edge_bps=edge_bps,
            net_edge_bps=net_edge_bps,
            pre_trade_signal=deviation.signal.value,
            ttl_sec=self.default_ttl_sec,
            sport=mapping.sport,
            runner_id=runner_id,
        )
