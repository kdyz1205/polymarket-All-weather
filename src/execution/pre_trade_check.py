"""
Pre-trade comparator — gates execution based on price deviation
between the model's signal price and the real market price.

Traffic-light system:
  GREEN  (<3% deviation)  → auto-queue for confirmation
  YELLOW (3-5%)           → queue with warning, human must confirm
  RED    (>5%)            → rejected, do not execute

All thresholds are configurable but ship with conservative defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.execution.quote_fetcher import LiveQuote


class Signal(Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


# Default thresholds (deviation percentage)
GREEN_MAX_PCT = 3.0
YELLOW_MAX_PCT = 5.0


@dataclass
class DeviationResult:
    """Result of comparing system price to market price."""
    signal: Signal
    system_price: float         # model's fair decimal odds
    market_price: float         # real market decimal odds (from mid)
    deviation_pct: float        # |system - market| / market * 100
    system_implied_prob: float  # 1/system_price
    market_implied_prob: float  # mid_price from Polymarket (0-1 scale)
    spread_bps: float           # market spread in basis points
    bid_depth: float
    ask_depth: float
    reason: str
    details: dict[str, Any] | None = None

    @property
    def is_executable(self) -> bool:
        return self.signal in (Signal.GREEN, Signal.YELLOW)

    @property
    def needs_human_review(self) -> bool:
        return self.signal == Signal.YELLOW


class PreTradeComparator:
    """Compares system signal price to real market price and gates execution."""

    def __init__(
        self,
        green_max_pct: float = GREEN_MAX_PCT,
        yellow_max_pct: float = YELLOW_MAX_PCT,
        min_depth: float = 50.0,         # minimum depth to consider valid
        max_spread_pct: float = 5.0,     # max spread as % of mid
    ):
        self.green_max_pct = green_max_pct
        self.yellow_max_pct = yellow_max_pct
        self.min_depth = min_depth
        self.max_spread_pct = max_spread_pct

    def compare(
        self,
        system_decimal_odds: float,
        quote: LiveQuote,
    ) -> DeviationResult:
        """Compare system's fair odds to real market quote.

        Args:
            system_decimal_odds: Model's fair decimal odds (e.g., 1.80)
            quote: Real-time market quote from LiveQuoteFetcher
        """
        # Handle fetch failures
        if not quote.is_valid:
            return DeviationResult(
                signal=Signal.RED,
                system_price=system_decimal_odds,
                market_price=0.0,
                deviation_pct=999.0,
                system_implied_prob=1.0 / system_decimal_odds if system_decimal_odds > 0 else 0,
                market_implied_prob=0.0,
                spread_bps=0.0,
                bid_depth=0.0,
                ask_depth=0.0,
                reason=f"invalid quote: {quote.error}",
            )

        market_decimal_odds = quote.decimal_odds
        system_implied = 1.0 / system_decimal_odds if system_decimal_odds > 0 else 0

        # Price deviation
        if market_decimal_odds > 0:
            deviation_pct = abs(system_decimal_odds - market_decimal_odds) / market_decimal_odds * 100
        else:
            deviation_pct = 999.0

        # Spread in bps
        spread_bps = (quote.spread / quote.mid_price * 10000) if quote.mid_price > 0 else 0

        # Determine signal
        if deviation_pct > self.yellow_max_pct:
            signal = Signal.RED
            reason = f"deviation {deviation_pct:.1f}% > {self.yellow_max_pct}% threshold"
        elif quote.bid_depth < self.min_depth or quote.ask_depth < self.min_depth:
            signal = Signal.RED
            reason = f"insufficient depth (bid={quote.bid_depth:.0f}, ask={quote.ask_depth:.0f})"
        elif quote.spread / quote.mid_price * 100 > self.max_spread_pct if quote.mid_price > 0 else True:
            signal = Signal.RED
            reason = f"spread too wide ({spread_bps:.0f}bps)"
        elif deviation_pct > self.green_max_pct:
            signal = Signal.YELLOW
            reason = f"deviation {deviation_pct:.1f}% — within yellow zone, needs human review"
        else:
            signal = Signal.GREEN
            reason = f"deviation {deviation_pct:.1f}% — within green zone"

        return DeviationResult(
            signal=signal,
            system_price=system_decimal_odds,
            market_price=market_decimal_odds,
            deviation_pct=deviation_pct,
            system_implied_prob=system_implied,
            market_implied_prob=quote.implied_prob,
            spread_bps=spread_bps,
            bid_depth=quote.bid_depth,
            ask_depth=quote.ask_depth,
            reason=reason,
        )
