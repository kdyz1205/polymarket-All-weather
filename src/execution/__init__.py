"""
Execution bridge — from internal signals to real Polymarket orders.

Modules:
  market_mapper     Map (sport, teams, market_type) → real market/token IDs
  quote_fetcher     Fetch real-time bid/ask/depth from Polymarket CLOB
  pre_trade_check   Price deviation gating (green / yellow / red)
  order_builder     Construct standardized order objects
  manual_executor   End-to-end signal → compare → confirm → queue flow
  auto_executor     Full auto: CLOB order submission + portfolio management
"""

from src.execution.market_mapper import MarketMapper, MarketMapping
from src.execution.quote_fetcher import LiveQuoteFetcher, LiveQuote
from src.execution.pre_trade_check import PreTradeComparator, DeviationResult, Signal
from src.execution.order_builder import OrderBuilder, ExecutionOrder
from src.execution.manual_executor import ManualExecutor, ExecutionDecision
from src.execution.auto_executor import AutoExecutor, TradeRecord, PortfolioState

__all__ = [
    "MarketMapper", "MarketMapping",
    "LiveQuoteFetcher", "LiveQuote",
    "PreTradeComparator", "DeviationResult", "Signal",
    "OrderBuilder", "ExecutionOrder",
    "ManualExecutor", "ExecutionDecision",
    "AutoExecutor", "TradeRecord", "PortfolioState",
]
