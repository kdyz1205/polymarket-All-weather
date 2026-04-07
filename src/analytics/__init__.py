"""
Analytics package — replay session measurement and attribution.

Sub-modules:
  A. session_metrics  — ROI, PnL, drawdown, win rate, Sharpe proxy
  B. execution_quality — fill ratio, edge capture, slippage, adverse selection
  C. event_attribution — per-event-type PnL impact, model quality, suspend stats
  D. risk_attribution  — PnL decomposition, root cause tagging, worst losses
  E. reporting         — unified terminal + JSON report
"""

from src.analytics.session_metrics import (
    FillRecord,
    MarketOutcome,
    RunnerPnL,
    SessionSummary,
    SessionAggregator,
)
from src.analytics.execution_quality import (
    OrderRecord,
    ExecutionMetrics,
    ExecutionAnalyzer,
)
from src.analytics.event_attribution import (
    EventSnapshot,
    EventTypeStats,
    EventAttribution,
    EventAttributionAnalyzer,
)
from src.analytics.risk_attribution import (
    PnLComponent,
    LossAttribution,
    RiskAttributionReport,
    RiskAttributionAnalyzer,
)
from src.analytics.reporting import ReplayReport, NoTradeDignostics

__all__ = [
    "FillRecord",
    "MarketOutcome",
    "RunnerPnL",
    "SessionSummary",
    "SessionAggregator",
    "OrderRecord",
    "ExecutionMetrics",
    "ExecutionAnalyzer",
    "EventSnapshot",
    "EventTypeStats",
    "EventAttribution",
    "EventAttributionAnalyzer",
    "PnLComponent",
    "LossAttribution",
    "RiskAttributionReport",
    "RiskAttributionAnalyzer",
    "ReplayReport",
    "NoTradeDignostics",
]
