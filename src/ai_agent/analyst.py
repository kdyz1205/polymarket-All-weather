"""
AI Analyst — provides real-time attribution analysis and market insights.

This is the "元认知与可解释性" module: AI reads factor performance,
market data, and PnL, then generates human-readable analysis explaining
what happened and why.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

from src.ai_agent.prompts.factor_generation import ATTRIBUTION_PROMPT
from src.core.event_bus import Events, bus
from src.core.models import (
    AgentAction,
    AgentDecision,
    FactorValue,
    MarketRegime,
    Tick,
)

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    market_id: str
    timestamp_ms: int
    analysis_text: str
    factor_attributions: dict[str, float]
    recommended_actions: list[str]


class AIAnalyst:
    """
    Periodically analyzes market performance and provides
    natural language attribution reports.
    """

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6") -> None:
        self.api_key = api_key
        self.model = model

    async def analyze_period(
        self,
        market_id: str,
        market_name: str,
        ticks: list[Tick],
        factor_values: dict[str, list[FactorValue]],
        pnl: float,
        regime: MarketRegime,
    ) -> AnalysisResult:
        """Generate attribution analysis for a recent period."""
        if len(ticks) < 10:
            return AnalysisResult(
                market_id=market_id,
                timestamp_ms=int(time.time() * 1000),
                analysis_text="Insufficient data for analysis.",
                factor_attributions={},
                recommended_actions=[],
            )

        # Compute market metrics
        first_price = ticks[0].mid_price
        last_price = ticks[-1].mid_price
        price_change_pct = ((last_price - first_price) / first_price * 100) if first_price > 0 else 0

        first_vol = ticks[0].volume_24h
        last_vol = ticks[-1].volume_24h
        vol_change_pct = ((last_vol - first_vol) / first_vol * 100) if first_vol > 0 else 0

        first_spread = ticks[0].spread
        last_spread = ticks[-1].spread
        spread_change_pct = ((last_spread - first_spread) / first_spread * 100) if first_spread > 0 else 0

        # Summarize factor performance
        factor_perf_lines = []
        attributions: dict[str, float] = {}
        for fid, values in factor_values.items():
            if not values:
                continue
            avg_value = sum(v.value for v in values) / len(values)
            avg_weight = sum(v.weight for v in values) / len(values)
            contribution = avg_value * avg_weight
            attributions[fid] = contribution
            factor_perf_lines.append(
                f"  - {fid}: avg_value={avg_value:.4f}, weight={avg_weight:.4f}, "
                f"contribution={contribution:.4f}"
            )

        factor_performance_text = "\n".join(factor_perf_lines) or "  No active factors."

        # Call LLM for analysis
        analysis_text = await self._call_llm_for_analysis(
            market_name=market_name,
            window=len(ticks),
            price_change_pct=price_change_pct,
            volume_change_pct=vol_change_pct,
            spread_change_pct=spread_change_pct,
            regime=regime,
            factor_performance=factor_performance_text,
            pnl=pnl,
        )

        result = AnalysisResult(
            market_id=market_id,
            timestamp_ms=int(time.time() * 1000),
            analysis_text=analysis_text,
            factor_attributions=attributions,
            recommended_actions=[],
        )

        # Publish as agent decision for the dashboard
        await bus.publish(
            Events.AGENT_DECISION,
            AgentDecision(
                action=AgentAction.LOG_INSIGHT,
                market_id=market_id,
                timestamp_ms=result.timestamp_ms,
                reasoning=analysis_text,
                details={"attributions": attributions, "pnl": pnl},
            ),
        )

        return result

    async def _call_llm_for_analysis(
        self,
        market_name: str,
        window: int,
        price_change_pct: float,
        volume_change_pct: float,
        spread_change_pct: float,
        regime: MarketRegime,
        factor_performance: str,
        pnl: float,
    ) -> str:
        import anthropic

        user_msg = ATTRIBUTION_PROMPT.format(
            market_name=market_name,
            window=window,
            price_change_pct=price_change_pct,
            volume_change_pct=volume_change_pct,
            spread_change_pct=spread_change_pct,
            regime=regime.value,
            factor_performance=factor_performance,
            pnl=pnl,
        )

        try:
            client = anthropic.AsyncAnthropic(api_key=self.api_key)
            response = await client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[{"role": "user", "content": user_msg}],
            )
            return response.content[0].text
        except Exception as e:
            logger.error("LLM analysis failed: %s", e)
            return f"Analysis unavailable: {e}"
