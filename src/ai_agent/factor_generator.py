"""
AI Factor Generator — uses Claude to create new factor expressions,
validate them in sandbox, and promote them to production.

This is the "自动化因子代码生成" module discussed in the architecture.
The full pipeline: Hypothesis -> Code Generation -> Sandbox Backtest ->
Self-Reflection (if failed) -> Grey-scale Deployment -> Full Activation.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

from src.ai_agent.prompts.factor_generation import (
    FACTOR_GENERATION_PROMPT,
    FACTOR_REFLECTION_PROMPT,
    SYSTEM_PROMPT,
)
from src.core.event_bus import Events, bus
from src.core.models import (
    AgentAction,
    AgentDecision,
    FactorDefinition,
    FactorStatus,
    MarketRegime,
    Tick,
)
from src.factors.engine import FactorEngine, FactorSandbox, SandboxResult

logger = logging.getLogger(__name__)


@dataclass
class MarketContext:
    """Snapshot of current market state for the AI to reason about."""

    market_id: str
    market_name: str
    price_trend: str  # "up", "down", "sideways"
    current_spread: float
    book_imbalance: float
    volume_ratio: float
    regime: MarketRegime
    active_factor_count: int
    recent_ics: dict[str, float]
    existing_expressions: dict[str, str]


class AIFactorGenerator:
    """
    Orchestrates the full AI factor generation pipeline.

    Lifecycle of a factor:
    1. AI generates hypothesis + expression
    2. Expression is validated (only allowed operators)
    3. Expression is backtested in sandbox
    4. If failed: AI reflects and retries (up to max_retries)
    5. If passed: factor is deployed with grey-scale weight
    6. After proving itself in live: promoted to full active
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6",
        allowed_operators: list[str] | None = None,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.allowed_operators = allowed_operators or list(
            __import__(
                "src.factors.operators.primitives", fromlist=["OPERATOR_REGISTRY"]
            ).OPERATOR_REGISTRY.keys()
        )
        self.max_retries = max_retries
        self._generation_count = 0

    async def generate_factors(
        self,
        context: MarketContext,
        historical_ticks: list[Tick],
        n_factors: int = 3,
    ) -> list[FactorDefinition]:
        """
        Full pipeline: generate -> validate -> sandbox -> reflect -> deploy.
        Returns list of factors that passed all gates.
        """
        logger.info(
            "[%s] Starting AI factor generation (requesting %d factors)",
            context.market_id,
            n_factors,
        )

        # Step 1: Ask AI to generate factor hypotheses
        raw_factors = await self._call_llm_for_factors(context, n_factors)

        approved: list[FactorDefinition] = []
        sandbox = FactorSandbox(historical_ticks)

        for raw in raw_factors:
            name = raw.get("name", f"ai_factor_{self._generation_count}")
            description = raw.get("description", "AI-generated factor")
            expression = raw.get("expression", "")

            # Step 2: Validate expression safety
            if not FactorEngine.validate_expression(expression):
                logger.warning("[%s] Expression failed validation: %s", context.market_id, expression)
                # Log the decision
                await bus.publish(
                    Events.AGENT_DECISION,
                    AgentDecision(
                        action=AgentAction.GENERATE_FACTOR,
                        market_id=context.market_id,
                        timestamp_ms=int(time.time() * 1000),
                        reasoning=f"Rejected factor '{name}': expression uses disallowed operators",
                        details={"expression": expression, "status": "rejected_validation"},
                    ),
                )
                continue

            # Step 3: Sandbox backtest
            result = sandbox.backtest(expression)

            # Step 4: Self-reflection loop if failed
            retries = 0
            while not result.passes_threshold and retries < self.max_retries:
                logger.info(
                    "[%s] Factor '%s' failed sandbox (%s), reflecting (attempt %d/%d)",
                    context.market_id,
                    name,
                    result.summary(),
                    retries + 1,
                    self.max_retries,
                )
                improved = await self._call_llm_for_reflection(
                    name, expression, result.summary()
                )
                if improved and improved.get("expression"):
                    expression = improved["expression"]
                    name = improved.get("name", name)
                    description = improved.get("description", description)
                    if FactorEngine.validate_expression(expression):
                        result = sandbox.backtest(expression)
                retries += 1

            # Step 5: Deploy or reject
            if result.passes_threshold:
                self._generation_count += 1
                factor_id = f"ai_{context.market_id}_{self._generation_count}"
                factor_def = FactorDefinition(
                    factor_id=factor_id,
                    name=name,
                    description=description,
                    expression=expression,
                    author="ai_agent",
                    status=FactorStatus.GREY,  # start in grey-scale
                    initial_weight=0.01,
                    metadata={
                        "ic": result.ic,
                        "ir": result.ir,
                        "sandbox_n": result.n_observations,
                    },
                )
                approved.append(factor_def)

                # Publish the creation event
                await bus.publish(Events.FACTOR_CREATED, factor_def)
                await bus.publish(
                    Events.AGENT_DECISION,
                    AgentDecision(
                        action=AgentAction.GENERATE_FACTOR,
                        market_id=context.market_id,
                        timestamp_ms=int(time.time() * 1000),
                        reasoning=(
                            f"Deployed new factor '{name}' in grey-scale mode. "
                            f"Sandbox IC={result.ic:.4f}, IR={result.ir:.4f}. "
                            f"Intuition: {description}"
                        ),
                        details={
                            "factor_id": factor_id,
                            "expression": expression,
                            "sandbox": result.summary(),
                        },
                    ),
                )
                logger.info(
                    "[%s] Factor '%s' APPROVED: %s",
                    context.market_id,
                    name,
                    result.summary(),
                )
            else:
                await bus.publish(
                    Events.AGENT_DECISION,
                    AgentDecision(
                        action=AgentAction.GENERATE_FACTOR,
                        market_id=context.market_id,
                        timestamp_ms=int(time.time() * 1000),
                        reasoning=(
                            f"Factor '{name}' rejected after {self.max_retries} reflection "
                            f"attempts. Final result: {result.summary()}"
                        ),
                        details={"expression": expression},
                    ),
                )
                logger.info(
                    "[%s] Factor '%s' REJECTED after reflection: %s",
                    context.market_id,
                    name,
                    result.summary(),
                )

        return approved

    # --------------------------------------------------
    # LLM Interaction
    # --------------------------------------------------

    async def _call_llm_for_factors(
        self, context: MarketContext, n_factors: int
    ) -> list[dict]:
        """Call Claude to generate factor hypotheses."""
        import anthropic

        system = SYSTEM_PROMPT.format(
            allowed_operators=", ".join(self.allowed_operators)
        )
        user_msg = FACTOR_GENERATION_PROMPT.format(
            market_name=context.market_name,
            market_id=context.market_id,
            price_trend=context.price_trend,
            current_spread=context.current_spread,
            book_imbalance=context.book_imbalance,
            volume_ratio=context.volume_ratio,
            regime=context.regime.value,
            active_factor_count=context.active_factor_count,
            recent_ics=json.dumps(context.recent_ics),
            existing_factors=json.dumps(context.existing_expressions, indent=2),
            n_factors=n_factors,
        )

        try:
            client = anthropic.AsyncAnthropic(api_key=self.api_key)
            response = await client.messages.create(
                model=self.model,
                max_tokens=2000,
                system=system,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = response.content[0].text
            # Parse JSON from response
            data = json.loads(text)
            return data.get("factors", [])
        except Exception as e:
            logger.error("LLM factor generation failed: %s", e)
            return []

    async def _call_llm_for_reflection(
        self, factor_name: str, expression: str, sandbox_result: str
    ) -> dict | None:
        """Ask Claude to reflect on a failed factor and improve it."""
        import anthropic

        system = SYSTEM_PROMPT.format(
            allowed_operators=", ".join(self.allowed_operators)
        )
        user_msg = FACTOR_REFLECTION_PROMPT.format(
            factor_name=factor_name,
            expression=expression,
            sandbox_result=sandbox_result,
        )

        try:
            client = anthropic.AsyncAnthropic(api_key=self.api_key)
            response = await client.messages.create(
                model=self.model,
                max_tokens=1000,
                system=system,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = response.content[0].text
            data = json.loads(text)
            factors = data.get("factors", [])
            return factors[0] if factors else None
        except Exception as e:
            logger.error("LLM reflection failed: %s", e)
            return None
