"""
Central event bus for inter-module communication.

All layers communicate through typed events on this bus, enabling loose coupling.
Data Layer publishes Ticks -> Factor Engine consumes Ticks, publishes FactorValues ->
AI Agent consumes FactorValues, publishes Decisions -> etc.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)

# Event handler type: async function that takes event data
EventHandler = Callable[[Any], Coroutine[Any, Any, None]]


class EventBus:
    """Async publish-subscribe event bus for the entire system."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[EventHandler]] = defaultdict(list)
        self._history: dict[str, list[Any]] = defaultdict(list)
        self._history_limit = 1000

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        self._subscribers[event_type].append(handler)
        logger.debug("Subscribed %s to event '%s'", handler.__qualname__, event_type)

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        self._subscribers[event_type] = [
            h for h in self._subscribers[event_type] if h is not handler
        ]

    async def publish(self, event_type: str, data: Any) -> None:
        # Store in history ring buffer
        history = self._history[event_type]
        history.append(data)
        if len(history) > self._history_limit:
            self._history[event_type] = history[-self._history_limit :]

        handlers = self._subscribers.get(event_type, [])
        if not handlers:
            return

        # Fire all handlers concurrently
        await asyncio.gather(
            *(self._safe_call(h, data) for h in handlers),
            return_exceptions=True,
        )

    async def _safe_call(self, handler: EventHandler, data: Any) -> None:
        try:
            await handler(data)
        except Exception:
            logger.exception("Error in event handler %s", handler.__qualname__)

    def get_history(self, event_type: str, limit: int = 100) -> list[Any]:
        return self._history.get(event_type, [])[-limit:]


# Singleton event bus for the application
bus = EventBus()


# Well-known event types
class Events:
    TICK = "tick"  # Tick
    ORDER_BOOK = "order_book"  # OrderBookSnapshot
    TRADE = "trade"  # Trade
    FACTOR_VALUE = "factor_value"  # FactorValue
    FACTOR_CREATED = "factor_created"  # FactorDefinition
    FACTOR_RETIRED = "factor_retired"  # FactorDefinition
    COMPOSITE_SIGNAL = "composite_signal"  # CompositeSignal
    AGENT_DECISION = "agent_decision"  # AgentDecision
    REGIME_CHANGE = "regime_change"  # RegimeState
    RISK_UPDATE = "risk_update"  # RiskSnapshot
    CIRCUIT_BREAK = "circuit_break"  # str (market_id)
    CIRCUIT_RESUME = "circuit_resume"  # str (market_id)
