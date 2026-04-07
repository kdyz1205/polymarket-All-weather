"""
Main orchestrator — boots the entire All-Weather Multi-Market AI Panel.

This is the entry point that wires everything together:
1. Loads config
2. Starts data feeds for each market
3. Starts MarketProcessors (one per market)
4. Starts the AI Agent (factor generation + decay monitor + analyst)
5. Starts the Risk Manager (global)
6. Starts the Dashboard server

Architecture (top to bottom):
┌──────────────────────────────────────────────────────┐
│                   Dashboard (FastAPI + WS)            │
│    Real-time market view + AI decision log + risk     │
├──────────────────────────────────────────────────────┤
│                AI Agent Layer                         │
│  ┌─────────────┐ ┌──────────────┐ ┌───────────────┐ │
│  │ Factor Gen   │ │ Decay Monitor│ │ AI Analyst    │ │
│  │ (Claude LLM) │ │ (IC tracking)│ │ (Attribution) │ │
│  └─────────────┘ └──────────────┘ └───────────────┘ │
├──────────────────────────────────────────────────────┤
│                Risk Manager                           │
│  Regime detection + Circuit breaker + Drawdown limits │
├──────────────────────────────────────────────────────┤
│         MarketProcessor (one per market)              │
│  ┌─────────────────────────────────────────────────┐ │
│  │ Factor Engine: validate + compute + aggregate    │ │
│  │ Operator Primitives: ts_mean, ts_corr, rank...  │ │
│  └─────────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────┤
│              Event Bus (async pub/sub)                │
├──────────────────────────────────────────────────────┤
│            Data Feeds (per market)                    │
│  Polymarket CLOB / Simulated / Custom                │
└──────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import yaml

from src.ai_agent.decay_monitor import FactorDecayMonitor
from src.core.market_processor import MarketConfig, MarketProcessor
from src.dashboard.server import app, dashboard_state
from src.data.feeds.polymarket_feed import SimulatedFeed
from src.risk.manager import RiskManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


async def run_system() -> None:
    """Boot and run the entire system."""
    config = load_config()
    logger.info("=" * 60)
    logger.info("ALL-WEATHER MULTI-MARKET AI PANEL")
    logger.info("Mode: %s", config["system"]["mode"])
    logger.info("=" * 60)

    # --- 1. Risk Manager (global) ---
    risk_cfg = config["risk"]
    risk_manager = RiskManager(
        max_drawdown_pct=risk_cfg["max_drawdown_pct"],
        per_market_max_drawdown_pct=risk_cfg["per_market_max_drawdown_pct"],
        volatility_circuit_breaker_z=risk_cfg["volatility_circuit_breaker_z"],
        regime_window=risk_cfg["regime_window"],
    )
    await risk_manager.start()

    # --- 2. Factor Decay Monitor ---
    factor_cfg = config["factor_engine"]
    decay_monitor = FactorDecayMonitor(
        min_ic_threshold=factor_cfg["min_ic_threshold"],
        decay_z_threshold=factor_cfg["decay_z_threshold"],
        max_factor_correlation=factor_cfg["max_factor_correlation"],
        ic_window=factor_cfg["ic_window"],
    )
    await decay_monitor.start()

    # --- 3. Dashboard state collector ---
    await dashboard_state.start()

    # --- 4. Per-market setup ---
    feeds = []
    processors = []

    for market_def in config["markets"]:
        if not market_def.get("enabled", True):
            continue

        market_id = market_def["id"]
        market_config = MarketConfig(
            market_id=market_id,
            market_type=market_def["type"],
            name=market_def["name"],
            endpoint=market_def.get("endpoint", ""),
            max_position_usd=market_def.get("max_position_usd", 1000),
            factor_ids=market_def.get("factors", []),
            extra=market_def,
        )

        # Create MarketProcessor
        processor = MarketProcessor(market_config)
        await processor.start()
        processors.append(processor)

        # Create data feed
        # In paper mode, use simulated feed; in live mode, use real API
        if config["system"]["mode"] == "paper":
            feed = SimulatedFeed(
                market_id=market_id,
                initial_price=0.5,
                volatility=0.003,
                tick_interval_ms=config["system"]["tick_interval_ms"],
            )
        else:
            from src.data.feeds.polymarket_feed import PolymarketFeed
            token_ids = market_def.get("token_ids", [])
            feed = PolymarketFeed(
                market_id=market_id,
                endpoint=market_def["endpoint"],
                token_id=token_ids[0] if token_ids else "",
                poll_interval_ms=config["system"]["tick_interval_ms"],
            )

        await feed.start()
        feeds.append(feed)
        logger.info("Market '%s' (%s) initialized", market_def["name"], market_id)

    logger.info("All %d markets initialized", len(processors))

    # --- 5. Start Dashboard server ---
    import uvicorn

    dashboard_cfg = config["dashboard"]
    server_config = uvicorn.Config(
        app,
        host=dashboard_cfg["host"],
        port=dashboard_cfg["port"],
        log_level="warning",
    )
    server = uvicorn.Server(server_config)

    logger.info("Dashboard available at http://localhost:%d", dashboard_cfg["port"])
    logger.info("=" * 60)

    # Run until interrupted
    try:
        await server.serve()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutting down...")
    finally:
        for feed in feeds:
            await feed.stop()
        for proc in processors:
            await proc.stop()
        await risk_manager.stop()
        await decay_monitor.stop()


def main() -> None:
    asyncio.run(run_system())


if __name__ == "__main__":
    main()
