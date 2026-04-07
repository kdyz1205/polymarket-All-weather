"""
Real-time Dashboard Server — serves the monitoring panel via WebSocket + REST.

The dashboard provides:
1. Real-time market prices and signals
2. Factor weights and IC values (live updating)
3. AI decision log (natural language)
4. Risk state with regime indicators (traffic lights)
5. Attribution analysis from AI Analyst
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from src.core.event_bus import Events, bus
from src.core.models import (
    AgentDecision,
    CompositeSignal,
    FactorValue,
    RegimeState,
    Tick,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="All-Weather Multi-Market AI Panel")


class DashboardState:
    """Aggregates all system state for the dashboard."""

    def __init__(self) -> None:
        self._latest_ticks: dict[str, Tick] = {}
        self._latest_signals: dict[str, CompositeSignal] = {}
        self._latest_factors: dict[str, dict[str, FactorValue]] = defaultdict(dict)
        self._latest_regimes: dict[str, RegimeState] = {}
        self._agent_log: list[AgentDecision] = []
        self._agent_log_limit = 200
        self._ws_clients: list[WebSocket] = []

    async def start(self) -> None:
        bus.subscribe(Events.TICK, self._on_tick)
        bus.subscribe(Events.COMPOSITE_SIGNAL, self._on_signal)
        bus.subscribe(Events.FACTOR_VALUE, self._on_factor)
        bus.subscribe(Events.REGIME_CHANGE, self._on_regime)
        bus.subscribe(Events.AGENT_DECISION, self._on_agent_decision)
        logger.info("Dashboard state collector started")

    async def _on_tick(self, tick: Tick) -> None:
        self._latest_ticks[tick.market_id] = tick
        await self._broadcast("tick", {
            "market_id": tick.market_id,
            "mid_price": tick.mid_price,
            "spread": tick.spread,
            "book_imbalance": tick.book_imbalance,
            "volume_24h": tick.volume_24h,
            "timestamp_ms": tick.timestamp_ms,
        })

    async def _on_signal(self, signal: CompositeSignal) -> None:
        self._latest_signals[signal.market_id] = signal
        await self._broadcast("signal", {
            "market_id": signal.market_id,
            "normalized_score": signal.normalized_score,
            "confidence": signal.confidence,
            "active_factor_count": signal.active_factor_count,
            "contributions": signal.factor_contributions,
        })

    async def _on_factor(self, fv: FactorValue) -> None:
        self._latest_factors[fv.market_id][fv.factor_id] = fv
        await self._broadcast("factor", {
            "market_id": fv.market_id,
            "factor_id": fv.factor_id,
            "value": fv.value,
            "weight": fv.weight,
            "ic_rolling": fv.ic_rolling,
            "status": fv.status.value,
        })

    async def _on_regime(self, state: RegimeState) -> None:
        self._latest_regimes[state.market_id] = state
        await self._broadcast("regime", {
            "market_id": state.market_id,
            "regime": state.regime.value,
            "confidence": state.confidence,
            "volatility_z": state.volatility_z,
            "liquidity_z": state.liquidity_z,
        })

    async def _on_agent_decision(self, decision: AgentDecision) -> None:
        self._agent_log.append(decision)
        if len(self._agent_log) > self._agent_log_limit:
            self._agent_log = self._agent_log[-self._agent_log_limit :]
        await self._broadcast("agent_decision", {
            "action": decision.action.value,
            "market_id": decision.market_id,
            "reasoning": decision.reasoning,
            "timestamp_ms": decision.timestamp_ms,
            "details": decision.details,
        })

    async def _broadcast(self, event_type: str, data: dict) -> None:
        message = json.dumps({"type": event_type, "data": data})
        disconnected = []
        for ws in self._ws_clients:
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self._ws_clients.remove(ws)

    def add_client(self, ws: WebSocket) -> None:
        self._ws_clients.append(ws)

    def remove_client(self, ws: WebSocket) -> None:
        if ws in self._ws_clients:
            self._ws_clients.remove(ws)

    def get_snapshot(self) -> dict[str, Any]:
        """Full system snapshot for initial page load."""
        return {
            "markets": {
                mid: {
                    "tick": {
                        "mid_price": t.mid_price,
                        "spread": t.spread,
                        "book_imbalance": t.book_imbalance,
                        "volume_24h": t.volume_24h,
                    } if (t := self._latest_ticks.get(mid)) else None,
                    "signal": {
                        "normalized_score": s.normalized_score,
                        "confidence": s.confidence,
                        "active_factor_count": s.active_factor_count,
                    } if (s := self._latest_signals.get(mid)) else None,
                    "regime": {
                        "regime": r.regime.value,
                        "volatility_z": r.volatility_z,
                        "liquidity_z": r.liquidity_z,
                    } if (r := self._latest_regimes.get(mid)) else None,
                    "factors": {
                        fid: {
                            "value": fv.value,
                            "weight": fv.weight,
                            "ic_rolling": fv.ic_rolling,
                            "status": fv.status.value,
                        }
                        for fid, fv in self._latest_factors.get(mid, {}).items()
                    },
                }
                for mid in set(
                    list(self._latest_ticks.keys())
                    + list(self._latest_signals.keys())
                    + list(self._latest_regimes.keys())
                )
            },
            "agent_log": [
                {
                    "action": d.action.value,
                    "market_id": d.market_id,
                    "reasoning": d.reasoning,
                    "timestamp_ms": d.timestamp_ms,
                }
                for d in self._agent_log[-50:]
            ],
        }


# Global dashboard state
dashboard_state = DashboardState()


# ============================================================
# REST Endpoints
# ============================================================


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content=DASHBOARD_HTML)


@app.get("/api/snapshot")
async def api_snapshot() -> dict:
    return dashboard_state.get_snapshot()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    dashboard_state.add_client(websocket)
    logger.info("WebSocket client connected")
    try:
        # Send initial snapshot
        await websocket.send_text(json.dumps({
            "type": "snapshot",
            "data": dashboard_state.get_snapshot(),
        }))
        # Keep connection alive
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        dashboard_state.remove_client(websocket)
        logger.info("WebSocket client disconnected")


# ============================================================
# Dashboard HTML (single-page app)
# ============================================================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>All-Weather Multi-Market AI Panel</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Courier New', monospace;
            background: #0a0a0a;
            color: #e0e0e0;
            padding: 20px;
        }
        h1 { color: #00ff88; margin-bottom: 20px; font-size: 1.4em; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 16px; }
        .card {
            background: #1a1a2e;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 16px;
        }
        .card h2 { color: #00ccff; font-size: 1em; margin-bottom: 12px; }
        .market-header { display: flex; justify-content: space-between; align-items: center; }
        .regime-light {
            width: 12px; height: 12px; border-radius: 50%;
            display: inline-block; margin-left: 8px;
        }
        .regime-normal { background: #00ff88; }
        .regime-high_volatility { background: #ffaa00; }
        .regime-low_liquidity { background: #ffaa00; }
        .regime-trending { background: #00ccff; }
        .regime-mean_reverting { background: #aa88ff; }
        .regime-crisis { background: #ff3333; animation: blink 0.5s infinite; }
        @keyframes blink { 50% { opacity: 0.3; } }
        .metric { display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #222; }
        .metric-label { color: #888; }
        .metric-value { color: #fff; font-weight: bold; }
        .signal-positive { color: #00ff88; }
        .signal-negative { color: #ff4444; }
        .signal-neutral { color: #888; }
        .factor-bar {
            height: 8px; background: #333; border-radius: 4px; margin: 4px 0;
            position: relative; overflow: hidden;
        }
        .factor-fill { height: 100%; border-radius: 4px; transition: width 0.3s; }
        .agent-log {
            max-height: 400px; overflow-y: auto;
            background: #0d0d1a; padding: 12px; border-radius: 4px;
            font-size: 0.85em; line-height: 1.6;
        }
        .log-entry { padding: 6px 0; border-bottom: 1px solid #1a1a2e; }
        .log-time { color: #666; }
        .log-action { color: #00ccff; font-weight: bold; }
        .log-reasoning { color: #ccc; }
        .circuit-broken { border-color: #ff3333 !important; }
        #connection-status {
            position: fixed; top: 10px; right: 10px;
            padding: 4px 12px; border-radius: 12px; font-size: 0.8em;
        }
        .connected { background: #00ff88; color: #000; }
        .disconnected { background: #ff3333; color: #fff; }
    </style>
</head>
<body>
    <div id="connection-status" class="disconnected">Disconnected</div>
    <h1>ALL-WEATHER MULTI-MARKET AI PANEL</h1>
    <div class="grid" id="markets-grid"></div>
    <div class="card" style="margin-top: 16px;">
        <h2>AI Agent Decision Log</h2>
        <div class="agent-log" id="agent-log"></div>
    </div>

    <script>
        const state = { markets: {}, agentLog: [] };

        function connectWS() {
            const ws = new WebSocket(`ws://${location.hostname}:${location.port}/ws`);
            const statusEl = document.getElementById('connection-status');

            ws.onopen = () => {
                statusEl.textContent = 'Connected';
                statusEl.className = 'connected';
            };
            ws.onclose = () => {
                statusEl.textContent = 'Disconnected';
                statusEl.className = 'disconnected';
                setTimeout(connectWS, 2000);
            };
            ws.onmessage = (e) => {
                const msg = JSON.parse(e.data);
                handleMessage(msg);
            };
        }

        function handleMessage(msg) {
            switch (msg.type) {
                case 'snapshot':
                    Object.assign(state, msg.data);
                    state.agentLog = msg.data.agent_log || [];
                    break;
                case 'tick':
                    if (!state.markets[msg.data.market_id]) state.markets[msg.data.market_id] = {};
                    state.markets[msg.data.market_id].tick = msg.data;
                    break;
                case 'signal':
                    if (!state.markets[msg.data.market_id]) state.markets[msg.data.market_id] = {};
                    state.markets[msg.data.market_id].signal = msg.data;
                    break;
                case 'regime':
                    if (!state.markets[msg.data.market_id]) state.markets[msg.data.market_id] = {};
                    state.markets[msg.data.market_id].regime = msg.data;
                    break;
                case 'factor':
                    if (!state.markets[msg.data.market_id]) state.markets[msg.data.market_id] = {};
                    if (!state.markets[msg.data.market_id].factors) state.markets[msg.data.market_id].factors = {};
                    state.markets[msg.data.market_id].factors[msg.data.factor_id] = msg.data;
                    break;
                case 'agent_decision':
                    state.agentLog.push(msg.data);
                    if (state.agentLog.length > 100) state.agentLog = state.agentLog.slice(-100);
                    break;
            }
            render();
        }

        function render() {
            const grid = document.getElementById('markets-grid');
            grid.innerHTML = '';
            for (const [mid, mdata] of Object.entries(state.markets)) {
                const tick = mdata.tick || {};
                const signal = mdata.signal || {};
                const regime = mdata.regime || { regime: 'normal' };
                const factors = mdata.factors || {};
                const isBroken = regime.regime === 'crisis';

                const signalClass = (signal.normalized_score || 0) > 0.1 ? 'signal-positive'
                    : (signal.normalized_score || 0) < -0.1 ? 'signal-negative' : 'signal-neutral';

                let factorsHtml = '';
                for (const [fid, fdata] of Object.entries(factors)) {
                    const pct = Math.abs(fdata.value || 0) * 100;
                    const color = (fdata.value || 0) > 0 ? '#00ff88' : '#ff4444';
                    factorsHtml += `
                        <div class="metric">
                            <span class="metric-label">${fid}</span>
                            <span class="metric-value" style="color:${color}">${(fdata.value||0).toFixed(4)} (w:${(fdata.weight||0).toFixed(3)})</span>
                        </div>
                        <div class="factor-bar"><div class="factor-fill" style="width:${pct}%;background:${color}"></div></div>
                    `;
                }

                grid.innerHTML += `
                    <div class="card ${isBroken ? 'circuit-broken' : ''}">
                        <div class="market-header">
                            <h2>${mid}</h2>
                            <span class="regime-light regime-${regime.regime}" title="${regime.regime}"></span>
                        </div>
                        <div class="metric"><span class="metric-label">Price</span><span class="metric-value">${(tick.mid_price||0).toFixed(4)}</span></div>
                        <div class="metric"><span class="metric-label">Spread</span><span class="metric-value">${(tick.spread||0).toFixed(6)}</span></div>
                        <div class="metric"><span class="metric-label">Imbalance</span><span class="metric-value">${(tick.book_imbalance||0).toFixed(4)}</span></div>
                        <div class="metric"><span class="metric-label">Signal</span><span class="metric-value ${signalClass}">${(signal.normalized_score||0).toFixed(4)}</span></div>
                        <div class="metric"><span class="metric-label">Confidence</span><span class="metric-value">${((signal.confidence||0)*100).toFixed(1)}%</span></div>
                        <div class="metric"><span class="metric-label">Regime</span><span class="metric-value">${regime.regime} (vol_z:${(regime.volatility_z||0).toFixed(2)})</span></div>
                        <h2 style="margin-top:12px">Active Factors (${signal.active_factor_count||0})</h2>
                        ${factorsHtml || '<div style="color:#666">No active factors</div>'}
                    </div>
                `;
            }

            // Render agent log
            const logEl = document.getElementById('agent-log');
            logEl.innerHTML = state.agentLog.slice().reverse().map(d => `
                <div class="log-entry">
                    <span class="log-time">${new Date(d.timestamp_ms).toLocaleTimeString()}</span>
                    <span class="log-action">[${d.action}]</span>
                    <span style="color:#888">${d.market_id}</span><br>
                    <span class="log-reasoning">${d.reasoning}</span>
                </div>
            `).join('');
        }

        connectWS();
    </script>
</body>
</html>
"""
