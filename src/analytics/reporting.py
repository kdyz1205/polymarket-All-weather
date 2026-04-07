"""
Sub-module E: Unified reporting — terminal + JSON export.

Combines all 4 analytics sub-modules into one ReplayReport that answers:
  - Did we make money? (SessionSummary)
  - Is execution eating our edge? (ExecutionMetrics)
  - Which events help/hurt? (EventAttribution)
  - Where exactly did we lose? (RiskAttributionReport)

Outputs:
  - Pretty-printed terminal report
  - JSON file for downstream consumption
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from src.analytics.session_metrics import SessionSummary
from src.analytics.execution_quality import ExecutionMetrics
from src.analytics.event_attribution import EventAttribution
from src.analytics.risk_attribution import RiskAttributionReport


@dataclass
class NoTradeDignostics:
    """When trades = 0, explains exactly why."""

    total_signals: int = 0
    total_passed: int = 0
    pass_rate: float = 0.0
    rejection_breakdown: dict[str, int] = field(default_factory=dict)
    binding_constraint: str = ""  # the single most common rejection reason
    strategy_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_signals": self.total_signals,
            "total_passed": self.total_passed,
            "pass_rate": round(self.pass_rate, 4),
            "binding_constraint": self.binding_constraint,
            "rejection_breakdown": dict(
                sorted(self.rejection_breakdown.items(), key=lambda x: -x[1])
            ),
            "strategy_config": self.strategy_config,
        }


@dataclass
class ReplayReport:
    """Complete replay session report combining all analytics."""

    session: SessionSummary = field(default_factory=SessionSummary)
    execution: ExecutionMetrics = field(default_factory=ExecutionMetrics)
    events: EventAttribution = field(default_factory=EventAttribution)
    risk: RiskAttributionReport = field(default_factory=RiskAttributionReport)
    no_trade: NoTradeDignostics | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "session": self.session.to_dict(),
            "execution": self.execution.to_dict(),
            "event_attribution": self.events.to_dict(),
            "risk_attribution": self.risk.to_dict(),
        }
        if self.no_trade is not None:
            d["no_trade_diagnostics"] = self.no_trade.to_dict()
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_json())

    def print_terminal(self) -> None:
        """Print a concise, readable terminal report."""
        s = self.session
        e = self.execution
        r = self.risk

        w = 66
        print()
        print("=" * w)
        print("  REPLAY ANALYTICS REPORT")
        print("=" * w)

        # --- Section 1: Session Summary ---
        print(f"  Session: {s.session_id} | {s.sport} | {s.market_id}")
        print(f"  Duration: {s.duration_sec:.0f}s | Fills: {s.total_fills} | Orders: {s.total_orders}")
        print("-" * w)
        print(f"  {'Gross PnL':<25} {s.gross_pnl:>+10.4f}")
        print(f"  {'Net PnL':<25} {s.net_pnl:>+10.4f}")
        print(f"  {'ROI':<25} {s.roi:>+10.4%}")
        print(f"  {'Turnover':<25} {s.turnover:>10.2f}")
        print(f"  {'Max Drawdown':<25} {s.max_drawdown:>10.4f}  ({s.max_drawdown_pct:.2%})")
        print(f"  {'Win Rate':<25} {s.win_rate:>10.2%}  ({s.win_count}W / {s.loss_count}L)")
        print(f"  {'Profit Factor':<25} {s.profit_factor:>10.2f}")
        print(f"  {'Sharpe Proxy':<25} {s.sharpe_proxy:>10.4f}")

        # --- Section 2: PnL Decomposition ---
        print()
        print("-" * w)
        print("  PnL DECOMPOSITION")
        print("-" * w)
        print(f"  {'Signal Edge':<25} {r.signal_edge:>+10.4f}  (theoretical)")
        print(f"  {'- Slippage':<25} {-r.slippage_cost:>+10.4f}  ({r.slippage_pct:>5.1f}% of edge)")
        print(f"  {'- Delay Damage':<25} {-r.delay_damage:>+10.4f}  ({r.delay_pct:>5.1f}% of edge)")
        print(f"  {'- Bad Fills':<25} {-r.bad_fills:>+10.4f}  ({r.bad_fills_pct:>5.1f}% of edge)")
        print(f"  {'- Hedging Cost':<25} {-r.hedging_cost:>+10.4f}  ({r.hedging_pct:>5.1f}% of edge)")
        print(f"  {'- Fees':<25} {-r.fees:>+10.4f}  ({r.fees_pct:>5.1f}% of edge)")
        print(f"  {'= Net PnL':<25} {r.net_pnl:>+10.4f}  ({r.edge_retained_pct:>5.1f}% retained)")

        # --- Section 3: Execution Quality ---
        print()
        print("-" * w)
        print("  EXECUTION QUALITY")
        print("-" * w)
        print(f"  {'Fill Ratio':<25} {e.fill_ratio:>10.2%}  ({e.filled_orders}/{e.total_orders})")
        print(f"  {'Cancel Ratio':<25} {e.cancel_ratio:>10.2%}")
        print(f"  {'Edge Capture Ratio':<25} {e.edge_capture_ratio:>10.4f}  *** KEY METRIC ***")
        print(f"  {'Avg Latency':<25} {e.avg_quote_to_fill_ms:>10.1f}ms")
        print(f"  {'P95 Latency':<25} {e.p95_quote_to_fill_ms:>10.1f}ms")
        print(f"  {'Avg Slippage':<25} {e.avg_slippage_bps:>10.2f}bps")
        print(f"  {'Adverse Selection':<25} {e.adverse_selection_ratio:>10.2%}")
        print(f"  {'Suspend Hit Rate':<25} {e.suspend_hit_ratio:>10.2%}")

        # --- Section 4: Model Quality ---
        print()
        print("-" * w)
        print("  MODEL QUALITY")
        print("-" * w)
        print(f"  {'Direction Accuracy':<25} {r.model_accuracy:>10.2%}  "
              f"({r.model_correct_direction}✓ / {r.model_wrong_direction}✗)")

        # --- Section 5: Loss Attribution ---
        if r.losses_by_cause:
            print()
            print("-" * w)
            print("  LOSS ATTRIBUTION (by root cause)")
            print("-" * w)
            sorted_causes = sorted(r.losses_by_cause.items(), key=lambda x: -x[1])
            for cause, amount in sorted_causes:
                count = r.loss_count_by_cause.get(cause, 0)
                print(f"  {cause:<25} {-amount:>+10.4f}  ({count} events)")

        # --- Section 6: Worst Losses ---
        if r.worst_losses:
            print()
            print("-" * w)
            print("  TOP WORST LOSSES")
            print("-" * w)
            for i, loss in enumerate(r.worst_losses[:5], 1):
                print(f"  {i}. {-loss.amount:>+8.4f}  [{loss.root_cause}] {loss.detail}")

        # --- Section 7: Event Attribution ---
        ev = self.events
        if ev.by_event_type:
            print()
            print("-" * w)
            print("  EVENT ATTRIBUTION (by type)")
            print("-" * w)
            for etype, stats in sorted(ev.by_event_type.items(),
                                       key=lambda x: abs(x[1].total_pnl_impact), reverse=True):
                led = f"led={stats.model_led_ratio:.0%}" if stats.count > 0 else ""
                conv = f"conv={stats.convergence_ratio:.0%}" if stats.count > 0 else ""
                print(f"  {etype:<20} n={stats.count:>3} | PnL={stats.total_pnl_impact:>+8.4f} "
                      f"| avg_jump={stats.avg_model_prob_jump:>+.4f} | {led} {conv}")

        if ev.by_game_phase:
            print()
            print(f"  {'Game Phase':<12} {'Events':>7} {'PnL Impact':>12} {'Suspends':>10}")
            for phase in ["early", "mid", "late", "clutch"]:
                if phase in ev.by_game_phase:
                    p = ev.by_game_phase[phase]
                    print(f"  {phase:<12} {p['event_count']:>7} {p['total_pnl_impact']:>+12.4f} "
                          f"{p['suspend_count']:>10}")

        # --- Section 8: No-Trade Diagnostics ---
        nt = self.no_trade
        if nt is not None and s.total_fills == 0:
            print()
            print("-" * w)
            print("  NO-TRADE DIAGNOSTICS")
            print("-" * w)
            print(f"  {'Strategy signals':<25} {nt.total_signals:>10}")
            print(f"  {'Passed all gates':<25} {nt.total_passed:>10}")
            print(f"  {'Pass rate':<25} {nt.pass_rate:>10.2%}")
            if nt.binding_constraint:
                print(f"  {'Binding constraint':<25} {nt.binding_constraint}")
            if nt.rejection_breakdown:
                print()
                print(f"  {'Rejection Reason':<35} {'Count':>8} {'%':>7}")
                total_rej = sum(nt.rejection_breakdown.values())
                for reason, count in sorted(nt.rejection_breakdown.items(), key=lambda x: -x[1]):
                    pct = count / total_rej * 100 if total_rej > 0 else 0
                    print(f"  {reason:<35} {count:>8} {pct:>6.1f}%")
        elif nt is not None and s.total_fills > 0:
            # Still show gate pass rate even when we have trades
            print()
            print("-" * w)
            print("  GATE DIAGNOSTICS")
            print("-" * w)
            print(f"  {'Strategy signals':<25} {nt.total_signals:>10}")
            print(f"  {'Passed all gates':<25} {nt.total_passed:>10}")
            print(f"  {'Pass rate':<25} {nt.pass_rate:>10.2%}")
            if nt.binding_constraint:
                print(f"  {'Binding constraint':<25} {nt.binding_constraint}")

        # --- Section 9: Per-Runner PnL ---
        if s.runner_pnls:
            print()
            print("-" * w)
            print("  PER-RUNNER PnL")
            print("-" * w)
            for rid, rp in s.runner_pnls.items():
                print(f"  {rid:<15} PnL={rp.actual_pnl:>+8.4f} | "
                      f"back={rp.back_stake:>7.2f} lay={rp.lay_stake:>7.2f} | "
                      f"fills={rp.fill_count}")

        print()
        print("=" * w)
