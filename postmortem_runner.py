"""
Postmortem runner — daily attribution and feature decay analysis.

Reads paper trading reports and produces:
  1. Alpha report:    which signals actually worked
  2. Execution report: how much edge was lost to fees, delay, slippage
  3. Risk report:      model errors vs execution errors
  4. Feature report:   which features are decaying over time

Usage:
  source .venv/bin/activate && python postmortem_runner.py [reports_dir]
"""

from __future__ import annotations

import glob
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class AlphaReport:
    """Which signals actually worked."""
    sport: str
    period: str
    total_signals: int = 0
    profitable_signals: int = 0
    signal_accuracy: float = 0.0
    mean_winning_edge_bps: float = 0.0
    mean_losing_edge_bps: float = 0.0
    best_quarter_or_inning: str = ""
    worst_quarter_or_inning: str = ""


@dataclass
class ExecutionReport:
    """How much edge was consumed by costs."""
    sport: str
    total_gross_edge: float = 0.0
    total_fees_estimated: float = 0.0
    total_delay_cost: float = 0.0
    total_net_pnl: float = 0.0
    fee_drag_pct: float = 0.0          # fees / gross edge
    delay_drag_pct: float = 0.0
    edge_retained_pct: float = 0.0


@dataclass
class RiskReport:
    """Model errors vs execution errors."""
    sport: str
    total_games: int = 0
    direction_correct: int = 0
    direction_accuracy: float = 0.0
    model_error_games: int = 0         # predicted wrong winner
    execution_error_games: int = 0     # predicted right but lost money
    mean_pnl_when_correct: float = 0.0
    mean_pnl_when_wrong: float = 0.0


@dataclass
class FeatureDecayReport:
    """Track whether features are decaying over time."""
    sport: str
    period: str
    metrics_by_window: dict[str, dict] = field(default_factory=dict)
    decaying_features: list[str] = field(default_factory=list)
    stable_features: list[str] = field(default_factory=list)


@dataclass
class PostmortemReport:
    """Full postmortem combining all sub-reports."""
    timestamp: str = ""
    alpha: AlphaReport | None = None
    execution: ExecutionReport | None = None
    risk: RiskReport | None = None
    feature_decay: FeatureDecayReport | None = None

    def print_summary(self) -> None:
        print(f"\n{'='*80}")
        print(f"  POSTMORTEM REPORT — {self.timestamp}")
        print(f"{'='*80}")

        if self.alpha:
            a = self.alpha
            print(f"\n  [ALPHA] {a.sport}")
            print(f"    Signals: {a.total_signals} total, {a.profitable_signals} profitable ({a.signal_accuracy:.1%})")
            print(f"    Win edge: {a.mean_winning_edge_bps:+.1f} bps avg")
            print(f"    Loss edge: {a.mean_losing_edge_bps:+.1f} bps avg")

        if self.execution:
            e = self.execution
            print(f"\n  [EXECUTION] {e.sport}")
            print(f"    Gross edge:     {e.total_gross_edge:+.2f}")
            print(f"    Fees estimated: {e.total_fees_estimated:+.2f} ({e.fee_drag_pct:.1%} of gross)")
            print(f"    Net PnL:        {e.total_net_pnl:+.2f}")
            print(f"    Edge retained:  {e.edge_retained_pct:.1%}")

        if self.risk:
            r = self.risk
            print(f"\n  [RISK] {r.sport}")
            print(f"    Games: {r.total_games}, Direction accuracy: {r.direction_accuracy:.1%}")
            print(f"    Model errors (wrong winner): {r.model_error_games}")
            print(f"    Execution errors (right winner, lost money): {r.execution_error_games}")
            print(f"    PnL when correct: {r.mean_pnl_when_correct:+.2f}")
            print(f"    PnL when wrong:   {r.mean_pnl_when_wrong:+.2f}")

        if self.feature_decay:
            fd = self.feature_decay
            print(f"\n  [FEATURE DECAY] {fd.sport}")
            if fd.decaying_features:
                print(f"    Decaying: {', '.join(fd.decaying_features)}")
            if fd.stable_features:
                print(f"    Stable:   {', '.join(fd.stable_features)}")

        print(f"\n{'='*80}\n")


def load_paper_reports(reports_dir: str = "reports",
                       sport: str | None = None) -> list[dict]:
    """Load all paper reports from the reports directory."""
    pattern = os.path.join(reports_dir, "paper_*.json")
    files = sorted(glob.glob(pattern))
    reports = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        if sport and data.get("sport") != sport:
            continue
        reports.append(data)
    return reports


def analyze_alpha(reports: list[dict], sport: str) -> AlphaReport:
    """Analyze signal quality from paper reports."""
    total_signals = 0
    profitable = 0
    winning_edges: list[float] = []
    losing_edges: list[float] = []

    for report in reports:
        for game in report.get("games", []):
            total_signals += game.get("total_signals", 0)
            if game.get("shadow_pnl", 0) > 0:
                profitable += 1
                winning_edges.append(game.get("shadow_pnl", 0))
            else:
                losing_edges.append(game.get("shadow_pnl", 0))

    total_games = sum(len(r.get("games", [])) for r in reports)
    return AlphaReport(
        sport=sport,
        period=f"{len(reports)} sessions",
        total_signals=total_signals,
        profitable_signals=profitable,
        signal_accuracy=profitable / max(total_games, 1),
        mean_winning_edge_bps=sum(winning_edges) / len(winning_edges) if winning_edges else 0,
        mean_losing_edge_bps=sum(losing_edges) / len(losing_edges) if losing_edges else 0,
    )


def analyze_execution(reports: list[dict], sport: str) -> ExecutionReport:
    """Analyze execution quality — fee and delay drag."""
    total_pnl = 0.0
    total_fills = 0
    total_orders = 0

    for report in reports:
        total_pnl += report.get("total_shadow_pnl", 0)
        total_fills += report.get("total_shadow_fills", 0)
        total_orders += report.get("total_shadow_orders", 0)

    # Estimate fees: 120 bps * avg_stake * fills (basketball)
    # For baseball: lower fees assumed
    fee_bps = 120.0 if sport == "basketball" else 50.0
    avg_stake = 30.0  # from config
    est_fees = total_fills * (fee_bps / 10000.0) * avg_stake

    gross = total_pnl + est_fees  # PnL before fees
    return ExecutionReport(
        sport=sport,
        total_gross_edge=gross,
        total_fees_estimated=est_fees,
        total_net_pnl=total_pnl,
        fee_drag_pct=est_fees / gross if gross > 0 else 0,
        edge_retained_pct=total_pnl / gross if gross > 0 else 0,
    )


def analyze_risk(reports: list[dict], sport: str) -> RiskReport:
    """Analyze model vs execution errors."""
    total = 0
    correct = 0
    model_errors = 0
    exec_errors = 0
    pnl_correct: list[float] = []
    pnl_wrong: list[float] = []

    for report in reports:
        for game in report.get("games", []):
            total += 1
            is_correct = game.get("direction_correct", False)
            pnl = game.get("shadow_pnl", 0)

            if is_correct:
                correct += 1
                pnl_correct.append(pnl)
                if pnl < 0:
                    exec_errors += 1  # right direction but lost money
            else:
                model_errors += 1
                pnl_wrong.append(pnl)

    return RiskReport(
        sport=sport,
        total_games=total,
        direction_correct=correct,
        direction_accuracy=correct / max(total, 1),
        model_error_games=model_errors,
        execution_error_games=exec_errors,
        mean_pnl_when_correct=sum(pnl_correct) / len(pnl_correct) if pnl_correct else 0,
        mean_pnl_when_wrong=sum(pnl_wrong) / len(pnl_wrong) if pnl_wrong else 0,
    )


def analyze_feature_decay(reports: list[dict], sport: str) -> FeatureDecayReport:
    """Check if performance is decaying over recent windows.

    Compares first half vs second half of available data.
    """
    all_games: list[dict] = []
    for r in reports:
        all_games.extend(r.get("games", []))

    if len(all_games) < 10:
        return FeatureDecayReport(sport=sport, period="insufficient data")

    mid = len(all_games) // 2
    first_half = all_games[:mid]
    second_half = all_games[mid:]

    def window_metrics(games: list[dict]) -> dict:
        pnls = [g.get("shadow_pnl", 0) for g in games]
        correct = sum(1 for g in games if g.get("direction_correct", False))
        return {
            "games": len(games),
            "mean_pnl": sum(pnls) / len(pnls) if pnls else 0,
            "accuracy": correct / len(games) if games else 0,
            "win_rate": sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0,
        }

    m1 = window_metrics(first_half)
    m2 = window_metrics(second_half)

    fd = FeatureDecayReport(
        sport=sport,
        period=f"{len(all_games)} games",
        metrics_by_window={"first_half": m1, "second_half": m2},
    )

    # Simple decay detection
    if m2["mean_pnl"] < m1["mean_pnl"] * 0.7:
        fd.decaying_features.append("overall_pnl")
    else:
        fd.stable_features.append("overall_pnl")

    if m2["accuracy"] < m1["accuracy"] - 0.05:
        fd.decaying_features.append("direction_accuracy")
    else:
        fd.stable_features.append("direction_accuracy")

    if m2["win_rate"] < m1["win_rate"] - 0.05:
        fd.decaying_features.append("win_rate")
    else:
        fd.stable_features.append("win_rate")

    return fd


def run_postmortem(reports_dir: str = "reports",
                   sport: str | None = None) -> list[PostmortemReport]:
    """Run full postmortem analysis on paper trading reports."""
    results: list[PostmortemReport] = []
    sports = [sport] if sport else ["basketball", "baseball"]

    for s in sports:
        reports = load_paper_reports(reports_dir, sport=s)
        if not reports:
            print(f"  No paper reports found for {s}")
            continue

        pm = PostmortemReport(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            alpha=analyze_alpha(reports, s),
            execution=analyze_execution(reports, s),
            risk=analyze_risk(reports, s),
            feature_decay=analyze_feature_decay(reports, s),
        )
        results.append(pm)

    return results


if __name__ == "__main__":
    rdir = sys.argv[1] if len(sys.argv) > 1 else "reports"
    sport = sys.argv[2] if len(sys.argv) > 2 else None

    reports = run_postmortem(rdir, sport)
    for pm in reports:
        pm.print_summary()

    if not reports:
        print("  No paper reports found. Run paper_runner.py first.")
