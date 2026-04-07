"""
Research factory orchestrator — unified nightly pipeline.

Runs the full research cycle:
  1. Generate candidate experiments
  2. Run replay/sweep/ablation evaluation
  3. Score and rank candidates
  4. Run paper trading for promoted strategies
  5. Run postmortem on paper results
  6. Update strategy registry
  7. Generate nightly report

Usage:
  source .venv/bin/activate && python orchestrator.py [mode]

Modes:
  nightly   — full pipeline (default)
  research  — only steps 1-3
  paper     — only step 4
  postmortem — only step 5
  report    — only step 7
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from dataclasses import dataclass, field, asdict

from src.research import (
    ExperimentSpec,
    ExperimentResult,
    GateResult,
    compute_promotion_score,
)
from src.research.registry import StrategyRegistry, StrategyEntry
from src.pricing.engine import BaseballFeatureFlags
from src.strategy import BasketballStrategyConfig


# ─── Step 1: Generate candidate experiments ───

def generate_candidates() -> list[ExperimentSpec]:
    """Generate a batch of candidate experiment specs.

    Currently generates systematic parameter variations.
    Future: LLM-assisted hypothesis generation.
    """
    candidates = []

    # Basketball: sweep fee thresholds
    for fee_bps in [80, 100, 120, 150]:
        for min_net in [30, 50, 80]:
            candidates.append(ExperimentSpec(
                name=f"bball_fee{fee_bps}_net{min_net}",
                sport="basketball",
                params={
                    "fee_bps_roundtrip": fee_bps,
                    "min_net_edge_bps": min_net,
                    "delay_penalty_bps_per_ms": 0.05,
                },
                n_games=30,
            ))

    # Baseball: compare flag combinations
    for flags_name, flags in {
        "trading_default": BaseballFeatureFlags.trading_default(),
        "full": BaseballFeatureFlags.research_full(),
        "leverage_only": BaseballFeatureFlags(
            enable_base_out_state=False, enable_leverage=True,
            enable_walkoff=False, enable_fatigue=False,
            enable_bullpen=False, enable_blowout_asymmetry=False,
        ),
        "blowout_only": BaseballFeatureFlags(
            enable_base_out_state=False, enable_leverage=False,
            enable_walkoff=False, enable_fatigue=False,
            enable_bullpen=False, enable_blowout_asymmetry=True,
        ),
    }.items():
        candidates.append(ExperimentSpec(
            name=f"baseball_{flags_name}",
            sport="baseball",
            feature_flags=asdict(flags),
            n_games=30,
        ))

    return candidates


# ─── Step 2: Run evaluation gates ───

def evaluate_candidate(spec: ExperimentSpec) -> ExperimentResult:
    """Run all evaluation gates for a candidate spec."""
    from ablation_runner import (
        run_one_game, merge_results, generate_game_path,
        BaseballFeatureFlags as BFF,
    )
    from paper_runner import paper_trade_basketball, paper_trade_baseball

    result = ExperimentResult(
        spec_id=spec.spec_id,
        spec_name=spec.name,
        sport=spec.sport,
    )

    # Gate 1: Replay (using ablation infrastructure for consistency)
    if spec.run_replay:
        if spec.sport == "baseball":
            flags = BFF(**spec.feature_flags) if spec.feature_flags else BFF.trading_default()
            game_results = []
            for i in range(spec.n_games):
                path, h_off, a_off = generate_game_path(spec.base_seed + i)
                book_seed = spec.base_seed + i + 10000
                r = run_one_game(flags, spec.name, path, h_off, a_off, book_seed)
                game_results.append(r)

            merged = merge_results(game_results)
            result.direction_accuracy = merged.accuracy
            result.mean_pnl = merged.mean_pnl
            result.std_pnl = merged.std_pnl
            result.total_fills = merged.total_fills
            result.total_signals = merged.total_signals
            result.pass_rate = merged.pass_rate
            result.edge_retained_pct = merged.edge_retained_pct
            result.mean_edge_bps = merged.mean_edge_bps

            replay_passed = merged.mean_pnl > 0
            result.gates.append(GateResult(
                gate_name="replay",
                passed=replay_passed,
                metrics={
                    "mean_pnl": merged.mean_pnl,
                    "accuracy": merged.accuracy,
                    "fills": float(merged.total_fills),
                },
            ))

        elif spec.sport == "basketball":
            # Build config from spec params so each candidate uses its own fee/net
            bball_config = BasketballStrategyConfig.paper_default()
            if spec.params:
                for k, v in spec.params.items():
                    if hasattr(bball_config, k):
                        setattr(bball_config, k, v)

            pnls = []
            correct = 0
            total = 0
            total_fills = 0
            total_signals = 0

            for i in range(spec.n_games):
                g = paper_trade_basketball(i, spec.base_seed + i, config=bball_config)
                pnls.append(g.shadow_pnl)
                if g.direction_correct:
                    correct += 1
                total += 1
                total_fills += g.shadow_fills
                total_signals += g.total_signals

            mean_pnl = sum(pnls) / len(pnls) if pnls else 0
            result.direction_accuracy = correct / max(total, 1)
            result.mean_pnl = mean_pnl
            result.total_fills = total_fills
            result.total_signals = total_signals
            if len(pnls) >= 2:
                m = mean_pnl
                result.std_pnl = (sum((x - m) ** 2 for x in pnls) / (len(pnls) - 1)) ** 0.5

            replay_passed = mean_pnl > 0
            result.gates.append(GateResult(
                gate_name="replay",
                passed=replay_passed,
                metrics={"mean_pnl": mean_pnl, "accuracy": result.direction_accuracy},
            ))

    # Compute promotion score
    result.promotion_score = compute_promotion_score(result, baseline_pnl=0)
    result.all_gates_passed = all(g.passed for g in result.gates)

    if result.all_gates_passed and result.promotion_score >= 20:
        result.recommended_status = "paper"
    elif result.mean_pnl > 0:
        result.recommended_status = "research"
    else:
        result.recommended_status = "retired"

    return result


# ─── Step 3: Rank and update registry ───

def update_registry_from_results(
    results: list[ExperimentResult],
    specs: list[ExperimentSpec],
    registry: StrategyRegistry,
) -> None:
    """Register new strategies and update metrics."""
    spec_map = {s.spec_id: s for s in specs}

    for result in results:
        spec = spec_map.get(result.spec_id)
        if not spec:
            continue

        entry = registry.get(result.spec_id)
        if not entry:
            entry = StrategyEntry(
                strategy_id=result.spec_id,
                name=result.spec_name,
                sport=result.sport,
                feature_flags=spec.feature_flags,
                params=spec.params,
            )
            registry.register(entry)

        registry.update_research_metrics(
            result.spec_id,
            pnl=result.mean_pnl,
            accuracy=result.direction_accuracy,
            games=spec.n_games,
            score=result.promotion_score,
        )

        # Auto-promote if gates passed
        if result.recommended_status == "paper" and entry.status == "research":
            promoted = registry.promote(result.spec_id, "paper")
            if promoted:
                print(f"    PROMOTED {result.spec_name} → paper (score={result.promotion_score:.1f})")

        elif result.recommended_status == "retired" and entry.status == "research":
            registry.retire(result.spec_id, reason="failed replay gate")


# ─── Step 7: Nightly report ───

def generate_nightly_report(
    results: list[ExperimentResult],
    registry: StrategyRegistry,
    paper_reports: list = None,
    postmortem_reports: list = None,
) -> str:
    """Generate and save nightly summary report."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append(f"{'='*80}")
    lines.append(f"  NIGHTLY RESEARCH REPORT — {ts}")
    lines.append(f"{'='*80}")

    # Research summary
    if results:
        lines.append(f"\n  [RESEARCH] {len(results)} candidates evaluated")
        passed = [r for r in results if r.all_gates_passed]
        lines.append(f"    Passed all gates: {len(passed)}")

        # Top 5 by promotion score
        ranked = sorted(results, key=lambda r: r.promotion_score, reverse=True)
        lines.append(f"\n    TOP CANDIDATES:")
        lines.append(f"    {'Name':<30} {'Sport':<10} {'PnL':>8} {'Acc':>7} {'Score':>7}")
        lines.append(f"    {'─'*65}")
        for r in ranked[:5]:
            lines.append(f"    {r.spec_name:<30} {r.sport:<10} "
                         f"{r.mean_pnl:>+8.2f} {r.direction_accuracy:>6.1%} "
                         f"{r.promotion_score:>7.1f}")

    # Registry status
    lines.append(f"\n  [REGISTRY]")
    for status in ["live", "paper", "research"]:
        entries = registry.list_by_status(status)
        lines.append(f"    {status}: {len(entries)} strategies")

    # Paper summary
    if paper_reports:
        lines.append(f"\n  [PAPER TRADING]")
        for pr in paper_reports:
            lines.append(f"    {pr.sport}: {pr.n_games} games, "
                         f"PnL={pr.total_shadow_pnl:+.2f}, "
                         f"Acc={pr.direction_accuracy:.1%}")

    # Postmortem summary
    if postmortem_reports:
        lines.append(f"\n  [POSTMORTEM]")
        for pm in postmortem_reports:
            if pm.risk:
                lines.append(f"    {pm.risk.sport}: model_errors={pm.risk.model_error_games}, "
                             f"exec_errors={pm.risk.execution_error_games}")
            if pm.feature_decay and pm.feature_decay.decaying_features:
                lines.append(f"    Decaying: {', '.join(pm.feature_decay.decaying_features)}")

    lines.append(f"\n{'='*80}")

    report_text = "\n".join(lines)

    # Save
    os.makedirs("reports", exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    path = f"reports/nightly_{date_str}.txt"
    with open(path, "w") as f:
        f.write(report_text)

    return report_text


# ─── Main orchestrator ───

def ensure_live_candidate(registry: StrategyRegistry) -> None:
    """Register basketball_fee80_net20 as a paper candidate if not present."""
    sid = "basketball_fee80_net20"
    if registry.get(sid):
        return
    cfg = BasketballStrategyConfig.paper_default()
    entry = StrategyEntry(
        strategy_id=sid,
        name="basketball_fee80_net20",
        sport="basketball",
        status="paper",
        params=cfg.to_dict(),
    )
    registry.register(entry)
    print(f"  Registered live candidate: {sid} (status=paper)")


def run_nightly() -> None:
    """Full nightly pipeline."""
    start = time.time()
    print(f"\n{'='*80}")
    print(f"  ORCHESTRATOR — NIGHTLY RUN — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    registry = StrategyRegistry()
    ensure_live_candidate(registry)

    # Step 1: Generate candidates
    print("  Step 1: Generating candidates...")
    candidates = generate_candidates()
    print(f"    Generated {len(candidates)} candidates")

    # Save experiment specs
    for spec in candidates:
        spec.save()

    # Step 2: Evaluate candidates
    print("\n  Step 2: Evaluating candidates...")
    results: list[ExperimentResult] = []
    for i, spec in enumerate(candidates):
        print(f"    [{i+1}/{len(candidates)}] {spec.name}...", end=" ", flush=True)
        result = evaluate_candidate(spec)
        results.append(result)
        result.save()
        status = "PASS" if result.all_gates_passed else "FAIL"
        print(f"{status} (pnl={result.mean_pnl:+.2f}, score={result.promotion_score:.1f})")

    # Step 3: Update registry
    print("\n  Step 3: Updating registry...")
    update_registry_from_results(results, candidates, registry)

    # Step 4: Paper trading for promoted strategies
    print("\n  Step 4: Paper trading...")
    paper_reports = []
    from paper_runner import run_paper
    paper_strategies = registry.list_by_status("paper")
    if paper_strategies:
        for entry in paper_strategies[:3]:  # max 3 paper strategies
            print(f"    Paper trading: {entry.name} ({entry.sport})")
            report = run_paper(sport=entry.sport, n_games=10)
            report.compute_aggregates()
            report.save()
            paper_reports.append(report)
            registry.update_paper_metrics(entry.strategy_id, report.mean_pnl_per_game, 1)
    else:
        print("    No strategies in paper status — running default paper sessions")
        for sport in ["basketball", "baseball"]:
            report = run_paper(sport=sport, n_games=10)
            report.compute_aggregates()
            report.save()
            paper_reports.append(report)

    # Step 5: Postmortem
    print("\n  Step 5: Running postmortem...")
    from postmortem_runner import run_postmortem
    postmortem_reports = run_postmortem()

    # Step 6: Generate report
    print("\n  Step 6: Generating nightly report...")
    report_text = generate_nightly_report(results, registry, paper_reports, postmortem_reports)
    print(report_text)

    elapsed = time.time() - start
    print(f"\n  Nightly run complete in {elapsed:.1f}s")
    registry.print_summary()


def run_research_only() -> None:
    """Only run candidate generation and evaluation."""
    registry = StrategyRegistry()
    ensure_live_candidate(registry)
    candidates = generate_candidates()
    print(f"  Generated {len(candidates)} candidates")

    results = []
    for i, spec in enumerate(candidates):
        print(f"  [{i+1}/{len(candidates)}] {spec.name}...", end=" ", flush=True)
        result = evaluate_candidate(spec)
        results.append(result)
        status = "PASS" if result.all_gates_passed else "FAIL"
        print(f"{status} (pnl={result.mean_pnl:+.2f})")

    update_registry_from_results(results, candidates, registry)
    registry.print_summary()


def run_paper_only() -> None:
    """Only run paper trading."""
    registry = StrategyRegistry()
    ensure_live_candidate(registry)
    from paper_runner import run_paper
    for sport in ["basketball", "baseball"]:
        report = run_paper(sport=sport, n_games=20)
        report.print_summary()
        report.save()


def run_postmortem_only() -> None:
    """Only run postmortem analysis."""
    from postmortem_runner import run_postmortem
    reports = run_postmortem()
    for pm in reports:
        pm.print_summary()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "nightly"

    if mode == "nightly":
        run_nightly()
    elif mode == "research":
        run_research_only()
    elif mode == "paper":
        run_paper_only()
    elif mode == "postmortem":
        run_postmortem_only()
    elif mode == "report":
        registry = StrategyRegistry()
        print(generate_nightly_report([], registry))
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python orchestrator.py [nightly|research|paper|postmortem|report]")
