"""
Experiment specification and result types for the research factory.

An ExperimentSpec is a fully reproducible description of a single experiment:
  - what sport
  - what strategy config
  - what feature flags
  - what parameters
  - what seeds / game count
  - what evaluation gates to run

An ExperimentResult is the output: metrics, pass/fail per gate, timestamp.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class ExperimentSpec:
    """Fully reproducible experiment description."""

    # Identity
    name: str                          # human-readable name
    sport: str                         # "basketball" or "baseball"
    strategy: str = "edge_back"        # strategy type

    # Feature flags (baseball only)
    feature_flags: dict[str, bool] = field(default_factory=dict)

    # Strategy parameters
    params: dict[str, Any] = field(default_factory=dict)

    # Experiment control
    n_games: int = 50
    base_seed: int = 42

    # Gates to run
    run_replay: bool = True
    run_sweep: bool = True
    run_ablation: bool = False
    run_robustness: bool = False

    # Metadata
    parent_spec: str = ""              # spec ID this was derived from
    hypothesis: str = ""               # what we expect to learn
    created_at: float = field(default_factory=time.time)

    @property
    def spec_id(self) -> str:
        """Deterministic hash of the spec content (excluding timestamps)."""
        content = {
            "sport": self.sport,
            "strategy": self.strategy,
            "feature_flags": self.feature_flags,
            "params": self.params,
            "n_games": self.n_games,
            "base_seed": self.base_seed,
        }
        raw = json.dumps(content, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["spec_id"] = self.spec_id
        return d

    def save(self, experiments_dir: str = "experiments") -> str:
        os.makedirs(experiments_dir, exist_ok=True)
        path = os.path.join(experiments_dir, f"{self.spec_id}.json")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "ExperimentSpec":
        with open(path) as f:
            d = json.load(f)
        d.pop("spec_id", None)
        return cls(**d)


@dataclass
class GateResult:
    """Result of a single evaluation gate."""
    gate_name: str                     # "replay", "sweep", "ablation", "robustness"
    passed: bool = False
    metrics: dict[str, float] = field(default_factory=dict)
    details: str = ""


@dataclass
class ExperimentResult:
    """Aggregated results for one experiment spec."""

    spec_id: str
    spec_name: str
    sport: str

    # Gate results
    gates: list[GateResult] = field(default_factory=list)

    # Summary metrics
    direction_accuracy: float = 0.0
    mean_pnl: float = 0.0
    std_pnl: float = 0.0
    total_fills: int = 0
    total_signals: int = 0
    pass_rate: float = 0.0
    edge_retained_pct: float = 0.0
    mean_edge_bps: float = 0.0

    # Promotion decision
    all_gates_passed: bool = False
    promotion_score: float = 0.0
    recommended_status: str = "research"  # research / paper / live / retired

    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def save(self, experiments_dir: str = "experiments") -> str:
        os.makedirs(experiments_dir, exist_ok=True)
        path = os.path.join(experiments_dir, f"{self.spec_id}_result.json")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path


# ─── Promotion scoring ───

def compute_promotion_score(result: ExperimentResult, baseline_pnl: float = 0.0) -> float:
    """Score a result for promotion decisions.

    Higher = better candidate for paper/live.
    Range: roughly -100 to +100.
    """
    score = 0.0

    # PnL improvement over baseline (most important)
    pnl_delta = result.mean_pnl - baseline_pnl
    score += min(pnl_delta * 0.5, 30.0)  # cap at 30 points

    # Edge retention
    if result.edge_retained_pct > 50:
        score += 15.0
    elif result.edge_retained_pct > 20:
        score += 8.0

    # Direction accuracy
    if result.direction_accuracy > 0.70:
        score += 15.0
    elif result.direction_accuracy > 0.65:
        score += 8.0
    elif result.direction_accuracy > 0.60:
        score += 3.0

    # Stability (low PnL variance is good)
    if result.std_pnl > 0 and result.mean_pnl > 0:
        sharpe_like = result.mean_pnl / result.std_pnl
        score += min(sharpe_like * 10, 20.0)

    # Complexity penalty (more gates that passed = more robust)
    gates_passed = sum(1 for g in result.gates if g.passed)
    total_gates = len(result.gates)
    if total_gates > 0:
        score += (gates_passed / total_gates) * 10.0

    # Turnover penalty (very high fill count may indicate overtrading)
    if result.total_fills > 0 and result.mean_pnl < 0:
        score -= 10.0  # losing money with lots of trades = bad

    return round(score, 2)
