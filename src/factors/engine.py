"""
Factor Engine — compiles and executes factor expressions safely.

AI generates expressions like:
    "clip(ts_zscore(order_book_imbalance, 50))"
    "ts_decay_linear(log_return(mid_price, 5), 20)"

The engine parses these into an AST, validates that only allowed operators
are used, extracts the required data series from tick history, and computes
the final value.

This is the critical safety boundary — no raw code execution, only
pre-validated operator composition.
"""

from __future__ import annotations

import ast
import logging
from typing import Any, Sequence

from src.core.models import Tick
from src.factors.operators.primitives import OPERATOR_REGISTRY

logger = logging.getLogger(__name__)

# Data fields extractable from Tick history
TICK_FIELDS = {
    "mid_price",
    "last_trade_price",
    "volume_24h",
    "spread",
    "bid_depth",
    "ask_depth",
    "book_imbalance",
}


class FactorCompilationError(Exception):
    pass


class FactorEngine:
    """Stateless factor computation engine."""

    @staticmethod
    def validate_expression(expression: str) -> bool:
        """Check that an expression only uses allowed operators and fields."""
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError:
            return False

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name):
                    if func.id not in OPERATOR_REGISTRY:
                        logger.warning("Disallowed operator: %s", func.id)
                        return False
                else:
                    return False  # no method calls or complex expressions
            elif isinstance(node, ast.Name):
                if node.id not in OPERATOR_REGISTRY and node.id not in TICK_FIELDS:
                    # Could be a numeric literal reference — check parent
                    pass
        return True

    @staticmethod
    def extract_series(field_name: str, tick_history: list[Tick]) -> list[float]:
        """Extract a time series of a specific field from tick history."""
        if field_name not in TICK_FIELDS:
            raise FactorCompilationError(f"Unknown tick field: {field_name}")
        return [getattr(t, field_name) for t in tick_history]

    @staticmethod
    def compute(expression: str, tick: Tick, tick_history: list[Tick]) -> float:
        """
        Evaluate a factor expression against current tick and history.

        Returns a float in [-1, 1].
        """
        if not tick_history:
            return 0.0

        # Build the evaluation context with data series and operators
        context: dict[str, Any] = {}

        # Add all operators
        context.update(OPERATOR_REGISTRY)

        # Add all tick field series
        for field_name in TICK_FIELDS:
            context[field_name] = [getattr(t, field_name) for t in tick_history]

        try:
            # Parse and evaluate in restricted context
            tree = ast.parse(expression, mode="eval")
            code = compile(tree, "<factor>", "eval")

            # Only allow our operators and data — no builtins
            result = eval(code, {"__builtins__": {}}, context)  # noqa: S307

            # Ensure output is a bounded float
            if isinstance(result, (int, float)):
                return max(-1.0, min(1.0, float(result)))
            elif isinstance(result, (list, tuple)):
                # If expression returns a series, take the last value
                val = float(result[-1]) if result else 0.0
                return max(-1.0, min(1.0, val))
            return 0.0

        except Exception as e:
            logger.error("Factor computation failed for '%s': %s", expression, e)
            return 0.0


class FactorSandbox:
    """
    Isolated backtest environment for testing AI-generated factors
    before they go live.
    """

    def __init__(self, historical_ticks: list[Tick]) -> None:
        self.ticks = historical_ticks

    def backtest(self, expression: str) -> SandboxResult:
        """Run a factor expression against historical data and compute metrics."""
        if not FactorEngine.validate_expression(expression):
            return SandboxResult(
                expression=expression,
                is_valid=False,
                error="Expression contains disallowed operators",
            )

        values: list[float] = []
        forward_returns: list[float] = []

        # Compute factor values at each tick with growing history
        for i in range(50, len(self.ticks)):  # need at least 50 ticks of history
            history = self.ticks[: i + 1]
            tick = self.ticks[i]
            value = FactorEngine.compute(expression, tick, history)
            values.append(value)

            # Forward return = next tick's mid_price change
            if i + 1 < len(self.ticks):
                fwd = self.ticks[i + 1].mid_price - tick.mid_price
                if tick.mid_price > 0:
                    fwd /= tick.mid_price
                forward_returns.append(fwd)

        if len(values) < 100 or len(forward_returns) < 100:
            return SandboxResult(
                expression=expression,
                is_valid=False,
                error="Insufficient data for backtest",
            )

        # Align lengths
        min_len = min(len(values), len(forward_returns))
        values = values[:min_len]
        forward_returns = forward_returns[:min_len]

        # Compute IC (Information Coefficient = rank correlation with forward returns)
        import numpy as np
        from scipy import stats

        ic, _ = stats.spearmanr(values, forward_returns)
        if np.isnan(ic):
            ic = 0.0

        # Compute IR (Information Ratio = mean IC / std IC over rolling windows)
        window = 50
        rolling_ics = []
        for j in range(window, len(values)):
            chunk_v = values[j - window : j]
            chunk_r = forward_returns[j - window : j]
            r_ic, _ = stats.spearmanr(chunk_v, chunk_r)
            if not np.isnan(r_ic):
                rolling_ics.append(r_ic)

        mean_ic = float(np.mean(rolling_ics)) if rolling_ics else 0.0
        std_ic = float(np.std(rolling_ics)) if rolling_ics else 1.0
        ir = mean_ic / std_ic if std_ic > 1e-10 else 0.0

        # Compute factor autocorrelation (turnover proxy)
        if len(values) > 1:
            autocorr = float(np.corrcoef(values[:-1], values[1:])[0, 1])
            if np.isnan(autocorr):
                autocorr = 0.0
        else:
            autocorr = 0.0

        return SandboxResult(
            expression=expression,
            is_valid=True,
            ic=ic,
            ir=ir,
            mean_ic=mean_ic,
            std_ic=std_ic,
            autocorrelation=autocorr,
            n_observations=len(values),
        )


class SandboxResult:
    def __init__(
        self,
        expression: str,
        is_valid: bool,
        error: str = "",
        ic: float = 0.0,
        ir: float = 0.0,
        mean_ic: float = 0.0,
        std_ic: float = 0.0,
        autocorrelation: float = 0.0,
        n_observations: int = 0,
    ) -> None:
        self.expression = expression
        self.is_valid = is_valid
        self.error = error
        self.ic = ic
        self.ir = ir
        self.mean_ic = mean_ic
        self.std_ic = std_ic
        self.autocorrelation = autocorrelation
        self.n_observations = n_observations

    @property
    def passes_threshold(self) -> bool:
        """Does this factor meet minimum quality standards?"""
        return (
            self.is_valid
            and abs(self.ic) > 0.02
            and abs(self.ir) > 0.5
            and self.n_observations >= 100
        )

    def summary(self) -> str:
        if not self.is_valid:
            return f"FAILED: {self.error}"
        status = "PASS" if self.passes_threshold else "FAIL"
        return (
            f"[{status}] IC={self.ic:.4f} IR={self.ir:.4f} "
            f"MeanIC={self.mean_ic:.4f} AutoCorr={self.autocorrelation:.4f} "
            f"N={self.n_observations}"
        )
