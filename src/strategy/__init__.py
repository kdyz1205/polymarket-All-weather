"""
Parameterized strategy configuration for basketball and baseball.

Each strategy has explicit gates that must ALL pass before an order is submitted.
Every gate rejection is tracked for no-trade diagnostics.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RejectReason(str, Enum):
    """Why an order was NOT placed."""
    EDGE_TOO_SMALL = "edge_below_threshold"
    LIQUIDITY_TOO_THIN = "liquidity_below_min"
    SPREAD_TOO_WIDE = "spread_above_max"
    COOLDOWN_ACTIVE = "cooldown_active"
    POSITION_LIMIT = "position_limit_reached"
    RATE_LIMIT = "orders_per_min_exceeded"
    RISK_GATE = "risk_engine_blocked"
    MARKET_STATE = "market_not_tradeable"
    INNING_FILTER = "inning_filtered_out"
    OUTS_FILTER = "outs_filtered_out"
    GAME_NOT_CLOSE = "game_not_close_enough"
    DELAY_PENALTY = "edge_after_delay_penalty_negative"
    KILL_SWITCH = "kill_switch_active"


@dataclass
class RejectionLog:
    """Tracks all order rejection reasons for diagnostics."""

    counts: dict[str, int] = field(default_factory=dict)
    total_signals: int = 0  # how many times strategy WANTED to trade
    total_passed: int = 0   # how many times all gates passed

    def reject(self, reason: RejectReason) -> None:
        self.counts[reason.value] = self.counts.get(reason.value, 0) + 1

    def signal(self) -> None:
        self.total_signals += 1

    def passed(self) -> None:
        self.total_passed += 1

    @property
    def pass_rate(self) -> float:
        return self.total_passed / self.total_signals if self.total_signals > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_signals": self.total_signals,
            "total_passed": self.total_passed,
            "pass_rate": round(self.pass_rate, 4),
            "rejection_breakdown": dict(
                sorted(self.counts.items(), key=lambda x: -x[1])
            ),
        }


@dataclass
class BasketballStrategyConfig:
    """Parameterized gates for basketball edge-back strategy."""

    # Edge threshold
    min_edge_bps: float = 300.0  # 3% = 300bps

    # Liquidity gate
    min_liquidity: float = 100.0  # minimum volume at best price

    # Spread gate
    max_spread_bps: float = 500.0  # max back-lay spread in bps

    # Cooldown
    cooldown_sec: float = 30.0  # min seconds between orders on same runner

    # Position limits
    max_position_per_runner: float = 500.0  # max total stake per runner
    max_total_position: float = 1500.0

    # Rate limit
    max_orders_per_min: int = 10

    # Sizing
    base_stake: float = 50.0

    # Delay penalty: reduce perceived edge by this factor * delay_ms
    delay_penalty_bps_per_ms: float = 0.1  # 0.1 bps per ms of delay

    # Quarter-specific
    enable_q1: bool = True
    enable_q2: bool = True
    enable_q3: bool = True
    enable_q4: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def aggressive(cls) -> BasketballStrategyConfig:
        """Low thresholds to maximise trade count for diagnostics."""
        return cls(
            min_edge_bps=50.0,
            min_liquidity=10.0,
            max_spread_bps=2000.0,
            cooldown_sec=5.0,
            max_position_per_runner=2000.0,
            max_total_position=5000.0,
            max_orders_per_min=60,
            base_stake=30.0,
            delay_penalty_bps_per_ms=0.0,
        )

    @classmethod
    def conservative(cls) -> BasketballStrategyConfig:
        """Tight thresholds, fewer but higher-quality trades."""
        return cls(
            min_edge_bps=500.0,
            min_liquidity=200.0,
            max_spread_bps=300.0,
            cooldown_sec=60.0,
            max_position_per_runner=300.0,
            max_total_position=800.0,
            max_orders_per_min=3,
            base_stake=25.0,
            delay_penalty_bps_per_ms=0.2,
        )


@dataclass
class BaseballStrategyConfig:
    """Parameterized gates for baseball edge-back strategy."""

    # Edge threshold
    min_edge_bps: float = 400.0  # 4% = 400bps

    # Liquidity gate
    min_liquidity: float = 80.0

    # Spread gate
    max_spread_bps: float = 600.0

    # Cooldown
    cooldown_sec: float = 45.0

    # Position limits
    max_position_per_runner: float = 400.0
    max_total_position: float = 1200.0

    # Rate limit
    max_orders_per_min: int = 8

    # Sizing
    base_stake: float = 30.0

    # Delay penalty
    delay_penalty_bps_per_ms: float = 0.08

    # Baseball-specific filters
    min_inning: int = 1        # don't trade before this inning
    max_inning: int = 9        # don't trade after this inning
    close_game_only: bool = False  # only trade when run differential <= threshold
    close_game_max_diff: int = 4   # max run differential for "close game"
    min_outs_in_inning: int = 0    # don't trade with fewer outs

    # Leverage-aware: trade more aggressively in high-leverage situations
    high_leverage_multiplier: float = 1.5  # edge multiplier in high leverage
    high_leverage_innings: list[int] = field(default_factory=lambda: [7, 8, 9])

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def aggressive(cls) -> BaseballStrategyConfig:
        """Low thresholds for diagnostics."""
        return cls(
            min_edge_bps=30.0,
            min_liquidity=5.0,
            max_spread_bps=3000.0,
            cooldown_sec=3.0,
            max_position_per_runner=2000.0,
            max_total_position=5000.0,
            max_orders_per_min=60,
            base_stake=20.0,
            delay_penalty_bps_per_ms=0.0,
            close_game_only=False,
        )

    @classmethod
    def conservative(cls) -> BaseballStrategyConfig:
        """Tight thresholds."""
        return cls(
            min_edge_bps=600.0,
            min_liquidity=150.0,
            max_spread_bps=400.0,
            cooldown_sec=90.0,
            max_position_per_runner=200.0,
            max_total_position=600.0,
            max_orders_per_min=2,
            base_stake=20.0,
            delay_penalty_bps_per_ms=0.15,
            close_game_only=True,
            close_game_max_diff=3,
        )


class StrategyGatekeeper:
    """
    Evaluates all gates for a potential order and returns pass/reject.

    Tracks every rejection for no-trade diagnostics. This is what makes
    the difference between "zero trades and no idea why" vs
    "zero trades because edge threshold was the binding constraint."
    """

    def __init__(self) -> None:
        self.rejection_log = RejectionLog()
        self._last_order_ts: dict[str, float] = {}  # runner_id -> last order timestamp
        self._order_timestamps: list[float] = []  # all order timestamps for rate limit
        self._position_sizes: dict[str, float] = {}  # runner_id -> current position

    def check_basketball(
        self,
        config: BasketballStrategyConfig,
        runner_id: str,
        edge_bps: float,
        best_back_price: float,
        best_lay_price: float,
        best_back_volume: float,
        current_sec: float,
        delay_ms: int,
        risk_allows: bool,
        kill_switch: bool,
        quarter: int,
    ) -> bool:
        """Check all basketball gates. Returns True if order should proceed."""
        self.rejection_log.signal()

        # Kill switch
        if kill_switch:
            self.rejection_log.reject(RejectReason.KILL_SWITCH)
            return False

        # Market state
        if best_back_price <= 0 or best_lay_price <= 0:
            self.rejection_log.reject(RejectReason.MARKET_STATE)
            return False

        # Quarter filter
        q_enabled = {1: config.enable_q1, 2: config.enable_q2,
                     3: config.enable_q3, 4: config.enable_q4}
        if not q_enabled.get(quarter, True):
            self.rejection_log.reject(RejectReason.MARKET_STATE)
            return False

        # Edge after delay penalty
        adjusted_edge_bps = edge_bps - (config.delay_penalty_bps_per_ms * delay_ms)
        if adjusted_edge_bps < config.min_edge_bps:
            self.rejection_log.reject(RejectReason.EDGE_TOO_SMALL)
            return False

        # Liquidity
        if best_back_volume < config.min_liquidity:
            self.rejection_log.reject(RejectReason.LIQUIDITY_TOO_THIN)
            return False

        # Spread
        if best_lay_price > 0 and best_back_price > 0:
            spread_bps = (best_lay_price - best_back_price) / best_back_price * 10000
            if spread_bps > config.max_spread_bps:
                self.rejection_log.reject(RejectReason.SPREAD_TOO_WIDE)
                return False

        # Cooldown
        last_ts = self._last_order_ts.get(runner_id, 0)
        if (current_sec - last_ts) < config.cooldown_sec:
            self.rejection_log.reject(RejectReason.COOLDOWN_ACTIVE)
            return False

        # Position limit
        current_pos = self._position_sizes.get(runner_id, 0)
        if current_pos + config.base_stake > config.max_position_per_runner:
            self.rejection_log.reject(RejectReason.POSITION_LIMIT)
            return False

        total_pos = sum(self._position_sizes.values())
        if total_pos + config.base_stake > config.max_total_position:
            self.rejection_log.reject(RejectReason.POSITION_LIMIT)
            return False

        # Rate limit
        cutoff = current_sec - 60
        recent = [t for t in self._order_timestamps if t > cutoff]
        if len(recent) >= config.max_orders_per_min:
            self.rejection_log.reject(RejectReason.RATE_LIMIT)
            return False

        # Risk engine
        if not risk_allows:
            self.rejection_log.reject(RejectReason.RISK_GATE)
            return False

        # All gates passed
        self.rejection_log.passed()
        self._last_order_ts[runner_id] = current_sec
        self._order_timestamps.append(current_sec)
        self._position_sizes[runner_id] = current_pos + config.base_stake
        return True

    def check_baseball(
        self,
        config: BaseballStrategyConfig,
        runner_id: str,
        edge_bps: float,
        best_back_price: float,
        best_lay_price: float,
        best_back_volume: float,
        current_sec: float,
        delay_ms: int,
        risk_allows: bool,
        kill_switch: bool,
        inning: int,
        outs: int,
        run_diff: int,
    ) -> bool:
        """Check all baseball gates. Returns True if order should proceed."""
        self.rejection_log.signal()

        if kill_switch:
            self.rejection_log.reject(RejectReason.KILL_SWITCH)
            return False

        if best_back_price <= 0 or best_lay_price <= 0:
            self.rejection_log.reject(RejectReason.MARKET_STATE)
            return False

        # Inning filter
        if inning < config.min_inning or inning > config.max_inning:
            self.rejection_log.reject(RejectReason.INNING_FILTER)
            return False

        # Outs filter
        if outs < config.min_outs_in_inning:
            self.rejection_log.reject(RejectReason.OUTS_FILTER)
            return False

        # Close game filter
        if config.close_game_only and abs(run_diff) > config.close_game_max_diff:
            self.rejection_log.reject(RejectReason.GAME_NOT_CLOSE)
            return False

        # High leverage adjustment
        effective_edge = edge_bps
        if inning in config.high_leverage_innings:
            effective_edge *= config.high_leverage_multiplier

        # Delay penalty
        adjusted_edge_bps = effective_edge - (config.delay_penalty_bps_per_ms * delay_ms)
        if adjusted_edge_bps < config.min_edge_bps:
            self.rejection_log.reject(RejectReason.EDGE_TOO_SMALL)
            return False

        # Liquidity
        if best_back_volume < config.min_liquidity:
            self.rejection_log.reject(RejectReason.LIQUIDITY_TOO_THIN)
            return False

        # Spread
        if best_lay_price > 0 and best_back_price > 0:
            spread_bps = (best_lay_price - best_back_price) / best_back_price * 10000
            if spread_bps > config.max_spread_bps:
                self.rejection_log.reject(RejectReason.SPREAD_TOO_WIDE)
                return False

        # Cooldown
        last_ts = self._last_order_ts.get(runner_id, 0)
        if (current_sec - last_ts) < config.cooldown_sec:
            self.rejection_log.reject(RejectReason.COOLDOWN_ACTIVE)
            return False

        # Position limit
        current_pos = self._position_sizes.get(runner_id, 0)
        if current_pos + config.base_stake > config.max_position_per_runner:
            self.rejection_log.reject(RejectReason.POSITION_LIMIT)
            return False

        total_pos = sum(self._position_sizes.values())
        if total_pos + config.base_stake > config.max_total_position:
            self.rejection_log.reject(RejectReason.POSITION_LIMIT)
            return False

        # Rate limit
        cutoff = current_sec - 60
        recent = [t for t in self._order_timestamps if t > cutoff]
        if len(recent) >= config.max_orders_per_min:
            self.rejection_log.reject(RejectReason.RATE_LIMIT)
            return False

        # Risk engine
        if not risk_allows:
            self.rejection_log.reject(RejectReason.RISK_GATE)
            return False

        self.rejection_log.passed()
        self._last_order_ts[runner_id] = current_sec
        self._order_timestamps.append(current_sec)
        self._position_sizes[runner_id] = current_pos + config.base_stake
        return True

    def record_fill(self, runner_id: str, size: float) -> None:
        """Update position tracking after a fill comes back."""
        # Position already counted at submission; fills confirm it.
        pass

    def reset(self) -> None:
        """Reset state for a new session."""
        self.rejection_log = RejectionLog()
        self._last_order_ts.clear()
        self._order_timestamps.clear()
        self._position_sizes.clear()


@dataclass
class PrematchMarketMakingConfig:
    """Conservative pre-match two-sided market-making strategy.

    Posts back and lay orders around fair price with a spread.
    Captures the spread if both sides get filled.
    """

    # Spread parameters
    half_spread_bps: float = 150.0  # half-spread on each side
    min_edge_bps: float = 50.0     # min edge to even consider quoting

    # Inventory management
    max_inventory_imbalance: float = 200.0  # max net position
    skew_bps_per_unit: float = 2.0  # skew quotes when position builds

    # Order size
    quote_size: float = 40.0

    # Only active pre-match
    stop_at_inplay: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}
