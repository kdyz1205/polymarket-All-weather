"""
Auto signal runner — scan cached markets, price, compare, generate orders.

Pipeline (fully automated):
  1. Read MarketCache → get today's active markets
  2. Read TeamCache + PlayerCache → build GameData for each
  3. Price each game (RealGamePricer: Log5 + injuries + HCA)
  4. Compare fair prob vs market price → find edge
  5. Pre-trade check (traffic light: GREEN/YELLOW/RED)
  6. Generate and queue actionable orders
  7. Save features to FeatureStore for learning

Execution modes:
  - "semi-auto": queue orders for manual confirmation (default)
  - "full-auto": execute immediately if conditions met (future)

Usage:
  python -m src.data.auto_signal_runner              # one-shot
  python -m src.data.auto_signal_runner --loop 300   # every 5 min
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime
from dataclasses import dataclass, asdict

from src.data.cache import (
    MarketCache, CachedMarket,
    TeamCache, CachedTeam,
    PlayerCache, CachedPlayer,
    OrderbookHistory,
    FeatureStore, GameFeatures,
)
from src.data.nba_client import (
    GameData, TeamGameData, PlayerInjury,
    estimate_player_impact,
)
from src.pricing.real_game_pricer import RealGamePricer, PricingResult
from real_signal_generator import RealSignal, generate_real_signals

from live_observer import (
    MicroLiveState, load_state, save_state, PendingOrder,
    queue_order, MAX_STAKE_PER_ORDER,
)

logger = logging.getLogger(__name__)

# Minimum edge (bps) to even consider a signal
MIN_RAW_EDGE_BPS = 50
# Fee assumption (roundtrip) for Polymarket
DEFAULT_FEE_BPS = 80.0
# Minimum net edge after fees to generate an actionable signal
MIN_NET_EDGE_BPS = 20.0
# Minimum confidence to act
MIN_CONFIDENCE = 0.50

SIGNALS_LOG = "micro_live_logs/real_signals.jsonl"


@dataclass
class ScanResult:
    """Result of scanning a single market."""
    slug: str
    home_team: str
    away_team: str
    fair_home_prob: float
    market_home_price: float
    edge_home_bps: float
    edge_away_bps: float
    best_side: str
    best_edge_bps: float
    net_edge_bps: float
    confidence: float
    signal_strength: str
    actionable: bool
    queued: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class AutoSignalRunner:
    """Scans all cached markets, prices them, generates signals."""

    def __init__(
        self,
        fee_bps: float = DEFAULT_FEE_BPS,
        min_net_edge: float = MIN_NET_EDGE_BPS,
        min_confidence: float = MIN_CONFIDENCE,
    ):
        self.fee_bps = fee_bps
        self.min_net_edge = min_net_edge
        self.min_confidence = min_confidence

        self._market_cache = MarketCache()
        self._team_cache = TeamCache()
        self._player_cache = PlayerCache()
        self._ob_history = OrderbookHistory()
        self._feature_store = FeatureStore()
        self._pricer = RealGamePricer()

    def scan_all(self, game_date: str | None = None) -> list[ScanResult]:
        """Scan all active markets for today, return results."""
        today = game_date or date.today().isoformat()
        markets = self._market_cache.get_today(today)

        if not markets:
            logger.info("No active markets for %s", today)
            return []

        results: list[ScanResult] = []
        for market in markets:
            try:
                result = self._scan_market(market)
                results.append(result)
            except Exception as e:
                logger.error("Failed to scan %s: %s", market.slug, e)
                results.append(ScanResult(
                    slug=market.slug,
                    home_team=market.home_team,
                    away_team=market.away_team,
                    fair_home_prob=0, market_home_price=market.home_price,
                    edge_home_bps=0, edge_away_bps=0,
                    best_side="none", best_edge_bps=0, net_edge_bps=0,
                    confidence=0, signal_strength="error",
                    actionable=False, reason=str(e),
                ))

        return results

    def scan_and_queue(self, game_date: str | None = None) -> list[ScanResult]:
        """Scan all markets and queue actionable signals.

        Returns all scan results (actionable ones will have queued=True).
        """
        results = self.scan_all(game_date)
        state = load_state()

        if state.kill_switch:
            logger.warning("Kill switch active: %s", state.kill_reason)
            for r in results:
                r.actionable = False
                r.reason = f"kill switch: {state.kill_reason}"
            return results

        for result in results:
            if not result.actionable:
                continue

            # Find the market in cache for token info
            market = self._market_cache.get(result.slug)
            if not market:
                result.reason = "market not in cache"
                result.actionable = False
                continue

            # Determine which side to trade
            side = result.best_side
            token_id = (market.home_token_id if side == "home"
                        else market.away_token_id)

            if not token_id:
                result.reason = "no token_id"
                result.actionable = False
                continue

            # Build and queue the order
            fair_odds = (1.0 / result.fair_home_prob if side == "home"
                         else 1.0 / (1.0 - result.fair_home_prob))

            order = PendingOrder(
                order_id=f"auto_{result.slug}_{side}_{int(time.time())}",
                timestamp=datetime.now().isoformat(),
                sport="basketball",
                market_id=result.slug,
                runner_id=side,
                side="buy",
                price=fair_odds,
                size=min(1.0, MAX_STAKE_PER_ORDER),
                edge_bps=result.best_edge_bps,
                net_edge_bps=result.net_edge_bps,
                fee_bps=self.fee_bps,
            )

            if queue_order(order, state):
                result.queued = True
            else:
                result.reason = "blocked by safety limits"

        save_state(state)
        return results

    def _scan_market(self, market: CachedMarket) -> ScanResult:
        """Scan a single market: build game data → price → compare."""
        # Build GameData from caches
        game = self._build_game_from_caches(market)

        # Price the game
        pricing = self._pricer.price(game)

        # Generate signals
        signals = generate_real_signals(game, pricing, fee_bps=self.fee_bps)

        # Find best side
        best_signal = max(signals, key=lambda s: s.net_edge_bps)
        actionable_signals = [s for s in signals if s.actionable]

        # Determine if actionable
        actionable = (
            best_signal.net_edge_bps >= self.min_net_edge
            and pricing.confidence >= self.min_confidence
            and best_signal.actionable
        )

        # Save features for learning
        self._save_features(market, pricing)

        return ScanResult(
            slug=market.slug,
            home_team=market.home_team,
            away_team=market.away_team,
            fair_home_prob=pricing.fair_home_prob,
            market_home_price=pricing.market_home_price,
            edge_home_bps=pricing.edge_home_bps,
            edge_away_bps=pricing.edge_away_bps,
            best_side=pricing.best_side,
            best_edge_bps=pricing.best_edge_bps,
            net_edge_bps=best_signal.net_edge_bps,
            confidence=pricing.confidence,
            signal_strength=best_signal.signal_strength,
            actionable=actionable,
        )

    def _build_game_from_caches(self, market: CachedMarket) -> GameData:
        """Assemble GameData from MarketCache + TeamCache + PlayerCache."""
        # Get team data from cache
        home_cached = self._team_cache.get_or_default(
            market.home_team, market.home_team
        )
        away_cached = self._team_cache.get_or_default(
            market.away_team, market.away_team
        )

        # Get injuries from player cache
        home_injuries = self._player_cache.get_team_injuries(market.home_team)
        away_injuries = self._player_cache.get_team_injuries(market.away_team)

        home_team = TeamGameData(
            abbreviation=market.home_team,
            full_name=home_cached.full_name or market.home_team,
            wins=home_cached.wins,
            losses=home_cached.losses,
            is_home=True,
            injuries=[
                PlayerInjury(
                    name=p.name,
                    team=p.team,
                    status=p.injury_status,
                    injury=p.injury_detail,
                    impact_rating=p.impact_rating,
                )
                for p in home_injuries
            ],
            offensive_rating=home_cached.offensive_rating,
            defensive_rating=home_cached.defensive_rating,
            pace=home_cached.pace,
            streak=home_cached.streak,
            last5_record=home_cached.last5,
        )

        away_team = TeamGameData(
            abbreviation=market.away_team,
            full_name=away_cached.full_name or market.away_team,
            wins=away_cached.wins,
            losses=away_cached.losses,
            is_home=False,
            injuries=[
                PlayerInjury(
                    name=p.name,
                    team=p.team,
                    status=p.injury_status,
                    injury=p.injury_detail,
                    impact_rating=p.impact_rating,
                )
                for p in away_injuries
            ],
            offensive_rating=away_cached.offensive_rating,
            defensive_rating=away_cached.defensive_rating,
            pace=away_cached.pace,
            streak=away_cached.streak,
            last5_record=away_cached.last5,
        )

        return GameData(
            game_id=market.slug,
            game_date=market.game_date,
            home_team=home_team,
            away_team=away_team,
            polymarket_slug=market.slug,
            polymarket_condition_id=market.condition_id,
            polymarket_home_token=market.home_token_id,
            polymarket_away_token=market.away_token_id,
            polymarket_home_price=market.home_price,
            polymarket_away_price=market.away_price,
        )

    def _save_features(self, market: CachedMarket, pricing: PricingResult) -> None:
        """Save feature vector for this game for future learning."""
        features = GameFeatures(
            slug=market.slug,
            game_date=market.game_date,
            timestamp=datetime.now().isoformat(),
            home_team=market.home_team,
            away_team=market.away_team,
            rating_diff=0.0,  # would need ELO
            win_pct_diff=pricing.home_win_pct - pricing.away_win_pct,
            home_injury_impact=pricing.home_injury_adj,
            away_injury_impact=pricing.away_injury_adj,
            injury_diff=pricing.away_injury_adj - pricing.home_injury_adj,
            home_court=pricing.home_court_adj,
            market_skew=pricing.edge_home_pct,
            fair_home_prob=pricing.fair_home_prob,
            market_home_prob=pricing.market_home_price,
            edge_home_bps=pricing.edge_home_bps,
            edge_away_bps=pricing.edge_away_bps,
            net_edge_bps=max(pricing.edge_home_bps, pricing.edge_away_bps) - self.fee_bps,
            signal_strength="strong" if pricing.best_edge_bps > 200 else
                            "moderate" if pricing.best_edge_bps > 100 else
                            "weak" if pricing.best_edge_bps > 50 else "none",
            confidence=pricing.confidence,
        )
        self._feature_store.save_features(features)

    def _log_signals(self, signals: list[RealSignal]) -> None:
        """Append signals to the log."""
        os.makedirs(os.path.dirname(SIGNALS_LOG) or ".", exist_ok=True)
        with open(SIGNALS_LOG, "a") as f:
            for s in signals:
                f.write(json.dumps(s.to_dict()) + "\n")


def run_scanner(loop_sec: int = 0, auto_queue: bool = True) -> None:
    """Run the auto signal scanner."""
    runner = AutoSignalRunner()

    while True:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n{'='*70}")
        print(f"  AUTO SIGNAL SCANNER — {ts}")
        print(f"{'='*70}")

        try:
            if auto_queue:
                results = runner.scan_and_queue()
            else:
                results = runner.scan_all()

            if not results:
                print("  No markets to scan.")
            else:
                actionable_count = sum(1 for r in results if r.actionable)
                queued_count = sum(1 for r in results if r.queued)

                print(f"\n  Scanned {len(results)} markets:")
                for r in results:
                    icon = ">>>" if r.actionable else "   "
                    q_icon = " [QUEUED]" if r.queued else ""
                    print(f"    {icon} {r.away_team}@{r.home_team}: "
                          f"fair={r.fair_home_prob:.1%} mkt={r.market_home_price:.1%} "
                          f"edge={r.best_edge_bps:+.0f}bps net={r.net_edge_bps:+.0f}bps "
                          f"{r.signal_strength} conf={r.confidence:.0%}"
                          f"{q_icon}")
                    if r.reason:
                        print(f"          reason: {r.reason}")

                print(f"\n  Summary: {actionable_count} actionable, "
                      f"{queued_count} queued")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

        if loop_sec <= 0:
            break

        print(f"\n  Next scan in {loop_sec}s...")
        time.sleep(loop_sec)


if __name__ == "__main__":
    import sys
    loop = 0
    no_queue = False

    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--loop" and i < len(sys.argv) - 1:
            loop = int(sys.argv[i + 1])
        elif arg == "--dry-run":
            no_queue = True

    run_scanner(loop_sec=loop, auto_queue=not no_queue)
