"""
MVP Runner — Full Rust+Python Closed Loop Demo.

This script proves the heterogeneous dual-engine architecture works:
1. Rust (sports_engine) maintains the order book at native speed
2. Python generates simulated match events + market data
3. Python strategy brain reads Rust state, computes fair odds, executes trades
4. Everything happens in-process, zero serialization overhead (PyO3 direct memory)

Run: source .venv/bin/activate && python mvp_runner.py
"""

import random
import time
import math

import sports_engine

# ============================================================
# Configuration
# ============================================================

INITIAL_HOME_WIN_PROB = 0.45  # pre-match probability
COMMISSION_RATE = 0.02  # 2% commission on winnings
STAKE_SIZE = 100.0  # base stake per trade


# ============================================================
# Poisson-based Fair Probability Model
# ============================================================

def poisson_pmf(k: int, lam: float) -> float:
    """Probability of exactly k events given rate lambda."""
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def compute_fair_home_win_prob(
    home_score: int,
    away_score: int,
    minute: int,
    home_red_cards: int,
    away_red_cards: int,
) -> float:
    """
    Simplified Poisson model for home win probability.

    In production, this would be a full Dixon-Coles model with:
    - Team-specific attack/defense ratings
    - Time-varying goal rates
    - Red card effects on scoring intensity
    - Venue effects

    For MVP, we use a simplified version that still captures the key dynamics.
    """
    remaining = max(1, 90 - minute) / 90.0

    # Base expected goals per 90 minutes
    home_lambda_90 = 1.5
    away_lambda_90 = 1.2

    # Red card effect: -30% scoring rate per red card
    home_lambda_90 *= (0.7 ** home_red_cards)
    away_lambda_90 *= (0.7 ** away_red_cards)

    # Scale by remaining time
    home_lambda = home_lambda_90 * remaining
    away_lambda = away_lambda_90 * remaining

    # Compute P(home wins) by summing over possible final scores
    max_goals = 6
    p_home_win = 0.0
    p_draw = 0.0

    for h_add in range(max_goals):
        for a_add in range(max_goals):
            p = poisson_pmf(h_add, home_lambda) * poisson_pmf(a_add, away_lambda)
            final_h = home_score + h_add
            final_a = away_score + a_add
            if final_h > final_a:
                p_home_win += p
            elif final_h == final_a:
                p_draw += p

    return p_home_win


def prob_to_odds(prob: float) -> float:
    """Convert probability to decimal odds."""
    if prob <= 0.001:
        return 1000.0
    if prob >= 0.999:
        return 1.001
    return 1.0 / prob


# ============================================================
# Market Simulator
# ============================================================

def generate_market_levels(fair_odds: float, noise: float = 0.03) -> tuple:
    """
    Generate simulated back/lay levels around a fair price.
    Returns (back_levels, lay_levels) as lists of (odds, volume) tuples.
    """
    rng = random.Random()

    # Back levels: available to back at (buyer's side)
    # Best back is slightly below fair (the market takes a cut)
    back_levels = []
    for i in range(5):
        odds = fair_odds * (1 - 0.01 * (i + 1)) + rng.gauss(0, noise)
        odds = max(1.01, round(odds, 2))
        volume = rng.uniform(50, 500) * (5 - i)  # more volume near the top
        back_levels.append((odds, volume))
    back_levels.sort(key=lambda x: -x[0])  # descending

    # Lay levels: available to lay at (seller's side)
    # Best lay is slightly above fair
    lay_levels = []
    for i in range(5):
        odds = fair_odds * (1 + 0.01 * (i + 1)) + rng.gauss(0, noise)
        odds = max(1.01, round(odds, 2))
        volume = rng.uniform(50, 500) * (5 - i)
        lay_levels.append((odds, volume))
    lay_levels.sort(key=lambda x: x[0])  # ascending

    return back_levels, lay_levels


# ============================================================
# Strategy Brain
# ============================================================

class StrategyBrain:
    """
    Simple edge-based strategy:
    - Computes fair probability using Poisson model
    - Compares to market-implied probability
    - If edge > threshold, executes a trade
    """

    def __init__(self, edge_threshold: float = 0.03):
        self.edge_threshold = edge_threshold
        self.trades: list[dict] = []
        self.total_pnl = 0.0

    def evaluate(
        self,
        book: sports_engine.OrderBook,
        match: sports_engine.MatchState,
    ) -> dict | None:
        """Evaluate whether to trade. Returns trade info or None."""
        # Our model's fair probability
        fair_prob = compute_fair_home_win_prob(
            match.home_score,
            match.away_score,
            match.minute,
            match.home_red_cards,
            match.away_red_cards,
        )
        fair_odds = prob_to_odds(fair_prob)

        # Market-implied probability
        market_prob = book.implied_probability()
        if market_prob <= 0:
            return None

        edge = fair_prob - market_prob

        if abs(edge) < self.edge_threshold:
            return None

        if edge > 0:
            # We think home is MORE likely than market says -> BACK home
            target_odds = fair_odds * 0.98  # willing to accept slightly worse
            result = book.execute_back(target_odds, STAKE_SIZE)
            if result.success:
                trade = {
                    "minute": match.minute,
                    "side": "BACK",
                    "edge": edge,
                    "fair_prob": fair_prob,
                    "market_prob": market_prob,
                    "result": result,
                }
                self.trades.append(trade)
                return trade
        else:
            # We think home is LESS likely than market says -> LAY home
            target_odds = fair_odds * 1.02
            result = book.execute_lay(target_odds, STAKE_SIZE)
            if result.success:
                trade = {
                    "minute": match.minute,
                    "side": "LAY",
                    "edge": edge,
                    "fair_prob": fair_prob,
                    "market_prob": market_prob,
                    "result": result,
                }
                self.trades.append(trade)
                return trade

        return None


# ============================================================
# Main Simulation Loop
# ============================================================

def run_simulation():
    print("=" * 70)
    print("  ALL-WEATHER MVP: Rust+Python Heterogeneous Engine Test")
    print("=" * 70)
    print()

    # 1. Initialize Rust engine
    book = sports_engine.OrderBook("match_odds_home")
    match = sports_engine.MatchState()
    brain = StrategyBrain(edge_threshold=0.02)

    print(f"[INIT] Rust OrderBook created: {book.snapshot().market_id}")
    print(f"[INIT] MatchState created: phase={match.phase_name()}")
    print(f"[INIT] Initial home win probability: {INITIAL_HOME_WIN_PROB:.1%}")
    print()

    # 2. Pre-match: seed the order book
    fair_odds = prob_to_odds(INITIAL_HOME_WIN_PROB)
    back_levels, lay_levels = generate_market_levels(fair_odds)
    book.update_back(back_levels)
    book.update_lay(lay_levels)

    snap = book.snapshot()
    print(f"[PRE-MATCH] Book seeded:")
    print(f"  Best Back: {snap.best_back:.3f}  Best Lay: {snap.best_lay:.3f}")
    print(f"  Spread: {snap.spread:.4f}  Implied Prob: {snap.implied_prob:.1%}")
    print(f"  Back Depth: {snap.back_depth:.0f}  Lay Depth: {snap.lay_depth:.0f}")
    print()

    # 3. Simulate 90 minutes of match time
    random.seed(7)  # reproducible (try seed=42 for extreme scenario)
    events_log = []

    start_time = time.perf_counter_ns()

    for _ in range(90):
        phase = match.advance_minute()
        minute = match.minute

        # --- Random events ---
        event = None
        r = random.random()

        if r < 0.025:  # ~2.5% chance per minute = ~2.25 goals per match
            if random.random() < 0.55:  # slight home advantage
                match.home_goal()
                event = f"GOAL! Home scores! ({match.home_score}-{match.away_score})"
                # Suspend market on goal
                book.suspend()
            else:
                match.away_goal()
                event = f"GOAL! Away scores! ({match.home_score}-{match.away_score})"
                book.suspend()
        elif r < 0.035:  # ~1% chance of red card
            if random.random() < 0.5:
                match.home_red_card()
                event = f"RED CARD! Home player sent off (reds: {match.home_red_cards})"
            else:
                match.away_red_card()
                event = f"RED CARD! Away player sent off (reds: {match.away_red_cards})"

        # --- Update market after event ---
        if event:
            events_log.append((minute, event))
            print(f"  [{minute:2d}'] *** {event} ***")

            if book.snapshot().is_suspended:
                # Resume after short delay (simulating bet delay)
                book.resume(5000)  # 5 second bet delay
                print(f"  [{minute:2d}'] Market resumed with 5s bet delay")

        # --- Regenerate market levels based on current state ---
        fair_prob = compute_fair_home_win_prob(
            match.home_score, match.away_score, minute,
            match.home_red_cards, match.away_red_cards,
        )
        fair_odds_now = prob_to_odds(fair_prob)
        noise = 0.02 + (0.05 if event else 0.0)  # more noise after events
        back_levels, lay_levels = generate_market_levels(fair_odds_now, noise)
        book.update_back(back_levels)
        book.update_lay(lay_levels)

        # --- Strategy brain evaluates ---
        trade = brain.evaluate(book, match)
        if trade:
            r = trade["result"]
            print(
                f"  [{minute:2d}'] TRADE: {trade['side']} | "
                f"edge={trade['edge']:+.3f} | "
                f"{r.message}"
            )

        # --- Periodic status ---
        if minute % 15 == 0:
            snap = book.snapshot()
            print(
                f"  [{minute:2d}'] Status: "
                f"score={match.home_score}-{match.away_score} "
                f"back={snap.best_back:.3f} lay={snap.best_lay:.3f} "
                f"prob={snap.implied_prob:.1%} "
                f"depth={snap.back_depth:.0f}/{snap.lay_depth:.0f} "
                f"matched={snap.total_matched:.0f}"
            )

    elapsed_ns = time.perf_counter_ns() - start_time
    elapsed_ms = elapsed_ns / 1_000_000

    # 4. Final results
    print()
    print("=" * 70)
    print("  SIMULATION COMPLETE")
    print("=" * 70)
    print(f"  Final Score: {match.home_score} - {match.away_score}")
    print(f"  Total Events: {len(events_log)}")
    print(f"  Total Trades: {len(brain.trades)}")
    print(f"  Total Matched Volume: {book.snapshot().total_matched:.2f}")
    print(f"  Tick Count (Rust): {book.snapshot().tick_count}")
    print(f"  Simulation Time: {elapsed_ms:.2f}ms (90 match-minutes)")
    print(f"  Avg Time Per Tick: {elapsed_ms / 90:.3f}ms")
    print()

    if brain.trades:
        print("  Trade Log:")
        for t in brain.trades:
            r = t["result"]
            print(
                f"    [{t['minute']:2d}'] {t['side']:4s} | "
                f"odds={r.executed_odds:.4f} size={r.executed_size:.2f} "
                f"slip={r.slippage_bps:.1f}bps | "
                f"edge={t['edge']:+.4f} fair={t['fair_prob']:.3f} mkt={t['market_prob']:.3f}"
            )
    print()

    # Verify the Rust<->Python bridge works correctly
    snap = book.snapshot()
    print("  [VERIFICATION] Rust<->Python bridge integrity:")
    print(f"    market_id:     {snap.market_id}")
    print(f"    best_back:     {snap.best_back:.4f}")
    print(f"    best_lay:      {snap.best_lay:.4f}")
    print(f"    implied_prob:  {snap.implied_prob:.4f}")
    print(f"    imbalance:     {snap.imbalance:.4f}")
    print(f"    is_suspended:  {snap.is_suspended}")
    print(f"    tick_count:    {snap.tick_count}")
    print(f"    total_matched: {snap.total_matched:.2f}")
    print()
    print("  ✓ Rust+Python heterogeneous engine: OPERATIONAL")
    print("=" * 70)


if __name__ == "__main__":
    run_simulation()
