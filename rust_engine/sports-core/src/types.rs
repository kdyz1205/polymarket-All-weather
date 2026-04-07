//! Core data types for the sports trading system.
//! Section 4 of the blueprint: Market, Runner, MatchState, LiveOrder, FairState.

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use crate::enums::*;

// ============================================================
// Tick ladder constants
// Betfair-style: ~350 possible price ticks from 1.01 to 1000
// We use a fixed array index (price_idx) everywhere for O(1) access.
// ============================================================

pub const MAX_LADDER_SIZE: usize = 400;
pub const NUM_LADDER_LEVELS: usize = 5; // visible depth levels

// ============================================================
// Section 4.1: Market
// ============================================================

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Market {
    #[pyo3(get, set)]
    pub market_id: String,
    #[pyo3(get, set)]
    pub sport: Sport,
    #[pyo3(get, set)]
    pub competition_id: String,
    #[pyo3(get, set)]
    pub event_id: String,
    #[pyo3(get, set)]
    pub market_type: String, // "match_odds", "moneyline", "spread", "totals"
    #[pyo3(get, set)]
    pub start_time_ms: i64,
    #[pyo3(get, set)]
    pub status: MarketStatus,
    #[pyo3(get, set)]
    pub in_play: bool,
    #[pyo3(get, set)]
    pub suspend_reason: SuspendReason,
    #[pyo3(get, set)]
    pub bet_delay_ms: u64,
    #[pyo3(get, set)]
    pub turn_in_play_time_ms: i64,
    #[pyo3(get, set)]
    pub settled_time_ms: i64,
    #[pyo3(get, set)]
    pub version: u64,
}

#[pymethods]
impl Market {
    #[new]
    pub fn new(market_id: String, sport: Sport, event_id: String, market_type: String) -> Self {
        Market {
            market_id,
            sport,
            competition_id: String::new(),
            event_id,
            market_type,
            start_time_ms: 0,
            status: MarketStatus::PreOpen,
            in_play: false,
            suspend_reason: SuspendReason::None,
            bet_delay_ms: 0,
            turn_in_play_time_ms: 0,
            settled_time_ms: 0,
            version: 0,
        }
    }

    pub fn is_tradeable(&self) -> bool {
        matches!(
            self.status,
            MarketStatus::OpenPrematch | MarketStatus::InPlay | MarketStatus::Reopened
        )
    }

    pub fn is_suspended(&self) -> bool {
        self.status == MarketStatus::Suspended
    }

    pub fn is_terminal(&self) -> bool {
        matches!(
            self.status,
            MarketStatus::Closed | MarketStatus::Settled | MarketStatus::Void
        )
    }
}

// ============================================================
// Section 4.2: Runner (single outcome in a market)
// For Match Odds / Moneyline: Home, Away (+ Draw for football)
// ============================================================

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LadderLevel {
    #[pyo3(get, set)]
    pub price_idx: u16,
    #[pyo3(get, set)]
    pub price: f64,
    #[pyo3(get, set)]
    pub volume: f64,
}

#[pymethods]
impl LadderLevel {
    #[new]
    pub fn new(price_idx: u16, price: f64, volume: f64) -> Self {
        LadderLevel {
            price_idx,
            price,
            volume,
        }
    }
}

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Runner {
    #[pyo3(get, set)]
    pub runner_id: String,
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub status: String, // "ACTIVE", "WINNER", "LOSER", "REMOVED"
    #[pyo3(get, set)]
    pub traded_volume_total: f64,
    // Best available prices
    #[pyo3(get, set)]
    pub best_back_price: f64,
    #[pyo3(get, set)]
    pub best_back_size: f64,
    #[pyo3(get, set)]
    pub best_lay_price: f64,
    #[pyo3(get, set)]
    pub best_lay_size: f64,
    #[pyo3(get, set)]
    pub last_traded_price: f64,
    // Full depth ladders
    pub atb_ladder: Vec<LadderLevel>, // available to back (best = highest price first)
    pub atl_ladder: Vec<LadderLevel>, // available to lay (best = lowest price first)
    pub trd_ladder: Vec<LadderLevel>, // traded volume at each price
}

#[pymethods]
impl Runner {
    #[new]
    pub fn new(runner_id: String, name: String) -> Self {
        Runner {
            runner_id,
            name,
            status: "ACTIVE".to_string(),
            traded_volume_total: 0.0,
            best_back_price: 0.0,
            best_back_size: 0.0,
            best_lay_price: 0.0,
            best_lay_size: 0.0,
            last_traded_price: 0.0,
            atb_ladder: Vec::new(),
            atl_ladder: Vec::new(),
            trd_ladder: Vec::new(),
        }
    }

    pub fn update_back_ladder(&mut self, levels: Vec<(f64, f64)>) {
        self.atb_ladder = levels
            .iter()
            .enumerate()
            .map(|(i, &(price, volume))| LadderLevel {
                price_idx: i as u16,
                price,
                volume,
            })
            .collect();
        if let Some(best) = self.atb_ladder.first() {
            self.best_back_price = best.price;
            self.best_back_size = best.volume;
        }
    }

    pub fn update_lay_ladder(&mut self, levels: Vec<(f64, f64)>) {
        self.atl_ladder = levels
            .iter()
            .enumerate()
            .map(|(i, &(price, volume))| LadderLevel {
                price_idx: i as u16,
                price,
                volume,
            })
            .collect();
        if let Some(best) = self.atl_ladder.first() {
            self.best_lay_price = best.price;
            self.best_lay_size = best.volume;
        }
    }

    pub fn spread(&self) -> f64 {
        if self.best_lay_price > 0.0 && self.best_back_price > 0.0 {
            self.best_lay_price - self.best_back_price
        } else {
            f64::MAX
        }
    }

    pub fn implied_probability(&self) -> f64 {
        if self.best_back_price > 0.0 && self.best_lay_price > 0.0 {
            let mid = (self.best_back_price + self.best_lay_price) / 2.0;
            1.0 / mid
        } else {
            0.0
        }
    }

    pub fn back_depth(&self) -> f64 {
        self.atb_ladder.iter().map(|l| l.volume).sum()
    }

    pub fn lay_depth(&self) -> f64 {
        self.atl_ladder.iter().map(|l| l.volume).sum()
    }

    pub fn imbalance(&self) -> f64 {
        let bd = self.back_depth();
        let ld = self.lay_depth();
        let total = bd + ld;
        if total < 1e-10 {
            return 0.0;
        }
        (bd - ld) / total
    }
}

// ============================================================
// Section 4.3: LiveOrder
// ============================================================

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LiveOrder {
    #[pyo3(get, set)]
    pub local_order_id: u64,
    #[pyo3(get, set)]
    pub exchange_order_id: String,
    #[pyo3(get, set)]
    pub market_id: String,
    #[pyo3(get, set)]
    pub runner_id: String,
    #[pyo3(get, set)]
    pub side: Side,
    #[pyo3(get, set)]
    pub price: f64,
    #[pyo3(get, set)]
    pub size: f64,
    #[pyo3(get, set)]
    pub remaining_size: f64,
    #[pyo3(get, set)]
    pub matched_size: f64,
    #[pyo3(get, set)]
    pub avg_matched_price: f64,
    #[pyo3(get, set)]
    pub status: OrderStatus,
    #[pyo3(get, set)]
    pub submit_ts_ms: i64,
    #[pyo3(get, set)]
    pub accepted_ts_ms: i64,
    #[pyo3(get, set)]
    pub delay_expire_ts_ms: i64,
    #[pyo3(get, set)]
    pub strategy_tag: String,
}

#[pymethods]
impl LiveOrder {
    #[new]
    pub fn new(
        local_order_id: u64,
        market_id: String,
        runner_id: String,
        side: Side,
        price: f64,
        size: f64,
        strategy_tag: String,
    ) -> Self {
        LiveOrder {
            local_order_id,
            exchange_order_id: String::new(),
            market_id,
            runner_id,
            side,
            price,
            size,
            remaining_size: size,
            matched_size: 0.0,
            avg_matched_price: 0.0,
            status: OrderStatus::Created,
            submit_ts_ms: 0,
            accepted_ts_ms: 0,
            delay_expire_ts_ms: 0,
            strategy_tag,
        }
    }

    pub fn is_active(&self) -> bool {
        matches!(
            self.status,
            OrderStatus::Live | OrderStatus::PartiallyMatched | OrderStatus::DelayPending
        )
    }

    pub fn is_terminal(&self) -> bool {
        matches!(
            self.status,
            OrderStatus::FullyMatched
                | OrderStatus::Cancelled
                | OrderStatus::Replaced
                | OrderStatus::Rejected
                | OrderStatus::Lapsed
        )
    }

    pub fn fill_pct(&self) -> f64 {
        if self.size > 0.0 {
            self.matched_size / self.size
        } else {
            0.0
        }
    }
}

// ============================================================
// Section 4.4: MatchState — sport-aware
// ============================================================

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MatchState {
    #[pyo3(get, set)]
    pub event_id: String,
    #[pyo3(get, set)]
    pub sport: Sport,
    #[pyo3(get, set)]
    pub phase: MatchPhase,
    #[pyo3(get, set)]
    pub match_clock_sec: f64,
    // Scores
    #[pyo3(get, set)]
    pub home_score: u32,
    #[pyo3(get, set)]
    pub away_score: u32,
    // Basketball-specific
    #[pyo3(get, set)]
    pub home_q1: u32,
    #[pyo3(get, set)]
    pub home_q2: u32,
    #[pyo3(get, set)]
    pub home_q3: u32,
    #[pyo3(get, set)]
    pub home_q4: u32,
    #[pyo3(get, set)]
    pub away_q1: u32,
    #[pyo3(get, set)]
    pub away_q2: u32,
    #[pyo3(get, set)]
    pub away_q3: u32,
    #[pyo3(get, set)]
    pub away_q4: u32,
    #[pyo3(get, set)]
    pub home_fouls: u32,
    #[pyo3(get, set)]
    pub away_fouls: u32,
    #[pyo3(get, set)]
    pub home_timeouts_remaining: u32,
    #[pyo3(get, set)]
    pub away_timeouts_remaining: u32,
    #[pyo3(get, set)]
    pub shot_clock_sec: f64,
    // Baseball-specific
    #[pyo3(get, set)]
    pub inning: u32,
    #[pyo3(get, set)]
    pub is_top_inning: bool,
    #[pyo3(get, set)]
    pub outs: u32,
    #[pyo3(get, set)]
    pub balls: u32,
    #[pyo3(get, set)]
    pub strikes: u32,
    #[pyo3(get, set)]
    pub runner_on_first: bool,
    #[pyo3(get, set)]
    pub runner_on_second: bool,
    #[pyo3(get, set)]
    pub runner_on_third: bool,
    #[pyo3(get, set)]
    pub home_hits: u32,
    #[pyo3(get, set)]
    pub away_hits: u32,
    #[pyo3(get, set)]
    pub home_errors: u32,
    #[pyo3(get, set)]
    pub away_errors: u32,
    // Football-specific
    #[pyo3(get, set)]
    pub home_red_cards: u32,
    #[pyo3(get, set)]
    pub away_red_cards: u32,
    #[pyo3(get, set)]
    pub home_corners: u32,
    #[pyo3(get, set)]
    pub away_corners: u32,
    // Universal
    #[pyo3(get, set)]
    pub last_event_ts_ms: i64,
}

#[pymethods]
impl MatchState {
    #[new]
    pub fn new(event_id: String, sport: Sport) -> Self {
        let timeouts = match sport {
            Sport::Basketball => 7, // NBA full timeouts
            _ => 0,
        };
        MatchState {
            event_id,
            sport,
            phase: MatchPhase::NotStarted,
            match_clock_sec: 0.0,
            home_score: 0,
            away_score: 0,
            home_q1: 0, home_q2: 0, home_q3: 0, home_q4: 0,
            away_q1: 0, away_q2: 0, away_q3: 0, away_q4: 0,
            home_fouls: 0,
            away_fouls: 0,
            home_timeouts_remaining: timeouts,
            away_timeouts_remaining: timeouts,
            shot_clock_sec: 24.0,
            inning: 0,
            is_top_inning: true,
            outs: 0,
            balls: 0,
            strikes: 0,
            runner_on_first: false,
            runner_on_second: false,
            runner_on_third: false,
            home_hits: 0,
            away_hits: 0,
            home_errors: 0,
            away_errors: 0,
            home_red_cards: 0,
            away_red_cards: 0,
            home_corners: 0,
            away_corners: 0,
            last_event_ts_ms: 0,
        }
    }

    pub fn is_live(&self) -> bool {
        !matches!(
            self.phase,
            MatchPhase::NotStarted | MatchPhase::Finished | MatchPhase::Abandoned
        )
    }

    pub fn is_finished(&self) -> bool {
        matches!(self.phase, MatchPhase::Finished | MatchPhase::Abandoned)
    }

    pub fn score_diff(&self) -> i32 {
        self.home_score as i32 - self.away_score as i32
    }

    /// Estimated remaining game time in seconds.
    pub fn remaining_sec(&self) -> f64 {
        match self.sport {
            Sport::Basketball => {
                // NBA: 4x12min = 48min = 2880sec
                let total = 2880.0;
                (total - self.match_clock_sec).max(0.0)
            }
            Sport::Baseball => {
                // Baseball has no clock; estimate by outs remaining
                // 9 innings x 6 outs = 54 total outs; rough estimate
                let total_outs = 54.0;
                let current_outs =
                    (self.inning.saturating_sub(1) as f64 * 6.0)
                    + if self.is_top_inning { 0.0 } else { 3.0 }
                    + self.outs as f64;
                let remaining_outs = (total_outs - current_outs).max(0.0);
                remaining_outs * 45.0 // ~45 sec per out avg
            }
            Sport::Football => {
                let total = 5400.0; // 90 min
                (total - self.match_clock_sec).max(0.0)
            }
        }
    }

    /// Base runner count (baseball).
    pub fn runners_on_base(&self) -> u32 {
        self.runner_on_first as u32
            + self.runner_on_second as u32
            + self.runner_on_third as u32
    }
}

// ============================================================
// Section 4.5: FairState (model output)
// ============================================================

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FairState {
    #[pyo3(get, set)]
    pub p_home: f64,
    #[pyo3(get, set)]
    pub p_away: f64,
    #[pyo3(get, set)]
    pub p_draw: f64, // 0.0 for basketball/baseball (no draw)
    #[pyo3(get, set)]
    pub fair_odds_home: f64,
    #[pyo3(get, set)]
    pub fair_odds_away: f64,
    #[pyo3(get, set)]
    pub fair_odds_draw: f64,
    #[pyo3(get, set)]
    pub uncertainty: f64,
    #[pyo3(get, set)]
    pub model_regime: ModelRegime,
    #[pyo3(get, set)]
    pub signal_strength: f64,
    #[pyo3(get, set)]
    pub timestamp_ms: i64,
}

#[pymethods]
impl FairState {
    #[new]
    pub fn new(p_home: f64, p_away: f64, p_draw: f64) -> Self {
        let oh = if p_home > 1e-6 { 1.0 / p_home } else { 999.0 };
        let oa = if p_away > 1e-6 { 1.0 / p_away } else { 999.0 };
        let od = if p_draw > 1e-6 { 1.0 / p_draw } else { 999.0 };
        FairState {
            p_home,
            p_away,
            p_draw,
            fair_odds_home: oh,
            fair_odds_away: oa,
            fair_odds_draw: od,
            uncertainty: 0.0,
            model_regime: ModelRegime::Normal,
            signal_strength: 0.0,
            timestamp_ms: 0,
        }
    }

    pub fn edge_home_back(&self, market_back_odds: f64) -> f64 {
        if market_back_odds <= 0.0 {
            return 0.0;
        }
        let market_prob = 1.0 / market_back_odds;
        self.p_home - market_prob
    }

    pub fn edge_away_back(&self, market_back_odds: f64) -> f64 {
        if market_back_odds <= 0.0 {
            return 0.0;
        }
        let market_prob = 1.0 / market_back_odds;
        self.p_away - market_prob
    }

    pub fn is_valid(&self) -> bool {
        let sum = self.p_home + self.p_away + self.p_draw;
        (sum - 1.0).abs() < 0.01
    }
}

// ============================================================
// Fill record
// ============================================================

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Fill {
    #[pyo3(get, set)]
    pub fill_id: u64,
    #[pyo3(get, set)]
    pub order_id: u64,
    #[pyo3(get, set)]
    pub market_id: String,
    #[pyo3(get, set)]
    pub runner_id: String,
    #[pyo3(get, set)]
    pub side: Side,
    #[pyo3(get, set)]
    pub price: f64,
    #[pyo3(get, set)]
    pub size: f64,
    #[pyo3(get, set)]
    pub timestamp_ms: i64,
}

// ============================================================
// Risk snapshot per outcome
// ============================================================

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OutcomePnl {
    #[pyo3(get, set)]
    pub runner_id: String,
    #[pyo3(get, set)]
    pub pnl_if_wins: f64,
    #[pyo3(get, set)]
    pub pnl_if_loses: f64,
}

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RiskSnapshot {
    #[pyo3(get, set)]
    pub market_id: String,
    #[pyo3(get, set)]
    pub timestamp_ms: i64,
    #[pyo3(get, set)]
    pub total_liability: f64,
    #[pyo3(get, set)]
    pub worst_case_loss: f64,
    #[pyo3(get, set)]
    pub best_case_profit: f64,
    pub outcome_pnls: Vec<OutcomePnl>,
    #[pyo3(get, set)]
    pub unmatched_exposure: f64,
    #[pyo3(get, set)]
    pub hedge_completion_pct: f64,
}

#[pymethods]
impl RiskSnapshot {
    #[new]
    pub fn new(market_id: String) -> Self {
        RiskSnapshot {
            market_id,
            timestamp_ms: 0,
            total_liability: 0.0,
            worst_case_loss: 0.0,
            best_case_profit: 0.0,
            outcome_pnls: Vec::new(),
            unmatched_exposure: 0.0,
            hedge_completion_pct: 0.0,
        }
    }
}

// ============================================================
// Execution intent (Section 8.4)
// ============================================================

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExecutionIntent {
    #[pyo3(get, set)]
    pub target_runner_id: String,
    #[pyo3(get, set)]
    pub target_side: Side,
    #[pyo3(get, set)]
    pub target_price: f64,
    #[pyo3(get, set)]
    pub target_size: f64,
    #[pyo3(get, set)]
    pub urgency: f64,       // 0.0 = passive, 1.0 = aggressive
    #[pyo3(get, set)]
    pub max_adverse_move: f64,
    #[pyo3(get, set)]
    pub stale_after_ms: u64,
    #[pyo3(get, set)]
    pub strategy_tag: String,
}

#[pymethods]
impl ExecutionIntent {
    #[new]
    pub fn new(
        target_runner_id: String,
        target_side: Side,
        target_price: f64,
        target_size: f64,
        strategy_tag: String,
    ) -> Self {
        ExecutionIntent {
            target_runner_id,
            target_side,
            target_price,
            target_size,
            urgency: 0.5,
            max_adverse_move: 0.05,
            stale_after_ms: 5000,
            strategy_tag,
        }
    }
}
