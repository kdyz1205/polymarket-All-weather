//! State machine transition logic.
//! Enforces valid transitions for market, match, and order states.

use sports_core::enums::*;
use sports_core::events::MatchEvent;
use sports_core::types::*;

use pyo3::prelude::*;

// ============================================================
// Market state transitions (Section 5.1)
// ============================================================

#[pyclass]
pub struct MarketStateMachine;

#[pymethods]
impl MarketStateMachine {
    #[new]
    pub fn new() -> Self { MarketStateMachine }

    /// Attempt a market state transition. Returns Ok(new_status) or Err(reason).
    #[staticmethod]
    pub fn transition(current: MarketStatus, target: MarketStatus) -> PyResult<MarketStatus> {
        let valid = match (current, target) {
            (MarketStatus::PreOpen, MarketStatus::OpenPrematch) => true,
            (MarketStatus::OpenPrematch, MarketStatus::InPlay) => true,
            (MarketStatus::OpenPrematch, MarketStatus::Suspended) => true,
            (MarketStatus::OpenPrematch, MarketStatus::Closed) => true,
            (MarketStatus::InPlay, MarketStatus::Suspended) => true,
            (MarketStatus::InPlay, MarketStatus::Closed) => true,
            (MarketStatus::Suspended, MarketStatus::Reopened) => true,
            (MarketStatus::Suspended, MarketStatus::Closed) => true,
            (MarketStatus::Reopened, MarketStatus::InPlay) => true,
            (MarketStatus::Reopened, MarketStatus::Suspended) => true,
            (MarketStatus::Reopened, MarketStatus::Closed) => true,
            (MarketStatus::Closed, MarketStatus::Settled) => true,
            (MarketStatus::Closed, MarketStatus::Void) => true,
            // Any -> Void is always allowed (exchange can void at any time)
            (_, MarketStatus::Void) => true,
            _ => false,
        };
        if valid {
            Ok(target)
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid market transition: {:?} -> {:?}",
                current, target
            )))
        }
    }

    /// Apply a market status change to a Market object. Mutates in place.
    #[staticmethod]
    pub fn apply_to_market(market: &mut Market, new_status: MarketStatus) -> PyResult<()> {
        let validated = MarketStateMachine::transition(market.status, new_status)?;
        market.status = validated;
        market.version += 1;
        if validated == MarketStatus::InPlay {
            market.in_play = true;
        }
        Ok(())
    }

    /// Suspend a market with a reason.
    #[staticmethod]
    pub fn suspend(market: &mut Market, reason: SuspendReason) -> PyResult<()> {
        MarketStateMachine::apply_to_market(market, MarketStatus::Suspended)?;
        market.suspend_reason = reason;
        Ok(())
    }

    /// Reopen a market after suspension.
    #[staticmethod]
    pub fn reopen(market: &mut Market, bet_delay_ms: u64) -> PyResult<()> {
        MarketStateMachine::apply_to_market(market, MarketStatus::Reopened)?;
        market.bet_delay_ms = bet_delay_ms;
        market.suspend_reason = SuspendReason::None;
        Ok(())
    }
}

// ============================================================
// Match state transitions (Section 5.2) — sport-aware
// ============================================================

#[pyclass]
pub struct MatchStateMachine;

#[pymethods]
impl MatchStateMachine {
    #[new]
    pub fn new() -> Self { MatchStateMachine }

    /// Apply a match event to the match state. Mutates in place.
    #[staticmethod]
    pub fn apply_event(state: &mut MatchState, event: &MatchEvent) -> PyResult<()> {
        state.last_event_ts_ms = event.timestamp_ms;

        match state.sport {
            Sport::Basketball => Self::apply_basketball(state, event),
            Sport::Baseball => Self::apply_baseball(state, event),
            Sport::Football => Self::apply_football(state, event),
        }
    }

    #[staticmethod]
    fn apply_basketball(state: &mut MatchState, event: &MatchEvent) -> PyResult<()> {
        let is_home = event.team == "home";

        match event.event_type {
            MatchEventType::PeriodStart => {
                state.phase = match state.match_clock_sec as u32 {
                    0..=719 => MatchPhase::Quarter1,
                    720..=1439 => MatchPhase::Quarter2,
                    1440..=2159 => MatchPhase::Quarter3,
                    2160..=2879 => MatchPhase::Quarter4,
                    _ => MatchPhase::Overtime,
                };
                state.home_fouls = 0; // reset per quarter
                state.away_fouls = 0;
            }
            MatchEventType::FieldGoal2 => {
                if is_home { state.home_score += 2; } else { state.away_score += 2; }
                Self::update_quarter_score(state, is_home, 2);
            }
            MatchEventType::FieldGoal3 => {
                if is_home { state.home_score += 3; } else { state.away_score += 3; }
                Self::update_quarter_score(state, is_home, 3);
            }
            MatchEventType::FreeThrow => {
                if is_home { state.home_score += 1; } else { state.away_score += 1; }
                Self::update_quarter_score(state, is_home, 1);
            }
            MatchEventType::Foul => {
                if is_home { state.home_fouls += 1; } else { state.away_fouls += 1; }
            }
            MatchEventType::TechnicalFoul => {
                if is_home { state.home_fouls += 1; } else { state.away_fouls += 1; }
            }
            MatchEventType::Timeout => {
                state.phase = MatchPhase::Timeout;
                if is_home {
                    state.home_timeouts_remaining = state.home_timeouts_remaining.saturating_sub(1);
                } else {
                    state.away_timeouts_remaining = state.away_timeouts_remaining.saturating_sub(1);
                }
            }
            MatchEventType::MatchEnd => {
                state.phase = MatchPhase::Finished;
            }
            _ => {}
        }
        Ok(())
    }

    #[staticmethod]
    fn update_quarter_score(state: &mut MatchState, is_home: bool, points: u32) {
        match state.phase {
            MatchPhase::Quarter1 => { if is_home { state.home_q1 += points; } else { state.away_q1 += points; } }
            MatchPhase::Quarter2 => { if is_home { state.home_q2 += points; } else { state.away_q2 += points; } }
            MatchPhase::Quarter3 => { if is_home { state.home_q3 += points; } else { state.away_q3 += points; } }
            MatchPhase::Quarter4 => { if is_home { state.home_q4 += points; } else { state.away_q4 += points; } }
            _ => {}
        }
    }

    #[staticmethod]
    fn apply_baseball(state: &mut MatchState, event: &MatchEvent) -> PyResult<()> {
        let is_home = event.team == "home";

        match event.event_type {
            MatchEventType::PeriodStart => {
                // New half-inning
                if state.is_top_inning && state.outs == 0 && state.inning > 0 {
                    state.is_top_inning = false;
                } else {
                    state.inning += 1;
                    state.is_top_inning = true;
                }
                state.outs = 0;
                state.balls = 0;
                state.strikes = 0;
                state.runner_on_first = false;
                state.runner_on_second = false;
                state.runner_on_third = false;
                state.phase = if state.is_top_inning {
                    MatchPhase::TopInning
                } else {
                    MatchPhase::BottomInning
                };
            }
            MatchEventType::Strikeout => {
                state.outs += 1;
                state.balls = 0;
                state.strikes = 0;
            }
            MatchEventType::Walk | MatchEventType::HitByPitch => {
                // Advance runners
                if state.runner_on_third && state.runner_on_second && state.runner_on_first {
                    // Bases loaded walk = run scored
                    if is_home { state.home_score += 1; } else { state.away_score += 1; }
                }
                if state.runner_on_second { state.runner_on_third = true; }
                if state.runner_on_first { state.runner_on_second = true; }
                state.runner_on_first = true;
                state.balls = 0;
                state.strikes = 0;
            }
            MatchEventType::Single => {
                // Score runner from 2nd/3rd, advance others
                if state.runner_on_third {
                    if is_home { state.home_score += 1; } else { state.away_score += 1; }
                    state.runner_on_third = false;
                }
                if state.runner_on_second {
                    state.runner_on_third = true;
                    state.runner_on_second = false;
                }
                if state.runner_on_first {
                    state.runner_on_second = true;
                }
                state.runner_on_first = true;
                if is_home { state.home_hits += 1; } else { state.away_hits += 1; }
                state.balls = 0;
                state.strikes = 0;
            }
            MatchEventType::Double => {
                let mut runs = 0u32;
                if state.runner_on_third { runs += 1; state.runner_on_third = false; }
                if state.runner_on_second { runs += 1; }
                if state.runner_on_first { runs += 1; state.runner_on_first = false; }
                if is_home { state.home_score += runs; } else { state.away_score += runs; }
                state.runner_on_third = false;
                state.runner_on_second = true;
                if is_home { state.home_hits += 1; } else { state.away_hits += 1; }
                state.balls = 0;
                state.strikes = 0;
            }
            MatchEventType::Triple => {
                let mut runs = 0u32;
                if state.runner_on_third { runs += 1; }
                if state.runner_on_second { runs += 1; }
                if state.runner_on_first { runs += 1; }
                if is_home { state.home_score += runs; } else { state.away_score += runs; }
                state.runner_on_first = false;
                state.runner_on_second = false;
                state.runner_on_third = true;
                if is_home { state.home_hits += 1; } else { state.away_hits += 1; }
                state.balls = 0;
                state.strikes = 0;
            }
            MatchEventType::HomeRun => {
                let mut runs = 1u32; // batter scores
                if state.runner_on_first { runs += 1; }
                if state.runner_on_second { runs += 1; }
                if state.runner_on_third { runs += 1; }
                if is_home { state.home_score += runs; } else { state.away_score += runs; }
                state.runner_on_first = false;
                state.runner_on_second = false;
                state.runner_on_third = false;
                if is_home { state.home_hits += 1; } else { state.away_hits += 1; }
                state.balls = 0;
                state.strikes = 0;
            }
            MatchEventType::DoublePlay => {
                state.outs += 2;
                state.runner_on_first = false;
                state.balls = 0;
                state.strikes = 0;
            }
            MatchEventType::Error => {
                if is_home { state.home_errors += 1; } else { state.away_errors += 1; }
            }
            MatchEventType::MatchEnd => {
                state.phase = MatchPhase::Finished;
            }
            _ => {}
        }

        // Check for 3 outs -> half-inning ends
        if state.outs >= 3 && !matches!(state.phase, MatchPhase::Finished) {
            state.phase = MatchPhase::InningBreak;
        }

        Ok(())
    }

    #[staticmethod]
    fn apply_football(state: &mut MatchState, event: &MatchEvent) -> PyResult<()> {
        let is_home = event.team == "home";

        match event.event_type {
            MatchEventType::Goal => {
                if is_home { state.home_score += 1; } else { state.away_score += 1; }
            }
            MatchEventType::RedCard => {
                if is_home { state.home_red_cards += 1; } else { state.away_red_cards += 1; }
            }
            MatchEventType::Corner => {
                if is_home { state.home_corners += 1; } else { state.away_corners += 1; }
            }
            MatchEventType::PeriodStart => {
                state.phase = if state.match_clock_sec < 2700.0 {
                    MatchPhase::FirstHalf
                } else {
                    MatchPhase::SecondHalf
                };
            }
            MatchEventType::PeriodEnd => {
                if state.phase == MatchPhase::FirstHalf {
                    state.phase = MatchPhase::HalfTime;
                }
            }
            MatchEventType::MatchEnd => {
                state.phase = MatchPhase::Finished;
            }
            _ => {}
        }
        Ok(())
    }
}

// ============================================================
// Order state transitions (Section 5.3)
// ============================================================

#[pyclass]
pub struct OrderStateMachine;

#[pymethods]
impl OrderStateMachine {
    #[new]
    pub fn new() -> Self { OrderStateMachine }

    #[staticmethod]
    pub fn transition(current: OrderStatus, target: OrderStatus) -> PyResult<OrderStatus> {
        let valid = match (current, target) {
            (OrderStatus::Created, OrderStatus::Submitted) => true,
            (OrderStatus::Submitted, OrderStatus::Accepted) => true,
            (OrderStatus::Submitted, OrderStatus::Rejected) => true,
            (OrderStatus::Accepted, OrderStatus::DelayPending) => true,
            (OrderStatus::Accepted, OrderStatus::Live) => true,
            (OrderStatus::DelayPending, OrderStatus::Live) => true,
            (OrderStatus::DelayPending, OrderStatus::Rejected) => true,
            (OrderStatus::DelayPending, OrderStatus::Lapsed) => true,
            (OrderStatus::Live, OrderStatus::PartiallyMatched) => true,
            (OrderStatus::Live, OrderStatus::FullyMatched) => true,
            (OrderStatus::Live, OrderStatus::CancelPending) => true,
            (OrderStatus::Live, OrderStatus::ReplacePending) => true,
            (OrderStatus::Live, OrderStatus::Lapsed) => true,
            (OrderStatus::PartiallyMatched, OrderStatus::FullyMatched) => true,
            (OrderStatus::PartiallyMatched, OrderStatus::CancelPending) => true,
            (OrderStatus::PartiallyMatched, OrderStatus::ReplacePending) => true,
            (OrderStatus::PartiallyMatched, OrderStatus::Lapsed) => true,
            (OrderStatus::CancelPending, OrderStatus::Cancelled) => true,
            (OrderStatus::CancelPending, OrderStatus::FullyMatched) => true, // raced
            (OrderStatus::ReplacePending, OrderStatus::Replaced) => true,
            (OrderStatus::ReplacePending, OrderStatus::Cancelled) => true, // replace failed
            _ => false,
        };
        if valid {
            Ok(target)
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid order transition: {:?} -> {:?}",
                current, target
            )))
        }
    }

    /// Submit an order: Created -> Submitted
    #[staticmethod]
    pub fn submit(order: &mut LiveOrder, now_ms: i64) -> PyResult<()> {
        order.status = Self::transition(order.status, OrderStatus::Submitted)?;
        order.submit_ts_ms = now_ms;
        Ok(())
    }

    /// Accept an order: Submitted -> Accepted
    #[staticmethod]
    pub fn accept(order: &mut LiveOrder, exchange_id: String, now_ms: i64) -> PyResult<()> {
        order.status = Self::transition(order.status, OrderStatus::Accepted)?;
        order.exchange_order_id = exchange_id;
        order.accepted_ts_ms = now_ms;
        Ok(())
    }

    /// Enter bet delay: Accepted -> DelayPending
    #[staticmethod]
    pub fn enter_delay(order: &mut LiveOrder, delay_ms: u64, now_ms: i64) -> PyResult<()> {
        order.status = Self::transition(order.status, OrderStatus::DelayPending)?;
        order.delay_expire_ts_ms = now_ms + delay_ms as i64;
        Ok(())
    }

    /// Go live: DelayPending/Accepted -> Live
    #[staticmethod]
    pub fn go_live(order: &mut LiveOrder) -> PyResult<()> {
        order.status = Self::transition(order.status, OrderStatus::Live)?;
        Ok(())
    }

    /// Record a fill (partial or full).
    #[staticmethod]
    pub fn record_fill(order: &mut LiveOrder, fill_price: f64, fill_size: f64) -> PyResult<()> {
        let prev_matched = order.matched_size;
        order.matched_size += fill_size;
        order.remaining_size -= fill_size;

        // Update average matched price
        if order.matched_size > 0.0 {
            order.avg_matched_price =
                (prev_matched * order.avg_matched_price + fill_size * fill_price)
                    / order.matched_size;
        }

        if order.remaining_size <= 1e-10 {
            order.remaining_size = 0.0;
            order.status = Self::transition(order.status, OrderStatus::FullyMatched)?;
        } else {
            // If currently Live, move to PartiallyMatched
            if order.status == OrderStatus::Live {
                order.status = Self::transition(order.status, OrderStatus::PartiallyMatched)?;
            }
        }
        Ok(())
    }
}
