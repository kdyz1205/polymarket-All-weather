//! Risk engine: per-outcome liability, worst-case PnL, kill switch logic.

use pyo3::prelude::*;
use sports_core::enums::*;
use sports_core::types::*;

/// Tracks all fills and computes per-outcome PnL.
#[pyclass]
pub struct RiskEngine {
    market_id: String,
    fills: Vec<Fill>,
    // Per-runner accumulators
    // runner_id -> (back_total_stake, back_avg_odds, lay_total_stake, lay_avg_odds)
    runner_positions: Vec<RunnerPosition>,
    // Limits
    max_market_liability: f64,
    max_outcome_liability: f64,
    max_unmatched_count: u32,
    kill_switch_active: bool,
}

#[derive(Clone, Debug)]
struct RunnerPosition {
    runner_id: String,
    back_stake: f64,
    back_weighted_odds: f64,
    lay_stake: f64,
    lay_weighted_odds: f64,
}

impl RunnerPosition {
    fn pnl_if_wins(&self) -> f64 {
        let back_profit = self.back_stake * (self.back_weighted_odds - 1.0);
        let lay_loss = self.lay_stake * (self.lay_weighted_odds - 1.0);
        back_profit - lay_loss
    }

    fn pnl_if_loses(&self) -> f64 {
        let back_loss = -self.back_stake;
        let lay_profit = self.lay_stake;
        back_loss + lay_profit
    }
}

#[pymethods]
impl RiskEngine {
    #[new]
    pub fn new(market_id: String, max_market_liability: f64, max_outcome_liability: f64) -> Self {
        RiskEngine {
            market_id,
            fills: Vec::new(),
            runner_positions: Vec::new(),
            max_market_liability,
            max_outcome_liability,
            max_unmatched_count: 50,
            kill_switch_active: false,
        }
    }

    /// Register a runner for risk tracking.
    pub fn add_runner(&mut self, runner_id: String) {
        self.runner_positions.push(RunnerPosition {
            runner_id,
            back_stake: 0.0,
            back_weighted_odds: 0.0,
            lay_stake: 0.0,
            lay_weighted_odds: 0.0,
        });
    }

    /// Record a fill and update positions.
    pub fn record_fill(&mut self, fill: &Fill) {
        self.fills.push(fill.clone());

        if let Some(pos) = self.runner_positions.iter_mut().find(|p| p.runner_id == fill.runner_id) {
            match fill.side {
                Side::Back => {
                    let prev_total = pos.back_stake;
                    pos.back_stake += fill.size;
                    if pos.back_stake > 0.0 {
                        pos.back_weighted_odds =
                            (prev_total * pos.back_weighted_odds + fill.size * fill.price)
                                / pos.back_stake;
                    }
                }
                Side::Lay => {
                    let prev_total = pos.lay_stake;
                    pos.lay_stake += fill.size;
                    if pos.lay_stake > 0.0 {
                        pos.lay_weighted_odds =
                            (prev_total * pos.lay_weighted_odds + fill.size * fill.price)
                                / pos.lay_stake;
                    }
                }
            }
        }
    }

    /// Compute full risk snapshot.
    pub fn snapshot(&self) -> RiskSnapshot {
        let mut outcomes: Vec<OutcomePnl> = Vec::new();
        let mut worst_loss = 0.0f64;
        let mut best_profit = f64::NEG_INFINITY;
        let mut total_liability = 0.0f64;

        for pos in &self.runner_positions {
            let pnl_win = pos.pnl_if_wins();
            let pnl_lose = pos.pnl_if_loses();

            outcomes.push(OutcomePnl {
                runner_id: pos.runner_id.clone(),
                pnl_if_wins: pnl_win,
                pnl_if_loses: pnl_lose,
            });

            // Worst case: this runner's worst outcome
            let worst = pnl_win.min(pnl_lose);
            worst_loss = worst_loss.min(worst);
            best_profit = best_profit.max(pnl_win.max(pnl_lose));

            // Liability: max possible loss on this runner
            let liability = (-worst).max(0.0);
            total_liability += liability;
        }

        RiskSnapshot {
            market_id: self.market_id.clone(),
            timestamp_ms: 0,
            total_liability,
            worst_case_loss: -worst_loss.min(0.0),
            best_case_profit: best_profit.max(0.0),
            outcome_pnls: outcomes,
            unmatched_exposure: 0.0,
            hedge_completion_pct: self.hedge_completion(),
        }
    }

    /// Check if a proposed trade would violate risk limits.
    pub fn check_limits(&self, runner_id: &str, side: Side, price: f64, size: f64) -> PyResult<bool> {
        if self.kill_switch_active {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Kill switch active — no new trades allowed",
            ));
        }

        // Simulate the fill
        let mut test_positions = self.runner_positions.clone();
        if let Some(pos) = test_positions.iter_mut().find(|p| p.runner_id == runner_id) {
            match side {
                Side::Back => {
                    let prev = pos.back_stake;
                    pos.back_stake += size;
                    if pos.back_stake > 0.0 {
                        pos.back_weighted_odds =
                            (prev * pos.back_weighted_odds + size * price) / pos.back_stake;
                    }
                }
                Side::Lay => {
                    let prev = pos.lay_stake;
                    pos.lay_stake += size;
                    if pos.lay_stake > 0.0 {
                        pos.lay_weighted_odds =
                            (prev * pos.lay_weighted_odds + size * price) / pos.lay_stake;
                    }
                }
            }
        }

        // Check total liability
        let mut total_liability = 0.0f64;
        for pos in &test_positions {
            let worst = pos.pnl_if_wins().min(pos.pnl_if_loses());
            total_liability += (-worst).max(0.0);
        }

        if total_liability > self.max_market_liability {
            return Ok(false);
        }

        // Check per-outcome
        for pos in &test_positions {
            let worst = pos.pnl_if_wins().min(pos.pnl_if_loses());
            if -worst > self.max_outcome_liability {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Activate kill switch — only allow flatten/hedge.
    pub fn activate_kill_switch(&mut self) {
        self.kill_switch_active = true;
    }

    pub fn deactivate_kill_switch(&mut self) {
        self.kill_switch_active = false;
    }

    pub fn is_kill_switch_active(&self) -> bool {
        self.kill_switch_active
    }

    fn hedge_completion(&self) -> f64 {
        // How balanced are back vs lay across runners
        let total_back: f64 = self.runner_positions.iter().map(|p| p.back_stake).sum();
        let total_lay: f64 = self.runner_positions.iter().map(|p| p.lay_stake).sum();
        let total = total_back + total_lay;
        if total < 1e-10 {
            return 1.0;
        }
        let min_side = total_back.min(total_lay);
        let max_side = total_back.max(total_lay);
        if max_side < 1e-10 {
            return 1.0;
        }
        min_side / max_side
    }
}
