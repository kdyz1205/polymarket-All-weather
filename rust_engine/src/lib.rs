use pyo3::prelude::*;

// ============================================================
// Core Order Book — O(1) array-based price level structure
// ============================================================

/// A single price level in the order book.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PriceLevel {
    #[pyo3(get)]
    pub odds: f64,
    #[pyo3(get)]
    pub volume: f64,
    #[pyo3(get)]
    pub order_count: u32,
}

/// Execution result returned to Python.
#[pyclass]
#[derive(Clone, Debug)]
pub struct ExecutionResult {
    #[pyo3(get)]
    pub success: bool,
    #[pyo3(get)]
    pub executed_odds: f64,
    #[pyo3(get)]
    pub executed_size: f64,
    #[pyo3(get)]
    pub slippage_bps: f64,
    #[pyo3(get)]
    pub message: String,
}

/// The core order book engine.
/// Maintains back (buy) and lay (sell) sides for a single market outcome.
///
/// In a real Betfair-style exchange:
///   - BACK = betting FOR the outcome (you want it to happen)
///   - LAY  = betting AGAINST the outcome (you want it NOT to happen)
///
/// Price is in decimal odds (e.g., 2.0 = even money = 50% implied probability).
#[pyclass]
pub struct OrderBook {
    /// Back side: sorted descending by odds (best back = highest odds first)
    back_levels: Vec<PriceLevel>,
    /// Lay side: sorted ascending by odds (best lay = lowest odds first)
    lay_levels: Vec<PriceLevel>,
    /// Market state
    market_id: String,
    is_suspended: bool,
    bet_delay_ms: u64,
    /// Stats
    total_matched: f64,
    tick_count: u64,
    last_update_ns: u64,
}

#[pymethods]
impl OrderBook {
    #[new]
    pub fn new(market_id: String) -> Self {
        OrderBook {
            back_levels: Vec::with_capacity(64),
            lay_levels: Vec::with_capacity(64),
            market_id,
            is_suspended: false,
            bet_delay_ms: 0,
            total_matched: 0.0,
            tick_count: 0,
            last_update_ns: 0,
        }
    }

    /// Update the entire back side of the book.
    /// Called by Python when new market data arrives.
    /// `levels` is a list of (odds, volume) tuples, sorted best-first.
    pub fn update_back(&mut self, levels: Vec<(f64, f64)>) {
        self.back_levels.clear();
        for (odds, volume) in levels {
            self.back_levels.push(PriceLevel {
                odds,
                volume,
                order_count: 1,
            });
        }
        self.tick_count += 1;
        self.last_update_ns = Self::now_ns();
    }

    /// Update the entire lay side of the book.
    pub fn update_lay(&mut self, levels: Vec<(f64, f64)>) {
        self.lay_levels.clear();
        for (odds, volume) in levels {
            self.lay_levels.push(PriceLevel {
                odds,
                volume,
                order_count: 1,
            });
        }
    }

    /// Get best back odds (highest available to back at).
    pub fn best_back_odds(&self) -> f64 {
        self.back_levels.first().map_or(0.0, |l| l.odds)
    }

    /// Get best lay odds (lowest available to lay at).
    pub fn best_lay_odds(&self) -> f64 {
        self.lay_levels.first().map_or(f64::MAX, |l| l.odds)
    }

    /// Spread between best back and best lay.
    pub fn spread(&self) -> f64 {
        let lay = self.best_lay_odds();
        let back = self.best_back_odds();
        if lay == f64::MAX || back == 0.0 {
            return f64::MAX;
        }
        lay - back
    }

    /// Implied probability from the mid-point of best back/lay.
    pub fn implied_probability(&self) -> f64 {
        let back = self.best_back_odds();
        let lay = self.best_lay_odds();
        if back <= 0.0 || lay == f64::MAX {
            return 0.0;
        }
        let mid = (back + lay) / 2.0;
        1.0 / mid
    }

    /// Total depth on back side.
    pub fn back_depth(&self) -> f64 {
        self.back_levels.iter().map(|l| l.volume).sum()
    }

    /// Total depth on lay side.
    pub fn lay_depth(&self) -> f64 {
        self.lay_levels.iter().map(|l| l.volume).sum()
    }

    /// Book imbalance: (back_depth - lay_depth) / total.
    /// Positive = more backing liquidity, negative = more laying.
    pub fn imbalance(&self) -> f64 {
        let back = self.back_depth();
        let lay = self.lay_depth();
        let total = back + lay;
        if total < 1e-10 {
            return 0.0;
        }
        (back - lay) / total
    }

    /// Suspend the market (e.g., goal scored, dangerous attack).
    pub fn suspend(&mut self) {
        self.is_suspended = true;
    }

    /// Resume the market after suspension.
    pub fn resume(&mut self, bet_delay_ms: u64) {
        self.is_suspended = false;
        self.bet_delay_ms = bet_delay_ms;
    }

    /// Execute a BACK order: you want to back at `target_odds` or better.
    /// Returns an ExecutionResult.
    pub fn execute_back(&mut self, target_odds: f64, stake: f64) -> ExecutionResult {
        if self.is_suspended {
            return ExecutionResult {
                success: false,
                executed_odds: 0.0,
                executed_size: 0.0,
                slippage_bps: 0.0,
                message: "Market suspended — cannot execute".to_string(),
            };
        }

        let best = self.best_back_odds();
        if best <= 0.0 || best < target_odds {
            return ExecutionResult {
                success: false,
                executed_odds: best,
                executed_size: 0.0,
                slippage_bps: 0.0,
                message: format!(
                    "No fill: best back {:.3} < target {:.3}",
                    best, target_odds
                ),
            };
        }

        // Walk the book to fill the order
        let mut remaining = stake;
        let mut total_cost = 0.0;
        let mut filled = 0.0;

        for level in self.back_levels.iter_mut() {
            if remaining <= 0.0 || level.odds < target_odds {
                break;
            }
            let fill_size = remaining.min(level.volume);
            total_cost += fill_size * level.odds;
            filled += fill_size;
            level.volume -= fill_size;
            remaining -= fill_size;
        }

        // Remove empty levels
        self.back_levels.retain(|l| l.volume > 1e-10);

        if filled < 1e-10 {
            return ExecutionResult {
                success: false,
                executed_odds: 0.0,
                executed_size: 0.0,
                slippage_bps: 0.0,
                message: "Zero fill — no liquidity at target".to_string(),
            };
        }

        let avg_odds = total_cost / filled;
        let slippage = if best > 0.0 {
            ((best - avg_odds) / best) * 10000.0
        } else {
            0.0
        };
        self.total_matched += filled;

        ExecutionResult {
            success: true,
            executed_odds: avg_odds,
            executed_size: filled,
            slippage_bps: slippage,
            message: format!(
                "BACK filled {:.2} @ avg odds {:.4} (slippage {:.1}bps)",
                filled, avg_odds, slippage
            ),
        }
    }

    /// Execute a LAY order: you want to lay at `target_odds` or better (lower).
    pub fn execute_lay(&mut self, target_odds: f64, stake: f64) -> ExecutionResult {
        if self.is_suspended {
            return ExecutionResult {
                success: false,
                executed_odds: 0.0,
                executed_size: 0.0,
                slippage_bps: 0.0,
                message: "Market suspended — cannot execute".to_string(),
            };
        }

        let best = self.best_lay_odds();
        if best == f64::MAX || best > target_odds {
            return ExecutionResult {
                success: false,
                executed_odds: best,
                executed_size: 0.0,
                slippage_bps: 0.0,
                message: format!(
                    "No fill: best lay {:.3} > target {:.3}",
                    best, target_odds
                ),
            };
        }

        let mut remaining = stake;
        let mut total_cost = 0.0;
        let mut filled = 0.0;

        for level in self.lay_levels.iter_mut() {
            if remaining <= 0.0 || level.odds > target_odds {
                break;
            }
            let fill_size = remaining.min(level.volume);
            total_cost += fill_size * level.odds;
            filled += fill_size;
            level.volume -= fill_size;
            remaining -= fill_size;
        }

        self.lay_levels.retain(|l| l.volume > 1e-10);

        if filled < 1e-10 {
            return ExecutionResult {
                success: false,
                executed_odds: 0.0,
                executed_size: 0.0,
                slippage_bps: 0.0,
                message: "Zero fill — no liquidity at target".to_string(),
            };
        }

        let avg_odds = total_cost / filled;
        let slippage = if best < f64::MAX {
            ((avg_odds - best) / best) * 10000.0
        } else {
            0.0
        };
        self.total_matched += filled;

        ExecutionResult {
            success: true,
            executed_odds: avg_odds,
            executed_size: filled,
            slippage_bps: slippage,
            message: format!(
                "LAY filled {:.2} @ avg odds {:.4} (slippage {:.1}bps)",
                filled, avg_odds, slippage
            ),
        }
    }

    /// Get full snapshot as a dict for Python dashboard.
    pub fn snapshot(&self) -> SnapshotData {
        SnapshotData {
            market_id: self.market_id.clone(),
            best_back: self.best_back_odds(),
            best_lay: self.best_lay_odds(),
            spread: self.spread(),
            implied_prob: self.implied_probability(),
            back_depth: self.back_depth(),
            lay_depth: self.lay_depth(),
            imbalance: self.imbalance(),
            is_suspended: self.is_suspended,
            bet_delay_ms: self.bet_delay_ms,
            total_matched: self.total_matched,
            tick_count: self.tick_count,
        }
    }

    #[staticmethod]
    fn now_ns() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64)
    }
}

/// Snapshot data exposed to Python.
#[pyclass]
#[derive(Clone, Debug)]
pub struct SnapshotData {
    #[pyo3(get)]
    pub market_id: String,
    #[pyo3(get)]
    pub best_back: f64,
    #[pyo3(get)]
    pub best_lay: f64,
    #[pyo3(get)]
    pub spread: f64,
    #[pyo3(get)]
    pub implied_prob: f64,
    #[pyo3(get)]
    pub back_depth: f64,
    #[pyo3(get)]
    pub lay_depth: f64,
    #[pyo3(get)]
    pub imbalance: f64,
    #[pyo3(get)]
    pub is_suspended: bool,
    #[pyo3(get)]
    pub bet_delay_ms: u64,
    #[pyo3(get)]
    pub total_matched: f64,
    #[pyo3(get)]
    pub tick_count: u64,
}

// ============================================================
// Match State Machine — tracks the game lifecycle
// ============================================================

#[pyclass]
#[derive(Clone, Debug, PartialEq)]
pub enum MatchPhase {
    PreMatch,
    FirstHalf,
    HalfTime,
    SecondHalf,
    ExtraTime,
    Finished,
}

#[pyclass]
pub struct MatchState {
    #[pyo3(get)]
    pub minute: u32,
    #[pyo3(get)]
    pub home_score: u32,
    #[pyo3(get)]
    pub away_score: u32,
    #[pyo3(get)]
    pub home_red_cards: u32,
    #[pyo3(get)]
    pub away_red_cards: u32,
    phase: MatchPhase,
}

#[pymethods]
impl MatchState {
    #[new]
    pub fn new() -> Self {
        MatchState {
            minute: 0,
            home_score: 0,
            away_score: 0,
            home_red_cards: 0,
            away_red_cards: 0,
            phase: MatchPhase::PreMatch,
        }
    }

    /// Advance the match clock by 1 minute. Returns the new phase.
    pub fn advance_minute(&mut self) -> String {
        self.minute += 1;
        self.phase = match self.minute {
            0 => MatchPhase::PreMatch,
            1..=45 => MatchPhase::FirstHalf,
            46 => MatchPhase::HalfTime,
            47..=90 => MatchPhase::SecondHalf,
            91..=120 => MatchPhase::ExtraTime,
            _ => MatchPhase::Finished,
        };
        format!("{:?}", self.phase)
    }

    pub fn home_goal(&mut self) {
        self.home_score += 1;
    }

    pub fn away_goal(&mut self) {
        self.away_score += 1;
    }

    pub fn home_red_card(&mut self) {
        self.home_red_cards += 1;
    }

    pub fn away_red_card(&mut self) {
        self.away_red_cards += 1;
    }

    pub fn phase_name(&self) -> String {
        format!("{:?}", self.phase)
    }

    pub fn is_live(&self) -> bool {
        matches!(
            self.phase,
            MatchPhase::FirstHalf | MatchPhase::SecondHalf | MatchPhase::ExtraTime
        )
    }

    pub fn is_finished(&self) -> bool {
        self.phase == MatchPhase::Finished
    }

    /// Remaining minutes (approximate).
    pub fn remaining_minutes(&self) -> u32 {
        if self.minute >= 90 {
            0
        } else {
            90 - self.minute
        }
    }
}

// ============================================================
// Python Module Registration
// ============================================================

#[pymodule]
fn sports_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OrderBook>()?;
    m.add_class::<PriceLevel>()?;
    m.add_class::<ExecutionResult>()?;
    m.add_class::<SnapshotData>()?;
    m.add_class::<MatchState>()?;
    m.add_class::<MatchPhase>()?;
    Ok(())
}
