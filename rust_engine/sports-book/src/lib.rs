//! Tick ladder order book with O(1) best-price access.
//! Maintains back/lay depth for each runner, supports delta updates,
//! and computes market microstructure features.

use pyo3::prelude::*;
use sports_core::types::*;

/// Full market book containing all runners.
#[pyclass]
pub struct MarketBook {
    #[pyo3(get)]
    pub market_id: String,
    pub runners: Vec<Runner>,
    tick_count: u64,
}

#[pymethods]
impl MarketBook {
    #[new]
    pub fn new(market_id: String) -> Self {
        MarketBook {
            market_id,
            runners: Vec::new(),
            tick_count: 0,
        }
    }

    /// Add a runner to this market book.
    pub fn add_runner(&mut self, runner_id: String, name: String) {
        self.runners.push(Runner::new(runner_id, name));
    }

    /// Get runner count.
    pub fn runner_count(&self) -> usize {
        self.runners.len()
    }

    /// Update back ladder for a runner. levels = [(price, volume), ...] best-first.
    pub fn update_runner_back(&mut self, runner_id: &str, levels: Vec<(f64, f64)>) -> PyResult<()> {
        let runner = self.find_runner_mut(runner_id)?;
        runner.update_back_ladder(levels);
        self.tick_count += 1;
        Ok(())
    }

    /// Update lay ladder for a runner.
    pub fn update_runner_lay(&mut self, runner_id: &str, levels: Vec<(f64, f64)>) -> PyResult<()> {
        let runner = self.find_runner_mut(runner_id)?;
        runner.update_lay_ladder(levels);
        Ok(())
    }

    /// Update traded volume for a runner.
    pub fn update_runner_traded(&mut self, runner_id: &str, price: f64, volume: f64) -> PyResult<()> {
        let runner = self.find_runner_mut(runner_id)?;
        runner.traded_volume_total += volume;
        runner.last_traded_price = price;
        Ok(())
    }

    /// Get snapshot of a single runner's book state.
    pub fn get_runner_snapshot(&self, runner_id: &str) -> PyResult<RunnerSnapshot> {
        let runner = self.find_runner(runner_id)?;
        Ok(RunnerSnapshot {
            runner_id: runner.runner_id.clone(),
            name: runner.name.clone(),
            best_back_price: runner.best_back_price,
            best_back_size: runner.best_back_size,
            best_lay_price: runner.best_lay_price,
            best_lay_size: runner.best_lay_size,
            spread: runner.spread(),
            implied_probability: runner.implied_probability(),
            back_depth: runner.back_depth(),
            lay_depth: runner.lay_depth(),
            imbalance: runner.imbalance(),
            traded_volume: runner.traded_volume_total,
            last_traded_price: runner.last_traded_price,
        })
    }

    /// Full market snapshot (all runners).
    pub fn snapshot(&self) -> MarketBookSnapshot {
        let runners: Vec<RunnerSnapshot> = self
            .runners
            .iter()
            .map(|r| RunnerSnapshot {
                runner_id: r.runner_id.clone(),
                name: r.name.clone(),
                best_back_price: r.best_back_price,
                best_back_size: r.best_back_size,
                best_lay_price: r.best_lay_price,
                best_lay_size: r.best_lay_size,
                spread: r.spread(),
                implied_probability: r.implied_probability(),
                back_depth: r.back_depth(),
                lay_depth: r.lay_depth(),
                imbalance: r.imbalance(),
                traded_volume: r.traded_volume_total,
                last_traded_price: r.last_traded_price,
            })
            .collect();

        // Overround = sum of implied probs (should be > 1.0 due to spread)
        let overround: f64 = runners.iter().map(|r| r.implied_probability).sum();

        MarketBookSnapshot {
            market_id: self.market_id.clone(),
            runners,
            overround,
            tick_count: self.tick_count,
        }
    }

    /// Simulate a back execution against a runner's back ladder.
    /// Walks the book, returns (avg_price, filled_size, slippage_bps).
    pub fn simulate_back_fill(
        &self,
        runner_id: &str,
        target_price: f64,
        stake: f64,
    ) -> PyResult<(f64, f64, f64)> {
        let runner = self.find_runner(runner_id)?;
        let mut remaining = stake;
        let mut total_cost = 0.0;
        let mut filled = 0.0;

        for level in &runner.atb_ladder {
            if remaining <= 0.0 || level.price < target_price {
                break;
            }
            let fill = remaining.min(level.volume);
            total_cost += fill * level.price;
            filled += fill;
            remaining -= fill;
        }

        if filled < 1e-10 {
            return Ok((0.0, 0.0, 0.0));
        }

        let avg_price = total_cost / filled;
        let best_price = runner.best_back_price;
        let slippage_bps = if best_price > 0.0 {
            ((best_price - avg_price) / best_price) * 10000.0
        } else {
            0.0
        };

        Ok((avg_price, filled, slippage_bps))
    }

    /// Simulate a lay execution against a runner's lay ladder.
    pub fn simulate_lay_fill(
        &self,
        runner_id: &str,
        target_price: f64,
        stake: f64,
    ) -> PyResult<(f64, f64, f64)> {
        let runner = self.find_runner(runner_id)?;
        let mut remaining = stake;
        let mut total_cost = 0.0;
        let mut filled = 0.0;

        for level in &runner.atl_ladder {
            if remaining <= 0.0 || level.price > target_price {
                break;
            }
            let fill = remaining.min(level.volume);
            total_cost += fill * level.price;
            filled += fill;
            remaining -= fill;
        }

        if filled < 1e-10 {
            return Ok((0.0, 0.0, 0.0));
        }

        let avg_price = total_cost / filled;
        let best_price = runner.best_lay_price;
        let slippage_bps = if best_price > 0.0 && best_price < f64::MAX {
            ((avg_price - best_price) / best_price) * 10000.0
        } else {
            0.0
        };

        Ok((avg_price, filled, slippage_bps))
    }

}

impl MarketBook {
    fn find_runner(&self, runner_id: &str) -> PyResult<&Runner> {
        self.runners
            .iter()
            .find(|r| r.runner_id == runner_id)
            .ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!("Runner not found: {}", runner_id))
            })
    }

    fn find_runner_mut(&mut self, runner_id: &str) -> PyResult<&mut Runner> {
        self.runners
            .iter_mut()
            .find(|r| r.runner_id == runner_id)
            .ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!("Runner not found: {}", runner_id))
            })
    }
}

// ============================================================
// Snapshot types for Python
// ============================================================

#[pyclass]
#[derive(Clone, Debug)]
pub struct RunnerSnapshot {
    #[pyo3(get)]
    pub runner_id: String,
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub best_back_price: f64,
    #[pyo3(get)]
    pub best_back_size: f64,
    #[pyo3(get)]
    pub best_lay_price: f64,
    #[pyo3(get)]
    pub best_lay_size: f64,
    #[pyo3(get)]
    pub spread: f64,
    #[pyo3(get)]
    pub implied_probability: f64,
    #[pyo3(get)]
    pub back_depth: f64,
    #[pyo3(get)]
    pub lay_depth: f64,
    #[pyo3(get)]
    pub imbalance: f64,
    #[pyo3(get)]
    pub traded_volume: f64,
    #[pyo3(get)]
    pub last_traded_price: f64,
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct MarketBookSnapshot {
    #[pyo3(get)]
    pub market_id: String,
    #[pyo3(get)]
    pub runners: Vec<RunnerSnapshot>,
    #[pyo3(get)]
    pub overround: f64,
    #[pyo3(get)]
    pub tick_count: u64,
}
