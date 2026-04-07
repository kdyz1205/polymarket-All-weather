//! Execution engine: converts strategy intents into concrete orders,
//! respects bet delay, handles partial fills, and manages order lifecycle.

use pyo3::prelude::*;
use sports_core::enums::*;
use sports_core::types::*;

/// Mock exchange for testing. Simulates order matching with configurable
/// fill probability, delay, and partial fills.
#[pyclass]
pub struct MockExchange {
    next_order_id: u64,
    next_fill_id: u64,
    orders: Vec<LiveOrder>,
    fills: Vec<Fill>,
    bet_delay_ms: u64,
    fill_probability: f64, // 0.0 to 1.0
}

#[pymethods]
impl MockExchange {
    #[new]
    pub fn new(bet_delay_ms: u64, fill_probability: f64) -> Self {
        MockExchange {
            next_order_id: 1,
            next_fill_id: 1,
            orders: Vec::new(),
            fills: Vec::new(),
            bet_delay_ms,
            fill_probability,
        }
    }

    /// Submit a new order. Returns local_order_id.
    pub fn submit_order(
        &mut self,
        market_id: String,
        runner_id: String,
        side: Side,
        price: f64,
        size: f64,
        strategy_tag: String,
        now_ms: i64,
    ) -> PyResult<u64> {
        let id = self.next_order_id;
        self.next_order_id += 1;

        let mut order = LiveOrder::new(id, market_id, runner_id, side, price, size, strategy_tag);
        order.submit_ts_ms = now_ms;
        order.status = OrderStatus::Submitted;

        // Simulate immediate acceptance
        order.status = OrderStatus::Accepted;
        order.accepted_ts_ms = now_ms;
        order.exchange_order_id = format!("EX-{}", id);

        // Enter bet delay if applicable
        if self.bet_delay_ms > 0 {
            order.status = OrderStatus::DelayPending;
            order.delay_expire_ts_ms = now_ms + self.bet_delay_ms as i64;
        } else {
            order.status = OrderStatus::Live;
        }

        self.orders.push(order);
        Ok(id)
    }

    /// Cancel an order by local_order_id.
    pub fn cancel_order(&mut self, local_order_id: u64) -> PyResult<bool> {
        if let Some(order) = self.orders.iter_mut().find(|o| o.local_order_id == local_order_id) {
            if order.is_active() {
                order.status = OrderStatus::Cancelled;
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Advance time: process delays, generate fills.
    /// Call this every tick to simulate exchange behavior.
    pub fn process_tick(&mut self, now_ms: i64, rng_seed: f64) -> Vec<Fill> {
        let mut new_fills = Vec::new();

        for order in self.orders.iter_mut() {
            // Check delay expiry
            if order.status == OrderStatus::DelayPending && now_ms >= order.delay_expire_ts_ms {
                order.status = OrderStatus::Live;
            }

            // Simulate fills for live orders
            if order.status == OrderStatus::Live || order.status == OrderStatus::PartiallyMatched {
                // Use deterministic "random" based on rng_seed
                let hash = (order.local_order_id as f64 * 0.618 + rng_seed).fract();
                if hash < self.fill_probability && order.remaining_size > 0.0 {
                    // Partial or full fill
                    let fill_fraction = 0.3 + hash * 0.7; // 30% to 100%
                    let fill_size = (order.remaining_size * fill_fraction).max(1.0).min(order.remaining_size);
                    let fill_price = order.price; // fill at order price

                    let fill = Fill {
                        fill_id: self.next_fill_id,
                        order_id: order.local_order_id,
                        market_id: order.market_id.clone(),
                        runner_id: order.runner_id.clone(),
                        side: order.side,
                        price: fill_price,
                        size: fill_size,
                        timestamp_ms: now_ms,
                    };
                    self.next_fill_id += 1;

                    // Update order
                    let prev_matched = order.matched_size;
                    order.matched_size += fill_size;
                    order.remaining_size -= fill_size;
                    if order.matched_size > 0.0 {
                        order.avg_matched_price =
                            (prev_matched * order.avg_matched_price + fill_size * fill_price)
                                / order.matched_size;
                    }

                    if order.remaining_size <= 1e-10 {
                        order.remaining_size = 0.0;
                        order.status = OrderStatus::FullyMatched;
                    } else {
                        order.status = OrderStatus::PartiallyMatched;
                    }

                    new_fills.push(fill.clone());
                    self.fills.push(fill);
                }
            }
        }

        new_fills
    }

    /// Get all active orders.
    pub fn active_orders(&self) -> Vec<LiveOrder> {
        self.orders.iter().filter(|o| o.is_active()).cloned().collect()
    }

    /// Get all fills.
    pub fn all_fills(&self) -> Vec<Fill> {
        self.fills.clone()
    }

    /// Get order by id.
    pub fn get_order(&self, local_order_id: u64) -> Option<LiveOrder> {
        self.orders.iter().find(|o| o.local_order_id == local_order_id).cloned()
    }

    /// Total number of orders submitted.
    pub fn order_count(&self) -> usize {
        self.orders.len()
    }

    /// Total number of fills.
    pub fn fill_count(&self) -> usize {
        self.fills.len()
    }
}
