//! Unified event model for the journal and replay system.
//! Every state change, order action, model output, and external event
//! is captured as a JournalEntry for complete replay capability.

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use crate::enums::*;

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MatchEvent {
    #[pyo3(get, set)]
    pub event_id: String,
    #[pyo3(get, set)]
    pub timestamp_ms: i64,
    #[pyo3(get, set)]
    pub event_type: MatchEventType,
    #[pyo3(get, set)]
    pub minute: f64,
    #[pyo3(get, set)]
    pub team: String, // "home" or "away"
    #[pyo3(get, set)]
    pub player: String,
    #[pyo3(get, set)]
    pub detail: String, // extra info, e.g. "3-pointer from corner"
}

#[pymethods]
impl MatchEvent {
    #[new]
    pub fn new(
        event_id: String,
        timestamp_ms: i64,
        event_type: MatchEventType,
        minute: f64,
        team: String,
    ) -> Self {
        MatchEvent {
            event_id,
            timestamp_ms,
            event_type,
            minute,
            team,
            player: String::new(),
            detail: String::new(),
        }
    }

    /// Is this a score-changing event?
    pub fn is_score_change(&self) -> bool {
        matches!(
            self.event_type,
            MatchEventType::FieldGoal2
                | MatchEventType::FieldGoal3
                | MatchEventType::FreeThrow
                | MatchEventType::HomeRun
                | MatchEventType::Single
                | MatchEventType::Double
                | MatchEventType::Triple
                | MatchEventType::Goal
        )
    }

    /// Impact magnitude: how much should this event shock the model?
    /// Returns (immediate_shock, duration_sec)
    pub fn impact_magnitude(&self) -> (f64, f64) {
        match self.event_type {
            // Basketball: individual plays have moderate impact
            MatchEventType::FieldGoal3 => (0.03, 5.0),
            MatchEventType::FieldGoal2 => (0.02, 3.0),
            MatchEventType::FreeThrow => (0.01, 2.0),
            MatchEventType::Turnover => (0.015, 5.0),
            MatchEventType::TechnicalFoul => (0.02, 10.0),
            // Baseball: higher variance per event
            MatchEventType::HomeRun => (0.15, 30.0),
            MatchEventType::Triple => (0.08, 20.0),
            MatchEventType::Double => (0.05, 15.0),
            MatchEventType::Single => (0.03, 10.0),
            MatchEventType::Strikeout => (0.02, 5.0),
            MatchEventType::Walk => (0.03, 8.0),
            MatchEventType::DoublePlay => (0.06, 10.0),
            MatchEventType::Error => (0.04, 10.0),
            MatchEventType::PitcherChange => (0.05, 60.0),
            // Football: discrete high-impact events
            MatchEventType::Goal => (0.30, 120.0),
            MatchEventType::RedCard => (0.15, 300.0),
            MatchEventType::Penalty => (0.20, 60.0),
            MatchEventType::ShotOnTarget => (0.02, 5.0),
            MatchEventType::DangerousAttack => (0.01, 3.0),
            // Period transitions
            MatchEventType::PeriodStart | MatchEventType::PeriodEnd => (0.0, 0.0),
            _ => (0.005, 2.0),
        }
    }
}

// ============================================================
// Journal entry types for replay
// ============================================================

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum JournalEntryType {
    MarketBookUpdate,
    MarketStateChange,
    MatchEventOccurred,
    ModelOutput,
    OrderSubmitted,
    OrderAccepted,
    OrderMatched,
    OrderCancelled,
    OrderRejected,
    RiskUpdate,
    StrategyDecision,
    SystemAlert,
}

#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JournalEntry {
    #[pyo3(get, set)]
    pub sequence_id: u64,
    #[pyo3(get, set)]
    pub timestamp_ms: i64,
    #[pyo3(get, set)]
    pub entry_type: JournalEntryType,
    #[pyo3(get, set)]
    pub market_id: String,
    #[pyo3(get, set)]
    pub payload_json: String, // serialized detail
}

#[pymethods]
impl JournalEntry {
    #[new]
    pub fn new(
        sequence_id: u64,
        timestamp_ms: i64,
        entry_type: JournalEntryType,
        market_id: String,
        payload_json: String,
    ) -> Self {
        JournalEntry {
            sequence_id,
            timestamp_ms,
            entry_type,
            market_id,
            payload_json,
        }
    }
}
