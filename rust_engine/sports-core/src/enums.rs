//! All enums for the sports trading system.
//! State machines are defined here; transition logic lives in sports-state.

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

// ============================================================
// Sport type — basketball & baseball first, football later
// ============================================================

#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Sport {
    Basketball,
    Baseball,
    Football,
}

// ============================================================
// Market state machine (Section 5.1)
// PRE_OPEN -> OPEN_PREMATCH -> IN_PLAY -> SUSPENDED -> REOPENED -> CLOSED -> SETTLED -> VOID
// ============================================================

#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarketStatus {
    PreOpen,
    OpenPrematch,
    InPlay,
    Suspended,
    Reopened,
    Closed,
    Settled,
    Void,
}

// ============================================================
// Match state machine (Section 5.2) — sport-agnostic phases
// Basketball: Q1, Q2, Q3, Q4, OT
// Baseball: Top/Bottom of innings 1-9+
// Football: FirstHalf, SecondHalf, ExtraTime, Penalties
// ============================================================

#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MatchPhase {
    NotStarted,
    // Basketball
    Quarter1,
    Quarter2,
    Quarter3,
    Quarter4,
    Overtime,
    // Baseball
    TopInning,
    BottomInning,
    InningBreak,
    // Football
    FirstHalf,
    HalfTime,
    SecondHalf,
    ExtraTime,
    Penalties,
    // Universal
    Timeout,
    Finished,
    Abandoned,
}

// ============================================================
// Order state machine (Section 5.3)
// CREATED -> SUBMITTED -> ACCEPTED -> DELAY_PENDING -> LIVE ->
//   PARTIALLY_MATCHED -> FULLY_MATCHED
//   CANCEL_PENDING -> CANCELLED
//   REPLACE_PENDING -> REPLACED
//   REJECTED | LAPSED
// ============================================================

#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrderStatus {
    Created,
    Submitted,
    Accepted,
    DelayPending,
    Live,
    PartiallyMatched,
    FullyMatched,
    CancelPending,
    Cancelled,
    ReplacePending,
    Replaced,
    Rejected,
    Lapsed,
}

#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Side {
    Back,
    Lay,
}

#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrderAction {
    PlaceLimit,
    CancelOrder,
    ReplaceOrder,
    HedgePosition,
    FlattenPosition,
    PanicExit,
    DoNothing,
}

// ============================================================
// Match events — covers basketball, baseball, football
// ============================================================

#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MatchEventType {
    // Universal
    MatchStart,
    PeriodStart,
    PeriodEnd,
    Timeout,
    MatchEnd,
    // Basketball
    FieldGoal2,
    FieldGoal3,
    FreeThrow,
    Rebound,
    Turnover,
    Steal,
    Block,
    Foul,
    TechnicalFoul,
    // Baseball
    Single,
    Double,
    Triple,
    HomeRun,
    Walk,
    Strikeout,
    HitByPitch,
    SacFly,
    DoublePlay,
    Error,
    StolenBase,
    WildPitch,
    PitcherChange,
    // Football
    Goal,
    RedCard,
    YellowCard,
    Shot,
    ShotOnTarget,
    Corner,
    DangerousAttack,
    Penalty,
    Substitution,
    VAR,
    Injury,
}

// ============================================================
// Model regime — what state is the market/match in
// ============================================================

#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelRegime {
    Normal,
    HighVolatility,
    LowLiquidity,
    EventShock,
    Suspended,
    GarbageTime,
    ClutchTime,
}

// ============================================================
// Suspend reason
// ============================================================

#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SuspendReason {
    None,
    Goal,
    RedCard,
    Penalty,
    VAR,
    InjuryStoppage,
    SystemError,
    ManualOverride,
    ScoreChange,
    PeriodTransition,
}
