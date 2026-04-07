//! PyO3 module registration — exposes the entire Rust workspace to Python
//! as a single `sports_engine` module.

use pyo3::prelude::*;

// Re-export from all crates
use sports_core::enums::*;
use sports_core::events::*;
use sports_core::types::*;
use sports_book::*;
use sports_state::*;
use sports_exec::*;
use sports_risk::*;
use sports_journal::*;

#[pymodule]
fn sports_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // --- Core enums ---
    m.add_class::<Sport>()?;
    m.add_class::<MarketStatus>()?;
    m.add_class::<MatchPhase>()?;
    m.add_class::<OrderStatus>()?;
    m.add_class::<Side>()?;
    m.add_class::<OrderAction>()?;
    m.add_class::<MatchEventType>()?;
    m.add_class::<ModelRegime>()?;
    m.add_class::<SuspendReason>()?;

    // --- Core types ---
    m.add_class::<Market>()?;
    m.add_class::<LadderLevel>()?;
    m.add_class::<Runner>()?;
    m.add_class::<LiveOrder>()?;
    m.add_class::<MatchState>()?;
    m.add_class::<FairState>()?;
    m.add_class::<Fill>()?;
    m.add_class::<OutcomePnl>()?;
    m.add_class::<RiskSnapshot>()?;
    m.add_class::<ExecutionIntent>()?;

    // --- Events ---
    m.add_class::<MatchEvent>()?;
    m.add_class::<JournalEntryType>()?;
    m.add_class::<JournalEntry>()?;

    // --- State machines ---
    m.add_class::<MarketStateMachine>()?;
    m.add_class::<MatchStateMachine>()?;
    m.add_class::<OrderStateMachine>()?;

    // --- Book ---
    m.add_class::<MarketBook>()?;
    m.add_class::<RunnerSnapshot>()?;
    m.add_class::<MarketBookSnapshot>()?;

    // --- Execution ---
    m.add_class::<MockExchange>()?;

    // --- Risk ---
    m.add_class::<RiskEngine>()?;

    // --- Journal ---
    m.add_class::<JournalWriter>()?;
    m.add_class::<JournalReader>()?;

    Ok(())
}
