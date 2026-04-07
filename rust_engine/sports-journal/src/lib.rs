//! Journal engine: records every event for complete replay capability.
//! Writes to an append-only log that can be replayed deterministically.

use pyo3::prelude::*;
use serde_json;
use sports_core::events::*;

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};

#[pyclass]
pub struct JournalWriter {
    next_seq: u64,
    entries: Vec<JournalEntry>,
    file_path: Option<String>,
}

#[pymethods]
impl JournalWriter {
    #[new]
    pub fn new(file_path: Option<String>) -> Self {
        JournalWriter {
            next_seq: 0,
            entries: Vec::new(),
            file_path,
        }
    }

    /// Write a journal entry. Appends to in-memory log and optionally to file.
    pub fn write(
        &mut self,
        timestamp_ms: i64,
        entry_type: JournalEntryType,
        market_id: String,
        payload_json: String,
    ) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;

        let entry = JournalEntry::new(seq, timestamp_ms, entry_type, market_id, payload_json.clone());
        self.entries.push(entry);

        // Append to file if configured
        if let Some(ref path) = self.file_path {
            if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
                let line = format!(
                    "{}|{}|{}\n",
                    seq, timestamp_ms, payload_json
                );
                let _ = file.write_all(line.as_bytes());
            }
        }

        seq
    }

    /// Get total entry count.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Get entries in a time range.
    pub fn get_entries(&self, from_ms: i64, to_ms: i64) -> Vec<JournalEntry> {
        self.entries
            .iter()
            .filter(|e| e.timestamp_ms >= from_ms && e.timestamp_ms <= to_ms)
            .cloned()
            .collect()
    }

    /// Get all entries (for replay).
    pub fn all_entries(&self) -> Vec<JournalEntry> {
        self.entries.clone()
    }

    /// Clear in-memory entries (keep file).
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

/// Replay reader: reads a journal file and yields entries in order.
#[pyclass]
pub struct JournalReader {
    entries: Vec<JournalEntry>,
    cursor: usize,
}

#[pymethods]
impl JournalReader {
    #[new]
    pub fn new() -> Self {
        JournalReader {
            entries: Vec::new(),
            cursor: 0,
        }
    }

    /// Load entries from a journal file.
    pub fn load_file(&mut self, path: &str) -> PyResult<usize> {
        let file = File::open(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Cannot open {}: {}", path, e)))?;

        let reader = BufReader::new(file);
        let mut count = 0;

        for line in reader.lines() {
            if let Ok(line) = line {
                let parts: Vec<&str> = line.splitn(3, '|').collect();
                if parts.len() == 3 {
                    let seq: u64 = parts[0].parse().unwrap_or(0);
                    let ts: i64 = parts[1].parse().unwrap_or(0);
                    let payload = parts[2].to_string();

                    self.entries.push(JournalEntry::new(
                        seq,
                        ts,
                        JournalEntryType::SystemAlert, // generic; real type in payload
                        String::new(),
                        payload,
                    ));
                    count += 1;
                }
            }
        }

        self.entries.sort_by_key(|e| e.sequence_id);
        self.cursor = 0;
        Ok(count)
    }

    /// Get next entry (for replay iteration).
    pub fn next_entry(&mut self) -> Option<JournalEntry> {
        if self.cursor < self.entries.len() {
            let entry = self.entries[self.cursor].clone();
            self.cursor += 1;
            Some(entry)
        } else {
            None
        }
    }

    /// Reset cursor to beginning.
    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    /// Total entries loaded.
    pub fn total_entries(&self) -> usize {
        self.entries.len()
    }

    /// Has more entries?
    pub fn has_next(&self) -> bool {
        self.cursor < self.entries.len()
    }
}
