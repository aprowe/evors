//! evors — trait-based evolutionary computation.

pub mod genetic;
pub mod observe;
pub mod strategies;
pub mod strategy;
pub mod trial;

pub use genetic::Genetic;
pub use observe::{CsvProgressObserver, NoObserver, Observer, Progress, RecordingObserver};
pub use strategy::{Behavior, EvolutionStrategy, Fitness};
pub use trial::{run, run_observed, Outcome, RunConfig, Stop};
