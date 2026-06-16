//! evors — trait-based evolutionary computation.

pub mod genetic;
pub mod observe;
pub mod population;
pub mod strategies;
pub mod strategy;
pub mod trial;

pub use genetic::Genetic;
pub use observe::{CsvProgressObserver, NoObserver, Observer, Progress, RecordingObserver};
pub use population::PopulationView;
pub use strategy::{Behavior, EvolutionStrategy, Fitness};
pub use trial::{Outcome, RunConfig, Stop, run, run_observed};
