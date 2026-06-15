//! evors — trait-based evolutionary computation.

pub mod genetic;
pub mod strategies;
pub mod strategy;
pub mod trial;

pub use genetic::Genetic;
pub use strategy::{Behavior, EvolutionStrategy, Fitness};
pub use trial::{Outcome, RunConfig, Stop, run};
