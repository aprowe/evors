//! evors — trait-based evolutionary computation.

pub mod genetic;
pub mod strategy;
pub mod trial;
pub mod strategies;

pub use genetic::Genetic;
pub use strategy::{Behavior, EvolutionStrategy, Fitness};
pub use trial::{run, Outcome, RunConfig, Stop};
