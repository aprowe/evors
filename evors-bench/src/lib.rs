//! evors-bench — benchmark evors strategies against hard 2-D minimizers.

pub mod de;
pub mod genetics;
pub mod harness;
pub mod problems;

pub use de::{DeInfo, Differential};
pub use genetics::VecGenetics;
pub use problems::{Problem, all};
