//! evors-neat — NEAT (NeuroEvolution of Augmenting Topologies) as evors
//! components.
//!
//! The NEAT genome and its structural operators are packaged as an
//! [`evors::Genetic`] implementation ([`NeatGenetics`]), so NEAT runs on evors'
//! existing strategies — chiefly `SoftStochastic` (SSEA), whose diversity
//! preservation stands in for NEAT speciation. [`network::activate`] turns a
//! genome into a feed-forward network for fitness evaluation.

pub mod genetics;
pub mod genome;
pub mod innovation;
pub mod network;

pub use genetics::{NeatGenetics, NeatParams};
pub use genome::{Activation, ConnGene, NeatGenome, NodeGene, NodeKind};
pub use network::activate;
