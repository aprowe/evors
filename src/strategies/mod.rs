pub mod classic;
pub mod custom;
pub mod soft;
pub mod speciated;

pub use classic::{ClassicPopulation, ClassicPopulationInfo};
pub use soft::{SoftStochastic, SoftStochasticInfo};
pub use speciated::{SpeciatedSoftStochastic, SpeciatedSoftStochasticInfo};
