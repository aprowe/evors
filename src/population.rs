//! A uniform view of a strategy's maintained population/pool, so tooling (e.g.
//! the benchmark coverage metric) can read members without branching per type.

use crate::genetic::Genetic;
use crate::strategies::{ClassicPopulation, SoftStochastic, SpeciatedSoftStochastic};
use crate::strategy::{Behavior, Fitness};

pub trait PopulationView {
    type Genome: Clone;
    /// Snapshot of the maintained population/pool genomes.
    fn members(&self) -> Vec<Self::Genome>;
}

impl<X: Genetic, S: Fitness + Behavior + Clone> PopulationView for SoftStochastic<X, S> {
    type Genome = X::Genome;
    fn members(&self) -> Vec<X::Genome> {
        self.pool_genomes()
    }
}

impl<X: Genetic, S: Fitness + Behavior + Clone> PopulationView for SpeciatedSoftStochastic<X, S> {
    type Genome = X::Genome;
    fn members(&self) -> Vec<X::Genome> {
        self.pool_genomes()
    }
}

impl<X: Genetic> PopulationView for ClassicPopulation<X> {
    type Genome = X::Genome;
    fn members(&self) -> Vec<X::Genome> {
        self.population()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategies::SoftStochasticInfo;
    use crate::trial::{RunConfig, Stop, run};
    use rand::Rng;

    struct UnitGenetics;
    impl Genetic for UnitGenetics {
        type Genome = f64;
        fn random(&self, rng: &mut impl Rng) -> f64 {
            rng.random::<f64>()
        }
        fn cross(&self, a: &f64, b: &f64, _r: &mut impl Rng) -> f64 {
            (a + b) / 2.0
        }
        fn mutate(&self, g: &f64, rng: &mut impl Rng) -> f64 {
            (g + rng.random_range(-0.05..0.05)).clamp(0.0, 1.0)
        }
        fn gene_dist(&self, a: &f64, b: &f64) -> f64 {
            (a - b).abs()
        }
    }

    #[test]
    fn members_populated_after_run() {
        let mut strat: SoftStochastic<UnitGenetics, f64> = SoftStochastic::new(
            UnitGenetics,
            SoftStochasticInfo {
                target: 15,
                seed: 2,
                ..Default::default()
            },
        );
        let cfg = RunConfig {
            stop: Stop::Trials(200),
            batch_size: 1,
            max_trials: None,
        };
        let _ = run(&cfg, &mut strat, |x: &f64| *x);
        assert!(
            !strat.members().is_empty(),
            "pool should be non-empty after a run"
        );
    }
}
