use crate::genetic::Genetic;

/// A pluggable evolutionary algorithm with an ask/tell interface.
///
/// The strategy is self-contained: it carries its genetics as the associated
/// type `Genetic`, owns that genetics value plus its own seeded RNG, and takes
/// no per-call arguments. The genome is reached via `Self::Genetic::Genome`.
pub trait EvolutionStrategy {
    type Genetic: Genetic;
    type Score;

    fn next(&mut self) -> <Self::Genetic as Genetic>::Genome;
    fn report(&mut self, genome: <Self::Genetic as Genetic>::Genome, score: Self::Score);
    fn best(&self) -> Option<(&<Self::Genetic as Genetic>::Genome, &Self::Score)>;
}

/// Opt-in: lets a strategy pull a scalar fitness out of an arbitrary score.
pub trait Fitness {
    fn fitness(&self) -> f64;
}

/// Opt-in: behavioral/solution-space distance for behavioral novelty.
pub trait Behavior {
    fn behavior_dist(&self, other: &Self) -> f64;
}

impl Fitness for f64 {
    fn fitness(&self) -> f64 {
        *self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f64_is_its_own_fitness() {
        assert_eq!(0.75_f64.fitness(), 0.75);
    }
}
