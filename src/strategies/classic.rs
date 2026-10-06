use crate::genetic::Genetic;
use crate::strategy::EvolutionStrategy;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// CreateInfo for `ClassicPopulation`.
#[derive(Clone, Copy)]
pub struct ClassicPopulationInfo {
    pub pop_size: usize,
    pub tournament: usize,
    pub mutation_rate: f64,
    pub seed: u64,
}

impl Default for ClassicPopulationInfo {
    fn default() -> Self {
        Self {
            pop_size: 100,
            tournament: 3,
            mutation_rate: 0.1,
            seed: 0,
        }
    }
}

/// A classic generational GA. `Score = f64` (higher is better).
/// Buffers a generation of reports, then tournament-selects parents and
/// produces offspring via cross + mutate. Owns its genetics and rng.
pub struct ClassicPopulation<X: Genetic> {
    genetics: X,
    rng: ChaCha8Rng,
    pop_size: usize,
    tournament: usize,
    mutation_rate: f64,
    current: Vec<(X::Genome, f64)>,
    pending: usize,
    best: Option<(X::Genome, f64)>,
    initialized: bool,
}

impl<X: Genetic> ClassicPopulation<X> {
    pub fn new(genetics: X, info: ClassicPopulationInfo) -> Self {
        Self {
            genetics,
            rng: ChaCha8Rng::seed_from_u64(info.seed),
            pop_size: info.pop_size.max(2),
            tournament: info.tournament.max(1),
            mutation_rate: info.mutation_rate.clamp(0.0, 1.0),
            current: Vec::new(),
            pending: 0,
            best: None,
            initialized: false,
        }
    }

    /// Snapshot of the current population's genomes — useful for instrumentation.
    pub fn population(&self) -> Vec<X::Genome> {
        self.current.iter().map(|(g, _)| g.clone()).collect()
    }

    /// Tournament selection from `current` (must be non-empty). Returns the
    /// winner's genome and fitness. Associated fn over explicit fields to keep
    /// borrows disjoint.
    fn select(
        current: &[(X::Genome, f64)],
        tournament: usize,
        rng: &mut ChaCha8Rng,
    ) -> (X::Genome, f64) {
        let mut best: Option<&(X::Genome, f64)> = None;
        for _ in 0..tournament {
            let cand = &current[rng.random_range(0..current.len())];
            if best.is_none_or(|b| cand.1 > b.1) {
                best = Some(cand);
            }
        }
        best.unwrap().clone()
    }
}

impl<X: Genetic> EvolutionStrategy for ClassicPopulation<X> {
    type Genetic = X;
    type Score = f64;

    fn next(&mut self) -> X::Genome {
        self.pending += 1;
        if !self.initialized || self.current.is_empty() {
            if self.pending >= self.pop_size {
                self.initialized = true;
            }
            return self.genetics.random(&mut self.rng);
        }
        let (g1, f1) = Self::select(&self.current, self.tournament, &mut self.rng);
        let (g2, f2) = Self::select(&self.current, self.tournament, &mut self.rng);
        // Pass the fitter parent as `a` (the preferred-parent convention).
        let (a, b) = if f1 >= f2 { (&g1, &g2) } else { (&g2, &g1) };
        let child = self.genetics.cross(a, b, &mut self.rng);
        if self.rng.random_bool(self.mutation_rate) {
            self.genetics.mutate(&child, &mut self.rng)
        } else {
            child
        }
    }

    fn report(&mut self, genome: X::Genome, score: f64) {
        if self.best.as_ref().is_none_or(|(_, s)| score > *s) {
            self.best = Some((genome.clone(), score));
        }
        self.current.push((genome, score));
        // Replace-worst (elitist steady-state): once the population is full,
        // evict the lowest-fitness member rather than the oldest. FIFO eviction
        // discards good genes and stalls on deceptive landscapes like XOR.
        while self.current.len() > self.pop_size {
            let (worst, _) = self
                .current
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();
            self.current.swap_remove(worst);
        }
        if self.pending >= self.pop_size {
            self.pending = 0; // begin a new generation of asks
        }
    }

    fn best(&self) -> Option<(&X::Genome, &f64)> {
        self.best.as_ref().map(|(g, s)| (g, s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::EvolutionStrategy;
    use crate::trial::{RunConfig, Stop, run};

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
    fn maximizes_toward_one() {
        // Fitness = the value itself; optimum is 1.0.
        let mut strat = ClassicPopulation::new(
            UnitGenetics,
            ClassicPopulationInfo {
                pop_size: 30,
                seed: 11,
                ..Default::default()
            },
        );
        let cfg = RunConfig {
            stop: Stop::Trials(3000),
            batch_size: 1,
            max_trials: None,
        };
        let out = run(&cfg, &mut strat, |x: &f64| *x);
        assert!(out.best_score.unwrap() > 0.95, "got {:?}", out.best_score);
    }

    #[test]
    fn tracks_best_monotonically() {
        let mut strat = ClassicPopulation::new(UnitGenetics, ClassicPopulationInfo::default());
        let _ = strat.next();
        strat.report(0.3, 0.3);
        strat.report(0.7, 0.7);
        strat.report(0.5, 0.5);
        assert_eq!(*strat.best().unwrap().1, 0.7);
    }
}
