use crate::genetic::Genetic;
use crate::strategy::EvolutionStrategy;

/// When to stop a run.
pub enum Stop<S> {
    /// Stop after this many batches (one "generation" per batch).
    Generations(usize),
    /// Stop after this many individual evaluations.
    Trials(usize),
    /// Stop when the predicate over the current best score returns true.
    Until(Box<dyn Fn(&S) -> bool>),
}

/// Run configuration (CreateInfo-style plain struct — no builder).
pub struct RunConfig<S> {
    pub stop: Stop<S>,
    /// Genomes pulled from `next()` per round before evaluation.
    pub batch_size: usize,
    /// Hard ceiling on total evaluations, applied regardless of `stop`.
    /// `None` means no cap (only safe with `Stop::Generations`/`Stop::Trials`).
    /// Always set this when using `Stop::Until` so a run that never reaches the
    /// target still terminates.
    pub max_trials: Option<usize>,
}

pub struct Outcome<G, S> {
    pub best_genome: Option<G>,
    pub best_score: Option<S>,
    pub generations: usize,
    pub trials: usize,
}

/// Evaluate a batch of genomes. Sequential build.
#[cfg(not(feature = "parallel"))]
fn evaluate<G, S, F>(batch: Vec<G>, fitness: &F) -> Vec<(G, S)>
where
    F: Fn(&G) -> S,
{
    batch.into_iter().map(|g| { let s = fitness(&g); (g, s) }).collect()
}

/// Evaluate a batch of genomes in parallel.
#[cfg(feature = "parallel")]
fn evaluate<G, S, F>(batch: Vec<G>, fitness: &F) -> Vec<(G, S)>
where
    F: Fn(&G) -> S + Sync,
    G: Send,
    S: Send,
{
    use rayon::prelude::*;
    batch.into_par_iter().map(|g| { let s = fitness(&g); (g, s) }).collect()
}

/// Drive a self-contained strategy against `fitness` until `cfg.stop`.
/// The strategy owns its genetics and rng, so the runner needs neither.
pub fn run<St, F>(
    cfg: &RunConfig<St::Score>,
    strategy: &mut St,
    fitness: F,
) -> Outcome<<St::Genetic as Genetic>::Genome, St::Score>
where
    St: EvolutionStrategy,
    F: Fn(&<St::Genetic as Genetic>::Genome) -> St::Score + Sync,
    <St::Genetic as Genetic>::Genome: Clone + Send,
    St::Score: Clone + Send,
{
    let mut generations = 0usize;
    let mut trials = 0usize;
    let batch_size = cfg.batch_size.max(1);

    loop {
        // Hard cap, applied regardless of `stop`.
        if let Some(cap) = cfg.max_trials {
            if trials >= cap { break; }
        }
        // Stop checks BEFORE producing more work.
        match &cfg.stop {
            Stop::Generations(g) if generations >= *g => break,
            Stop::Trials(t) if trials >= *t => break,
            Stop::Until(pred) => {
                if let Some((_, s)) = strategy.best() {
                    if pred(s) { break; }
                }
            }
            _ => {}
        }

        // Ask for a batch against the current strategy snapshot.
        let batch: Vec<<St::Genetic as Genetic>::Genome> =
            (0..batch_size).map(|_| strategy.next()).collect();

        // Evaluate, then report sequentially so steady-state strategies update per-report.
        for (g, s) in evaluate(batch, &fitness) {
            strategy.report(g, s);
            trials += 1;
        }
        generations += 1;
    }

    let (best_genome, best_score) = match strategy.best() {
        Some((g, s)) => (Some(g.clone()), Some(s.clone())),
        None => (None, None),
    };
    Outcome { best_genome, best_score, generations, trials }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genetic::Genetic;
    use crate::strategy::EvolutionStrategy;
    use rand::Rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    struct UnitGenetics;
    impl Genetic for UnitGenetics {
        type Genome = f64;
        fn random(&self, rng: &mut impl Rng) -> f64 { rng.random::<f64>() }
        fn cross(&self, a: &f64, b: &f64, _r: &mut impl Rng) -> f64 { (a + b) / 2.0 }
        fn mutate(&self, g: &f64, rng: &mut impl Rng) -> f64 { (g + rng.random_range(-0.1..0.1)).clamp(0.0, 1.0) }
        fn gene_dist(&self, a: &f64, b: &f64) -> f64 { (a - b).abs() }
    }

    /// Trivial self-contained strategy: emits randoms, keeps the single best.
    struct KeepBest {
        genetics: UnitGenetics,
        rng: ChaCha8Rng,
        best: Option<(f64, f64)>, // (genome, score)
    }
    impl KeepBest {
        fn new(seed: u64) -> Self {
            Self { genetics: UnitGenetics, rng: ChaCha8Rng::seed_from_u64(seed), best: None }
        }
    }
    impl EvolutionStrategy for KeepBest {
        type Genetic = UnitGenetics;
        type Score = f64;
        fn next(&mut self) -> f64 {
            self.genetics.random(&mut self.rng)
        }
        fn report(&mut self, genome: f64, score: f64) {
            if self.best.map_or(true, |(_, s)| score > s) {
                self.best = Some((genome, score));
            }
        }
        fn best(&self) -> Option<(&f64, &f64)> {
            self.best.as_ref().map(|(g, s)| (g, s))
        }
    }

    #[test]
    fn stops_after_trial_budget() {
        let mut strat = KeepBest::new(3);
        let cfg = RunConfig { stop: Stop::Trials(50), batch_size: 1, max_trials: None };
        let out = run(&cfg, &mut strat, |x: &f64| *x);
        assert_eq!(out.trials, 50);
        assert!(out.best_score.unwrap() >= 0.0);
    }

    #[test]
    fn batch_size_rounds_up_trials() {
        let mut strat = KeepBest::new(3);
        let cfg = RunConfig { stop: Stop::Generations(3), batch_size: 4, max_trials: None };
        let out = run(&cfg, &mut strat, |x: &f64| *x);
        assert_eq!(out.generations, 3);
        assert_eq!(out.trials, 12);
    }

    #[test]
    fn until_predicate_stops_run() {
        let mut strat = KeepBest::new(3);
        let cfg = RunConfig { stop: Stop::Until(Box::new(|s: &f64| *s > 0.9)), batch_size: 1, max_trials: None };
        let out = run(&cfg, &mut strat, |x: &f64| *x);
        assert!(out.best_score.unwrap() > 0.9);
    }
}
