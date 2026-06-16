use crate::genetic::Genetic;
use crate::strategy::{EvolutionStrategy, Fitness};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// CreateInfo for `SoftStochastic`.
pub struct SoftStochasticInfo {
    pub target: usize,
    pub temperature: f64,
    pub novelty_weight: f64, // w in [0, 1]
    pub knn: usize,
    pub seed: u64,
}

impl Default for SoftStochasticInfo {
    fn default() -> Self {
        Self {
            target: 100,
            temperature: 0.2,
            novelty_weight: 0.5,
            knn: 5,
            seed: 0,
        }
    }
}

struct Member<G> {
    genome: G,
    fitness: f64,
    /// Cached genetic novelty (mean gene_dist to knn nearest). Refreshed exactly
    /// for a newly inserted member and periodically in bulk for the whole pool,
    /// so the per-report cull stays O(pool) instead of O(pool²).
    novelty: f64,
}

/// Stochastic Soft Evolution Algorithm: steady-state, soft-culling, no hard
/// cutoffs. `Score: Fitness`. v1 novelty is genetic (`gene_dist`). Owns its
/// genetics and rng.
pub struct SoftStochastic<X: Genetic> {
    genetics: X,
    rng: ChaCha8Rng,
    target: usize,
    temperature: f64,
    novelty_weight: f64,
    knn: usize,
    pool: Vec<Member<X::Genome>>,
    best: Option<(X::Genome, f64)>,
    /// Bulk-refresh every this many culls (amortizes the O(pool²) recompute).
    refresh_interval: usize,
    since_refresh: usize,
}

impl<X: Genetic> SoftStochastic<X> {
    pub fn new(genetics: X, info: SoftStochasticInfo) -> Self {
        Self {
            genetics,
            rng: ChaCha8Rng::seed_from_u64(info.seed),
            target: info.target.max(2),
            temperature: info.temperature.max(1e-6),
            novelty_weight: info.novelty_weight.clamp(0.0, 1.0),
            knn: info.knn.max(1),
            pool: Vec::new(),
            best: None,
            refresh_interval: (info.target.max(2) / 8).max(4),
            since_refresh: 0,
        }
    }

    /// Current pool size — useful for instrumentation.
    pub fn pool_size(&self) -> usize {
        self.pool.len()
    }

    /// Snapshot of the maintained pool's genomes — useful for instrumentation
    /// and visualising the diverse solution set.
    pub fn pool_genomes(&self) -> Vec<X::Genome> {
        self.pool.iter().map(|m| m.genome.clone()).collect()
    }

    /// Boltzmann (softmax over fitness/temperature) selection index.
    fn boltzmann_pick(pool: &[Member<X::Genome>], temperature: f64, rng: &mut ChaCha8Rng) -> usize {
        let max_f = pool.iter().map(|m| m.fitness).fold(f64::MIN, f64::max);
        let weights: Vec<f64> = pool
            .iter()
            .map(|m| ((m.fitness - max_f) / temperature).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        let mut pick = rng.random::<f64>() * total;
        for (i, w) in weights.iter().enumerate() {
            pick -= w;
            if pick <= 0.0 {
                return i;
            }
        }
        pool.len() - 1
    }

    /// Mean gene_dist from `genome` to its up-to-knn nearest pool neighbors.
    /// `skip` excludes one index (a member measuring against itself).
    fn genetic_novelty(
        genetics: &X,
        pool: &[Member<X::Genome>],
        knn: usize,
        genome: &X::Genome,
        skip: Option<usize>,
    ) -> f64 {
        let mut dists: Vec<f64> = pool
            .iter()
            .enumerate()
            .filter(|(i, _)| Some(*i) != skip)
            .map(|(_, m)| genetics.gene_dist(genome, &m.genome))
            .collect();
        if dists.is_empty() {
            return 0.0;
        }
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let k = knn.min(dists.len());
        dists.iter().take(k).sum::<f64>() / k as f64
    }

    /// Recompute every member's cached novelty from scratch. O(pool²); called
    /// only every `refresh_interval` culls to bound staleness without paying the
    /// quadratic cost on every report.
    fn refresh_novelty(&mut self) {
        let novs: Vec<f64> = (0..self.pool.len())
            .map(|i| {
                Self::genetic_novelty(
                    &self.genetics,
                    &self.pool,
                    self.knn,
                    &self.pool[i].genome,
                    Some(i),
                )
            })
            .collect();
        for (m, n) in self.pool.iter_mut().zip(novs) {
            m.novelty = n;
        }
    }

    /// Soft cull, run from `report` when over capacity. Uses cached novelty so
    /// the survival pass is O(pool); the cache is bulk-refreshed periodically.
    fn soft_cull(&mut self) {
        self.since_refresh += 1;
        if self.since_refresh >= self.refresh_interval {
            self.refresh_novelty();
            self.since_refresh = 0;
        }

        let (min_f, max_f) = self.pool.iter().fold((f64::MAX, f64::MIN), |(lo, hi), m| {
            (lo.min(m.fitness), hi.max(m.fitness))
        });
        let fspan = (max_f - min_f).max(1e-9);
        let (min_n, max_n) = self.pool.iter().fold((f64::MAX, f64::MIN), |(lo, hi), m| {
            (lo.min(m.novelty), hi.max(m.novelty))
        });
        let nspan = (max_n - min_n).max(1e-9);

        let pressure = (self.pool.len() as f64 / self.target as f64).max(1e-3);
        let w = self.novelty_weight;
        let best_fit = self.best.as_ref().map(|(_, f)| *f);

        let drained: Vec<Member<X::Genome>> = std::mem::take(&mut self.pool);
        let mut survivors: Vec<Member<X::Genome>> = Vec::with_capacity(drained.len());
        let mut kept_best = false;
        for m in drained.into_iter() {
            // Keep exactly one copy of the global best unconditionally.
            if !kept_best && best_fit.map_or(false, |bf| (m.fitness - bf).abs() < f64::EPSILON) {
                kept_best = true;
                survivors.push(m);
                continue;
            }
            let nf = (m.fitness - min_f) / fspan;
            let nn = (m.novelty - min_n) / nspan;
            let combined = ((1.0 - w) * nf + w * nn).clamp(0.0, 1.0);
            let p_survive = combined.powf(pressure).clamp(0.0, 1.0);
            if self.rng.random_bool(p_survive) {
                survivors.push(m);
            }
        }
        self.pool = survivors;
    }
}

impl<X: Genetic> EvolutionStrategy for SoftStochastic<X> {
    type Genetic = X;
    type Score = f64;

    fn next(&mut self) -> X::Genome {
        if self.pool.is_empty() {
            return self.genetics.random(&mut self.rng);
        }
        let ratio = self.pool.len() as f64 / self.target as f64;
        let p_random = (1.0 - ratio).clamp(0.05, 0.9);
        if self.rng.random_bool(p_random) {
            return self.genetics.random(&mut self.rng);
        }
        let pi = Self::boltzmann_pick(&self.pool, self.temperature, &mut self.rng);
        let parent = self.pool[pi].genome.clone();
        let action: f64 = self.rng.random();
        if action < 0.6 {
            self.genetics.mutate(&parent, &mut self.rng)
        } else if action < 0.9 {
            let oi = Self::boltzmann_pick(&self.pool, self.temperature, &mut self.rng);
            // Pass the fitter parent as `a` (the preferred-parent convention).
            let (a, b) = if self.pool[pi].fitness >= self.pool[oi].fitness {
                (parent, self.pool[oi].genome.clone())
            } else {
                (self.pool[oi].genome.clone(), parent)
            };
            self.genetics.cross(&a, &b, &mut self.rng)
        } else {
            parent // replay
        }
    }

    fn report(&mut self, genome: X::Genome, score: f64) {
        let fit = score.fitness();
        if self.best.as_ref().map_or(true, |(_, s)| fit > *s) {
            self.best = Some((genome.clone(), fit));
        }
        // Compute the new member's novelty against the current pool once, O(pool).
        let novelty = Self::genetic_novelty(&self.genetics, &self.pool, self.knn, &genome, None);
        self.pool.push(Member {
            genome,
            fitness: fit,
            novelty,
        });
        if self.pool.len() > self.target {
            self.soft_cull();
        }
    }

    fn best(&self) -> Option<(&X::Genome, &f64)> {
        self.best.as_ref().map(|(g, s)| (g, s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn pool_grows_from_empty_and_improves() {
        let mut strat = SoftStochastic::new(
            UnitGenetics,
            SoftStochasticInfo {
                target: 20,
                seed: 5,
                ..Default::default()
            },
        );
        let cfg = RunConfig {
            stop: Stop::Trials(200),
            batch_size: 1,
            max_trials: None,
        };
        let out = run(&cfg, &mut strat, |x: &f64| *x);
        assert!(out.best_score.unwrap() > 0.8, "got {:?}", out.best_score);
    }

    #[test]
    fn pool_does_not_grow_unbounded() {
        let mut strat = SoftStochastic::new(
            UnitGenetics,
            SoftStochasticInfo {
                target: 10,
                seed: 1,
                ..Default::default()
            },
        );
        for _ in 0..500 {
            let g = strat.next();
            strat.report(g, g);
        }
        // Soft cull keeps the pool near target, never runaway.
        assert!(
            strat.pool.len() <= 10 * 3,
            "pool grew to {}",
            strat.pool.len()
        );
    }
}
