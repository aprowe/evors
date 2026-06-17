use crate::genetic::Genetic;
use crate::strategy::{Behavior, EvolutionStrategy, Fitness};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// CreateInfo for `SoftStochastic`.
#[derive(Clone, Copy)]
pub struct SoftStochasticInfo {
    pub target: usize,
    pub temperature: f64,
    /// Weight of novelty vs fitness in survival, `[0, 1]`.
    pub novelty_weight: f64,
    /// Within the novelty term, weight of *behavioral* (solution-space) novelty
    /// vs *genetic* novelty, `[0, 1]`. 0 = pure genetic novelty (the default and
    /// the only behavior for scalar scores).
    pub behavior_weight: f64,
    pub knn: usize,
    pub seed: u64,
}

impl Default for SoftStochasticInfo {
    fn default() -> Self {
        Self {
            target: 100,
            temperature: 0.2,
            novelty_weight: 0.5,
            behavior_weight: 0.0,
            knn: 5,
            seed: 0,
        }
    }
}

struct Member<G, S> {
    genome: G,
    score: S,
    fitness: f64,
    /// Cached genetic novelty (stochastic weighted-kNN over gene_dist; see
    /// `weighted_knn`). Re-sampled each refresh, so it fluctuates — ergodic.
    gnovelty: f64,
    /// Cached behavioral novelty (stochastic weighted-kNN over behavior_dist);
    /// only maintained when `behavior_weight > 0`.
    bnovelty: f64,
}

/// Stochastic Soft Evolution Algorithm: steady-state, soft-culling, no hard
/// cutoffs. Generic over a score `S: Fitness + Behavior`; novelty blends genetic
/// distance (`gene_dist`) and behavioral distance (`S::behavior_dist`). Owns its
/// genetics and rng.
pub struct SoftStochastic<X: Genetic, S> {
    genetics: X,
    rng: ChaCha8Rng,
    target: usize,
    temperature: f64,
    novelty_weight: f64,
    behavior_weight: f64,
    knn: usize,
    pool: Vec<Member<X::Genome, S>>,
    best: Option<(X::Genome, S)>,
    /// Bulk-refresh every this many culls (amortizes the O(pool²) recompute).
    refresh_interval: usize,
    since_refresh: usize,
}

impl<X: Genetic, S: Fitness + Behavior + Clone> SoftStochastic<X, S> {
    pub fn new(genetics: X, info: SoftStochasticInfo) -> Self {
        Self {
            genetics,
            rng: ChaCha8Rng::seed_from_u64(info.seed),
            target: info.target.max(2),
            temperature: info.temperature.max(1e-6),
            novelty_weight: info.novelty_weight.clamp(0.0, 1.0),
            behavior_weight: info.behavior_weight.clamp(0.0, 1.0),
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

    /// Live convergence controls (e.g. driven by a GUI).
    pub fn set_novelty_weight(&mut self, w: f64) {
        self.novelty_weight = w.clamp(0.0, 1.0);
    }
    pub fn set_behavior_weight(&mut self, w: f64) {
        self.behavior_weight = w.clamp(0.0, 1.0);
    }
    pub fn set_temperature(&mut self, t: f64) {
        self.temperature = t.max(1e-6);
    }

    /// Snapshot of the maintained pool's genomes — useful for instrumentation
    /// and visualising the diverse solution set.
    pub fn pool_genomes(&self) -> Vec<X::Genome> {
        self.pool.iter().map(|m| m.genome.clone()).collect()
    }

    /// Mutable access to the owned genetics (e.g. to live-tune mutation rates).
    pub fn genetics_mut(&mut self) -> &mut X {
        &mut self.genetics
    }

    /// Run `f` over each pool genome without cloning — cheap instrumentation.
    pub fn pool_fold<T>(&self, init: T, mut f: impl FnMut(T, &X::Genome) -> T) -> T {
        self.pool.iter().fold(init, |acc, m| f(acc, &m.genome))
    }

    /// The current pool's highest-fitness genome (changes live as the pool
    /// churns — useful for a display that should reflect ongoing exploration,
    /// unlike the monotonic all-time `best`).
    pub fn pool_best_genome(&self) -> Option<X::Genome> {
        self.pool
            .iter()
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
            .map(|m| m.genome.clone())
    }

    /// Boltzmann (softmax over fitness/temperature) selection index.
    fn boltzmann_pick(
        pool: &[Member<X::Genome, S>],
        temperature: f64,
        rng: &mut ChaCha8Rng,
    ) -> usize {
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

    /// **Stochastic weighted kNN.** Instead of the exact `knn` nearest, draw
    /// `knn` neighbours by proximity-weighted sampling (weight `exp(−d/h)`, with
    /// bandwidth `h` = the k-th nearest distance, auto-scaling to local density)
    /// and return their mean distance. The randomness makes novelty **ergodic** —
    /// every genome has a fluctuating, never-zero chance of scoring novel, so rare
    /// lineages ("slivers") survive given enough time — and softens the hard
    /// k-cutoff into a soft proximity gradient. Sampling is Efraimidis–Spirakis
    /// (key `= u^(1/w)`; the `k` largest keys form the sample).
    fn weighted_knn(
        pool: &[Member<X::Genome, S>],
        knn: usize,
        skip: Option<usize>,
        rng: &mut ChaCha8Rng,
        mut dist: impl FnMut(&Member<X::Genome, S>) -> f64,
    ) -> f64 {
        let dists: Vec<f64> = pool
            .iter()
            .enumerate()
            .filter(|(i, _)| Some(*i) != skip)
            .map(|(_, m)| dist(m))
            .collect();
        if dists.is_empty() {
            return 0.0;
        }
        let k = knn.min(dists.len());
        // Bandwidth = the k-th nearest distance (local scale).
        let mut sorted = dists.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let h = sorted[k - 1].max(1e-9);
        // Efraimidis–Spirakis weighted sampling without replacement: key = u^(1/w),
        // the k items with the largest keys are the proximity-weighted sample.
        let mut keyed: Vec<(f64, f64)> = dists
            .iter()
            .map(|&d| {
                let w = (-d / h).exp().max(1e-12);
                let u: f64 = rng.random::<f64>().max(1e-12);
                (u.powf(1.0 / w), d)
            })
            .collect();
        keyed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        keyed.iter().take(k).map(|(_, d)| d).sum::<f64>() / k as f64
    }

    fn genetic_novelty(
        genetics: &X,
        pool: &[Member<X::Genome, S>],
        knn: usize,
        genome: &X::Genome,
        skip: Option<usize>,
        rng: &mut ChaCha8Rng,
    ) -> f64 {
        Self::weighted_knn(pool, knn, skip, rng, |m| {
            genetics.gene_dist(genome, &m.genome)
        })
    }

    fn behavioral_novelty(
        pool: &[Member<X::Genome, S>],
        knn: usize,
        score: &S,
        skip: Option<usize>,
        rng: &mut ChaCha8Rng,
    ) -> f64 {
        Self::weighted_knn(pool, knn, skip, rng, |m| score.behavior_dist(&m.score))
    }

    /// Recompute every member's cached novelty from scratch. O(pool²); called
    /// only every `refresh_interval` culls to bound staleness.
    fn refresh_novelty(&mut self) {
        let use_b = self.behavior_weight > 0.0;
        let g: Vec<f64> = (0..self.pool.len())
            .map(|i| {
                Self::genetic_novelty(
                    &self.genetics,
                    &self.pool,
                    self.knn,
                    &self.pool[i].genome,
                    Some(i),
                    &mut self.rng,
                )
            })
            .collect();
        let b: Vec<f64> = (0..self.pool.len())
            .map(|i| {
                if use_b {
                    Self::behavioral_novelty(
                        &self.pool,
                        self.knn,
                        &self.pool[i].score,
                        Some(i),
                        &mut self.rng,
                    )
                } else {
                    0.0
                }
            })
            .collect();
        for (m, (gn, bn)) in self.pool.iter_mut().zip(g.into_iter().zip(b)) {
            m.gnovelty = gn;
            m.bnovelty = bn;
        }
    }

    /// Soft cull, run from `report` when over capacity. O(pool) using cached
    /// novelty; the cache is bulk-refreshed periodically.
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
        let (min_g, max_g) = self.pool.iter().fold((f64::MAX, f64::MIN), |(lo, hi), m| {
            (lo.min(m.gnovelty), hi.max(m.gnovelty))
        });
        let gspan = (max_g - min_g).max(1e-9);
        let wb = self.behavior_weight;
        let (min_b, bspan) = if wb > 0.0 {
            let (lo, hi) = self.pool.iter().fold((f64::MAX, f64::MIN), |(lo, hi), m| {
                (lo.min(m.bnovelty), hi.max(m.bnovelty))
            });
            (lo, (hi - lo).max(1e-9))
        } else {
            (0.0, 1.0)
        };

        let pressure = (self.pool.len() as f64 / self.target as f64).max(1e-3);
        let w = self.novelty_weight;
        let best_fit = self.best.as_ref().map(|(_, s)| s.fitness());

        let drained: Vec<Member<X::Genome, S>> = std::mem::take(&mut self.pool);
        let mut survivors: Vec<Member<X::Genome, S>> = Vec::with_capacity(drained.len());
        let mut kept_best = false;
        for m in drained.into_iter() {
            // Keep exactly one copy of the global best unconditionally.
            if !kept_best && best_fit.map_or(false, |bf| (m.fitness - bf).abs() < f64::EPSILON) {
                kept_best = true;
                survivors.push(m);
                continue;
            }
            let nf = (m.fitness - min_f) / fspan;
            let gnorm = (m.gnovelty - min_g) / gspan;
            // bw == 0 -> novelty is exactly genetic (preserves scalar-score behavior).
            let nn = if wb > 0.0 {
                let bnorm = (m.bnovelty - min_b) / bspan;
                (1.0 - wb) * gnorm + wb * bnorm
            } else {
                gnorm
            };
            let combined = ((1.0 - w) * nf + w * nn).clamp(0.0, 1.0);
            let p_survive = combined.powf(pressure).clamp(0.0, 1.0);
            if self.rng.random_bool(p_survive) {
                survivors.push(m);
            }
        }
        self.pool = survivors;
    }
}

impl<X: Genetic, S: Fitness + Behavior + Clone> EvolutionStrategy for SoftStochastic<X, S> {
    type Genetic = X;
    type Score = S;

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

    fn report(&mut self, genome: X::Genome, score: S) {
        let fit = score.fitness();
        if self.best.as_ref().map_or(true, |(_, s)| fit > s.fitness()) {
            self.best = Some((genome.clone(), score.clone()));
        }
        // Compute the new member's novelty against the current pool once, O(pool).
        let gnovelty = Self::genetic_novelty(
            &self.genetics,
            &self.pool,
            self.knn,
            &genome,
            None,
            &mut self.rng,
        );
        let bnovelty = if self.behavior_weight > 0.0 {
            Self::behavioral_novelty(&self.pool, self.knn, &score, None, &mut self.rng)
        } else {
            0.0
        };
        self.pool.push(Member {
            genome,
            score,
            fitness: fit,
            gnovelty,
            bnovelty,
        });
        if self.pool.len() > self.target {
            self.soft_cull();
        }
    }

    fn best(&self) -> Option<(&X::Genome, &S)> {
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
        let mut strat: SoftStochastic<UnitGenetics, f64> = SoftStochastic::new(
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
        let mut strat: SoftStochastic<UnitGenetics, f64> = SoftStochastic::new(
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
