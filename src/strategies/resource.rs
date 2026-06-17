use crate::genetic::Genetic;
use crate::strategy::{EvolutionStrategy, Fitness};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// CreateInfo for `ResourceSoftStochastic`.
#[derive(Clone, Copy)]
pub struct ResourceSoftStochasticInfo {
    pub target: usize,
    pub temperature: f64,
    /// Dimensionless sharing-radius multiplier. Each refresh,
    /// `σ = relative_width × median(pairwise gene_dist over the pool)`, so the
    /// niche width auto-scales to whatever `gene_dist` returns for this genome
    /// type — no absolute units are hardcoded. Typical range 0.1–0.3.
    pub relative_width: f64,
    /// Assortative-mating locality: how strongly crossover prefers a mate close
    /// (by `gene_dist`) to the first parent. 0 = no bias (unbiased selection).
    pub mate_locality: f64,
    pub seed: u64,
}

impl Default for ResourceSoftStochasticInfo {
    fn default() -> Self {
        Self {
            target: 100,
            temperature: 0.2,
            relative_width: 0.15,
            mate_locality: 1.0,
            seed: 0,
        }
    }
}

struct Member<G, S> {
    genome: G,
    score: S,
    fitness: f64,
    /// Cached crowding: Σ_j sh(gene_dist(self, j)) over the pool (incl. self ⇒ ≥ 1).
    crowding: f64,
}

/// Small positive floor added to the shifted raw fitness so realized fitness is
/// strictly positive even when every pool member shares the same raw fitness.
const REALIZED_EPS: f64 = 1e-9;

/// Resource Soft Evolution Algorithm: the SSEA steady-state soft cull, but the
/// survival/selection metric is **shared fitness** — positively-shifted raw
/// fitness divided by crowding (instantaneous fitness sharing). Crowded regions
/// split their resource more ways, so they stop growing and the population
/// settles into a stable per-niche carrying capacity (no novelty score). Pairs
/// with assortative mating to decouple niches. Owns its genetics and rng.
pub struct ResourceSoftStochastic<X: Genetic, S> {
    genetics: X,
    rng: ChaCha8Rng,
    target: usize,
    temperature: f64,
    relative_width: f64,
    mate_locality: f64,
    /// Current sharing radius; recomputed each refresh from the pool's spread.
    sigma: f64,
    pool: Vec<Member<X::Genome, S>>,
    best: Option<(X::Genome, S)>,
    /// Bulk-refresh σ + all crowding every this many culls (amortizes O(pool²)).
    refresh_interval: usize,
    since_refresh: usize,
}

impl<X: Genetic, S: Fitness + Clone> ResourceSoftStochastic<X, S> {
    pub fn new(genetics: X, info: ResourceSoftStochasticInfo) -> Self {
        Self {
            genetics,
            rng: ChaCha8Rng::seed_from_u64(info.seed),
            target: info.target.max(2),
            temperature: info.temperature.max(1e-6),
            relative_width: info.relative_width.max(1e-6),
            mate_locality: info.mate_locality.max(0.0),
            sigma: 1e-6,
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

    /// The current sharing radius (changes as the population spreads/condenses).
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// Live convergence controls (e.g. driven by a GUI).
    pub fn set_temperature(&mut self, t: f64) {
        self.temperature = t.max(1e-6);
    }
    pub fn set_relative_width(&mut self, w: f64) {
        self.relative_width = w.max(1e-6);
    }
    pub fn set_mate_locality(&mut self, l: f64) {
        self.mate_locality = l.max(0.0);
    }

    /// Snapshot of the maintained pool's genomes — instrumentation / display.
    pub fn pool_genomes(&self) -> Vec<X::Genome> {
        self.pool.iter().map(|m| m.genome.clone()).collect()
    }

    /// Mutable access to the owned genetics (e.g. to live-tune mutation rates).
    pub fn genetics_mut(&mut self) -> &mut X {
        &mut self.genetics
    }

    /// Sharing radius from the pool's own spread: `relative_width × median pairwise
    /// gene_dist`. Returns a small positive floor when the pool is too small.
    fn compute_sigma(genetics: &X, pool: &[Member<X::Genome, S>], relative_width: f64) -> f64 {
        let n = pool.len();
        if n < 2 {
            return 1e-6;
        }
        let mut dists: Vec<f64> = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                dists.push(genetics.gene_dist(&pool[i].genome, &pool[j].genome));
            }
        }
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = dists[dists.len() / 2];
        (relative_width * med).max(1e-6)
    }

    /// Crowding of `genome`: Σ_j sh(gene_dist) with triangular kernel
    /// `sh(d) = max(0, 1 − d/σ)`, plus 1.0 for self-sharing (so crowding ≥ 1 and
    /// realized fitness can never divide by zero). `skip` excludes a pool index
    /// when recomputing crowding for an existing member.
    fn crowding_of(
        genetics: &X,
        pool: &[Member<X::Genome, S>],
        sigma: f64,
        genome: &X::Genome,
        skip: Option<usize>,
    ) -> f64 {
        let mut c = 0.0;
        for (i, m) in pool.iter().enumerate() {
            if Some(i) == skip {
                continue;
            }
            let d = genetics.gene_dist(genome, &m.genome);
            if d < sigma {
                c += 1.0 - d / sigma;
            }
        }
        c + 1.0
    }

    /// Per-member realized (shared) fitness: positively-shifted raw fitness over
    /// crowding. The shift (`−min_f + ε`) makes sharing well-defined for the
    /// gauntlet's non-positive `fitness = −objective`.
    fn realized_vec(&self) -> Vec<f64> {
        let min_f = self.pool.iter().map(|m| m.fitness).fold(f64::MAX, f64::min);
        self.pool
            .iter()
            .map(|m| (m.fitness - min_f + REALIZED_EPS) / m.crowding)
            .collect()
    }

    /// Boltzmann (softmax over realized/temperature) selection index.
    fn boltzmann_pick(realized: &[f64], temperature: f64, rng: &mut ChaCha8Rng) -> usize {
        let max_f = realized.iter().cloned().fold(f64::MIN, f64::max);
        let weights: Vec<f64> = realized
            .iter()
            .map(|r| ((r - max_f) / temperature).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        let mut pick = rng.random::<f64>() * total;
        for (i, w) in weights.iter().enumerate() {
            pick -= w;
            if pick <= 0.0 {
                return i;
            }
        }
        realized.len() - 1
    }

    /// Pick a mate for parent `a`: weight ∝ Boltzmann(realized) × exp(−locality·dist).
    fn mate_pick(
        genetics: &X,
        pool: &[Member<X::Genome, S>],
        realized: &[f64],
        a: &X::Genome,
        temperature: f64,
        locality: f64,
        rng: &mut ChaCha8Rng,
    ) -> usize {
        let max_f = realized.iter().cloned().fold(f64::MIN, f64::max);
        let weights: Vec<f64> = pool
            .iter()
            .zip(realized.iter())
            .map(|(m, r)| {
                let bf = ((r - max_f) / temperature).exp();
                let d = genetics.gene_dist(a, &m.genome);
                bf * (-locality * d).exp()
            })
            .collect();
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return pool.len() - 1;
        }
        let mut pick = rng.random::<f64>() * total;
        for (i, w) in weights.iter().enumerate() {
            pick -= w;
            if pick <= 0.0 {
                return i;
            }
        }
        pool.len() - 1
    }

    /// Recompute σ and every member's cached crowding from scratch. O(pool²);
    /// called only every `refresh_interval` culls to bound staleness.
    fn refresh_crowding(&mut self) {
        let sigma = Self::compute_sigma(&self.genetics, &self.pool, self.relative_width);
        self.sigma = sigma;
        let c: Vec<f64> = (0..self.pool.len())
            .map(|i| {
                Self::crowding_of(
                    &self.genetics,
                    &self.pool,
                    sigma,
                    &self.pool[i].genome,
                    Some(i),
                )
            })
            .collect();
        for (m, cr) in self.pool.iter_mut().zip(c) {
            m.crowding = cr;
        }
    }

    /// Soft cull, run from `report` when over capacity. Ranks survival by
    /// normalized realized (shared) fitness so crowded members are culled first.
    fn soft_cull(&mut self) {
        // Refresh on the first cull (since_refresh == 0) and every interval after,
        // so σ/crowding are real from the start.
        if self.since_refresh == 0 || self.since_refresh >= self.refresh_interval {
            self.refresh_crowding();
            self.since_refresh = 0;
        }
        self.since_refresh += 1;

        let realized = self.realized_vec();
        let (min_r, max_r) = realized
            .iter()
            .fold((f64::MAX, f64::MIN), |(lo, hi), &r| (lo.min(r), hi.max(r)));
        let rspan = (max_r - min_r).max(1e-9);
        let pressure = (self.pool.len() as f64 / self.target as f64).max(1e-3);
        let best_fit = self.best.as_ref().map(|(_, s)| s.fitness());

        let drained: Vec<Member<X::Genome, S>> = std::mem::take(&mut self.pool);
        let mut survivors: Vec<Member<X::Genome, S>> = Vec::with_capacity(drained.len());
        let mut kept_best = false;
        for (m, r) in drained.into_iter().zip(realized) {
            // Keep exactly one copy of the global best unconditionally.
            if !kept_best && best_fit.map_or(false, |bf| (m.fitness - bf).abs() < f64::EPSILON) {
                kept_best = true;
                survivors.push(m);
                continue;
            }
            let norm = ((r - min_r) / rspan).clamp(0.0, 1.0);
            let p_survive = norm.powf(pressure).clamp(0.0, 1.0);
            if self.rng.random_bool(p_survive) {
                survivors.push(m);
            }
        }
        self.pool = survivors;
    }
}

impl<X: Genetic, S: Fitness + Clone> EvolutionStrategy for ResourceSoftStochastic<X, S> {
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
        let realized = self.realized_vec();
        let pi = Self::boltzmann_pick(&realized, self.temperature, &mut self.rng);
        let parent = self.pool[pi].genome.clone();
        let action: f64 = self.rng.random();
        if action < 0.6 {
            self.genetics.mutate(&parent, &mut self.rng)
        } else if action < 0.9 {
            let oi = Self::mate_pick(
                &self.genetics,
                &self.pool,
                &realized,
                &parent,
                self.temperature,
                self.mate_locality,
                &mut self.rng,
            );
            // Pass the higher-realized parent as `a` (preferred-parent convention).
            let (a, b) = if realized[pi] >= realized[oi] {
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
        let crowding = Self::crowding_of(&self.genetics, &self.pool, self.sigma, &genome, None);
        self.pool.push(Member {
            genome,
            score,
            fitness: fit,
            crowding,
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

    fn mk(x: f64) -> Member<f64, f64> {
        Member {
            genome: x,
            score: x,
            fitness: x,
            crowding: 1.0,
        }
    }

    #[test]
    fn sigma_scales_with_spread() {
        let g = UnitGenetics;
        let tight: Vec<Member<f64, f64>> = [0.50, 0.51, 0.49, 0.50].into_iter().map(mk).collect();
        let spread: Vec<Member<f64, f64>> = [0.0, 0.3, 0.6, 1.0].into_iter().map(mk).collect();
        let s_tight = ResourceSoftStochastic::<UnitGenetics, f64>::compute_sigma(&g, &tight, 0.15);
        let s_spread =
            ResourceSoftStochastic::<UnitGenetics, f64>::compute_sigma(&g, &spread, 0.15);
        assert!(
            s_spread > s_tight,
            "sigma should grow with spread: tight={s_tight} spread={s_spread}"
        );
    }

    #[test]
    fn crowding_penalizes_clusters() {
        let g = UnitGenetics;
        // Four genomes clustered around 0.5, one loner at 0.0.
        let pool: Vec<Member<f64, f64>> =
            [0.50, 0.50, 0.51, 0.49, 0.0].into_iter().map(mk).collect();
        let sigma = 0.1;
        let crowded =
            ResourceSoftStochastic::<UnitGenetics, f64>::crowding_of(&g, &pool, sigma, &0.50, None);
        let loner =
            ResourceSoftStochastic::<UnitGenetics, f64>::crowding_of(&g, &pool, sigma, &0.0, None);
        assert!(
            crowded > loner,
            "clustered genome should be more crowded: crowded={crowded} loner={loner}"
        );
    }

    #[test]
    fn mate_locality_zero_matches_unbiased() {
        let g = UnitGenetics;
        let pool: Vec<Member<f64, f64>> = [(0.1, 0.2), (0.9, 0.8), (0.5, 0.5)]
            .into_iter()
            .map(|(x, f)| Member {
                genome: x,
                score: f,
                fitness: f,
                crowding: 1.0,
            })
            .collect();
        let realized: Vec<f64> = pool.iter().map(|m| m.fitness).collect();
        // With locality 0 the distance factor is exp(0)=1, so mate_pick uses the
        // same weights as boltzmann_pick and consumes one rng draw — identical pick.
        for seed in 0..5u64 {
            let mut r1 = ChaCha8Rng::seed_from_u64(seed);
            let mut r2 = ChaCha8Rng::seed_from_u64(seed);
            let i = ResourceSoftStochastic::<UnitGenetics, f64>::boltzmann_pick(
                &realized, 0.2, &mut r1,
            );
            let j = ResourceSoftStochastic::<UnitGenetics, f64>::mate_pick(
                &g,
                &pool,
                &realized,
                &pool[0].genome,
                0.2,
                0.0,
                &mut r2,
            );
            assert_eq!(i, j, "locality=0 should match unbiased pick (seed {seed})");
        }
    }

    #[test]
    fn resource_pool_grows_and_improves() {
        let mut strat: ResourceSoftStochastic<UnitGenetics, f64> = ResourceSoftStochastic::new(
            UnitGenetics,
            ResourceSoftStochasticInfo {
                target: 20,
                seed: 5,
                mate_locality: 2.0,
                ..Default::default()
            },
        );
        let cfg = RunConfig {
            stop: Stop::Trials(300),
            batch_size: 1,
            max_trials: None,
        };
        let out = run(&cfg, &mut strat, |x: &f64| *x);
        assert!(out.best_score.unwrap() > 0.8, "got {:?}", out.best_score);
    }
}
