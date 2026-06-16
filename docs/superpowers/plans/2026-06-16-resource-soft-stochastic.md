# ResourceSoftStochastic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `ResourceSoftStochastic` evors strategy that drives diversity through instantaneous fitness sharing (realized = raw/crowding, self-scaling σ, no novelty score) and validate it reaches a stable, non-oscillating equilibrium on the two-circles landscape.

**Architecture:** A self-contained strategy file mirroring `SpeciatedSoftStochastic` (so the tuned `SoftStochastic` RNG stream stays untouched). The SSEA steady-state soft-cull is reused, but the survival/selection metric is **shared fitness**: each member's positively-shifted raw fitness divided by its crowding (Σ of a triangular sharing kernel over genome-space neighbours). The sharing radius σ is recomputed each refresh from the population's own median pairwise `gene_dist`, so nothing is hardcoded in absolute units. Crossover uses assortative mating (reused kernel) so niches decouple into independent equilibria.

**Tech Stack:** Rust (edition 2024), rand 0.9 + rand_chacha 0.9 (`ChaCha8Rng`), the `evors` core traits (`Genetic`, `EvolutionStrategy`, `Fitness`, `PopulationView`, `Observer`), `evors-bench` harness, Python/matplotlib (`scripts/plot.py`) for charts. Build/test via the `rtk` wrapper.

---

## File Structure

- **Create** `src/strategies/resource.rs` — the `ResourceSoftStochastic` strategy + `ResourceSoftStochasticInfo` + unit tests. One clear responsibility: instantaneous-fitness-sharing steady-state evolution.
- **Modify** `src/strategies/mod.rs` — register and re-export the new module.
- **Modify** `src/population.rs` — `PopulationView` impl so the strategy drops into the bench harness.
- **Modify** `evors-bench/src/harness.rs` — `Strat::Resource` variant, `run_config` arm, two default configs.
- **Modify** `evors-bench/src/two_circles.rs` — add an `occupancy()` helper (left/right/gap counts).
- **Create** `evors-bench/tests/resource.rs` — the equilibrium property test + sphere sanity test.
- **Modify** `evors-bench/examples/two_circles.rs` — also evolve the resource strategy; dump its final pool and an occupancy-over-time CSV.
- **Modify** `scripts/plot.py` — three-panel two-circles scatter, occupancy-over-time chart, resource series on the coverage bar chart.

---

### Task 1: ResourceSoftStochastic strategy

**Files:**
- Create: `src/strategies/resource.rs`
- Modify: `src/strategies/mod.rs`

- [ ] **Step 1: Create the strategy file**

Create `src/strategies/resource.rs` with the full contents below. This mirrors `src/strategies/speciated.rs` but replaces the novelty machinery with crowding-based shared fitness.

```rust
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
            .map(|i| Self::crowding_of(&self.genetics, &self.pool, sigma, &self.pool[i].genome, Some(i)))
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
        let s_tight =
            ResourceSoftStochastic::<UnitGenetics, f64>::compute_sigma(&g, &tight, 0.15);
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
```

- [ ] **Step 2: Register and re-export the module**

Modify `src/strategies/mod.rs`. Add the module declaration and re-export, keeping the existing entries:

```rust
pub mod classic;
pub mod custom;
pub mod resource;
pub mod soft;
pub mod speciated;

pub use classic::{ClassicPopulation, ClassicPopulationInfo};
pub use resource::{ResourceSoftStochastic, ResourceSoftStochasticInfo};
pub use soft::{SoftStochastic, SoftStochasticInfo};
pub use speciated::{SpeciatedSoftStochastic, SpeciatedSoftStochasticInfo};
```

- [ ] **Step 3: Run the strategy unit tests**

Run: `rtk cargo test -p evors resource`
Expected: PASS — `sigma_scales_with_spread`, `crowding_penalizes_clusters`, `mate_locality_zero_matches_unbiased`, `resource_pool_grows_and_improves` all green.

- [ ] **Step 4: Format and commit**

```bash
rtk cargo fmt
rtk git add src/strategies/resource.rs src/strategies/mod.rs
rtk git commit -m "feat: add ResourceSoftStochastic (instantaneous fitness sharing)"
```

---

### Task 2: PopulationView impl

**Files:**
- Modify: `src/population.rs`

- [ ] **Step 1: Add the import**

In `src/population.rs`, extend the `strategies` import to include the new type:

```rust
use crate::strategies::{ClassicPopulation, ResourceSoftStochastic, SoftStochastic, SpeciatedSoftStochastic};
```

- [ ] **Step 2: Add the impl**

After the `SpeciatedSoftStochastic` impl block (around line 26), add:

```rust
impl<X: Genetic, S: Fitness + Clone> PopulationView for ResourceSoftStochastic<X, S> {
    type Genome = X::Genome;
    fn members(&self) -> Vec<X::Genome> {
        self.pool_genomes()
    }
}
```

Note: unlike the SSEA/Speciated impls, the bound is `S: Fitness + Clone` (no `Behavior`), matching the strategy's own bounds. The existing `use crate::strategy::{Behavior, Fitness};` line already imports `Fitness`; leave it unchanged (`Behavior` is still used by the other impls).

- [ ] **Step 3: Build to verify it compiles**

Run: `rtk cargo build -p evors`
Expected: builds clean, no warnings about unused imports.

- [ ] **Step 4: Commit**

```bash
rtk git add src/population.rs
rtk git commit -m "feat: PopulationView for ResourceSoftStochastic"
```

---

### Task 3: Gauntlet harness integration

**Files:**
- Modify: `evors-bench/src/harness.rs`

- [ ] **Step 1: Extend the imports**

In `evors-bench/src/harness.rs`, update the `evors::strategies` import to add the new types:

```rust
use evors::strategies::{
    ClassicPopulation, ClassicPopulationInfo, ResourceSoftStochastic, ResourceSoftStochasticInfo,
    SoftStochastic, SoftStochasticInfo, SpeciatedSoftStochastic, SpeciatedSoftStochasticInfo,
};
```

- [ ] **Step 2: Add the `Strat` variant**

Add a `Resource` variant to the `Strat` enum (after `Speciated`):

```rust
#[derive(Clone)]
pub enum Strat {
    Classic(ClassicPopulationInfo),
    Ssea(SoftStochasticInfo),
    Speciated(SpeciatedSoftStochasticInfo),
    Resource(ResourceSoftStochasticInfo),
    De(DeInfo),
}
```

- [ ] **Step 3: Add the `run_config` arm**

In `run_config`, add a match arm (after the `Strat::Speciated` arm):

```rust
        Strat::Resource(info) => {
            let mut info = *info;
            info.seed = seed;
            let mut strat = ResourceSoftStochastic::new(
                VecGenetics::new(p.bounds.clone(), cfg.mutation_step),
                info,
            );
            run_one(p, &mut strat, budget)
        }
```

- [ ] **Step 4: Add default configs**

In `default_configs`, after the speciated loop and before the `de` loop, add:

```rust
    for &(target, rw, loc) in &[(300usize, 0.1f64, 3.0f64), (300, 0.2, 3.0)] {
        v.push(Config {
            label: format!("resource(target={target},rw={rw},loc={loc})"),
            strat: Strat::Resource(ResourceSoftStochasticInfo {
                target,
                temperature: 0.2,
                relative_width: rw,
                mate_locality: loc,
                seed: 0,
            }),
            mutation_step: 0.3,
        });
    }
```

- [ ] **Step 5: Build the bench crate**

Run: `rtk cargo build -p evors-bench`
Expected: builds clean.

- [ ] **Step 6: Commit**

```bash
rtk git add evors-bench/src/harness.rs
rtk git commit -m "feat: run ResourceSoftStochastic across the gauntlet"
```

---

### Task 4: Two-circles occupancy helper + suite tests

**Files:**
- Modify: `evors-bench/src/two_circles.rs`
- Create: `evors-bench/tests/resource.rs`

- [ ] **Step 1: Add the `occupancy` helper**

Append to `evors-bench/src/two_circles.rs`:

```rust
/// Counts of pool members in the left ring (`x < −(C−1.5)`), the right ring
/// (`x > C−1.5`), and the central dead zone (`|x| < GAP`). Positional only — no
/// fitness gate — so it tracks where the population actually sits over time.
pub fn occupancy(pool: &[Vec<f64>]) -> (usize, usize, usize) {
    let left = pool.iter().filter(|x| x[0] < -(C - 1.5)).count();
    let right = pool.iter().filter(|x| x[0] > C - 1.5).count();
    let gap = pool.iter().filter(|x| x[0].abs() < GAP).count();
    (left, right, gap)
}
```

- [ ] **Step 2: Create the suite tests**

Create `evors-bench/tests/resource.rs`:

```rust
//! ResourceSoftStochastic equilibrium + sanity tests.

use evors::population::PopulationView;
use evors::strategies::{ResourceSoftStochastic, ResourceSoftStochasticInfo};
use evors::{RunConfig, Stop, run};
use evors_bench::VecGenetics;
use evors_bench::two_circles::{between_fraction, bounds, covers_both, objective, occupancy};

fn fitness(x: &Vec<f64>) -> f64 {
    -objective(x)
}

#[test]
fn resource_reaches_stable_equilibrium() {
    let mut strat = ResourceSoftStochastic::new(
        VecGenetics::new(bounds(), 0.3),
        ResourceSoftStochasticInfo {
            target: 300,
            temperature: 0.2,
            relative_width: 0.15,
            mate_locality: 3.0,
            seed: 1,
        },
    );

    // Run in chunks, sampling each niche's occupancy over time. `run` advances
    // the same strategy each call, so this is one continuous 40k-eval run.
    let chunks = 40usize;
    let per = 1000usize;
    let mut left: Vec<f64> = Vec::new();
    let mut right: Vec<f64> = Vec::new();
    for _ in 0..chunks {
        let cfg = RunConfig {
            stop: Stop::Trials(per),
            batch_size: 1,
            max_trials: None,
        };
        let _ = run(&cfg, &mut strat, fitness);
        let pool = strat.members();
        let (l, r, _g) = occupancy(&pool);
        left.push(l as f64);
        right.push(r as f64);
    }

    // Diversity: both rings covered, dead zone thinned.
    let pool = strat.members();
    assert!(covers_both(&pool), "resource should cover both rings");
    let bf = between_fraction(&pool);
    println!("resource between-fraction = {bf:.3}");
    assert!(bf < 0.15, "dead zone not thinned: between-fraction {bf:.3}");

    // Stability: in the back half of the run, each niche's occupancy is steady
    // (no boom-bust). Assert it never empties and its relative spread is bounded.
    let half = chunks / 2;
    let spread = |xs: &[f64]| -> f64 {
        let s = &xs[half..];
        let mean = s.iter().sum::<f64>() / s.len() as f64;
        let lo = s.iter().cloned().fold(f64::MAX, f64::min);
        let hi = s.iter().cloned().fold(f64::MIN, f64::max);
        assert!(lo > 0.0, "a niche went empty (boom-bust): {s:?}");
        (hi - lo) / mean.max(1e-9)
    };
    let sl = spread(&left);
    let sr = spread(&right);
    println!("back-half relative spread: left={sl:.2} right={sr:.2}");
    assert!(sl < 0.8, "left niche oscillated: relative spread {sl:.2}");
    assert!(sr < 0.8, "right niche oscillated: relative spread {sr:.2}");
}

#[test]
fn resource_solves_sphere() {
    use evors_bench::harness::{Config, Strat, run_config, succeeded};
    use evors_bench::problems::all;
    let p = all().into_iter().find(|p| p.name == "sphere").unwrap();
    let cfg = Config {
        label: "resource".into(),
        strat: Strat::Resource(ResourceSoftStochasticInfo {
            target: 100,
            temperature: 0.1,
            relative_width: 0.15,
            mate_locality: 2.0,
            seed: 0,
        }),
        mutation_step: 0.3,
    };
    let r = run_config(&p, &cfg, 8000, 1);
    assert!(succeeded(&p, &r), "resource should solve sphere");
}
```

- [ ] **Step 3: Run the suite tests and read the printed diagnostics**

Run: `rtk cargo test -p evors-bench --test resource -- --nocapture`
Expected: both tests PASS. The `--nocapture` flag surfaces the `between-fraction` and `relative spread` prints.

- [ ] **Step 4: Tighten the thresholds if needed**

The thresholds (`bf < 0.15`, spread `< 0.8`) are starting guards. Read the printed values from Step 3:
- If `between-fraction` is comfortably under 0.15 (e.g. ~0.05–0.10), tighten to a value ~0.03 above the observed number so the test stays meaningful but not flaky.
- If either `relative spread` is well under 0.8 (e.g. ~0.3–0.5), tighten to ~0.15 above the observed max.
- If a threshold is *exceeded*, do NOT just loosen it — first try `mate_locality = 5.0` (stronger niche decoupling) or `relative_width = 0.2`, re-run, and only adjust thresholds once the behaviour is genuinely stable. Record the final observed numbers in a one-line comment above each assert.

Re-run Step 3 after any change until green.

- [ ] **Step 5: Commit**

```bash
rtk cargo fmt
rtk git add evors-bench/src/two_circles.rs evors-bench/tests/resource.rs
rtk git commit -m "test: ResourceSoftStochastic reaches stable two-circles equilibrium"
```

---

### Task 5: Two-circles example — resource pool + occupancy CSV

**Files:**
- Modify: `evors-bench/examples/two_circles.rs`

- [ ] **Step 1: Rewrite the example to chunk all runs and emit the new CSVs**

Replace the entire contents of `evors-bench/examples/two_circles.rs` with:

```rust
//! Two-circles demo: evolve plain SSEA, Speciated SSEA, and Resource (fitness
//! sharing) on a landscape that rewards being on either of two unit rings
//! (centred at x = ±5). Dumps the final pools for the scatter chart, and an
//! occupancy-over-time CSV (SSEA vs Resource) for the equilibrium chart. Run:
//! `cargo run --release -p evors-bench --example two_circles`

use evors::population::PopulationView;
use evors::strategies::{
    ResourceSoftStochastic, ResourceSoftStochasticInfo, SoftStochastic, SoftStochasticInfo,
    SpeciatedSoftStochastic, SpeciatedSoftStochasticInfo,
};
use evors::strategy::EvolutionStrategy;
use evors::{RunConfig, Stop, run};
use evors_bench::VecGenetics;
use evors_bench::two_circles::{between_fraction, bounds, objective, occupancy};
use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};

const CHUNK: usize = 1_000;
const CHUNKS: usize = 40; // 40k evaluations total

fn fitness(x: &Vec<f64>) -> f64 {
    -objective(x)
}

/// Drive `strat` for CHUNKS×CHUNK evals, recording per-chunk niche occupancy.
/// Returns (final_pool, occupancy_rows) where each row is (trials, left, right, gap).
fn run_chunked<St>(strat: &mut St) -> (Vec<Vec<f64>>, Vec<(usize, usize, usize, usize)>)
where
    St: EvolutionStrategy<Score = f64> + PopulationView<Genome = Vec<f64>>,
    St::Genetic: evors::genetic::Genetic<Genome = Vec<f64>>,
{
    let mut rows = Vec::with_capacity(CHUNKS);
    for c in 0..CHUNKS {
        let cfg = RunConfig {
            stop: Stop::Trials(CHUNK),
            batch_size: 1,
            max_trials: None,
        };
        let _ = run(&cfg, strat, fitness);
        let (l, r, g) = occupancy(&strat.members());
        rows.push(((c + 1) * CHUNK, l, r, g));
    }
    (strat.members(), rows)
}

fn dump_pool(path: &str, pool: &[Vec<f64>]) {
    let f = File::create(path).unwrap();
    let mut w = BufWriter::new(f);
    writeln!(w, "x,y,fit").unwrap();
    for p in pool {
        writeln!(w, "{},{},{}", p[0], p[1], -objective(p)).unwrap();
    }
}

fn main() {
    create_dir_all("data/bench").unwrap();

    let mut ssea = SoftStochastic::new(
        VecGenetics::new(bounds(), 0.3),
        SoftStochasticInfo {
            target: 300,
            novelty_weight: 0.6,
            temperature: 0.2,
            behavior_weight: 0.0,
            knn: 5,
            seed: 1,
        },
    );
    let (ssea_pool, ssea_occ) = run_chunked(&mut ssea);

    let mut spec = SpeciatedSoftStochastic::new(
        VecGenetics::new(bounds(), 0.3),
        SpeciatedSoftStochasticInfo {
            target: 300,
            novelty_weight: 0.6,
            temperature: 0.2,
            behavior_weight: 0.0,
            knn: 5,
            seed: 1,
            mate_locality: 5.0,
            archive_prob: 0.05,
            speciation_pressure: 0.6,
        },
    );
    let (spec_pool, _spec_occ) = run_chunked(&mut spec);

    let mut res = ResourceSoftStochastic::new(
        VecGenetics::new(bounds(), 0.3),
        ResourceSoftStochasticInfo {
            target: 300,
            temperature: 0.2,
            relative_width: 0.15,
            mate_locality: 3.0,
            seed: 1,
        },
    );
    let (res_pool, res_occ) = run_chunked(&mut res);

    dump_pool("data/bench/two_circles_ssea.csv", &ssea_pool);
    dump_pool("data/bench/two_circles_speciated.csv", &spec_pool);
    dump_pool("data/bench/two_circles_resource.csv", &res_pool);

    // Occupancy over time: SSEA vs Resource (the equilibrium comparison).
    let f = File::create("data/bench/two_circles_occupancy.csv").unwrap();
    let mut w = BufWriter::new(f);
    writeln!(
        w,
        "trials,ssea_left,ssea_right,ssea_gap,resource_left,resource_right,resource_gap"
    )
    .unwrap();
    for (a, b) in ssea_occ.iter().zip(res_occ.iter()) {
        writeln!(
            w,
            "{},{},{},{},{},{},{}",
            a.0, a.1, a.2, a.3, b.1, b.2, b.3
        )
        .unwrap();
    }

    println!(
        "between-fraction  ssea={:.3}  speciated={:.3}  resource={:.3}",
        between_fraction(&ssea_pool),
        between_fraction(&spec_pool),
        between_fraction(&res_pool),
    );
    println!("wrote data/bench/two_circles_{{ssea,speciated,resource}}.csv + _occupancy.csv");
}
```

- [ ] **Step 2: Run the example**

Run: `rtk cargo run --release -p evors-bench --example two_circles`
Expected: prints three between-fractions (resource clearly below ssea) and writes four CSVs under `data/bench/`.

- [ ] **Step 3: Commit**

```bash
rtk cargo fmt
rtk git add evors-bench/examples/two_circles.rs
rtk git commit -m "feat: two_circles example dumps resource pool + occupancy timeseries"
```

---

### Task 6: Charts — three-panel scatter, occupancy, coverage series

**Files:**
- Modify: `scripts/plot.py`

- [ ] **Step 1: Add a resource colour constant**

In `scripts/plot.py`, just after the existing colour constants (`CLASSIC`, `SSEA`), add:

```python
RESOURCE = "#e76f51"  # orange
```

- [ ] **Step 2: Make the two-circles block a three-panel scatter**

Find the "Two-circles" block near the end (it loads `two_circles_ssea.csv` and `two_circles_speciated.csv` and builds a `1, 2` subplot). Replace that whole block with:

```python
    # Two-circles: SSEA vs Speciated vs Resource final pools (dead-zone thinning).
    tc_s = "data/bench/two_circles_ssea.csv"
    tc_p = "data/bench/two_circles_speciated.csv"
    tc_r = "data/bench/two_circles_resource.csv"
    if os.path.exists(tc_s) and os.path.exists(tc_p) and os.path.exists(tc_r):
        def load_xy(path):
            with open(path) as f:
                rows = list(csv.reader(f))
            return np.array([[float(r[0]), float(r[1])] for r in rows[1:]]) if len(rows) > 1 else np.empty((0, 2))
        a, b, c = load_xy(tc_s), load_xy(tc_p), load_xy(tc_r)
        th = np.linspace(0, 2 * np.pi, 200)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), subplot_kw={"aspect": "equal"})
        for ax, data, title, col in [
            (axes[0], a, "SoftStochastic (SSEA)", SSEA),
            (axes[1], b, "Speciated SSEA", "#5e548e"),
            (axes[2], c, "Resource (fitness sharing)", RESOURCE),
        ]:
            for cx in (-5.0, 5.0):
                ax.plot(cx + np.cos(th), np.sin(th), color="gray", ls="--", lw=1, alpha=0.6)
            ax.axvspan(-3, 3, color="#d1495b", alpha=0.08)  # the dead zone |x|<3
            if len(data):
                betw = np.mean(np.abs(data[:, 0]) < 3.0)
                ax.scatter(data[:, 0], data[:, 1], s=10, c=col, alpha=0.7, edgecolors="none")
                ax.set_title(f"{title}\n{betw*100:.0f}% stranded in the dead zone")
            ax.set_xlim(-7, 7); ax.set_ylim(-3, 3)
        fig.suptitle("Two unit circles — diversity mechanisms vs the between-basin dead zone", y=1.02)
        fig.tight_layout()
        fig.savefig("charts/bench/two_circles.png", bbox_inches="tight")
        plt.close(fig)
        print("wrote charts/bench/two_circles.png")

    # Occupancy over time: Resource settles to a stable equilibrium; SSEA is noisier.
    occ_path = "data/bench/two_circles_occupancy.csv"
    if os.path.exists(occ_path):
        hdr, od = load_csv(occ_path)
        col = {name: i for i, name in enumerate(hdr)}
        t = od[:, col["trials"]]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t, od[:, col["ssea_left"]], color=SSEA, lw=1.6, label="SSEA left ring")
        ax.plot(t, od[:, col["ssea_right"]], color=SSEA, lw=1.6, ls="--", label="SSEA right ring")
        ax.plot(t, od[:, col["resource_left"]], color=RESOURCE, lw=1.8, label="Resource left ring")
        ax.plot(t, od[:, col["resource_right"]], color=RESOURCE, lw=1.8, ls="--", label="Resource right ring")
        ax.set_xlabel("evaluations (trials)")
        ax.set_ylabel("pool members in each ring")
        ax.set_title("Two-circles niche occupancy over time — Resource reaches a stable equilibrium")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig("charts/bench/occupancy.png")
        plt.close(fig)
        print("wrote charts/bench/occupancy.png")
```

- [ ] **Step 3: Add the resource series to the coverage bar chart**

Find the coverage-bars block (`if os.path.exists("data/bench/results.csv"):`). Replace the series/bar section (from `classic = [best_cov(...)` through the `ax.bar(...)` calls) with five evenly-spaced series:

```python
        classic = [best_cov(p, "classic") for p in probs]
        ssea = [best_cov(p, "ssea(") for p in probs]
        de = [best_cov(p, "de") for p in probs]
        spec = [best_cov(p, "speciated") for p in probs]
        resource = [best_cov(p, "resource") for p in probs]
        x = np.arange(len(probs))
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.bar(x - 0.32, classic, 0.15, label="ClassicPopulation", color="#d1495b")
        ax.bar(x - 0.16, de, 0.15, label="Differential Evolution", color="#e9c46a")
        ax.bar(x + 0.00, ssea, 0.15, label="SoftStochastic (SSEA)", color="#2a9d8f")
        ax.bar(x + 0.16, spec, 0.15, label="Speciated SSEA", color="#5e548e")
        ax.bar(x + 0.32, resource, 0.15, label="Resource (sharing)", color="#e76f51")
        ax.set_xticks(x); ax.set_xticklabels(probs, rotation=30, ha="right")
        ax.set_ylabel("basins covered (mean)")
        ax.set_title("Basin coverage by strategy (diversity-driven strategies find all optima)")
        ax.legend()
        fig.tight_layout()
        fig.savefig("charts/bench/coverage.png")
        plt.close(fig)
        print("wrote charts/bench/coverage.png")
```

- [ ] **Step 4: Regenerate the charts**

Run: `python scripts/plot.py`
Expected: prints `wrote charts/bench/two_circles.png` and `wrote charts/bench/occupancy.png` (plus the existing chart lines). The two-circles PNG now has three panels; `occupancy.png` shows the resource lines flat in the back half versus noisier SSEA lines.

(If `data/bench/results.csv` is stale or missing the resource series, optionally run `rtk cargo run --release -p evors-bench` first to regenerate the full sweep, then re-run `python scripts/plot.py`. The sweep is long; the two-circles + occupancy charts only need the example from Task 5.)

- [ ] **Step 5: Commit**

```bash
rtk git add scripts/plot.py
rtk git commit -m "feat: charts for resource fitness sharing (3-panel scatter, occupancy, coverage)"
```

---

## Final verification

- [ ] **Run the full evors test suite**

Run: `rtk cargo test`
Expected: all suites green, including the new `resource` unit tests and the `resource` integration suite. No regressions in the existing XOR/circle/NEAT/speciation tests (their RNG streams are untouched).

- [ ] **Confirm formatting**

Run: `rtk cargo fmt --check`
Expected: no diff.

---

## Self-review notes (for the implementer)

- **σ is never hardcoded in absolute units** — `compute_sigma` derives it from `median pairwise gene_dist × relative_width` every refresh. This is the spec's headline requirement; do not introduce an absolute `share_sigma`.
- **No novelty/behavior term** — `ResourceSoftStochasticInfo` deliberately omits `novelty_weight`, `behavior_weight`, and `knn`. Survival and selection both use realized (shared) fitness only.
- **Positive-fitness shift** lives in `realized_vec` (`fitness − min_f + REALIZED_EPS`); the gauntlet's `fitness = −objective` is non-positive, so the shift is load-bearing — don't remove it.
- **Method/field name `sigma`**: the field is `self.sigma`; the calculation is the associated fn `compute_sigma` (distinct name to avoid confusion). Keep them distinct.
- **The equilibrium test is the point of this feature.** If thresholds need adjusting (Task 4 Step 4), prefer tuning `mate_locality`/`relative_width` to restore stability over loosening the asserts, and record the observed numbers in comments.
