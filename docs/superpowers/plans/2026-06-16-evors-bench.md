# evors-bench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `evors-bench`, a headless benchmark suite that sweeps evors' strategies + parameter grids against hard 2-D minimization functions over many seeds, reporting best/error, success rate, evals-to-target, convergence curves, and basin coverage — plus two reusable evors-core interfaces (a runner `Observer` and a `PopulationView` trait).

**Architecture:** Two additive evors-core changes: `run_observed` emits a scalar-friendly `Progress` to an `Observer` each eval (sinks: recording, CSV, future UDP), and `PopulationView` unifies the pool snapshot. The bench crate defines a `Problem` descriptor + function set, a `VecGenetics` (`Genome = Vec<f64>`), a sweep harness that runs each (problem × config × seed) via `run_observed`, computes metrics (incl. coverage of distinct local minima via `PopulationView::members`), and writes a console table + CSVs + charts.

**Tech Stack:** Rust (edition 2024), `rand` 0.9, `rand_chacha` 0.9, evors workspace, Python/matplotlib for charts.

---

## File Structure

```
src/                          # evors core (additive)
  observe.rs                  # Progress, Observer, NoObserver, RecordingObserver, CsvProgressObserver
  population.rs               # PopulationView trait + impls for the two strategies
  trial.rs                    # add run_observed; run delegates
  lib.rs                      # re-exports
evors-bench/
  Cargo.toml
  src/
    lib.rs
    problems.rs               # Problem + the function set
    genetics.rs               # VecGenetics
    harness.rs                # Config, run_one, sweep, RunResult, metrics
    main.rs                   # runner: sweep -> console table + CSVs
  tests/
    bench.rs                  # integration: Sphere success; SSEA coverage > Classic
scripts/plot.py               # add bench charts
```

---

## Task 1: evors core — runner observer

**Files:**
- Create: `src/observe.rs`
- Modify: `src/trial.rs`, `src/lib.rs`

- [ ] **Step 1: Create `src/observe.rs`**

```rust
//! Progress observation for `run_observed`. The `Progress` event is kept
//! scalar-friendly so sinks can serialize it cheaply (e.g. stream over UDP to a
//! topic, or to CSV). Sinks implement `Observer`.

use crate::strategy::Fitness;
use std::io::Write;

/// One evaluation's worth of progress.
pub struct Progress<'a, G, S> {
    pub trials: usize,
    pub generation: usize,
    pub genome: &'a G,
    pub score: &'a S,
    /// The strategy's best *before* this eval was folded in (may be `None`
    /// early). Sinks that want best-so-far including the current eval should
    /// track it from `score` themselves (see `RecordingObserver`).
    pub best_score: Option<&'a S>,
}

/// A sink for progress events.
pub trait Observer<G, S> {
    fn on_eval(&mut self, p: Progress<'_, G, S>);
}

/// Does nothing; the default for plain `run`.
pub struct NoObserver;
impl<G, S> Observer<G, S> for NoObserver {
    fn on_eval(&mut self, _p: Progress<'_, G, S>) {}
}

/// Records best-so-far fitness per evaluation (drives convergence curves).
#[derive(Default)]
pub struct RecordingObserver {
    pub best_fitness: Vec<f64>,
    cur_best: f64,
    started: bool,
}

impl RecordingObserver {
    pub fn new() -> Self {
        Self { best_fitness: Vec::new(), cur_best: f64::MIN, started: false }
    }
}

impl<G, S: Fitness> Observer<G, S> for RecordingObserver {
    fn on_eval(&mut self, p: Progress<'_, G, S>) {
        let f = p.score.fitness();
        if !self.started || f > self.cur_best {
            self.cur_best = f;
            self.started = true;
        }
        self.best_fitness.push(self.cur_best);
    }
}

/// Streams `trials,fitness,best` rows to any writer. The seed of a future
/// UDP/topic sink — same trait, different transport.
pub struct CsvProgressObserver<W> {
    w: W,
    header: bool,
    cur_best: f64,
}

impl<W: Write> CsvProgressObserver<W> {
    pub fn new(w: W) -> Self {
        Self { w, header: false, cur_best: f64::MIN }
    }
}

impl<W: Write, G, S: Fitness> Observer<G, S> for CsvProgressObserver<W> {
    fn on_eval(&mut self, p: Progress<'_, G, S>) {
        if !self.header {
            let _ = writeln!(self.w, "trials,fitness,best");
            self.header = true;
        }
        let f = p.score.fitness();
        if f > self.cur_best {
            self.cur_best = f;
        }
        let _ = writeln!(self.w, "{},{},{}", p.trials, f, self.cur_best);
    }
}
```

- [ ] **Step 2: Add `run_observed` and delegate `run` in `src/trial.rs`**

Add `use crate::observe::{NoObserver, Observer, Progress};` at the top. Replace the body of `run` and add `run_observed`:

```rust
/// Drive a strategy against `fitness`, emitting progress to `obs` each eval.
pub fn run_observed<St, F, O>(
    cfg: &RunConfig<St::Score>,
    strategy: &mut St,
    fitness: F,
    obs: &mut O,
) -> Outcome<<St::Genetic as Genetic>::Genome, St::Score>
where
    St: EvolutionStrategy,
    F: Fn(&<St::Genetic as Genetic>::Genome) -> St::Score + Sync,
    <St::Genetic as Genetic>::Genome: Clone + Send,
    St::Score: Clone + Send,
    O: Observer<<St::Genetic as Genetic>::Genome, St::Score>,
{
    let mut generations = 0usize;
    let mut trials = 0usize;
    let batch_size = cfg.batch_size.max(1);

    loop {
        if let Some(cap) = cfg.max_trials {
            if trials >= cap {
                break;
            }
        }
        match &cfg.stop {
            Stop::Generations(g) if generations >= *g => break,
            Stop::Trials(t) if trials >= *t => break,
            Stop::Until(pred) => {
                if let Some((_, s)) = strategy.best() {
                    if pred(s) {
                        break;
                    }
                }
            }
            _ => {}
        }

        let batch: Vec<<St::Genetic as Genetic>::Genome> =
            (0..batch_size).map(|_| strategy.next()).collect();

        for (g, s) in evaluate(batch, &fitness) {
            trials += 1;
            // Observe BEFORE report (no clone): refs to the current genome/score.
            obs.on_eval(Progress {
                trials,
                generation: generations,
                genome: &g,
                score: &s,
                best_score: strategy.best().map(|(_, bs)| bs),
            });
            strategy.report(g, s);
        }
        generations += 1;
    }

    let (best_genome, best_score) = match strategy.best() {
        Some((g, s)) => (Some(g.clone()), Some(s.clone())),
        None => (None, None),
    };
    Outcome { best_genome, best_score, generations, trials }
}

/// Drive a strategy with no observation (see `run_observed`).
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
    run_observed(cfg, strategy, fitness, &mut NoObserver)
}
```

Delete the old `run` body (the loop is now in `run_observed`). Keep the existing `evaluate` functions and `Stop`/`RunConfig`/`Outcome` unchanged.

- [ ] **Step 3: Re-export in `src/lib.rs`**

```rust
pub mod observe;
```
and add to the existing re-exports:
```rust
pub use observe::{CsvProgressObserver, NoObserver, Observer, Progress, RecordingObserver};
pub use trial::{run, run_observed, Outcome, RunConfig, Stop};
```

- [ ] **Step 4: Add a test to `src/observe.rs`**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::genetic::Genetic;
    use crate::strategies::{SoftStochastic, SoftStochasticInfo};
    use crate::trial::{run_observed, RunConfig, Stop};
    use rand::Rng;

    struct UnitGenetics;
    impl Genetic for UnitGenetics {
        type Genome = f64;
        fn random(&self, rng: &mut impl Rng) -> f64 { rng.random::<f64>() }
        fn cross(&self, a: &f64, b: &f64, _r: &mut impl Rng) -> f64 { (a + b) / 2.0 }
        fn mutate(&self, g: &f64, rng: &mut impl Rng) -> f64 { (g + rng.random_range(-0.05..0.05)).clamp(0.0, 1.0) }
        fn gene_dist(&self, a: &f64, b: &f64) -> f64 { (a - b).abs() }
    }

    #[test]
    fn recording_observer_tracks_monotone_best() {
        let mut strat: SoftStochastic<UnitGenetics, f64> = SoftStochastic::new(
            UnitGenetics,
            SoftStochasticInfo { target: 20, seed: 1, ..Default::default() },
        );
        let cfg = RunConfig { stop: Stop::Trials(300), batch_size: 1, max_trials: None };
        let mut rec = RecordingObserver::new();
        let _ = run_observed(&cfg, &mut strat, |x: &f64| *x, &mut rec);
        assert_eq!(rec.best_fitness.len(), 300, "one record per eval");
        // best-so-far must be non-decreasing.
        for w in rec.best_fitness.windows(2) {
            assert!(w[1] >= w[0], "best-so-far regressed: {} -> {}", w[0], w[1]);
        }
    }
}
```

- [ ] **Step 5: Run tests + the existing suite (no regressions)**

Run: `cargo test -p evors`
Expected: PASS (new observe test + all existing tests, since `run` is behavior-preserving).

- [ ] **Step 6: Commit**

```bash
rtk git add src/observe.rs src/trial.rs src/lib.rs
rtk git commit -m "feat: add run_observed + Observer (recording/CSV sinks)"
```

---

## Task 2: evors core — `PopulationView`

**Files:**
- Create: `src/population.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Create `src/population.rs`**

```rust
//! A uniform view of a strategy's maintained population/pool, so tooling (e.g.
//! the benchmark coverage metric) can read members without branching per type.

use crate::genetic::Genetic;
use crate::strategies::{ClassicPopulation, SoftStochastic};

pub trait PopulationView {
    type Genome: Clone;
    /// Snapshot of the maintained population/pool genomes.
    fn members(&self) -> Vec<Self::Genome>;
}

impl<X: Genetic, S> PopulationView for SoftStochastic<X, S> {
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
    use crate::trial::{run, RunConfig, Stop};
    use rand::Rng;

    struct UnitGenetics;
    impl Genetic for UnitGenetics {
        type Genome = f64;
        fn random(&self, rng: &mut impl Rng) -> f64 { rng.random::<f64>() }
        fn cross(&self, a: &f64, b: &f64, _r: &mut impl Rng) -> f64 { (a + b) / 2.0 }
        fn mutate(&self, g: &f64, rng: &mut impl Rng) -> f64 { (g + rng.random_range(-0.05..0.05)).clamp(0.0, 1.0) }
        fn gene_dist(&self, a: &f64, b: &f64) -> f64 { (a - b).abs() }
    }

    #[test]
    fn members_populated_after_run() {
        let mut strat: SoftStochastic<UnitGenetics, f64> = SoftStochastic::new(
            UnitGenetics,
            SoftStochasticInfo { target: 15, seed: 2, ..Default::default() },
        );
        let cfg = RunConfig { stop: Stop::Trials(200), batch_size: 1, max_trials: None };
        let _ = run(&cfg, &mut strat, |x: &f64| *x);
        assert!(!strat.members().is_empty(), "pool should be non-empty after a run");
    }
}
```

- [ ] **Step 2: Re-export in `src/lib.rs`**

```rust
pub mod population;
pub use population::PopulationView;
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p evors population`
Expected: PASS (1 test).

- [ ] **Step 4: Commit**

```bash
rtk git add src/population.rs src/lib.rs
rtk git commit -m "feat: add PopulationView trait unifying pool snapshots"
```

---

## Task 3: evors-bench scaffold + problems

**Files:**
- Create: `evors-bench/Cargo.toml`, `evors-bench/src/lib.rs`, `evors-bench/src/problems.rs`
- Modify: root `Cargo.toml` (workspace members)

- [ ] **Step 1: Add the member to root `Cargo.toml`**

```toml
[workspace]
members = [".", "evors-neat", "evors-cppn", "evors-bench"]
```

- [ ] **Step 2: Create `evors-bench/Cargo.toml`**

```toml
[package]
name = "evors-bench"
version = "0.1.0"
edition = "2024"

[dependencies]
evors = { path = ".." }
rand = "0.9"
rand_chacha = "0.9"
```

- [ ] **Step 3: Create `evors-bench/src/problems.rs`**

```rust
//! Standard continuous **minimization** test functions (2-D), each with its
//! bounds, known global minimum, global optima, and notable local minima (for
//! basin coverage). `value_eps` is the success tolerance on objective value;
//! `basin_eps` is the L2 distance a pool member must be within to "occupy" a
//! local minimum.

use std::f64::consts::PI;

pub struct Problem {
    pub name: &'static str,
    pub dim: usize,
    pub bounds: Vec<(f64, f64)>,
    pub objective: fn(&[f64]) -> f64,
    pub global_min: f64,
    pub optima: Vec<Vec<f64>>,
    pub local_minima: Vec<Vec<f64>>,
    pub value_eps: f64,
    pub basin_eps: f64,
}

// ---- objective functions ----

fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum()
}
fn rosenbrock(x: &[f64]) -> f64 {
    let (a, b) = (x[0], x[1]);
    (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2)
}
fn rastrigin(x: &[f64]) -> f64 {
    10.0 * x.len() as f64
        + x.iter().map(|v| v * v - 10.0 * (2.0 * PI * v).cos()).sum::<f64>()
}
fn eggcrate(x: &[f64]) -> f64 {
    let (a, b) = (x[0], x[1]);
    a * a + b * b + 25.0 * (a.sin().powi(2) + b.sin().powi(2))
}
fn ackley(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let s1 = x.iter().map(|v| v * v).sum::<f64>() / n;
    let s2 = x.iter().map(|v| (2.0 * PI * v).cos()).sum::<f64>() / n;
    -20.0 * (-0.2 * s1.sqrt()).exp() - s2.exp() + 20.0 + std::f64::consts::E
}
fn griewank(x: &[f64]) -> f64 {
    let s = x.iter().map(|v| v * v).sum::<f64>() / 4000.0;
    let p: f64 = x.iter().enumerate().map(|(i, v)| (v / ((i + 1) as f64).sqrt()).cos()).product();
    s - p + 1.0
}
fn schwefel(x: &[f64]) -> f64 {
    418.982_887_272_433_8 * x.len() as f64
        - x.iter().map(|v| v * v.abs().sqrt().sin()).sum::<f64>()
}
fn dropwave(x: &[f64]) -> f64 {
    let r2 = x[0] * x[0] + x[1] * x[1];
    -(1.0 + (12.0 * r2.sqrt()).cos()) / (0.5 * r2 + 2.0)
}
fn himmelblau(x: &[f64]) -> f64 {
    let (a, b) = (x[0], x[1]);
    (a * a + b - 11.0).powi(2) + (a + b * b - 7.0).powi(2)
}
fn cross_in_tray(x: &[f64]) -> f64 {
    let (a, b) = (x[0], x[1]);
    let inner = (100.0 - (a * a + b * b).sqrt() / PI).abs();
    -0.0001 * ((a.sin() * b.sin() * inner.exp()).abs() + 1.0).powf(0.1)
}

// ---- helpers ----

fn b2(lo: f64, hi: f64) -> Vec<(f64, f64)> {
    vec![(lo, hi), (lo, hi)]
}

/// Integer-lattice points within `[lo, hi]^2` (used for grid-structured minima).
fn int_grid(lo: f64, hi: f64) -> Vec<Vec<f64>> {
    let mut v = Vec::new();
    let a = lo.ceil() as i64;
    let b = hi.floor() as i64;
    for i in a..=b {
        for j in a..=b {
            v.push(vec![i as f64, j as f64]);
        }
    }
    v
}

/// Multiples-of-π lattice within `[lo, hi]^2` (Egg Crate local minima).
fn pi_grid(lo: f64, hi: f64) -> Vec<Vec<f64>> {
    let mut v = Vec::new();
    let k = (lo / PI).ceil() as i64;
    let m = (hi / PI).floor() as i64;
    for i in k..=m {
        for j in k..=m {
            v.push(vec![i as f64 * PI, j as f64 * PI]);
        }
    }
    v
}

/// The full benchmark set.
pub fn all() -> Vec<Problem> {
    let himmel = vec![
        vec![3.0, 2.0],
        vec![-2.805118, 3.131312],
        vec![-3.779310, -3.283186],
        vec![3.584428, -1.848126],
    ];
    let cross = vec![
        vec![1.349406, 1.349406],
        vec![1.349406, -1.349406],
        vec![-1.349406, 1.349406],
        vec![-1.349406, -1.349406],
    ];
    vec![
        Problem { name: "sphere", dim: 2, bounds: b2(-5.0, 5.0), objective: sphere,
            global_min: 0.0, optima: vec![vec![0.0, 0.0]], local_minima: vec![vec![0.0, 0.0]],
            value_eps: 1e-3, basin_eps: 0.3 },
        Problem { name: "rosenbrock", dim: 2, bounds: b2(-2.0, 2.0), objective: rosenbrock,
            global_min: 0.0, optima: vec![vec![1.0, 1.0]], local_minima: vec![vec![1.0, 1.0]],
            value_eps: 1e-2, basin_eps: 0.2 },
        Problem { name: "rastrigin", dim: 2, bounds: b2(-5.12, 5.12), objective: rastrigin,
            global_min: 0.0, optima: vec![vec![0.0, 0.0]], local_minima: int_grid(-5.0, 5.0),
            value_eps: 1e-2, basin_eps: 0.3 },
        Problem { name: "eggcrate", dim: 2, bounds: b2(-5.0, 5.0), objective: eggcrate,
            global_min: 0.0, optima: vec![vec![0.0, 0.0]], local_minima: pi_grid(-5.0, 5.0),
            value_eps: 1e-2, basin_eps: 0.5 },
        Problem { name: "ackley", dim: 2, bounds: b2(-5.0, 5.0), objective: ackley,
            global_min: 0.0, optima: vec![vec![0.0, 0.0]], local_minima: int_grid(-4.0, 4.0),
            value_eps: 1e-2, basin_eps: 0.3 },
        Problem { name: "griewank", dim: 2, bounds: b2(-10.0, 10.0), objective: griewank,
            global_min: 0.0, optima: vec![vec![0.0, 0.0]], local_minima: int_grid(-9.0, 9.0),
            value_eps: 1e-2, basin_eps: 0.5 },
        Problem { name: "schwefel", dim: 2, bounds: b2(-500.0, 500.0), objective: schwefel,
            global_min: 0.0, optima: vec![vec![420.9687, 420.9687]],
            local_minima: vec![vec![420.9687, 420.9687], vec![420.9687, -302.5249],
                vec![-302.5249, 420.9687], vec![-302.5249, -302.5249]],
            value_eps: 1.0, basin_eps: 20.0 },
        Problem { name: "dropwave", dim: 2, bounds: b2(-5.12, 5.12), objective: dropwave,
            global_min: -1.0, optima: vec![vec![0.0, 0.0]], local_minima: vec![vec![0.0, 0.0]],
            value_eps: 1e-2, basin_eps: 0.3 },
        Problem { name: "himmelblau", dim: 2, bounds: b2(-5.0, 5.0), objective: himmelblau,
            global_min: 0.0, optima: himmel.clone(), local_minima: himmel,
            value_eps: 1e-2, basin_eps: 0.4 },
        Problem { name: "cross_in_tray", dim: 2, bounds: b2(-10.0, 10.0), objective: cross_in_tray,
            global_min: -2.062612, optima: cross.clone(), local_minima: cross,
            value_eps: 1e-2, basin_eps: 0.4 },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn objective_matches_global_min_at_each_optimum() {
        for p in all() {
            for opt in &p.optima {
                let v = (p.objective)(opt);
                assert!(
                    (v - p.global_min).abs() < p.value_eps.max(1e-2),
                    "{}: objective({:?}) = {} != global_min {}",
                    p.name, opt, v, p.global_min
                );
            }
        }
    }

    #[test]
    fn local_minima_lie_within_bounds() {
        for p in all() {
            for m in &p.local_minima {
                for (v, (lo, hi)) in m.iter().zip(&p.bounds) {
                    assert!(*v >= *lo - 1e-9 && *v <= *hi + 1e-9, "{}: {:?} out of bounds", p.name, m);
                }
            }
        }
    }
}
```

- [ ] **Step 4: Create `evors-bench/src/lib.rs`**

```rust
//! evors-bench — benchmark evors strategies against hard 2-D minimizers.

pub mod genetics;
pub mod harness;
pub mod problems;

pub use genetics::VecGenetics;
pub use problems::{all, Problem};
```

(`genetics` and `harness` are added in later tasks; comment those two `pub mod`/`pub use` lines out until then so it compiles, or implement Task 4 next.)

- [ ] **Step 5: Run the problem tests**

Run: `cargo test -p evors-bench problems`
Expected: PASS (2 tests). (Temporarily comment the `genetics`/`harness` modules in `lib.rs`.)

- [ ] **Step 6: Commit**

```bash
rtk git add Cargo.toml evors-bench/
rtk git commit -m "feat(bench): scaffold + minimization problem set"
```

---

## Task 4: `VecGenetics`

**Files:**
- Create: `evors-bench/src/genetics.rs`
- Modify: `evors-bench/src/lib.rs` (enable the `genetics` module)

- [ ] **Step 1: Create `evors-bench/src/genetics.rs`**

```rust
//! Real-vector genetics for continuous optimization. `Genome = Vec<f64>`; here
//! the genome *is* the solution, so `gene_dist` is the solution distance that
//! drives SSEA's basin spreading.

use evors::Genetic;
use rand::Rng;

pub struct VecGenetics {
    bounds: Vec<(f64, f64)>,
    /// Per-coordinate uniform mutation half-range.
    step: f64,
}

impl VecGenetics {
    pub fn new(bounds: Vec<(f64, f64)>, step: f64) -> Self {
        Self { bounds, step: step.max(1e-9) }
    }
}

impl Genetic for VecGenetics {
    type Genome = Vec<f64>;

    fn random(&self, rng: &mut impl Rng) -> Vec<f64> {
        self.bounds.iter().map(|(lo, hi)| rng.random_range(*lo..*hi)).collect()
    }

    fn cross(&self, a: &Vec<f64>, b: &Vec<f64>, rng: &mut impl Rng) -> Vec<f64> {
        a.iter().zip(b).map(|(x, y)| if rng.random_bool(0.5) { *x } else { *y }).collect()
    }

    fn mutate(&self, g: &Vec<f64>, rng: &mut impl Rng) -> Vec<f64> {
        g.iter()
            .zip(&self.bounds)
            .map(|(v, (lo, hi))| (v + rng.random_range(-self.step..self.step)).clamp(*lo, *hi))
            .collect()
    }

    fn gene_dist(&self, a: &Vec<f64>, b: &Vec<f64>) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn random_and_mutate_stay_in_bounds() {
        let g = VecGenetics::new(vec![(-2.0, 2.0), (0.0, 1.0)], 0.5);
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let mut genome = g.random(&mut rng);
        for _ in 0..200 {
            genome = g.mutate(&genome, &mut rng);
            assert!(genome[0] >= -2.0 && genome[0] <= 2.0);
            assert!(genome[1] >= 0.0 && genome[1] <= 1.0);
        }
    }

    #[test]
    fn gene_dist_is_euclidean() {
        let g = VecGenetics::new(vec![(0.0, 1.0), (0.0, 1.0)], 0.1);
        assert!((g.gene_dist(&vec![0.0, 0.0], &vec![3.0, 4.0]) - 5.0).abs() < 1e-9);
    }
}
```

- [ ] **Step 2: Enable the module in `lib.rs`** (uncomment `pub mod genetics;` and `pub use genetics::VecGenetics;`).

- [ ] **Step 3: Run tests**

Run: `cargo test -p evors-bench genetics`
Expected: PASS (2 tests).

- [ ] **Step 4: Commit**

```bash
rtk git add evors-bench/src/genetics.rs evors-bench/src/lib.rs
rtk git commit -m "feat(bench): VecGenetics real-vector genetics"
```

---

## Task 5: harness — single run + metrics

**Files:**
- Create: `evors-bench/src/harness.rs`
- Modify: `evors-bench/src/lib.rs` (enable `harness`)

- [ ] **Step 1: Create `evors-bench/src/harness.rs`**

```rust
//! Run a strategy on a problem and compute metrics.

use crate::genetics::VecGenetics;
use crate::problems::Problem;
use evors::genetic::Genetic;
use evors::population::PopulationView;
use evors::strategies::{
    ClassicPopulation, ClassicPopulationInfo, SoftStochastic, SoftStochasticInfo,
};
use evors::strategy::EvolutionStrategy;
use evors::{run_observed, RecordingObserver, RunConfig, Stop};

/// The result of one (problem, config, seed) run.
pub struct RunResult {
    /// best-so-far *objective value* (minimized) per evaluation.
    pub convergence: Vec<f64>,
    pub final_pool: Vec<Vec<f64>>,
    pub best_x: Vec<f64>,
    pub best_value: f64,
}

/// Which strategy + its parameters (seed is injected by the harness).
#[derive(Clone)]
pub enum Strat {
    Classic(ClassicPopulationInfo),
    Ssea(SoftStochasticInfo),
}

/// A full config: a strategy spec, the genetics mutation step, and a label.
#[derive(Clone)]
pub struct Config {
    pub label: String,
    pub strat: Strat,
    pub mutation_step: f64,
}

fn run_one<St>(p: &Problem, strat: &mut St, budget: usize) -> RunResult
where
    St: EvolutionStrategy<Score = f64> + PopulationView<Genome = Vec<f64>>,
    St::Genetic: Genetic<Genome = Vec<f64>>,
{
    let cfg = RunConfig { stop: Stop::Trials(budget), batch_size: 1, max_trials: None };
    let mut rec = RecordingObserver::new();
    let objective = p.objective;
    let out = run_observed(&cfg, strat, move |x: &Vec<f64>| -objective(x), &mut rec);

    // Recorded fitness is -objective; convert to minimized objective value.
    let convergence: Vec<f64> = rec.best_fitness.iter().map(|f| -f).collect();
    let best_value = out.best_score.map(|f| -f).unwrap_or(f64::INFINITY);
    let best_x = out.best_genome.unwrap_or_default();
    RunResult { convergence, final_pool: strat.members(), best_x, best_value }
}

/// Run a config on a problem at a given seed.
pub fn run_config(p: &Problem, cfg: &Config, budget: usize, seed: u64) -> RunResult {
    match &cfg.strat {
        Strat::Classic(info) => {
            let genetics = VecGenetics::new(p.bounds.clone(), cfg.mutation_step);
            let mut info = info.clone();
            info.seed = seed;
            let mut strat = ClassicPopulation::new(genetics, info);
            run_one(p, &mut strat, budget)
        }
        Strat::Ssea(info) => {
            let genetics = VecGenetics::new(p.bounds.clone(), cfg.mutation_step);
            let mut info = info.clone();
            info.seed = seed;
            let mut strat = SoftStochastic::new(genetics, info);
            run_one(p, &mut strat, budget)
        }
    }
}

// ---- metrics over a single RunResult ----

pub fn error(p: &Problem, r: &RunResult) -> f64 {
    r.best_value - p.global_min
}

pub fn succeeded(p: &Problem, r: &RunResult) -> bool {
    error(p, r) < p.value_eps
}

/// First eval index reaching within `value_eps` of the optimum (`None` if never).
pub fn evals_to_target(p: &Problem, r: &RunResult) -> Option<usize> {
    r.convergence.iter().position(|v| v - p.global_min < p.value_eps)
}

/// Distinct local minima with at least one final-pool member within `basin_eps`.
pub fn coverage(p: &Problem, r: &RunResult) -> usize {
    p.local_minima
        .iter()
        .filter(|m| {
            r.final_pool.iter().any(|x| {
                x.iter().zip(m.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt() < p.basin_eps
            })
        })
        .count()
}
```

- [ ] **Step 2: Enable the module in `lib.rs`** (uncomment `pub mod harness;`).

- [ ] **Step 3: Add `ClassicPopulationInfo: Clone` + `SoftStochasticInfo: Clone` if missing**

These CreateInfo structs must derive `Clone` (the harness clones them per seed). Check `src/strategies/classic.rs` and `src/strategies/soft.rs`; add `#[derive(Clone)]` to `ClassicPopulationInfo` and `SoftStochasticInfo` if not present. `SoftStochasticInfo` contains only `Copy` scalars, `ClassicPopulationInfo` likewise — `#[derive(Clone, Copy)]` is fine.

Run: `cargo build -p evors-bench`
Expected: builds.

- [ ] **Step 4: Add a harness test**

Append to `harness.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::problems::all;

    fn problem(name: &str) -> Problem {
        all().into_iter().find(|p| p.name == name).unwrap()
    }

    #[test]
    fn ssea_solves_sphere() {
        let p = problem("sphere");
        let cfg = Config {
            label: "ssea".into(),
            strat: Strat::Ssea(SoftStochasticInfo {
                target: 80, novelty_weight: 0.2, ..Default::default()
            }),
            mutation_step: 0.3,
        };
        let r = run_config(&p, &cfg, 6000, 7);
        assert!(succeeded(&p, &r), "error {} not < {}", error(&p, &r), p.value_eps);
        assert!(coverage(&p, &r) >= 1);
    }
}
```

- [ ] **Step 5: Run the test**

Run: `cargo test -p evors-bench harness`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
rtk git add evors-bench/src/harness.rs evors-bench/src/lib.rs src/strategies/classic.rs src/strategies/soft.rs
rtk git commit -m "feat(bench): single-run harness + metrics (error/success/evals/coverage)"
```

---

## Task 6: sweep + aggregation + reporting

**Files:**
- Modify: `evors-bench/src/harness.rs` (add sweep + aggregate)
- Create: `evors-bench/src/main.rs`

- [ ] **Step 1: Add aggregation to `harness.rs`**

```rust
/// Aggregated metrics for one (problem, config) over many seeds.
pub struct Summary {
    pub problem: String,
    pub config: String,
    pub mean_best: f64,
    pub mean_error: f64,
    pub success_rate: f64,
    pub median_evals: Option<usize>,
    pub mean_coverage: f64,
    /// Mean best-so-far value per eval, across seeds (for charts).
    pub mean_convergence: Vec<f64>,
    /// Final pool of the first seed (for the scatter chart).
    pub sample_pool: Vec<Vec<f64>>,
}

fn median(mut xs: Vec<usize>) -> Option<usize> {
    if xs.is_empty() {
        return None;
    }
    xs.sort_unstable();
    Some(xs[xs.len() / 2])
}

/// Run `seeds` repetitions of a config on a problem and aggregate.
pub fn summarize(p: &Problem, cfg: &Config, budget: usize, seeds: u64) -> Summary {
    let runs: Vec<RunResult> = (0..seeds).map(|s| run_config(p, cfg, budget, s)).collect();
    let n = seeds as f64;

    let mean_best = runs.iter().map(|r| r.best_value).sum::<f64>() / n;
    let mean_error = runs.iter().map(|r| error(p, r)).sum::<f64>() / n;
    let success_rate = runs.iter().filter(|r| succeeded(p, r)).count() as f64 / n;
    let median_evals = median(runs.iter().filter_map(|r| evals_to_target(p, r)).collect());
    let mean_coverage = runs.iter().map(|r| coverage(p, r) as f64).sum::<f64>() / n;

    let len = runs.iter().map(|r| r.convergence.len()).min().unwrap_or(0);
    let mean_convergence: Vec<f64> = (0..len)
        .map(|i| runs.iter().map(|r| r.convergence[i]).sum::<f64>() / n)
        .collect();

    Summary {
        problem: p.name.to_string(),
        config: cfg.label.clone(),
        mean_best,
        mean_error,
        success_rate,
        median_evals,
        mean_coverage,
        mean_convergence,
        sample_pool: runs.into_iter().next().map(|r| r.final_pool).unwrap_or_default(),
    }
}

/// The default config grid: ClassicPopulation vs SoftStochastic, a few params each.
pub fn default_configs() -> Vec<Config> {
    let mut v = Vec::new();
    for &(pop, tour, mr) in &[(100usize, 2usize, 0.6f64), (200, 3, 0.8)] {
        v.push(Config {
            label: format!("classic(pop={pop},t={tour},mr={mr})"),
            strat: Strat::Classic(ClassicPopulationInfo {
                pop_size: pop, tournament: tour, mutation_rate: mr, seed: 0,
            }),
            mutation_step: 0.3,
        });
    }
    for &(target, nw, temp) in &[(150usize, 0.2f64, 0.1f64), (200, 0.5, 0.15), (300, 0.7, 0.2)] {
        v.push(Config {
            label: format!("ssea(target={target},nw={nw},T={temp})"),
            strat: Strat::Ssea(SoftStochasticInfo {
                target, novelty_weight: nw, temperature: temp, behavior_weight: 0.0,
                knn: 5, seed: 0,
            }),
            mutation_step: 0.3,
        });
    }
    v
}
```

- [ ] **Step 2: Create `evors-bench/src/main.rs`**

```rust
//! Run the benchmark sweep: print a per-problem leaderboard and write CSVs.

use evors_bench::harness::{default_configs, summarize, Summary};
use evors_bench::problems::all;
use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};

const BUDGET: usize = 20_000;
const SEEDS: u64 = 15;

fn main() {
    create_dir_all("data/bench").unwrap();
    let configs = default_configs();
    let problems = all();
    let mut results: Vec<Summary> = Vec::new();

    for p in &problems {
        println!("\n=== {} (global_min {:.4}) ===", p.name, p.global_min);
        println!(
            "{:<34} {:>10} {:>10} {:>8} {:>10} {:>9}",
            "config", "best", "error", "succ%", "med-evals", "coverage"
        );
        let mut summaries: Vec<Summary> =
            configs.iter().map(|c| summarize(p, c, BUDGET, SEEDS)).collect();
        // Leaderboard: best error first, then highest coverage.
        summaries.sort_by(|a, b| {
            a.mean_error
                .partial_cmp(&b.mean_error)
                .unwrap()
                .then(b.mean_coverage.partial_cmp(&a.mean_coverage).unwrap())
        });
        for s in &summaries {
            let evals = s.median_evals.map(|e| e.to_string()).unwrap_or_else(|| "-".into());
            println!(
                "{:<34} {:>10.4} {:>10.4} {:>7.0}% {:>10} {:>9.1}",
                s.config, s.mean_best, s.mean_error, s.success_rate * 100.0, evals, s.mean_coverage
            );
        }
        write_convergence(p.name, &summaries);
        write_scatter(p.name, &summaries);
        results.extend(summaries);
    }
    write_results(&results);
    println!("\nwrote data/bench/results.csv + per-problem convergence/scatter CSVs");
}

fn write_results(results: &[Summary]) {
    let f = File::create("data/bench/results.csv").unwrap();
    let mut w = BufWriter::new(f);
    writeln!(w, "problem,config,mean_best,mean_error,success_rate,median_evals,mean_coverage").unwrap();
    for s in results {
        let evals = s.median_evals.map(|e| e.to_string()).unwrap_or_default();
        writeln!(
            w, "{},{},{},{},{},{},{}",
            s.problem, s.config, s.mean_best, s.mean_error, s.success_rate, evals, s.mean_coverage
        ).unwrap();
    }
}

fn write_convergence(problem: &str, summaries: &[Summary]) {
    let f = File::create(format!("data/bench/{problem}_convergence.csv")).unwrap();
    let mut w = BufWriter::new(f);
    let labels: Vec<&str> = summaries.iter().map(|s| s.config.as_str()).collect();
    writeln!(w, "eval,{}", labels.join(",")).unwrap();
    let len = summaries.iter().map(|s| s.mean_convergence.len()).min().unwrap_or(0);
    for i in (0..len).step_by(50) {
        let row: Vec<String> = summaries.iter().map(|s| s.mean_convergence[i].to_string()).collect();
        writeln!(w, "{i},{}", row.join(",")).unwrap();
    }
}

fn write_scatter(problem: &str, summaries: &[Summary]) {
    let f = File::create(format!("data/bench/{problem}_scatter.csv")).unwrap();
    let mut w = BufWriter::new(f);
    writeln!(w, "config,x,y").unwrap();
    for s in summaries {
        for x in &s.sample_pool {
            if x.len() >= 2 {
                writeln!(w, "{},{},{}", s.config, x[0], x[1]).unwrap();
            }
        }
    }
}
```

- [ ] **Step 3: Build**

Run: `cargo build -p evors-bench`
Expected: builds (binary `evors-bench`).

- [ ] **Step 4: Commit**

```bash
rtk git add evors-bench/src/harness.rs evors-bench/src/main.rs
rtk git commit -m "feat(bench): sweep, aggregation, console table + CSV output"
```

---

## Task 7: integration test — the diversity assertion

**Files:**
- Create: `evors-bench/tests/bench.rs`

- [ ] **Step 1: Write the test**

```rust
//! Integration: both strategies solve an easy problem; SSEA covers more basins
//! than the greedy GA on a multi-optimum problem (the diversity assertion).

use evors::strategies::{ClassicPopulationInfo, SoftStochasticInfo};
use evors_bench::harness::{coverage, run_config, succeeded, Config, Strat};
use evors_bench::problems::all;
use evors_bench::Problem;

fn problem(name: &str) -> Problem {
    all().into_iter().find(|p| p.name == name).unwrap()
}

fn classic(step: f64) -> Config {
    Config {
        label: "classic".into(),
        strat: Strat::Classic(ClassicPopulationInfo {
            pop_size: 200, tournament: 3, mutation_rate: 0.6, seed: 0,
        }),
        mutation_step: step,
    }
}
fn ssea(step: f64) -> Config {
    Config {
        label: "ssea".into(),
        strat: Strat::Ssea(SoftStochasticInfo {
            target: 250, novelty_weight: 0.6, temperature: 0.15, behavior_weight: 0.0,
            knn: 5, seed: 0,
        }),
        mutation_step: step,
    }
}

#[test]
fn both_solve_sphere() {
    let p = problem("sphere");
    assert!(succeeded(&p, &run_config(&p, &classic(0.3), 8000, 1)));
    assert!(succeeded(&p, &run_config(&p, &ssea(0.3), 8000, 1)));
}

#[test]
fn ssea_covers_more_basins_than_classic_on_himmelblau() {
    // Himmelblau has 4 distinct global minima; SSEA should find several, the
    // greedy GA collapses onto ~1.
    let p = problem("himmelblau");
    let ssea_cov = coverage(&p, &run_config(&p, &ssea(0.25), 15000, 3));
    let classic_cov = coverage(&p, &run_config(&p, &classic(0.25), 15000, 3));
    assert!(
        ssea_cov > classic_cov,
        "ssea covered {ssea_cov}/4 basins, classic {classic_cov}/4"
    );
    assert!(ssea_cov >= 3, "ssea should find most of the 4 optima, got {ssea_cov}");
}
```

- [ ] **Step 2: Run it**

Run: `cargo test -p evors-bench --test bench`
Expected: PASS. If `ssea_cov` is short, tune `target`/`novelty_weight`/`mutation_step`/budget in the test (these are the documented levers) — do not weaken the `ssea_cov > classic_cov` core assertion.

- [ ] **Step 3: Commit**

```bash
rtk git add evors-bench/tests/bench.rs
rtk git commit -m "test(bench): sphere success + SSEA basin-coverage > classic"
```

---

## Task 8: charts + full run

**Files:**
- Modify: `scripts/plot.py`
- Run: the bench binary + plot

- [ ] **Step 1: Append bench charts to `scripts/plot.py`**

```python
# ---------------- bench: convergence + coverage + scatter ----------------
import glob
if os.path.isdir("data/bench"):
    os.makedirs("charts/bench", exist_ok=True)

    # Per-problem convergence (one line per config).
    for path in glob.glob("data/bench/*_convergence.csv"):
        name = os.path.basename(path).replace("_convergence.csv", "")
        with open(path) as f:
            rows = list(csv.reader(f))
        header, data = rows[0], np.array([[float(x) for x in r] for r in rows[1:]])
        if data.size == 0:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        evals = data[:, 0]
        for j, label in enumerate(header[1:], start=1):
            ax.plot(evals, data[:, j], lw=1.4, label=label)
        ax.set_xlabel("evaluations")
        ax.set_ylabel("best-so-far objective (min)")
        ax.set_title(f"{name} — convergence")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(f"charts/bench/{name}_convergence.png")
        plt.close(fig)
    print("wrote charts/bench/*_convergence.png")

    # Coverage bars from results.csv (classic vs ssea max per problem).
    if os.path.exists("data/bench/results.csv"):
        with open("data/bench/results.csv") as f:
            rows = list(csv.DictReader(f))
        probs = sorted({r["problem"] for r in rows})
        def best_cov(prob, kind):
            vals = [float(r["mean_coverage"]) for r in rows
                    if r["problem"] == prob and r["config"].startswith(kind)]
            return max(vals) if vals else 0.0
        classic = [best_cov(p, "classic") for p in probs]
        ssea = [best_cov(p, "ssea") for p in probs]
        x = np.arange(len(probs))
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(x - 0.2, classic, 0.4, label="ClassicPopulation", color="#d1495b")
        ax.bar(x + 0.2, ssea, 0.4, label="SoftStochastic", color="#2a9d8f")
        ax.set_xticks(x); ax.set_xticklabels(probs, rotation=30, ha="right")
        ax.set_ylabel("basins covered (mean)")
        ax.set_title("Basin coverage — Classic vs SSEA")
        ax.legend()
        fig.tight_layout()
        fig.savefig("charts/bench/coverage.png")
        plt.close(fig)
        print("wrote charts/bench/coverage.png")

    # Final-pool scatter (best classic vs best ssea config) for multimodal probs.
    for path in glob.glob("data/bench/*_scatter.csv"):
        name = os.path.basename(path).replace("_scatter.csv", "")
        with open(path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        # pick one classic and one ssea config present in the scatter.
        def pts(kind):
            xs = [(float(r["x"]), float(r["y"])) for r in rows if r["config"].startswith(kind)]
            return np.array(xs) if xs else np.empty((0, 2))
        c, s = pts("classic"), pts("ssea")
        if len(c) == 0 and len(s) == 0:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), subplot_kw={"aspect": "equal"})
        for ax, data, title, col in [
            (axes[0], c, "ClassicPopulation", "#d1495b"),
            (axes[1], s, "SoftStochastic", "#2a9d8f"),
        ]:
            if len(data):
                ax.scatter(data[:, 0], data[:, 1], s=10, c=col, alpha=0.6, edgecolors="none")
            ax.set_title(title)
        fig.suptitle(f"{name} — final population spread")
        fig.tight_layout()
        fig.savefig(f"charts/bench/{name}_scatter.png")
        plt.close(fig)
    print("wrote charts/bench/*_scatter.png")
```

- [ ] **Step 2: Run the full benchmark + charts**

Run:
```bash
cargo run --release -p evors-bench
python scripts/plot.py
```
Expected: console leaderboard per problem; `data/bench/*.csv` written; `charts/bench/*_convergence.png` and `charts/bench/coverage.png` written. Eyeball `coverage.png` — SSEA bars should exceed Classic on the multimodal problems (rastrigin, eggcrate, himmelblau, cross_in_tray).

- [ ] **Step 3: Commit**

```bash
rtk git add scripts/plot.py
rtk git commit -m "feat(bench): convergence + coverage charts; full sweep run"
```

---

## Notes for the Executor

- evors core changes are **additive**: `run` delegates to `run_observed(.., &mut NoObserver)`; observing happens **before** `report` using refs (no clone), and `RecordingObserver` tracks best-so-far itself — so the common path stays clone-free and the existing tuned tests (XOR/circle/NEAT/cppn) must remain byte-identical. Re-run `cargo test --workspace` after Task 1–2.
- `data/` is gitignored — only the code and `scripts/plot.py` are committed; charts under `charts/bench/` can be committed if you want them in the repo.
- Tune SSEA/classic params or budget if a benchmark assertion is flaky; never weaken the `ssea_cov > classic_cov` diversity assertion.
- `Score = f64`, `behavior_weight = 0` throughout (genome == solution ⇒ genetic novelty *is* solution novelty).
