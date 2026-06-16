# evors-bench — Design Spec

**Date:** 2026-06-16
**Status:** Approved (design), pending implementation plan
**Depends on:** `evors` (core), with two small backward-compatible core additions.

## Summary

`evors-bench` is a headless benchmark suite that runs evors' strategies (and a
grid of their parameters) against a slew of hard continuous **minimization**
functions (Egg Crate, Rastrigin, Ackley, …), records metrics over many seeds,
and emits a console leaderboard, CSVs, and charts. Its headline metric is
**basin coverage** — how many distinct local minima the final population covers —
which is exactly what `SoftStochastic` (SSEA) is built to win and what a plain GA
collapses on.

Two reusable interfaces are added to evors core: a **progress observer** on the
runner and a **`PopulationView`** trait. These also give evors a real
reporting/logging surface (an observer that prints is a logger; a future sink can
stream events over UDP to a topic, or to CSV).

## Core additions to `evors`

### 1. Progress observer on the runner

```rust
/// One evaluation's worth of progress. Kept scalar-friendly so sinks can
/// serialize it cheaply (e.g. stream over UDP to a topic, or to CSV).
pub struct Progress<'a, G, S> {
    pub trials: usize,
    pub generation: usize,
    pub genome: &'a G,
    pub score: &'a S,
    pub best_score: Option<&'a S>,
}

pub trait Observer<G, S> {
    fn on_eval(&mut self, p: Progress<'_, G, S>);
}

/// No-op sink; `run` uses this.
pub struct NoObserver;
impl<G, S> Observer<G, S> for NoObserver { fn on_eval(&mut self, _: Progress<'_, G, S>) {} }

pub fn run_observed<St, F, O>(
    cfg: &RunConfig<St::Score>, strategy: &mut St, fitness: F, obs: &mut O,
) -> Outcome<<St::Genetic as Genetic>::Genome, St::Score>
where
    St: EvolutionStrategy,
    F: Fn(&<St::Genetic as Genetic>::Genome) -> St::Score + Sync,
    <St::Genetic as Genetic>::Genome: Clone + Send,
    St::Score: Clone + Send,
    O: Observer<<St::Genetic as Genetic>::Genome, St::Score>;

// existing `run` delegates: run_observed(cfg, strategy, fitness, &mut NoObserver)
```

The runner calls `obs.on_eval(..)` after each `report`, passing the just-evaluated
genome/score, the running trial/generation counters, and the strategy's current
`best()`. This is backward compatible (plain `run` is unchanged behaviorally).

**Sinks** (this project ships the first two; the trait makes the rest trivial):
- `RecordingObserver` — collects best-so-far per eval in memory (drives the
  harness metrics and convergence charts).
- A streaming **CSV** observer (writes progress rows) — supports the "CSV output"
  future direction.
- Future: a **UDP/topic** sink streaming `Progress` scalars to a signal bus — not
  built now, but the `Progress` shape (scalars + refs) is chosen to make it easy.

### 2. `PopulationView` trait

```rust
pub trait PopulationView {
    type Genome: Clone;
    /// Snapshot of the maintained population/pool genomes.
    fn members(&self) -> Vec<Self::Genome>;
}
```
Implemented for `SoftStochastic` (returns `pool_genomes()`) and
`ClassicPopulation` (returns `population()`). Lets the coverage metric read the
final pool generically instead of branching per strategy type. The existing
inherent methods stay (or delegate to the trait).

## evors-bench crate

### Problem model

```rust
pub struct Problem {
    pub name: &'static str,
    pub dim: usize,                     // default 2
    pub bounds: Vec<(f64, f64)>,        // per-dim [lo, hi]
    pub objective: fn(&[f64]) -> f64,   // to MINIMIZE
    pub global_min: f64,                // known f*
    pub optima: Vec<Vec<f64>>,          // global optima (for success / error)
    pub local_minima: Vec<Vec<f64>>,    // notable local minima (for basin coverage)
}
```

Problem set (2-D unless noted), each with correct bounds, `global_min`, and
optima: **Sphere** (easy control), **Rosenbrock** (deceptive valley),
**Rastrigin**, **Egg Crate**, **Ackley**, **Griewank**, **Schwefel**,
**Drop-Wave**, **Himmelblau** (4 global), **Cross-in-Tray** (4 global). For
grid-structured functions (Rastrigin, Egg Crate) `local_minima` is the lattice
within bounds; for multi-global ones it is that handful. A unit test asserts
`objective(optimum) ≈ global_min` for every problem.

### Genetics — `VecGenetics`

`impl evors::Genetic<Genome = Vec<f64>>`, constructed from a problem's bounds plus
a mutation step:
- `random` → uniform within bounds
- `mutate` → Gaussian step per coordinate, clamped to bounds (step = param)
- `cross` → per-gene uniform crossover
- `gene_dist` → Euclidean

Here **genome == solution**, so `gene_dist` *is* solution distance — genetic
novelty alone drives basin spreading. `Score = f64`, `behavior_weight = 0`
(exactly like the unit-circle benchmark).

**Minimization adapter:** evors maximizes, so `fitness = -objective(x)`. Reported
**best value** = `-best_fitness`; **error** = `best_value - global_min`.

### Harness

A `Config` = a strategy kind plus its parameters. Per-strategy **parameter grids**:
- `ClassicPopulation`: `pop_size`, `tournament`, `mutation_rate`
- `SoftStochastic`: `target`, `novelty_weight`, `temperature` (+ genetics mutation step)

For every **problem × config × seed** (N seeds, e.g. 15; fixed eval budget, e.g.
20 000), the harness calls `run_observed` with a `RecordingObserver` to capture
the convergence curve, then reads the final pool via `PopulationView::members`.

`RunResult { convergence: Vec<f64>, final_pool: Vec<Vec<f64>>, best_x: Vec<f64>, best_value: f64 }`.

### Metrics (averaged across seeds)

- **best value** = `-best_fitness`; **error** = `best_value - global_min`
- **success rate** = fraction of seeds with `error < ε`
- **evals-to-target** = median first eval with `error < ε` (∞ if never reached)
- **coverage** = number of distinct `local_minima` with ≥1 final-pool member
  within `ε_basin` — the diversity metric (SSEA's strength)
- **convergence** = best-so-far vs evals, averaged over seeds

`ε` and `ε_basin` are per-problem (scaled to bounds); defaults documented.

### Reporting

- **Console**: per-problem leaderboard table — `config | best | error | success% |
  med-evals | coverage`.
- **CSV** under `data/bench/`:
  - `results.csv` — one row per problem×config with all metrics.
  - per-problem convergence CSVs (eval, mean best-so-far per strategy).
  - per-(problem, strategy) final-pool scatter CSVs (x, y, value).
- **Charts** via `scripts/plot.py` additions, under `charts/bench/`:
  - convergence curves (Classic vs SSEA best config) per problem.
  - a **coverage bar chart** (Classic vs SSEA across problems).
  - **final-pool scatter over the function contour** for the multimodal
    problems — the visual showing SSEA covering all basins vs Classic collapsing.

## Testing

- **Unit:** `objective(optimum) ≈ global_min` for every problem; `VecGenetics`
  stays within bounds and `gene_dist` is Euclidean; `RecordingObserver` length
  equals the eval count; `PopulationView::members` non-empty after a run.
- **Integration:**
  - On **Sphere**, both strategies reach success (`error < ε`) within budget.
  - On a **multimodal** problem (Rastrigin or Egg Crate), **SSEA coverage >
    ClassicPopulation coverage** — the diversity assertion, generalizing the
    unit-circle test.
- **Regression:** the existing evors suite still passes after the core additions
  (the observer/PopulationView changes are additive).

## Scope / non-goals (v1)

- 2-D functions (dimension is a field; N-D is a later extension).
- Ships `RecordingObserver` (+ a CSV-writing observer); the **UDP/topic** sink is
  future work — only the `Progress` shape is designed for it now.
- No GUI (headless harness + reports), consistent with the chosen deliverable.
