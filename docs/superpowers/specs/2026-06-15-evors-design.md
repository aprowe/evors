# evors — Design Spec

**Date:** 2026-06-15
**Status:** Approved (design), pending implementation plan

## Summary

`evors` is a generic, trait-based **evolutionary computation** crate for Rust. It separates three concerns so that different *genetics* and different *evolutionary algorithms* can be mixed and matched for the same problem:

1. **`Genetic`** — an *instruction set* (strategy object) describing how to create, recombine, mutate, and measure distance between genomes. Implemented on a descriptor struct, **not** on the genome type, so multiple genetics can target the same genome type.
2. **`EvolutionStrategy`** — a pluggable evolutionary algorithm with an **ask/tell** interface (`next` / `report`). Owns its own population/archive and its own notion of `Score`.
3. **`Trial` runner** — drives a strategy against a fitness function until a stop condition, with optional batched/parallel evaluation.

The core stays **score-agnostic**: fitness, novelty, and gene-novelty are concerns of *concrete strategies*, expressed via extra trait bounds the strategy declares — never baked into the core.

## Core Traits

### `Genetic`

```rust
pub trait Genetic {
    type Genome: Clone;

    fn random(&self, rng: &mut impl Rng) -> Self::Genome;
    fn cross(&self, a: &Self::Genome, b: &Self::Genome, rng: &mut impl Rng) -> Self::Genome;
    fn mutate(&self, g: &Self::Genome, rng: &mut impl Rng) -> Self::Genome;
    fn gene_dist(&self, a: &Self::Genome, b: &Self::Genome) -> f64; // genotypic distance
}
```

- Implemented on a **descriptor struct** (the "instructions"), not on the genome. Two structs (e.g. `BitFlipGenetics`, `BitSwapGenetics`) can both have `type Genome = Vec<bool>` with different behavior.
- `gene_dist` provides **genotypic** distance. (Behavioral/solution distance is separate — see Scoring.)
- All randomness flows through a caller-supplied `&mut impl Rng` so runs are seedable/reproducible.

### `EvolutionStrategy`

```rust
pub trait EvolutionStrategy<X: Genetic> {
    type Score;

    fn next(&mut self, genetics: &X, rng: &mut impl Rng) -> X::Genome;
    fn report(&mut self, genome: X::Genome, score: Self::Score);
    fn best(&self) -> Option<(&X::Genome, &Self::Score)>;
}
```

- **Ask/tell.** `next` hands out one genome to evaluate; `report` feeds its score back.
- The strategy does **not** hold a reference to `genetics` — it is passed into `next` each call.
- `Score` is an associated type. It may be rich (`f64`, `(fitness, novelty)`, behavior descriptors, multi-objective vectors). Because a rich score has no universal ordering, the **strategy owns `best()`** — the runner never ranks scores itself.

### Scoring (strategy-imposed, not core)

The fitness closure produces the `Score`: `F: Fn(&X::Genome) -> S`. Concrete strategies that need to interpret the score declare extra bounds. Two tiny opt-in helper traits ship in the crate for strategies to use:

```rust
pub trait Fitness  { fn fitness(&self) -> f64; }                     // scalar extraction
pub trait Behavior { fn behavior_dist(&self, other: &Self) -> f64; } // behavioral/solution distance
```

- **Genetic novelty** is always available via `Genetic::gene_dist`.
- **Behavioral novelty** requires `Score: Behavior`, declared as a bound by strategies that use it.
- The core `EvolutionStrategy` imposes **none** of these — they live entirely in concrete strategies.

## Runner (`Trial`)

```rust
pub enum Stop<S> {
    Generations(usize),
    Trials(usize),
    Until(Box<dyn Fn(&S) -> bool>), // predicate over best score so far
}

pub struct RunConfig<S> {
    pub stop: Stop<S>,
    pub batch_size: usize, // pull N from next() per round before evaluating
    pub seed: Option<u64>,
}

pub struct Outcome<G, S> {
    pub best_genome: Option<G>,
    pub best_score: Option<S>,
    pub generations: usize,
    pub trials: usize,
}

pub fn run<X, St, F, S>(
    cfg: &RunConfig<S>,
    genetics: &X,
    strategy: &mut St,
    fitness: F,
) -> Outcome<X::Genome, S>
where
    X: Genetic,
    St: EvolutionStrategy<X, Score = S>,
    F: Fn(&X::Genome) -> S;
```

**Loop:** pull `batch_size` genomes via `next()` (against the current strategy snapshot) → evaluate `fitness` on each (rayon-parallel behind the `parallel` feature) → `report()` each back (sequentially, so steady-state strategies update per-report) → increment `trials` by batch size, `generations` by one per batch → query `strategy.best()` → check `stop`.

- Seeding: `seed` initializes a `rand_chacha` RNG for reproducibility.
- `batch_size = 1` gives the purest sequential behavior; larger batches trade per-report immediacy for parallel evaluation.

## Built-in Strategies

### `ClassicPopulation` (reference)

- `Score = f64`. Generational GA: buffers a generation's reports, then **tournament-selects** parents and produces the next generation via `genetics.cross` + `genetics.mutate`.
- Serves as the baseline to benchmark the others against.

### `SoftStochastic` (SSEA — "Stochastic Soft Evolution Algorithm")

The crate's headline algorithm. **Steady-state, soft-culling, statistical — no hard cutoffs.**

- **State:** a pool of scored individuals (not removed when handed out).
- **`next`:** stochastically chooses an action weighted by pool size + individual scores:
  - new random (`genetics.random`),
  - mutate a pool member (`genetics.mutate`),
  - crossbreed a pair (`genetics.cross`),
  - replay an existing genome (clone from pool; runner re-evaluates and re-reports it).
  When `pool_size < target` (stagnant) the action distribution biases toward **new random** (explore); when `pool_size > target` (flourishing) it biases toward exploit.
- **`report`:** adds the individual, then runs a **soft cull** — every pool member gets a survival probability `p = f(score, novelty, pool_size / target)` with no hard cutoff. Novel+low-score can survive; stale+high-score can survive; novel+high-score ≈ near-certain survival. Novelty blends genetic novelty (`gene_dist`) and, if `Score: Behavior`, behavioral novelty.
- **Bounds:** `Score: Fitness` (always), `Score: Behavior` (optional, enables behavioral novelty).
- **Config:** `target_pool_size`, softness/temperature, `novelty_weight`.
- **Implementability:** confirmed — every operation maps onto the core traits; no core changes needed beyond the opt-in `Fitness`/`Behavior` helper traits.

### `custom.rs`

Named placeholder module for a future user-defined algorithm.

## Benchmarks

### Benchmark 1 — XOR convergence (optimization test)

- **Genome:** weights of a 2-2-1 neural net.
- **Fitness:** `1 - MSE` over the four XOR cases (`f64`).
- **Pass:** the strategy drives best fitness to solve all four cases within N evaluations.
- **Coverage:** both `ClassicPopulation` and `SoftStochastic` should pass — proves they can locate a single global optimum.

### Benchmark 2 — unit circle (diversity / niching test)

- **Genome:** a point `(x, y)`.
- **Fitness:** `1 - (x² + y² - 1)²` — the **entire unit circle** (radius 1) is the global optimum.
- **Pass (SSEA):** bin high-fitness solutions (fitness ≥ threshold near 1) by **angle** into sectors (e.g. 12); pass when (nearly) all sectors are populated with roughly even counts (low coefficient of variation across sectors).
- **Contrast:** `ClassicPopulation` is *expected to collapse* onto one arc and fail this metric — demonstrating SSEA's diversity advantage.

## Project Layout

```
src/
  lib.rs            // crate root + re-exports (replaces main.rs — this is a library)
  genetic.rs        // Genetic trait
  strategy.rs       // EvolutionStrategy trait + Fitness/Behavior helper traits
  trial.rs          // run(), RunConfig, Stop, Outcome
  strategies/
    mod.rs
    classic.rs      // ClassicPopulation
    soft.rs         // SoftStochastic (SSEA)
    custom.rs       // placeholder
examples/
  evolve_string.rs  // worked demo: evolve a String toward a target
benches/ or tests/
  xor.rs            // Benchmark 1
  unit_circle.rs    // Benchmark 2
```

## Dependencies

- `rand` (+ `rand_chacha` for seeded, reproducible runs).
- `rayon` — optional, behind a `parallel` feature, for batched parallel fitness evaluation.

## Testing Strategy

- **TDD throughout.** Unit tests per trait and per strategy.
- Integration tests:
  - `evolve_string` provably converges toward a target string (monotonic best-fitness improvement + reaches target within N generations).
  - **Benchmark 1 (XOR):** both strategies reach a solution within an eval budget.
  - **Benchmark 2 (unit circle):** SSEA achieves even angular sector coverage; ClassicPopulation does not (contrast assertion).

## Open Items for Implementation

- Exact functional form of SSEA's `next` action-weighting and `report` survival probability (the statistical kernel) — to be specified during implementation with the user.
- Choice of behavior descriptor type for the unit-circle benchmark (likely the point itself, with Euclidean `behavior_dist`).
