# evors

A small, trait-based **evolutionary computation** crate for Rust. evors keeps the
*genetics* (how genomes are built and varied) separate from the *evolutionary
algorithm* (how a population is searched), so you can mix and match them — and
plug in your own algorithm behind one clean ask/tell interface.

## The three core pieces

### `Genetic` — an instruction set for a genome type

Implemented on a **descriptor struct**, not on the genome itself, so two
different genetics can target the same genome type.

```rust
pub trait Genetic {
    type Genome: Clone;
    fn random(&self, rng: &mut impl Rng) -> Self::Genome;
    fn cross(&self, a: &Self::Genome, b: &Self::Genome, rng: &mut impl Rng) -> Self::Genome;
    fn mutate(&self, g: &Self::Genome, rng: &mut impl Rng) -> Self::Genome;
    fn gene_dist(&self, a: &Self::Genome, b: &Self::Genome) -> f64; // genotypic distance
}
```

### `EvolutionStrategy` — a pluggable algorithm (ask/tell)

The strategy is **self-contained**: it carries its genetics as an associated
type and owns both that genetics value and its own seeded RNG. `next`/`report`
take no per-call arguments.

```rust
pub trait EvolutionStrategy {
    type Genetic: Genetic;
    type Score;
    fn next(&mut self) -> <Self::Genetic as Genetic>::Genome;        // ask
    fn report(&mut self, genome: <Self::Genetic as Genetic>::Genome, score: Self::Score); // tell
    fn best(&self) -> Option<(&<Self::Genetic as Genetic>::Genome, &Self::Score)>;
}
```

`Score` is whatever the fitness function returns — `f64`, `(fitness, novelty)`,
a behavior descriptor, a multi-objective vector. The core stays score-agnostic;
strategies that need to interpret a score declare opt-in bounds via the helper
traits `Fitness` (scalar extraction) and `Behavior` (behavioral distance).

### `run` — the runner

```rust
pub fn run<St, F>(cfg: &RunConfig<St::Score>, strategy: &mut St, fitness: F)
    -> Outcome<<St::Genetic as Genetic>::Genome, St::Score>;
```

`RunConfig { stop, batch_size, max_trials }` drives the strategy until `stop`
(N generations, N evaluations, or an `Until` predicate over the best score).
**`max_trials` is a hard ceiling that always applies** — set it whenever you use
`Stop::Until` so a run that never reaches the target still terminates. Batches
are evaluated in parallel behind the `parallel` feature.

## Construction: CreateInfo, not builders

Each strategy is built from its genetics plus a Vulkan-style **CreateInfo** struct
(`Default` + struct-update), keeping the genetics as a separate required argument:

```rust
let strat = SoftStochastic::new(PointGenetics, SoftStochasticInfo {
    target: 300, novelty_weight: 0.6, ..Default::default()
});
```

## Built-in strategies

- **`ClassicPopulation`** — an elitist, replace-worst steady-state GA with
  tournament selection. The optimization baseline.
- **`SoftStochastic` (SSEA)** — the headline algorithm. Steady-state, **soft
  culling with no hard cutoffs**: every pool member survives with probability
  `combined^pressure`, where `combined` blends normalized fitness and genetic
  novelty and `pressure = pool_size / target`. A small pool (stagnant) biases
  `next` toward random exploration; a large pool (flourishing) biases toward
  exploitation. Novel-but-weak individuals can survive; so can strong-but-stale
  ones — which preserves diversity a greedy GA destroys.

## Quickstart

```bash
cargo run --example evolve_string        # evolve a string toward a target
cargo run --release --example bench_data # write benchmark CSVs into ./data
python scripts/plot.py                   # render charts into ./charts
cargo test                               # unit + integration + benchmarks
```

## Benchmarks

- **XOR convergence** (`tests/xor.rs`) — evolve the 9 weights of a 2-2-1 net to
  solve XOR. Both strategies reach fitness `> 0.98`. Classic converges fastest
  with low selection pressure; SSEA pays for its diversity but still escapes the
  deceptive ~0.875 plateau.
- **Unit circle** (`tests/unit_circle.rs`) — fitness `1 − (x² + y² − 1)²`, so the
  whole ring is optimal. SSEA populates all 12 angular sectors; the greedy GA
  collapses onto a single arc. This is the diversity test SSEA is built to win.

See `charts/` for rendered convergence and solution-spread figures.

## Features

- `parallel` — evaluate each batch's fitness in parallel via `rayon`.

## Workspace

This repo is a Cargo workspace:

- **`evors`** (this crate) — the core traits, strategies, and runner.
- **[`evors-neat`](evors-neat/)** — NEAT (topology-evolving networks) implemented
  as an `evors::Genetic`, running on SSEA. Demonstrates the abstractions
  generalize beyond fixed-shape genomes; solves XOR with a minimal single-hidden-node
  network (`charts/neat_xor.png`).
- **[`evors-cppn`](evors-cppn/)** — evolve images with CPPN-NEAT, rendered and
  scored on the GPU (egui + glow). Sub-project A (the headless GPU eval core) is
  done; the evolution loop and egui app follow.

`cargo test --workspace` runs everything. The `evors-cppn` GPU tests require a
real OpenGL 4.3 context.
