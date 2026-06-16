# evors-neat — Design Spec

**Date:** 2026-06-15
**Status:** Approved (design), implementing
**Parent:** builds on `evors` (see `2026-06-15-evors-design.md`)

## Summary

`evors-neat` is a sibling crate that implements **NEAT** (NeuroEvolution of
Augmenting Topologies) as evors components, to prove evors generalizes beyond
fixed-shape genomes. NEAT's topology-evolving network genome, structural
mutation with innovation tracking, innovation-aligned crossover, and
compatibility distance are packaged as an `evors::Genetic` implementation. NEAT
then runs on evors' **existing** strategies — chiefly `SoftStochastic` (SSEA),
whose novelty/diversity preservation stands in for NEAT speciation.

This is sub-project 1 of two. Sub-project 2 (a scenario/benchmark suite) gets its
own cycle later.

## Workspace

Convert the repo to a Cargo workspace with minimal disruption: `evors` stays the
root package *and* a workspace member; `evors-neat` is a sibling directory.

```
evors/
  Cargo.toml          # [package] evors + [workspace] members = [".", "evors-neat"]
  src/ ...            # evors core (small change below)
  evors-neat/
    Cargo.toml        # depends on evors via path = ".."
    src/
      lib.rs
      genome.rs       # NeatGenome: node genes + connection genes
      innovation.rs   # Innovations registry (historical markings)
      genetics.rs     # NeatGenetics: impl evors::Genetic
      network.rs      # phenotype: feed-forward activation of a genome
```

## evors core change: fitter-parent crossover

Canonical NEAT inherits disjoint/excess genes from the **more-fit** parent, but
`Genetic::cross(&self, a, b, rng)` carries no fitness. Rather than change the
signature, adopt a **convention**: `a` is the *preferred* (fitter) parent, and
impls may bias inheritance toward it. The strategies — which know parent fitness
at selection time — order parents fitter-first:

- `ClassicPopulation::select` returns `(genome, fitness)`; `next` passes the
  fitter of the two parents as `a`.
- `SoftStochastic` compares `pool[pi].fitness` vs `pool[oi].fitness` and passes
  the fitter as `a`.

No trait signature change, no `Score` coupling. Harmless for symmetric genetics
(point midpoint, net uniform crossover); slightly changes string crossover
ordering — the existing benchmarks are re-run to confirm they still pass.

## Genome

```rust
pub enum NodeKind { Input, Bias, Output, Hidden }
pub struct NodeGene { pub id: u64, pub kind: NodeKind }
pub struct ConnGene  { pub innov: u64, pub from: u64, pub to: u64, pub weight: f64, pub enabled: bool }

#[derive(Clone)]
pub struct NeatGenome { pub nodes: Vec<NodeGene>, pub conns: Vec<ConnGene> }
```

Fixed node-id layout: inputs `0..I`, bias `I` (constant 1.0), outputs
`I+1..=I+O`, hidden ids allocated from the registry starting at `I+O+1`.

## Innovation registry (interior-mutable)

`NeatGenetics` owns the historical markings behind a `RefCell` (strategy ops are
sequential, so it never aliases):

```rust
pub struct Innovations {
    next_innov: u64,
    next_node: u64,
    conn_ids: HashMap<(u64, u64), u64>,           // (from,to) -> innovation
    split_ids: HashMap<(u64, u64), (u64,u64,u64)>, // split of (from,to) -> (new_node, innov_in, innov_out)
}
```

- `conn_innov(from,to)` returns a stable innovation id for an edge, so identical
  new edges in different genomes align.
- `split(from,to)` returns a stable `(new_node, innov_in, innov_out)` so the same
  connection split in different genomes produces the same node id and innovations
  — the historical marking that makes crossover alignment work.

## NeatGenetics: the `Genetic` impl

`type Genome = NeatGenome`. Constructed CreateInfo-style:
`NeatGenetics::new(inputs, outputs, NeatParams { .. })` where `NeatParams` holds
weight-perturb σ, weight-mutate probability, add-connection / add-node
probabilities, and compatibility coefficients `c1`,`c2`,`c3`,`c_weight`.

- **`random`** — minimal genome: inputs+bias fully connected to outputs with
  random weights, no hidden nodes. Initial connections draw stable innovation
  ids from the registry, so the founding population is innovation-aligned.
- **`mutate`** — clone, then: with `weight_mutate_prob`, perturb each weight by
  `N(0, σ)` (occasional reset); with `add_conn_prob`, add a **cycle-free** edge
  between two currently-unconnected nodes (reachability check keeps the network
  feed-forward); with `add_node_prob`, pick an enabled connection, disable it,
  and split it via `registry.split` into `from→new (w=1)` and `new→to (w=old)`.
- **`cross`** — `a` is the fitter parent. Align connections by innovation:
  matching genes take weight/enabled from a random parent; disjoint/excess genes
  are inherited from `a` only. Child node set = the fixed I/O/bias nodes plus any
  hidden node referenced by an inherited connection (node kind taken from `a`,
  falling back to `b`). A gene disabled in the chosen parent re-enables with
  probability 0.25.
- **`gene_dist`** — compatibility distance
  `δ = c1·E/N + c2·D/N + c_weight·W̄`, where `E`=excess, `D`=disjoint genes
  (relative to each genome's max innovation), `W̄`=mean weight difference of
  matching genes, `N`=max(gene counts) (clamped to ≥1). This is exactly the
  diversity signal SSEA consumes via novelty.

## Phenotype: `network.rs`

Build a feed-forward evaluator from a genome:

- Topologically order nodes over the enabled connections (Kahn's algorithm; the
  acyclic invariant is guaranteed by the mutation cycle check).
- `activate(&genome, inputs) -> Vec<f64>`: set input node values and bias = 1.0;
  in topological order each non-input node sums `weight × from_value` over its
  enabled incoming connections and applies a steep sigmoid `1/(1+e^(-4.9x))`
  (the classic NEAT activation); return the output nodes' values.

## Validation: XOR via SSEA

The success scenario (reuses the XOR problem shape from evors):

- `NeatGenetics::new(2, 1, ..)`; fitness = `1 − MSE` over the four XOR cases,
  evaluated by activating the evolved network.
- Run on `SoftStochastic` with `gene_dist` (compatibility) as the novelty metric.
- **Pass:** evolves a network with fitness `> 0.95` that classifies all four XOR
  cases correctly, within a bounded evaluation budget (`RunConfig.max_trials`).
- A `bench_data`-style example records best-fitness vs evaluations and the
  champion's hidden-node count over time (topology growth), rendered by the
  existing `scripts/plot.py` pattern.

## Testing

- **Unit:** registry id stability (same edge/split → same ids); `random` builds a
  valid minimal genome; `mutate` add-node/add-connection keep the graph
  feed-forward (no cycles) and well-formed; `cross` of two genomes yields a valid
  child whose disjoint/excess genes come from the fitter parent; `gene_dist` is
  zero for identical genomes and grows with structural difference; `network`
  activates XOR-shaped genomes correctly on a hand-built solution.
- **Integration:** NEAT-on-SSEA solves XOR (the pass criterion above).
- **Regression:** the full evors suite still passes after the fitter-parent change.

## Scope / simplifications (v1)

- **Feed-forward only** — `add-connection` rejects cycles so activation is a
  topological pass. Recurrent NEAT is a later extension.
- No canonical speciation — SSEA's diversity preservation substitutes for it
  (the whole point of running NEAT on evors).
- Single-threaded strategy ops (RefCell registry); fitness evaluation still
  parallelizes under the `parallel` feature.
