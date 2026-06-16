# evors-neat

NEAT (NeuroEvolution of Augmenting Topologies) implemented as [evors](../)
components. The point of this crate is to prove evors generalizes beyond
fixed-shape genomes: NEAT's topology-evolving network genome and its structural
operators are packaged as an `evors::Genetic`, so NEAT runs on evors' **existing**
strategies — chiefly `SoftStochastic` (SSEA), whose diversity preservation stands
in for NEAT speciation.

## What's here

- **`NeatGenome`** — node genes + connection genes (with innovation numbers).
- **`NeatGenetics`** — `impl evors::Genetic`:
  - `random` — minimal genome (inputs+bias fully connected to outputs).
  - `mutate` — weight perturbation + **add-connection** (cycle-free) + **add-node**
    (split a connection), with innovation numbers from a shared registry.
  - `cross` — innovation-aligned recombination; disjoint/excess genes inherited
    from the fitter parent (via evors' `a = preferred parent` convention).
  - `gene_dist` — NEAT **compatibility distance** `c1·E/N + c2·D/N + c3·W̄`, which
    is exactly the novelty signal SSEA consumes.
- **`activate`** — feed-forward phenotype evaluation of a genome.

## Result

NEAT-on-SSEA solves XOR with a **minimal, single-hidden-node** topology:

```bash
cargo run --release -p evors-neat --example neat_xor   # writes data/neat_xor.csv
python ../scripts/plot.py                              # renders charts/neat_xor.png
cargo test -p evors-neat
```

The `charts/neat_xor.png` figure shows best fitness pinned at the **0.75
plateau** (the linear ceiling — XOR isn't linearly separable) until the champion
**gains a hidden node**, at which point it jumps and solves (~12.5k evaluations).
Topology augmentation is what breaks the plateau.

## v1 scope

- **Feed-forward only** — `add-connection` rejects cycles so activation is a
  single topological pass. Recurrent NEAT is a later extension.
- No canonical speciation — SSEA's diversity substitutes for it (the whole point).
- Single-threaded strategy ops (the innovation registry is a `RefCell`); fitness
  evaluation still parallelizes under evors' `parallel` feature.
