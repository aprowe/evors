# SpeciatedSoftStochastic — Design Spec

**Date:** 2026-06-16
**Status:** Approved (design), pending implementation plan
**Depends on:** `evors` core (new strategy), `evors-bench` (gauntlet + tests).

## Summary

`SpeciatedSoftStochastic` is a new evors strategy: the SSEA machinery (steady-state
soft culling, genetic + behavioral novelty) with **one change — assortative
mating**. Crossover prefers genetically-close mates, so "species" emerge softly
(no species IDs, no fixed count). The payoff: on a multi-basin landscape, far
fewer "throwaway" organisms stranded *between* basins, while still covering all of
them. It runs across the existing minimizer gauntlet alongside Classic / SSEA /
DE, and a two-circles test in the suite validates the speciation property.

## The strategy

New file `src/strategies/speciated.rs`. To avoid touching the tuned
`SoftStochastic` (its RNG stream is locked by the XOR/circle/NEAT tests), this is
a self-contained strategy with the same pool/cull/novelty logic, differing only
in parent selection during crossover. The duplication is deliberate and isolated.

```rust
#[derive(Clone, Copy)]
pub struct SpeciatedSoftStochasticInfo {
    pub target: usize,
    pub temperature: f64,
    pub novelty_weight: f64,
    pub behavior_weight: f64,
    pub knn: usize,
    pub seed: u64,
    /// Assortative-mating strength. 0 = mate by fitness only (identical to SSEA);
    /// larger = breed preferentially with genetically-close individuals.
    pub mate_locality: f64,
}

pub struct SpeciatedSoftStochastic<X: Genetic, S> { /* same fields as SoftStochastic + mate_locality */ }
```

- **`next()` — the only behavioral change.** The crossover action picks parent
  **A** by Boltzmann selection (as SSEA does). Parent **B** is then drawn with
  weight `∝ exp((fitness_B − max_f)/temperature) · exp(−mate_locality ·
  gene_dist(A, B))` — fitness-weighted *and* distance-discounted toward A. With
  `mate_locality = 0` the `exp(−0) = 1` factor vanishes and B's distribution
  matches SSEA's Boltzmann pick exactly. The fitter of A/B is passed first
  (preferred-parent convention). The random / mutate / replay actions and the
  soft cull are byte-for-byte the same as SSEA.
- **Emergent species:** organisms breed with neighbors, so cross-cluster
  offspring (which land in dead zones between basins) become rare; clusters
  self-organize with no explicit species structure.
- Implements `EvolutionStrategy` (`Score = S`) and `PopulationView`, so it drops
  into `run`/`run_observed` and the bench harness unchanged. A `set_mate_locality`
  live setter is included for parity with SSEA's setters.

Re-exported from `src/strategies/mod.rs` as `SpeciatedSoftStochastic` /
`SpeciatedSoftStochasticInfo`.

## Validation — two unit circles

A landscape rewarding being on *either* of two unit rings:

```text
ring(u, v) = 1 − (u² + v² − 1)²
two_circles(x, y) = −max(ring(x + C, y), ring(x − C, y))   // minimized; global_min = −1
```
with centers at `x = ±C` (C = 5), bounds `x ∈ [−7, 7]`, `y ∈ [−2, 2]`. Lives as a
helper in `evors-bench` (it reuses `VecGenetics`).

**"Between" metric:** the fraction of the final pool stranded in the central dead
zone, `between_fraction = |{ m : |m.x| < gap }| / pool_size` with `gap = 3`.

**Coverage guard:** "covers both rings" = at least one pool member on each ring
(a high-fitness member with `x < −C+1.5` and one with `x > C−1.5`).

## Tests (in the suite)

In `evors-bench/tests/speciation.rs`:

1. **`speciation_reduces_between_population`** — run plain `SoftStochastic` and
   `SpeciatedSoftStochastic` (with `mate_locality > 0`) on two-circles for a fixed
   budget/seed. Assert **both** cover both rings, and
   `speciated_between_fraction < ssea_between_fraction` by a clear margin. This is
   the core property test.
2. **`speciated_solves_sphere`** — sanity: the new strategy still optimizes
   (reaches success on the sphere problem), proving the cull/novelty machinery is
   intact.

Plus a unit test in `speciated.rs` (e.g. `mate_locality_zero_matches_unbiased`)
checking that with `mate_locality = 0` the mate-selection weights reduce to the
Boltzmann weights (no distance bias).

## Gauntlet integration

- Add a `Speciated(SpeciatedSoftStochasticInfo)` variant to the bench `Strat`
  enum; `run_config` builds `SpeciatedSoftStochastic::new(VecGenetics, info)` and
  runs it via `run_one` (it satisfies the `EvolutionStrategy<Score=f64> +
  PopulationView<Genome=Vec<f64>>` bounds).
- Add 1–2 speciated configs (e.g. `target=300, novelty_weight=0.6,
  mate_locality∈{1.0, 3.0}`) to `default_configs`, so the new strategy runs across
  all 10 gauntlet problems and appears in the console leaderboard, `results.csv`,
  convergence charts (automatic — they plot every config), and the coverage bar
  chart (add a fourth "speciated" series).
- **Expectation:** comparable basin coverage to SSEA but with tighter, less
  scattered final pools (the speciation effect is most visible on the two-circles
  scatter and the multimodal scatter charts, not necessarily the scalar metrics).

## Charts

- `evors-bench` example `two_circles` writes `data/bench/two_circles_ssea.csv` and
  `..._speciated.csv` (final-pool `x,y`); `scripts/plot.py` renders
  `charts/bench/two_circles.png` — side by side, showing plain SSEA's smear
  through the center vs speciated's clean two-blob separation, with the dead-zone
  band marked.
- The existing coverage bar chart gains a speciated series.

## Implementation note — the explore↔tidy tension, and three knobs

The two-circles experiment exposed a genuine tension. **Assortative mating alone
barely thins the dead zone** (~0.16 for both SSEA and speciated): SSEA's cull
*rewards novelty*, and the empty gap between basins is the most novel region, so
novelty-driven survival keeps refilling it regardless of who breeds. Worse, the
gap population is dominated by SSEA's constant random-exploration injection
(≈5% of steps), so it is partly *structural* — those organisms **are** the
exploration. You cannot have strong exploration and an empty gap simultaneously.

The final strategy therefore exposes three knobs and lets the user pick a point on
the spectrum:

1. **`mate_locality`** — assortative mating (breed with close kin); forms species.
2. **`archive_prob`** — a novelty **archive**: each evaluated genome is sampled
   into an archive with this probability, and novelty is measured against pool ∪
   archive. Explored regions (incl. the gap) stop being novel, so organisms push
   to fresh frontiers instead of camping — **this preserves exploration**. Its
   dead-zone thinning is real but modest/noisy on its own (it can't beat the
   re-injection rate).
3. **`speciation_pressure`** — a fitness gate on the novelty bonus:
   ```
   combined = (1 − w)·fitness + w·novelty·fitness^speciation_pressure
   ```
   `0` = ungated novelty (max exploration, populated gap); `1` = novelty scaled by
   fitness (tidy species, less exploration). It interpolates between the two.

**Chosen demo/test point:** `mate_locality = 5`, `archive_prob = 0.05`,
`speciation_pressure = 0.6` — keeps the archive's exploration while reliably
thinning the gap on all tested seeds (between-fraction ~0.17 → ~0.12–0.13, both
rings still covered). `0.8` overshoots and goes non-monotonic/noisy. Across the
gauntlet, speciated keeps strong coverage of the good basins (rastrigin ~24,
griewank ~58) — close to SSEA, far above DE/Classic.

## Scope / non-goals

- Three tunable knobs vs SSEA: distance-biased mating, a novelty archive, and a
  fitness gate (`speciation_pressure`). The experiment showed no single mechanism
  both keeps full exploration and strongly empties the gap — the gate is the dial.
- No species IDs, counts, or explicit fitness sharing.
- Deliberate code duplication of the SSEA core to keep the tuned `SoftStochastic`
  untouched; a future DRY refactor (shared cull/novelty module) is out of scope.
