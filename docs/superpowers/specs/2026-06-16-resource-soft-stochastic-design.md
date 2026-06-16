# ResourceSoftStochastic — Design Spec

**Date:** 2026-06-16
**Status:** Approved (design), pending implementation plan
**Depends on:** `evors` core (new strategy), `evors-bench` (gauntlet + tests + charts).

## Summary

`ResourceSoftStochastic` is a new evors strategy that drives diversity through
**instantaneous fitness sharing** rather than a novelty score. Fitness is treated
as a limited resource: individuals close together in genome space split the
resource between them, so crowded regions yield low *realized* fitness and stop
growing. Because the feedback is instantaneous (no depletable stock, no
regeneration delay), the population settles into a **stable equilibrium** — the
Ideal Free Distribution across niches — instead of the boom-bust waves a greedy
depletable-resource model produces.

It reuses the SSEA steady-state soft-cull machinery, but the survival metric is
shared fitness (`raw / crowding`) with **no novelty or behavior term**. It pairs
with assortative (local) mating so cross-niche offspring stop refilling the dead
zone, letting each niche reach its own independent equilibrium. The sharing radius
σ is **never hardcoded in absolute units** — it self-scales from the population's
own spread each refresh.

## The strategy

New file `src/strategies/resource.rs`. Like `SpeciatedSoftStochastic`, this is a
self-contained strategy that copies the SSEA pool/cull skeleton so the tuned
`SoftStochastic` (whose RNG stream is locked by the XOR/circle/NEAT tests) stays
untouched. The duplication is deliberate and isolated. It implements
`EvolutionStrategy` (`Score = S`) and `PopulationView`, so it drops into
`run`/`run_observed` and the bench harness unchanged.

The score bound is just **`S: Fitness`** — no `Behavior`, since the resource model
works purely on genome distance.

```rust
#[derive(Clone, Copy)]
pub struct ResourceSoftStochasticInfo {
    pub target: usize,
    pub temperature: f64,
    /// Dimensionless sharing radius multiplier. σ = relative_width × median pairwise
    /// gene_dist over the pool, recomputed each refresh. The only niche-width knob;
    /// carries across problems because it scales with whatever gene_dist returns.
    pub relative_width: f64,
    /// Assortative-mating strength. 0 = mate by realized fitness only (unbiased);
    /// larger = breed preferentially with genetically-close individuals.
    pub mate_locality: f64,
    pub seed: u64,
}

pub struct ResourceSoftStochastic<X: Genetic, S> { /* pool + cached crowding + σ + rng */ }
```

Notably **absent** vs `SoftStochasticInfo`: `novelty_weight`, `behavior_weight`,
`knn`. The resource model uses none of them.

### Survival metric — shared fitness

SSEA ranks survivors by `combined = (1−w)·fitness + w·novelty`. Resource ranks by
**shared fitness**:

```text
realized(i)  = shifted_raw(i) / crowding(i)
crowding(i)  = Σ_j  sh(gene_dist(i, j))      // over whole pool, includes self ⇒ ≥ 1
sh(d)        = max(0, 1 − d/σ)               // triangular: hard radius ⇒ crisp niches
```

The soft cull's Boltzmann survival weight becomes
`exp((realized(i) − max_realized) / temperature)` — the same non-greedy
steady-state cull, just fed shared fitness. More neighbors inside σ ⇒ higher
crowding ⇒ resource split more ways ⇒ lower realized ⇒ the niche stops growing.
That negative feedback **is** the per-niche carrying capacity, and because it is
instantaneous it converges rather than oscillates.

### Two correctness details

1. **Positive-fitness shift.** Fitness sharing assumes fitness > 0, but the
   gauntlet uses `fitness = −objective ≤ 0`. So before dividing, shift the pool to
   a positive range: `shifted_raw(i) = raw(i) − min_raw_in_pool + ε` (small
   `ε > 0`). Crowding then genuinely *penalizes* density (more crowded → smaller
   positive value → lower survival weight). The shift is recomputed each refresh
   from the current pool's min.

2. **Self-scaling σ (no hardcoded radius).** Each refresh,
   `σ = relative_width × median(pairwise gene_dist over the pool)`. σ auto-tracks
   the population: as it spreads, σ grows; as niches condense, the median is
   dominated by *cross-niche* separation (set by the landscape), so σ stays stable
   and within-niche members (≪ σ apart) fully share. The median may be computed
   over a capped random sample of pairs (e.g. up to ~2000 pairs) when the pool is
   large, to keep cost bounded; over the default `target = 300` the full O(n²) is
   fine and matches SSEA's existing novelty-refresh cost.

### Caching / refresh cadence

Crowding (per pool member) and σ are cached and **bulk-refreshed every `target/8`
culls**, exactly mirroring SSEA's novelty cache. Between refreshes, an incoming
genome's crowding is computed against the current pool on insert (O(n)) so the
negative feedback stays responsive; the periodic bulk refresh recomputes σ, the
positive shift, and all crowding values together.

### Mating — assortative / local

Reuse Speciated's `mate_pick`. Crossover picks parent **A** by Boltzmann selection
on realized fitness; parent **B** is drawn with weight
`∝ exp((realized_B − max_realized)/temperature) · exp(−mate_locality · gene_dist(A, B))`.
The fitter of A/B is passed first (preferred-parent convention). With
`mate_locality = 0` the distance factor `exp(−0) = 1` vanishes and B's
distribution is unbiased. Local mating keeps cross-niche offspring (which land in
dead zones) rare, so each niche reaches its own independent stable equilibrium.

The random / mutate / replay actions are the same as SSEA. A `set_mate_locality`
and `set_relative_width` live setter are included for parity with SSEA's setters.

Re-exported from `src/strategies/mod.rs` as `ResourceSoftStochastic` /
`ResourceSoftStochasticInfo`. A `PopulationView` impl is added in
`src/population.rs` (bound `S: Fitness + Clone`, matching how the pool genomes are
stored).

## Validation — the equilibrium story

The point is not merely "covers both circles" (Speciated already does that) — it
is **reaches a stable equilibrium with no boom-bust**. Validation therefore
measures niche occupancy *over time*.

Landscape: the existing two-circles helper in `evors-bench` (rings centered at
`x = ±C`, `C = 5`, bounds `x ∈ [−7, 7]`, `y ∈ [−2, 2]`, reuses `VecGenetics`).
Niche occupancy at any moment, over the current pool:

```text
left  = |{ m : m.x < −C + 1.5 }|
right = |{ m : m.x >  C − 1.5 }|
gap   = |{ m : |m.x| < 3 }|        // the dead zone (matches between_fraction)
```

## Tests (in the suite)

In `evors-bench/tests/resource.rs`:

1. **`resource_reaches_stable_equilibrium`** (core test) — run
   `ResourceSoftStochastic` on two-circles for a fixed budget/seed, sampling
   occupancy every N trials via the `Observer`. Assert over the **back half** of
   the run:
   - **Diversity:** both `left` and `right` occupied in every back-half sample,
     and `between_fraction` stays low (clear margin below the SSEA baseline level
     established in the speciated work, ~0.17).
   - **Stability:** the left and right occupancy counts are *steady* — relative
     spread `(max − min) / mean` over the back-half samples is below a threshold
     (e.g. < 0.5), i.e. no boom-bust oscillation. This is the property that
     distinguishes the resource model from a greedy depletable one.
2. **`resource_solves_sphere`** — sanity: the strategy still optimizes (reaches
   success on the sphere problem), proving the cull machinery is intact.

Plus unit tests in `resource.rs`:

- **`sigma_scales_with_spread`** — σ computed from a spread-out pool is strictly
  larger than σ from a tight pool (self-scaling works).
- **`crowding_penalizes_clusters`** — a genome inside a tight cluster has higher
  crowding than a spatially isolated genome.
- **`mate_locality_zero_matches_unbiased`** — with `mate_locality = 0` the
  mate-selection weights reduce to the unbiased (no distance bias) weights.

## Gauntlet integration

- Add a `Resource(ResourceSoftStochasticInfo)` variant to the bench `Strat` enum;
  `run_config` builds `ResourceSoftStochastic::new(VecGenetics, info)` and runs it
  via `run_one` (it satisfies `EvolutionStrategy<Score=f64> +
  PopulationView<Genome=Vec<f64>>`).
- Add 1–2 resource configs to `default_configs` (e.g.
  `target = 300, relative_width ∈ {0.1, 0.2}, mate_locality = 3`), so the strategy
  runs across all 10 gauntlet problems and appears in the console leaderboard,
  `results.csv`, the convergence charts (automatic), and the coverage bar chart
  (add a resource series).
- **Expectation:** basin coverage comparable to SSEA/Speciated, accuracy below DE
  (the diversity tax). The standout is *diversity with no novelty score* plus a
  stable pool — most visible on the two-circles scatter and the occupancy
  time-series, not necessarily the scalar metrics.

## Charts

- Extend the `evors-bench` `two_circles` example to also evolve and dump the
  **resource** final pool to `data/bench/two_circles_resource.csv` (alongside the
  existing `_ssea` / `_speciated` CSVs). `scripts/plot.py` renders a **three-panel
  scatter** in `charts/bench/two_circles.png` — SSEA smear / Speciated two-blobs /
  Resource two-blobs — with the dead-zone band marked.
- **Occupancy-over-time chart:** the example (or the equilibrium test) writes
  left/right/gap counts vs trials to `data/bench/two_circles_occupancy.csv`;
  `scripts/plot.py` renders it to `charts/bench/occupancy.png`. The resource lines
  should flat-line in the back half — the visual proof of equilibrium — contrasted
  against plain SSEA's noisier occupancy.
- The existing coverage bar chart gains a resource series.

## Scope / non-goals

- One new strategy with four tunable knobs (`target`, `temperature`,
  `relative_width`, `mate_locality`). No novelty score, no behavioral distance, no
  `knn`.
- Sharing kernel is fixed as triangular `max(0, 1 − d/σ)`; α (sharpness) is not
  exposed. σ is self-scaling; an absolute σ is explicitly rejected.
- No depletable-stock / regeneration model — the whole point is the instantaneous
  (delay-free) variant that does not oscillate.
- Deliberate code duplication of the SSEA core (shared with the Speciated rationale)
  to keep the tuned `SoftStochastic` untouched; a future DRY refactor (shared
  cull module across SSEA/Speciated/Resource) is out of scope.
