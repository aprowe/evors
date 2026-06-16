# evors-sandbox (sub-project A) — headless simulation core

**Date:** 2026-06-15
**Status:** Approved (design), ready for planning
**Parent project:** Evolution Sandbox (`SANDBOX.md`) — a continuous, shared-world
artificial-life simulation where organisms with evolving NEAT brains hunt food
and each other. This spec covers **sub-project A** only: the headless,
deterministic simulation core. Sub-projects C (evolution glue) and D (egui/glow
live viewer) follow in their own cycles.

## Project decomposition

The full project splits into independent specs, built in order:

- **A — Simulation core (this spec).** World, organisms, food, cone-vision ray
  tracing, energy/movement physics, eating, fitness accounting. Headless,
  deterministic, no GPU, no GUI. Controllers are a trait, so A is testable with
  hand-coded brains before NEAT or a viewer exist.
- **C — Evolution integration.** One `evors::EvolutionStrategy` per species
  (`N` independent gene pools); diet (carnivore/herbivore) fixed per species.
  Continuous steady-state ask/tell: the world holds a fixed number of live
  organisms; each death triggers `report(genome, fitness)` on the dead
  organism's species pool, then `next()` to spawn a replacement at a random
  location. No generations, no barriers — the strategy's live pool *is* the
  population, which is exactly what SSEA's soft-cull pool was built for.
- **D — egui/glow live viewer.** Render the world, vision cones, and per-organism
  inspection, reusing the soap-in-a-tub GL patterns and the egui app from
  `evors-cppn` sub-project C.

(Sub-project B — recurrent NEAT brains — is **deferred**. Brains are plain
feed-forward `evors-neat` for now; the recurrent/memory/oscillator inputs are
parked as a documented later extension, see *Deferred inputs* below.)

## Workspace

Add `evors-sandbox` as a workspace member (sibling to `evors-cppn`). Dependencies:
`evors` (path), `evors-neat` (path), `rand` 0.9, `rand_chacha` 0.9. No GL, no
egui in sub-project A.

## World model and tick

The world is a **fixed-timestep, seeded** simulation — fully reproducible from a
config + seed. A `World` owns:

- a `ChaCha` RNG seeded from config,
- a flat `Vec` of **organisms** and a flat `Vec` of **food**, each addressed by a
  stable generational id,
- rectangular world bounds; organisms **clamp at walls** (no toroidal wrap),
- config: species/organism counts, food density, energy and motion constants.

Core entry point:

```rust
impl World {
    /// Advance one tick; return the organisms that died this tick.
    fn step(&mut self, dt: f32) -> Vec<Death>;
}
```

`step` runs, in order:

1. For each organism: build `Senses` → call its `Controller` → get `Motor`.
2. Integrate movement, drain energy (passive + movement cost).
3. Resolve collisions (carnivore↔herbivore eating, herbivore↔food eating).
4. Age food; spawn new food up to target density.
5. Mark `energy ≤ 0` as dead; collect dead organisms into the returned
   `Vec<Death>` (each carries species id + finalized fitness).

A itself **never touches `evors` strategies** — the caller (sub-project C)
decides what to do with deaths. This keeps A a clean, independently testable
unit: feed it a world + controllers, step it, assert on outcomes.

### Controllers are a trait

```rust
trait Controller { fn sense_act(&mut self, senses: &Senses) -> Motor; }
```

A can run with hand-coded controllers (e.g. "turn toward the brightest ray")
before any network exists; C plugs in a NEAT-backed controller wrapping
`evors_neat::network::activate`. Determinism + this trait boundary are what make
A verifiable on its own.

## Senses (vision) and motor

**Vision is a cone of `R` rays** (config `ray_count`, `vision_length`,
`vision_arc`). Rays fan evenly across `vision_arc`, centered on heading, each of
length `vision_length`. For each ray, find the **nearest circle** (food *or* other
organism) intersected within range and record:

- **distance** → normalized `1 - hit_dist/vision_length` ∈ `[0,1]` (`0` = nothing
  seen, `1` = touching); "closer = stronger" is easier for nets to learn.
- **hue** → the hit circle's color ∈ `[0,1]` (`0` if nothing hit).

Rays see **both food and other organisms**; an organism distinguishes them only
by hue (no separate type channel). `Senses` is therefore `2·R` vision values plus
`energy_ratio` (current/max).

```rust
struct Senses { rays: Vec<RayHit>, energy_ratio: f32 } // rays.len() == R
struct RayHit { distance: f32, hue: f32 }
```

**Motor: 4 outputs** — `forward, back, turn_left, turn_right`, each ∈ `[0,1]`
(output nodes are sigmoid). Net thrust = `forward - back`; net angular velocity =
`turn_right - turn_left`. Opposing pairs (rather than two signed outputs) match
the spec's four-action framing and let an organism independently "want" to brake.
Thrust and turn scale by config `max_speed` / `max_turn_rate`, times `dt`.

```rust
struct Motor { forward: f32, back: f32, turn_left: f32, turn_right: f32 }
```

So a species' NEAT genome is fixed-shape at the I/O boundary: **`2R + 1` inputs,
`4` outputs**; topology evolves between.

### Deferred inputs (later, with recurrence)

When feed-forward graduates to recurrent (sub-project B), the input vector gains:
random, sin1/sin2/sin3 oscillators (each with a phase/freq gene), and
memory1/memory2/memory3 (recurrent state). The encoding appends these to the end
of the input vector, so adding them later does not reshape the existing inputs.

## Energy, eating, food

**Energy economy.** Each organism has `energy ∈ [0, max]`. Per tick it loses a
**passive drain** (constant × dt) plus a **movement cost** (proportional to
`|thrust|` and `|turn|` × dt). Zero → dead. Newborns spawn at a configured
starting energy.

**Aging.** Each organism tracks `age` (seconds alive). Energy gained from eating
is scaled by a decay factor that **shrinks as age grows** (e.g. `gain ×
decay^age`). Against constant drain, an old organism eventually starves no matter
how well it plays — guaranteeing turnover so the gene pool keeps flowing (no
immortal organism camping an optimum). The decay applies to the *organism's* age,
not the food item's.

**Eating (collision resolution).** Circle-overlap tests (sum of radii):

- **Herbivore ∩ food:** food consumed; herbivore gains `food_energy × age_decay`.
- **Carnivore ∩ herbivore:** herbivore dies immediately (a `Death` this tick,
  reported to its species pool by C); carnivore gains a share of the herbivore's
  energy × its own `age_decay`.

A **uniform spatial grid** buckets organisms and food by cell; vision queries and
eating tests only inspect neighboring cells, keeping cost near-linear in
population rather than O(n²). Built in from the start, not retrofitted.

**Food generation.** Food spawns at random positions up to a **target density**
(`step` tops it back up as it is eaten), each with a configured size and a hue
from the food palette.

## Fitness

Accumulated live, finalized at death:

```
fitness = time_alive  +  closest_approach_bonus
```

- `time_alive` — seconds survived (dominant term once organisms are competent).
- `closest_approach_bonus` — tracks the **nearest the organism ever got to its
  food target** (food for herbivores, prey for carnivores), mapped so its
  **maximum equals the energy one meal would grant**. This gives gradient to a
  population that cannot yet eat at all (drifting toward food scores above sitting
  still), capped at "one eat" so it only fuels early generations and never
  dominates `time_alive`.

## Physical genes

Per the spec, a genome carries physical/phenotype genes alongside the brain
topology: **color (hue)** and **size** (pre-determined per organism). Sin
phase/freq and memory phase genes are part of the *deferred* recurrent inputs and
are not encoded in sub-project A. In A these physical traits can be fixed per
species in config; how they are encoded as evolvable genes is finalized in C
(where the NEAT genome + a small physical-gene struct become the species genome).

## Validation (headless milestone)

All tests are pure CPU, deterministic, no GPU:

1. **Determinism** — two `World`s built from the same config + seed produce
   bit-identical state after `K` steps.
2. **Vision correctness** — with a hand-placed food circle, the ray that should
   hit it reports the expected normalized distance and hue within `1e-4`; rays
   that miss report `0`.
3. **Vision range/arc** — food outside `vision_length` or outside `vision_arc`
   is not seen.
4. **Energy & death** — passive drain alone starves a stationary organism in the
   expected number of ticks; movement costs more than rest; `step` returns the
   organism in `Vec<Death>` on the tick it crosses zero.
5. **Eating** — a herbivore overlapping food gains energy and the food is
   removed; a carnivore overlapping a herbivore kills it (herbivore appears in
   `Vec<Death>`) and gains energy.
6. **Aging** — identical eats yield strictly less energy at higher age.
7. **Food density** — after eating, `step` restores food toward target density.
8. **Fitness** — `closest_approach_bonus` is monotonic in nearest approach and
   capped at one-meal energy; `time_alive` accumulates with ticks.
9. **Spatial grid** — grid-based neighbor queries return the same hits as a brute
   O(n²) reference on random layouts.
10. **Hand-coded controller smoke test** — a "seek brightest ray" herbivore in a
    food-rich world survives longer than a do-nothing controller.

## Scope (sub-project A)

- Headless only — no GPU, no egui, no `evors` strategy wiring (that is C).
- Feed-forward brains via the `Controller` trait; recurrent/memory/oscillator
  inputs deferred to B.
- Clamp-at-walls bounds; uniform spatial grid for neighbor queries.
- Deterministic, fixed-timestep `step(dt) -> Vec<Death>`.
