# evors-sandbox Sub-Project A (Headless Sim Core) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `evors-sandbox`, a headless deterministic artificial-life simulation core — world, cone-vision ray tracing, energy/eating/aging physics, a uniform spatial grid, and fitness accounting — exposing `World::step(dt) -> Vec<Death>` behind a `Controller` trait.

**Architecture:** A new workspace crate depending on `evors`/`evors-neat`. The `World` owns a seeded `ChaCha8Rng`, a `Vec<Organism>`, and a `Vec<Food>`. `step` senses → acts → integrates → eats → ages food → reports deaths. Organism brains are abstracted behind a `Controller` trait so the core is testable with hand-coded controllers; sub-projects C/D wire NEAT and a viewer later. No GPU, no egui.

**Tech Stack:** Rust (edition 2024), `rand` 0.9 (`random_range`/`random_bool`), `rand_chacha` 0.9 (`ChaCha8Rng::seed_from_u64`).

**Spec:** `docs/superpowers/specs/2026-06-15-evors-sandbox-A-sim-core-design.md`

---

## File Structure

```
evors-sandbox/
  Cargo.toml
  src/
    lib.rs        # module wiring + re-exports
    geom.rs       # Vec2 + ray-circle intersection
    config.rs     # SimConfig + SpeciesConfig
    organism.rs   # Diet, Organism, Food, Senses/RayHit/Motor, Controller, Death
    grid.rs       # SpatialGrid (uniform spatial hashing)
    vision.rs     # sense(): cone ray tracing -> Senses
    world.rs      # World + step()
  tests/
    sim.rs        # determinism + hand-coded-controller smoke test
```

Plus one modification to the root `Cargo.toml` workspace member list.

**Type conventions used throughout (defined in early tasks, referenced later):**
- `Vec2 { x: f32, y: f32 }` with `new/add/sub/scale/len/dot` (Task 1).
- `Diet::{Herbivore, Carnivore}` (Task 3).
- `Organism` fields: `pos, heading, energy, age, species, size, hue, diet, max_energy, time_alive, best_approach, controller` (Task 3).
- `Food { pos, size, hue }` (Task 3).
- `Senses { rays: Vec<RayHit>, energy_ratio: f32 }`, `RayHit { distance, hue }`, `Motor { forward, back, turn_left, turn_right }` (Task 3).
- `Controller::sense_act(&mut self, &Senses) -> Motor` (Task 3).
- `Death { species: usize, fitness: f64 }` (Task 3).

---

## Task 0: Crate scaffold + workspace member

**Files:**
- Create: `evors-sandbox/Cargo.toml`
- Create: `evors-sandbox/src/lib.rs`
- Modify: `Cargo.toml` (root workspace `members`)

- [ ] **Step 1: Create the crate manifest**

`evors-sandbox/Cargo.toml`:

```toml
[package]
name = "evors-sandbox"
version = "0.1.0"
edition = "2024"

[dependencies]
evors = { path = ".." }
evors-neat = { path = "../evors-neat" }
rand = "0.9"
rand_chacha = "0.9"
```

- [ ] **Step 2: Create a placeholder lib so it compiles**

`evors-sandbox/src/lib.rs`:

```rust
//! evors-sandbox — a headless, deterministic artificial-life simulation core.
//!
//! Organisms with pluggable [`Controller`] brains live in a shared [`World`],
//! sense food and each other via cone ray tracing, spend energy to move, eat,
//! age, and die. `World::step` advances the simulation one fixed tick and
//! returns the organisms that died (for an evolution loop to report on).

pub mod geom;
```

(Remaining modules are added by later tasks. Only `geom` exists after Task 1; until then keep `lib.rs` empty of `pub mod` lines so it compiles. For Step 2, use an empty body:)

```rust
//! evors-sandbox — a headless, deterministic artificial-life simulation core.
```

- [ ] **Step 3: Register the crate in the workspace**

In root `Cargo.toml`, change:

```toml
[workspace]
members = [".", "evors-neat", "evors-cppn"]
```

to:

```toml
[workspace]
members = [".", "evors-neat", "evors-cppn", "evors-sandbox"]
```

- [ ] **Step 4: Verify it builds**

Run: `rtk cargo build -p evors-sandbox`
Expected: compiles cleanly (an empty-library warning is fine).

- [ ] **Step 5: Commit**

```bash
rtk git add evors-sandbox/Cargo.toml evors-sandbox/src/lib.rs Cargo.toml Cargo.lock
rtk git commit -m "Scaffold evors-sandbox crate (workspace member A)"
```

---

## Task 1: Geometry — Vec2 and ray-circle intersection

**Files:**
- Create: `evors-sandbox/src/geom.rs`
- Modify: `evors-sandbox/src/lib.rs` (add `pub mod geom;`)

- [ ] **Step 1: Write the failing tests**

`evors-sandbox/src/geom.rs`:

```rust
//! 2D vector math and ray-circle intersection used by vision and collisions.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub fn new(x: f32, y: f32) -> Self {
        Vec2 { x, y }
    }
    pub fn add(self, o: Vec2) -> Vec2 {
        Vec2::new(self.x + o.x, self.y + o.y)
    }
    pub fn sub(self, o: Vec2) -> Vec2 {
        Vec2::new(self.x - o.x, self.y - o.y)
    }
    pub fn scale(self, s: f32) -> Vec2 {
        Vec2::new(self.x * s, self.y * s)
    }
    pub fn dot(self, o: Vec2) -> f32 {
        self.x * o.x + self.y * o.y
    }
    pub fn len(self) -> f32 {
        self.dot(self).sqrt()
    }
}

/// Distance along a **unit-direction** ray from `origin` to its first
/// intersection with the circle (`center`, `radius`), or `None` if the ray
/// misses or the hit lies beyond `max_len` (or behind the origin).
///
/// `dir` must be normalized.
pub fn ray_circle(origin: Vec2, dir: Vec2, max_len: f32, center: Vec2, radius: f32) -> Option<f32> {
    let oc = origin.sub(center);
    let b = oc.dot(dir);
    let c = oc.dot(oc) - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    let mut t = -b - sq;
    if t < 0.0 {
        t = -b + sq; // origin inside the circle: take the exit/forward root
    }
    if t < 0.0 || t > max_len {
        None
    } else {
        Some(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec_ops() {
        let a = Vec2::new(3.0, 4.0);
        assert_eq!(a.len(), 5.0);
        assert_eq!(a.add(Vec2::new(1.0, 1.0)), Vec2::new(4.0, 5.0));
        assert_eq!(a.sub(Vec2::new(1.0, 1.0)), Vec2::new(2.0, 3.0));
        assert_eq!(a.scale(2.0), Vec2::new(6.0, 8.0));
        assert_eq!(a.dot(Vec2::new(1.0, 0.0)), 3.0);
    }

    #[test]
    fn ray_hits_circle_ahead() {
        // Ray from origin pointing +x; circle centered at (10,0) r=1 -> hit at t=9.
        let t = ray_circle(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), 100.0, Vec2::new(10.0, 0.0), 1.0);
        assert!((t.unwrap() - 9.0).abs() < 1e-4);
    }

    #[test]
    fn ray_misses_circle_to_the_side() {
        let t = ray_circle(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), 100.0, Vec2::new(10.0, 5.0), 1.0);
        assert!(t.is_none());
    }

    #[test]
    fn ray_ignores_circle_behind() {
        let t = ray_circle(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), 100.0, Vec2::new(-10.0, 0.0), 1.0);
        assert!(t.is_none());
    }

    #[test]
    fn ray_respects_max_len() {
        let t = ray_circle(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), 5.0, Vec2::new(10.0, 0.0), 1.0);
        assert!(t.is_none());
    }
}
```

- [ ] **Step 2: Wire the module**

In `evors-sandbox/src/lib.rs`, ensure the body contains:

```rust
//! evors-sandbox — a headless, deterministic artificial-life simulation core.

pub mod geom;
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `rtk cargo test -p evors-sandbox geom`
Expected: 5 tests pass. (They are written to pass against the implementation above; this task ships impl + tests together since the math is the unit.)

- [ ] **Step 4: Commit**

```bash
rtk git add evors-sandbox/src/geom.rs evors-sandbox/src/lib.rs
rtk git commit -m "evors-sandbox: Vec2 + ray-circle intersection"
```

---

## Task 2: Config structs

**Files:**
- Create: `evors-sandbox/src/config.rs`
- Modify: `evors-sandbox/src/lib.rs`

- [ ] **Step 1: Write the config module**

`evors-sandbox/src/config.rs`:

```rust
//! Simulation configuration. All tunables live here; a `SimConfig` plus its
//! `seed` fully determines a run.

use crate::organism::Diet;

/// World-wide simulation parameters.
#[derive(Clone, Debug)]
pub struct SimConfig {
    pub width: f32,
    pub height: f32,

    // Vision (shared by all organisms for now).
    pub ray_count: usize,
    pub vision_length: f32,
    pub vision_arc: f32, // total fan width, radians

    // Motion.
    pub max_speed: f32,     // world-units / second at full thrust
    pub max_turn_rate: f32, // radians / second at full turn

    // Energy.
    pub passive_drain: f32, // energy / second, always applied
    pub move_cost: f32,     // energy per (speed-unit · second)
    pub age_decay: f32,     // per-second multiplicative decay of eat gain, in (0,1]

    // Food.
    pub food_count: usize, // target number of food items (density)
    pub food_size: f32,
    pub food_hue: f32,
    pub food_energy: f32,

    pub seed: u64,
    pub species: Vec<SpeciesConfig>,
}

/// One species' fixed traits. Diet never mutates; each species gets its own
/// gene pool in sub-project C.
#[derive(Clone, Debug)]
pub struct SpeciesConfig {
    pub diet: Diet,
    pub count: usize, // how many of this species are kept alive
    pub size: f32,
    pub hue: f32,
    pub max_energy: f32,
    pub start_energy: f32,
}

impl Default for SimConfig {
    fn default() -> Self {
        SimConfig {
            width: 1000.0,
            height: 1000.0,
            ray_count: 7,
            vision_length: 200.0,
            vision_arc: std::f32::consts::FRAC_PI_2, // 90 degrees
            max_speed: 120.0,
            max_turn_rate: std::f32::consts::PI, // half-turn per second
            passive_drain: 1.0,
            move_cost: 0.02,
            age_decay: 0.98,
            food_count: 60,
            food_size: 6.0,
            food_hue: 0.33, // green
            food_energy: 40.0,
            seed: 1,
            species: Vec::new(),
        }
    }
}
```

- [ ] **Step 2: Wire the module**

Add to `evors-sandbox/src/lib.rs`:

```rust
pub mod config;
pub mod organism;
```

(Declare `organism` here too; Task 3 fills it. Order does not matter to the compiler.)

- [ ] **Step 3: Verify it builds after Task 3**

This task cannot compile until `organism::Diet` exists. Proceed to Task 3, then build both together. (Commit happens at the end of Task 3.)

---

## Task 3: Core types — Diet, Organism, Food, sensory/motor, Controller, Death

**Files:**
- Create: `evors-sandbox/src/organism.rs`

- [ ] **Step 1: Write the organism module with a unit test**

`evors-sandbox/src/organism.rs`:

```rust
//! Entities and the sensory/motor interface between an organism and its brain.

use crate::geom::Vec2;

/// Whether an organism eats food (`Herbivore`) or other organisms
/// (`Carnivore`). Fixed per species.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Diet {
    Herbivore,
    Carnivore,
}

/// One vision ray's reading: normalized closeness and the hue of what it hit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit {
    /// `1 - hit_dist/vision_length` in `[0,1]`; `0.0` means nothing seen.
    pub distance: f32,
    /// Hue in `[0,1]` of the nearest hit circle; `0.0` if nothing seen.
    pub hue: f32,
}

/// The full input an organism's brain receives each tick.
#[derive(Clone, Debug)]
pub struct Senses {
    pub rays: Vec<RayHit>,
    pub energy_ratio: f32, // energy / max_energy, in [0,1]
}

/// The brain's output: four opposing drives in `[0,1]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Motor {
    pub forward: f32,
    pub back: f32,
    pub turn_left: f32,
    pub turn_right: f32,
}

/// A pluggable brain. Implemented by hand-coded controllers (tests) and by a
/// NEAT-backed controller in sub-project C.
pub trait Controller {
    fn sense_act(&mut self, senses: &Senses) -> Motor;
}

/// A living organism.
pub struct Organism {
    pub pos: Vec2,
    pub heading: f32, // radians
    pub energy: f32,
    pub age: f32, // seconds alive
    pub species: usize,
    pub size: f32,
    pub hue: f32,
    pub diet: Diet,
    pub max_energy: f32,

    // Fitness accumulators, finalized at death.
    pub time_alive: f32,
    /// Smallest surface distance ever recorded to a valid target; starts at
    /// +infinity (never approached anything).
    pub best_approach: f32,

    pub controller: Box<dyn Controller>,
}

impl Organism {
    /// Build an organism at `pos` with `heading`, taking its fixed traits from a
    /// species and the given controller.
    pub fn new(
        species: usize,
        pos: Vec2,
        heading: f32,
        size: f32,
        hue: f32,
        diet: Diet,
        max_energy: f32,
        start_energy: f32,
        controller: Box<dyn Controller>,
    ) -> Self {
        Organism {
            pos,
            heading,
            energy: start_energy,
            age: 0.0,
            species,
            size,
            hue,
            diet,
            max_energy,
            time_alive: 0.0,
            best_approach: f32::INFINITY,
            controller,
        }
    }

    pub fn energy_ratio(&self) -> f32 {
        (self.energy / self.max_energy).clamp(0.0, 1.0)
    }
}

/// A food pellet.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Food {
    pub pos: Vec2,
    pub size: f32,
    pub hue: f32,
}

/// Returned by `World::step` for each organism that died this tick.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Death {
    pub species: usize,
    pub fitness: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Idle;
    impl Controller for Idle {
        fn sense_act(&mut self, _: &Senses) -> Motor {
            Motor { forward: 0.0, back: 0.0, turn_left: 0.0, turn_right: 0.0 }
        }
    }

    #[test]
    fn organism_starts_with_expected_state() {
        let o = Organism::new(
            0, Vec2::new(1.0, 2.0), 0.0, 5.0, 0.5, Diet::Herbivore, 100.0, 50.0, Box::new(Idle),
        );
        assert_eq!(o.energy, 50.0);
        assert_eq!(o.energy_ratio(), 0.5);
        assert_eq!(o.age, 0.0);
        assert!(o.best_approach.is_infinite());
    }
}
```

- [ ] **Step 2: Run tests to verify the module compiles and passes**

Run: `rtk cargo test -p evors-sandbox organism`
Expected: `organism_starts_with_expected_state` passes; `config.rs` now compiles (it imports `Diet`).

- [ ] **Step 3: Commit (config + organism together)**

```bash
rtk git add evors-sandbox/src/config.rs evors-sandbox/src/organism.rs evors-sandbox/src/lib.rs
rtk git commit -m "evors-sandbox: config + core entity/sensory/motor types"
```

---

## Task 4: Uniform spatial grid

**Files:**
- Create: `evors-sandbox/src/grid.rs`
- Modify: `evors-sandbox/src/lib.rs` (add `pub mod grid;`)

The grid buckets external entities (addressed by index) into cells, so neighbor queries inspect only nearby cells instead of every entity.

- [ ] **Step 1: Write the failing tests**

`evors-sandbox/src/grid.rs`:

```rust
//! A uniform spatial grid for near-linear neighbor queries.
//!
//! Entities are inserted by index + position; `candidates` returns every index
//! in the cells overlapping a query disc's bounding box (a superset — callers
//! filter by exact distance).

use crate::geom::Vec2;

pub struct SpatialGrid {
    cell: f32,
    cols: usize,
    rows: usize,
    cells: Vec<Vec<usize>>,
}

impl SpatialGrid {
    /// A grid covering `[0,width] x [0,height]` with square cells of side
    /// `cell`. `cell` should be >= the largest query radius for best results.
    pub fn new(width: f32, height: f32, cell: f32) -> Self {
        let cols = ((width / cell).ceil() as usize).max(1);
        let rows = ((height / cell).ceil() as usize).max(1);
        SpatialGrid { cell, cols, rows, cells: vec![Vec::new(); cols * rows] }
    }

    pub fn clear(&mut self) {
        for c in &mut self.cells {
            c.clear();
        }
    }

    fn cell_of(&self, pos: Vec2) -> (usize, usize) {
        let cx = ((pos.x / self.cell).floor() as isize).clamp(0, self.cols as isize - 1) as usize;
        let cy = ((pos.y / self.cell).floor() as isize).clamp(0, self.rows as isize - 1) as usize;
        (cx, cy)
    }

    pub fn insert(&mut self, idx: usize, pos: Vec2) {
        let (cx, cy) = self.cell_of(pos);
        self.cells[cy * self.cols + cx].push(idx);
    }

    /// All indices in cells overlapping the bounding box of the disc
    /// (`center`, `radius`). A superset of the true neighbors.
    pub fn candidates(&self, center: Vec2, radius: f32) -> Vec<usize> {
        let min = Vec2::new(center.x - radius, center.y - radius);
        let max = Vec2::new(center.x + radius, center.y + radius);
        let (min_x, min_y) = self.cell_of(min);
        let (max_x, max_y) = self.cell_of(max);
        let mut out = Vec::new();
        for cy in min_y..=max_y {
            for cx in min_x..=max_x {
                out.extend_from_slice(&self.cells[cy * self.cols + cx]);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand::Rng;

    /// Exact neighbors within `radius` via the grid (candidates + distance filter).
    fn grid_neighbors(grid: &SpatialGrid, pts: &[Vec2], center: Vec2, radius: f32) -> Vec<usize> {
        let mut v: Vec<usize> = grid
            .candidates(center, radius)
            .into_iter()
            .filter(|&i| pts[i].sub(center).len() <= radius)
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Exact neighbors via brute force.
    fn brute_neighbors(pts: &[Vec2], center: Vec2, radius: f32) -> Vec<usize> {
        let mut v: Vec<usize> = (0..pts.len())
            .filter(|&i| pts[i].sub(center).len() <= radius)
            .collect();
        v.sort_unstable();
        v
    }

    #[test]
    fn grid_matches_brute_force() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let (w, h) = (500.0_f32, 500.0_f32);
        let pts: Vec<Vec2> = (0..400)
            .map(|_| Vec2::new(rng.random_range(0.0..w), rng.random_range(0.0..h)))
            .collect();

        let mut grid = SpatialGrid::new(w, h, 50.0);
        for (i, p) in pts.iter().enumerate() {
            grid.insert(i, *p);
        }

        for _ in 0..20 {
            let center = Vec2::new(rng.random_range(0.0..w), rng.random_range(0.0..h));
            let radius = rng.random_range(10.0..50.0);
            assert_eq!(
                grid_neighbors(&grid, &pts, center, radius),
                brute_neighbors(&pts, center, radius),
            );
        }
    }
}
```

- [ ] **Step 2: Wire the module**

Add `pub mod grid;` to `evors-sandbox/src/lib.rs`.

- [ ] **Step 3: Run the test**

Run: `rtk cargo test -p evors-sandbox grid`
Expected: `grid_matches_brute_force` passes (grid candidates + filter equal brute force).

- [ ] **Step 4: Commit**

```bash
rtk git add evors-sandbox/src/grid.rs evors-sandbox/src/lib.rs
rtk git commit -m "evors-sandbox: uniform spatial grid with brute-force parity test"
```

---

## Task 5: Vision — cone ray tracing

**Files:**
- Create: `evors-sandbox/src/vision.rs`
- Modify: `evors-sandbox/src/lib.rs` (add `pub mod vision;`)

`sense` takes pre-filtered candidate index lists (so it is testable without a grid; the world passes grid candidates). It traces `ray_count` rays across `vision_arc` centered on heading and, per ray, records the nearest circle hit among candidate food and candidate organisms.

- [ ] **Step 1: Write the failing tests**

`evors-sandbox/src/vision.rs`:

```rust
//! Cone-based vision: trace rays, return per-ray nearest-hit distance + hue.

use crate::config::SimConfig;
use crate::geom::{Vec2, ray_circle};
use crate::organism::{Food, Organism, RayHit, Senses};

/// Direction unit vector for ray `i` of `ray_count`, fanned across `arc`
/// centered on `heading`.
fn ray_dir(heading: f32, arc: f32, i: usize, ray_count: usize) -> Vec2 {
    let offset = if ray_count <= 1 {
        0.0
    } else {
        -arc / 2.0 + arc * (i as f32 / (ray_count - 1) as f32)
    };
    let a = heading + offset;
    Vec2::new(a.cos(), a.sin())
}

/// Compute `seer`'s senses. `food_candidates` / `org_candidates` are indices
/// into `food` / `organisms` to test; `self_idx` is excluded from organisms.
pub fn sense(
    seer: &Organism,
    self_idx: usize,
    cfg: &SimConfig,
    food: &[Food],
    food_candidates: &[usize],
    organisms: &[Organism],
    org_candidates: &[usize],
) -> Senses {
    let mut rays = Vec::with_capacity(cfg.ray_count);
    for i in 0..cfg.ray_count {
        let dir = ray_dir(seer.heading, cfg.vision_arc, i, cfg.ray_count);
        let mut best_t = f32::INFINITY;
        let mut best_hue = 0.0;

        for &fi in food_candidates {
            let f = &food[fi];
            if let Some(t) = ray_circle(seer.pos, dir, cfg.vision_length, f.pos, f.size) {
                if t < best_t {
                    best_t = t;
                    best_hue = f.hue;
                }
            }
        }
        for &oi in org_candidates {
            if oi == self_idx {
                continue;
            }
            let o = &organisms[oi];
            if let Some(t) = ray_circle(seer.pos, dir, cfg.vision_length, o.pos, o.size) {
                if t < best_t {
                    best_t = t;
                    best_hue = o.hue;
                }
            }
        }

        let distance = if best_t.is_finite() {
            1.0 - best_t / cfg.vision_length
        } else {
            0.0
        };
        rays.push(RayHit { distance, hue: best_hue });
    }

    Senses { rays, energy_ratio: seer.energy_ratio() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SimConfig;
    use crate::organism::{Controller, Diet, Motor, Organism};

    struct Idle;
    impl Controller for Idle {
        fn sense_act(&mut self, _: &Senses) -> Motor {
            Motor { forward: 0.0, back: 0.0, turn_left: 0.0, turn_right: 0.0 }
        }
    }

    fn seer_at(pos: Vec2, heading: f32) -> Organism {
        Organism::new(0, pos, heading, 5.0, 0.5, Diet::Herbivore, 100.0, 50.0, Box::new(Idle))
    }

    fn cfg_1ray(length: f32, arc: f32) -> SimConfig {
        SimConfig { ray_count: 1, vision_length: length, vision_arc: arc, ..SimConfig::default() }
    }

    #[test]
    fn center_ray_sees_food_dead_ahead() {
        // Single ray, organism at origin facing +x, food 100 units ahead.
        let cfg = cfg_1ray(200.0, 0.0);
        let seer = seer_at(Vec2::new(0.0, 0.0), 0.0);
        let food = vec![Food { pos: Vec2::new(100.0, 0.0), size: 10.0, hue: 0.33 }];
        let s = sense(&seer, 0, &cfg, &food, &[0], &[], &[]);
        // Surface hit at t = 90 -> distance = 1 - 90/200 = 0.55.
        assert!((s.rays[0].distance - 0.55).abs() < 1e-3);
        assert!((s.rays[0].hue - 0.33).abs() < 1e-6);
        assert!((s.energy_ratio - 0.5).abs() < 1e-6);
    }

    #[test]
    fn nothing_seen_reads_zero() {
        let cfg = cfg_1ray(200.0, 0.0);
        let seer = seer_at(Vec2::new(0.0, 0.0), 0.0);
        let food = vec![Food { pos: Vec2::new(0.0, 100.0), size: 10.0, hue: 0.33 }]; // 90 deg to the side
        let s = sense(&seer, 0, &cfg, &food, &[0], &[], &[]);
        assert_eq!(s.rays[0].distance, 0.0);
        assert_eq!(s.rays[0].hue, 0.0);
    }

    #[test]
    fn food_beyond_range_not_seen() {
        let cfg = cfg_1ray(50.0, 0.0); // range 50, food at 100
        let seer = seer_at(Vec2::new(0.0, 0.0), 0.0);
        let food = vec![Food { pos: Vec2::new(100.0, 0.0), size: 10.0, hue: 0.33 }];
        let s = sense(&seer, 0, &cfg, &food, &[0], &[], &[]);
        assert_eq!(s.rays[0].distance, 0.0);
    }

    #[test]
    fn food_outside_arc_not_seen() {
        // 3 rays across a narrow 20-degree arc, facing +x; food straight up is
        // outside every ray.
        let cfg = SimConfig {
            ray_count: 3,
            vision_length: 200.0,
            vision_arc: 20.0_f32.to_radians(),
            ..SimConfig::default()
        };
        let seer = seer_at(Vec2::new(0.0, 0.0), 0.0);
        let food = vec![Food { pos: Vec2::new(0.0, 100.0), size: 5.0, hue: 0.33 }];
        let s = sense(&seer, 0, &cfg, &food, &[0], &[], &[]);
        assert!(s.rays.iter().all(|r| r.distance == 0.0));
    }

    #[test]
    fn nearest_circle_wins() {
        let cfg = cfg_1ray(200.0, 0.0);
        let seer = seer_at(Vec2::new(0.0, 0.0), 0.0);
        let food = vec![
            Food { pos: Vec2::new(150.0, 0.0), size: 5.0, hue: 0.1 },
            Food { pos: Vec2::new(50.0, 0.0), size: 5.0, hue: 0.9 },
        ];
        let s = sense(&seer, 0, &cfg, &food, &[0, 1], &[], &[]);
        assert!((s.rays[0].hue - 0.9).abs() < 1e-6); // the closer one
    }
}
```

- [ ] **Step 2: Wire the module**

Add `pub mod vision;` to `evors-sandbox/src/lib.rs`.

- [ ] **Step 3: Run the tests**

Run: `rtk cargo test -p evors-sandbox vision`
Expected: all 5 vision tests pass.

- [ ] **Step 4: Commit**

```bash
rtk git add evors-sandbox/src/vision.rs evors-sandbox/src/lib.rs
rtk git commit -m "evors-sandbox: cone-vision ray tracing (distance + hue per ray)"
```

---

## Task 6: World — construction, food spawning, and a random-position helper

**Files:**
- Create: `evors-sandbox/src/world.rs`
- Modify: `evors-sandbox/src/lib.rs` (add `pub mod world;` and re-exports)

This task builds the `World` shell: it owns the config, RNG, organisms, and food, tops food up to target density, and exposes a deterministic random-position helper. `step` is added in Tasks 7-9.

- [ ] **Step 1: Write the failing tests**

`evors-sandbox/src/world.rs`:

```rust
//! The simulation world: owns entities + RNG and advances them via `step`.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::config::SimConfig;
use crate::geom::Vec2;
use crate::organism::{Death, Food, Organism};

pub struct World {
    pub cfg: SimConfig,
    pub organisms: Vec<Organism>,
    pub food: Vec<Food>,
    rng: ChaCha8Rng,
}

impl World {
    /// Create a world seeded from `cfg.seed`, with food filled to target
    /// density and no organisms yet (the caller adds them via `add_organism`).
    pub fn new(cfg: SimConfig) -> Self {
        let rng = ChaCha8Rng::seed_from_u64(cfg.seed);
        let mut w = World { cfg, organisms: Vec::new(), food: Vec::new(), rng };
        w.replenish_food();
        w
    }

    /// A uniformly random point inside the world bounds (advances the RNG).
    pub fn random_position(&mut self) -> Vec2 {
        Vec2::new(
            self.rng.random_range(0.0..self.cfg.width),
            self.rng.random_range(0.0..self.cfg.height),
        )
    }

    /// A uniformly random heading in `[0, 2π)` (advances the RNG).
    pub fn random_heading(&mut self) -> f32 {
        self.rng.random_range(0.0..std::f32::consts::TAU)
    }

    pub fn add_organism(&mut self, o: Organism) {
        self.organisms.push(o);
    }

    /// Spawn food until `food.len() == cfg.food_count`.
    pub(crate) fn replenish_food(&mut self) {
        while self.food.len() < self.cfg.food_count {
            let pos = Vec2::new(
                self.rng.random_range(0.0..self.cfg.width),
                self.rng.random_range(0.0..self.cfg.height),
            );
            let (size, hue) = (self.cfg.food_size, self.cfg.food_hue);
            self.food.push(Food { pos, size, hue });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> SimConfig {
        SimConfig { food_count: 25, seed: 7, ..SimConfig::default() }
    }

    #[test]
    fn new_world_fills_food_to_target() {
        let w = World::new(cfg());
        assert_eq!(w.food.len(), 25);
        assert!(w.organisms.is_empty());
    }

    #[test]
    fn replenish_tops_back_up_after_eating() {
        let mut w = World::new(cfg());
        w.food.truncate(10);
        w.replenish_food();
        assert_eq!(w.food.len(), 25);
    }

    #[test]
    fn same_seed_same_food_layout() {
        let a = World::new(cfg());
        let b = World::new(cfg());
        assert_eq!(a.food, b.food);
    }
}
```

- [ ] **Step 2: Wire the module + re-exports**

Add to `evors-sandbox/src/lib.rs`:

```rust
pub mod world;

pub use config::{SimConfig, SpeciesConfig};
pub use organism::{Controller, Death, Diet, Food, Motor, Organism, RayHit, Senses};
pub use world::World;
```

- [ ] **Step 3: Run the tests**

Run: `rtk cargo test -p evors-sandbox world`
Expected: 3 tests pass.

- [ ] **Step 4: Commit**

```bash
rtk git add evors-sandbox/src/world.rs evors-sandbox/src/lib.rs
rtk git commit -m "evors-sandbox: World shell — entities, RNG, food density"
```

---

## Task 7: Movement, energy drain, aging, and death

**Files:**
- Modify: `evors-sandbox/src/world.rs` (add a `step` that does sense→act→integrate→drain→age→death, no eating yet)

This task adds a first `step` covering the per-organism update and death collection. Eating (Task 8) and fitness's approach term (Task 9) layer on after.

- [ ] **Step 1: Add the failing tests**

Append to the `tests` module in `evors-sandbox/src/world.rs`:

```rust
    use crate::geom::Vec2;
    use crate::organism::{Controller, Diet, Motor, Senses};

    struct Idle;
    impl Controller for Idle {
        fn sense_act(&mut self, _: &Senses) -> Motor {
            Motor { forward: 0.0, back: 0.0, turn_left: 0.0, turn_right: 0.0 }
        }
    }
    struct FullThrust;
    impl Controller for FullThrust {
        fn sense_act(&mut self, _: &Senses) -> Motor {
            Motor { forward: 1.0, back: 0.0, turn_left: 0.0, turn_right: 0.0 }
        }
    }

    fn herbivore(pos: Vec2, energy: f32, c: Box<dyn Controller>) -> Organism {
        let mut o = Organism::new(0, pos, 0.0, 5.0, 0.5, Diet::Herbivore, 100.0, energy, c);
        o.energy = energy;
        o
    }

    fn empty_world() -> World {
        // No food, no species interactions — isolate movement/energy.
        World::new(SimConfig { food_count: 0, seed: 1, ..SimConfig::default() })
    }

    #[test]
    fn idle_organism_starves_from_passive_drain() {
        let mut w = empty_world();
        w.add_organism(herbivore(Vec2::new(500.0, 500.0), 2.5, Box::new(Idle)));
        // passive_drain = 1.0/s; dt = 1.0 -> 2.5 energy lasts 3 ticks.
        assert!(w.step(1.0).is_empty()); // 1.5 left
        assert!(w.step(1.0).is_empty()); // 0.5 left
        let dead = w.step(1.0); // -> dead
        assert_eq!(dead.len(), 1);
        assert_eq!(dead[0].species, 0);
        assert!(w.organisms.is_empty());
    }

    #[test]
    fn moving_costs_more_than_resting() {
        let mut idle_w = empty_world();
        idle_w.add_organism(herbivore(Vec2::new(500.0, 500.0), 100.0, Box::new(Idle)));
        idle_w.step(1.0);
        let idle_energy = idle_w.organisms[0].energy;

        let mut move_w = empty_world();
        move_w.add_organism(herbivore(Vec2::new(500.0, 500.0), 100.0, Box::new(FullThrust)));
        move_w.step(1.0);
        let move_energy = move_w.organisms[0].energy;

        assert!(move_energy < idle_energy);
    }

    #[test]
    fn thrust_moves_along_heading_and_ages() {
        let mut w = empty_world();
        w.add_organism(herbivore(Vec2::new(100.0, 100.0), 100.0, Box::new(FullThrust)));
        w.step(0.5);
        let o = &w.organisms[0];
        // heading 0 -> +x; moved max_speed(120)*0.5 = 60 units.
        assert!((o.pos.x - 160.0).abs() < 1e-3);
        assert!((o.pos.y - 100.0).abs() < 1e-3);
        assert!((o.age - 0.5).abs() < 1e-6);
        assert!((o.time_alive - 0.5).abs() < 1e-6);
    }

    #[test]
    fn movement_clamps_at_walls() {
        let mut w = empty_world();
        // Near the right wall, full thrust along +x.
        w.add_organism(herbivore(Vec2::new(w.cfg.width - 5.0, 100.0), 100.0, Box::new(FullThrust)));
        w.step(1.0);
        assert!(w.organisms[0].pos.x <= w.cfg.width);
        assert!(w.organisms[0].pos.x >= 0.0);
    }
```

- [ ] **Step 2: Implement `step` (movement/energy/age/death)**

Add these methods inside `impl World` in `evors-sandbox/src/world.rs` (above the `#[cfg(test)]`). This version has no eating yet; Task 8 extends it.

```rust
    /// Advance the simulation one tick of `dt` seconds; return organisms that
    /// died this tick (energy depleted), each with finalized fitness.
    pub fn step(&mut self, dt: f32) -> Vec<Death> {
        self.update_organisms(dt);
        self.replenish_food();
        self.collect_dead()
    }

    /// Sense → act → integrate motion → spend energy → age, for every organism.
    fn update_organisms(&mut self, dt: f32) {
        let cfg = self.cfg.clone();
        // Snapshot positions so vision (added in a later task) and motion read a
        // consistent frame; for now each organism updates independently.
        for i in 0..self.organisms.len() {
            // Senses are wired to vision in Task 9; for movement/energy a blank
            // sense vector is sufficient and keeps controllers callable.
            let senses = crate::organism::Senses {
                rays: vec![crate::organism::RayHit { distance: 0.0, hue: 0.0 }; cfg.ray_count],
                energy_ratio: self.organisms[i].energy_ratio(),
            };
            let m = self.organisms[i].controller.sense_act(&senses);

            let thrust = (m.forward - m.back).clamp(-1.0, 1.0);
            let turn = (m.turn_right - m.turn_left).clamp(-1.0, 1.0);

            let o = &mut self.organisms[i];
            o.heading += turn * cfg.max_turn_rate * dt;
            let speed = thrust * cfg.max_speed;
            let dir = Vec2::new(o.heading.cos(), o.heading.sin());
            o.pos = o.pos.add(dir.scale(speed * dt));
            o.pos.x = o.pos.x.clamp(0.0, cfg.width);
            o.pos.y = o.pos.y.clamp(0.0, cfg.height);

            let move_cost = speed.abs() * cfg.move_cost * dt;
            o.energy -= cfg.passive_drain * dt + move_cost;
            o.age += dt;
            o.time_alive += dt;
        }
    }

    /// Remove organisms with depleted energy, returning their deaths.
    fn collect_dead(&mut self) -> Vec<Death> {
        let mut deaths = Vec::new();
        let mut i = 0;
        while i < self.organisms.len() {
            if self.organisms[i].energy <= 0.0 {
                let o = self.organisms.swap_remove(i);
                deaths.push(Death { species: o.species, fitness: self.fitness_of(&o) });
            } else {
                i += 1;
            }
        }
        deaths
    }

    /// Finalized fitness. The closest-approach term is added in Task 9; for now
    /// fitness is time alive.
    fn fitness_of(&self, o: &Organism) -> f64 {
        o.time_alive as f64
    }
```

Note: `swap_remove` reorders the `organisms` vector. That is fine for correctness (organisms are unordered), but means death order within a tick is not positional — tests assert on counts/species, not order.

- [ ] **Step 3: Run the tests**

Run: `rtk cargo test -p evors-sandbox world`
Expected: all world tests (Task 6 + Task 7) pass.

- [ ] **Step 4: Commit**

```bash
rtk git add evors-sandbox/src/world.rs
rtk git commit -m "evors-sandbox: step() — movement, energy drain, aging, death"
```

---

## Task 8: Eating — herbivore↔food, carnivore↔herbivore, with aging decay

**Files:**
- Modify: `evors-sandbox/src/world.rs` (add an eating phase to `step`)

- [ ] **Step 1: Add the failing tests**

Append to the `tests` module in `evors-sandbox/src/world.rs`:

```rust
    fn carnivore(pos: Vec2, energy: f32) -> Organism {
        Organism::new(1, pos, 0.0, 8.0, 0.0, Diet::Carnivore, 200.0, energy, Box::new(Idle))
    }

    #[test]
    fn herbivore_eats_overlapping_food_and_gains_energy() {
        let mut w = World::new(SimConfig { food_count: 0, seed: 1, ..SimConfig::default() });
        let before = 50.0;
        w.add_organism(herbivore(Vec2::new(100.0, 100.0), before, Box::new(Idle)));
        // Food overlapping the organism (within size sum).
        w.food.push(Food { pos: Vec2::new(101.0, 100.0), size: w.cfg.food_size, hue: w.cfg.food_hue });
        w.step(0.0); // dt=0 isolates eating from drain
        assert!(w.food.is_empty());
        // Gained food_energy * age_decay^age (age still 0) minus zero drain.
        assert!((w.organisms[0].energy - (before + w.cfg.food_energy)).abs() < 1e-3);
    }

    #[test]
    fn carnivore_eats_overlapping_herbivore() {
        let mut w = World::new(SimConfig { food_count: 0, seed: 1, ..SimConfig::default() });
        w.add_organism(herbivore(Vec2::new(200.0, 200.0), 30.0, Box::new(Idle))); // index 0, species 0
        w.add_organism(carnivore(Vec2::new(205.0, 200.0), 80.0)); // index 1, species 1, overlaps
        let dead = w.step(0.0);
        // The herbivore died (eaten) and is reported under its species (0).
        assert_eq!(dead.len(), 1);
        assert_eq!(dead[0].species, 0);
        // One organism (the carnivore) remains and gained energy.
        assert_eq!(w.organisms.len(), 1);
        assert_eq!(w.organisms[0].diet, Diet::Carnivore);
        assert!(w.organisms[0].energy > 80.0);
    }

    #[test]
    fn eat_gain_shrinks_with_age() {
        // Two herbivores, same setup, different ages; older gains less.
        let mut w = World::new(SimConfig { food_count: 0, seed: 1, ..SimConfig::default() });

        let mut young = herbivore(Vec2::new(100.0, 100.0), 50.0, Box::new(Idle));
        young.age = 0.0;
        let mut old = herbivore(Vec2::new(800.0, 800.0), 50.0, Box::new(Idle));
        old.age = 100.0;
        w.add_organism(young);
        w.add_organism(old);
        w.food.push(Food { pos: Vec2::new(100.0, 100.0), size: w.cfg.food_size, hue: w.cfg.food_hue });
        w.food.push(Food { pos: Vec2::new(800.0, 800.0), size: w.cfg.food_size, hue: w.cfg.food_hue });

        w.step(0.0);
        let young_gain = w.organisms.iter().find(|o| o.pos.x < 500.0).unwrap().energy - 50.0;
        let old_gain = w.organisms.iter().find(|o| o.pos.x > 500.0).unwrap().energy - 50.0;
        assert!(old_gain < young_gain);
        assert!(old_gain > 0.0);
    }
```

- [ ] **Step 2: Implement the eating phase**

First, update the imports at the top of `evors-sandbox/src/world.rs` — `resolve_eating` and `nearest_target_dist` need `Diet`:

```rust
use crate::organism::{Death, Diet, Food, Organism};
```

In `evors-sandbox/src/world.rs`, change `step` to call `resolve_eating` between organism updates and death collection:

```rust
    pub fn step(&mut self, dt: f32) -> Vec<Death> {
        self.update_organisms(dt);
        let mut deaths = self.resolve_eating();
        self.replenish_food();
        deaths.extend(self.collect_dead());
        deaths
    }
```

Add the eating method inside `impl World` (uses the spatial grid for neighbor candidates):

```rust
    /// Resolve eating collisions. Herbivores eat overlapping food; carnivores
    /// eat overlapping herbivores (which die immediately). Returns deaths from
    /// predation. Energy gains are scaled by the eater's age decay.
    fn resolve_eating(&mut self) -> Vec<Death> {
        use crate::grid::SpatialGrid;
        let cfg = self.cfg.clone();
        let cell = cfg.vision_length.max(cfg.food_size * 4.0).max(1.0);

        // --- Herbivore eats food ---
        let mut food_grid = SpatialGrid::new(cfg.width, cfg.height, cell);
        for (fi, f) in self.food.iter().enumerate() {
            food_grid.insert(fi, f.pos);
        }
        let mut eaten_food = vec![false; self.food.len()];
        for i in 0..self.organisms.len() {
            if self.organisms[i].diet != Diet::Herbivore {
                continue;
            }
            let (pos, reach, age) =
                (self.organisms[i].pos, self.organisms[i].size, self.organisms[i].age);
            let mut gain = 0.0;
            for fi in food_grid.candidates(pos, reach + cfg.food_size) {
                if eaten_food[fi] {
                    continue;
                }
                let f = &self.food[fi];
                if pos.sub(f.pos).len() <= reach + f.size {
                    eaten_food[fi] = true;
                    gain += cfg.food_energy * cfg.age_decay.powf(age);
                }
            }
            if gain > 0.0 {
                let o = &mut self.organisms[i];
                o.energy = (o.energy + gain).min(o.max_energy);
            }
        }
        // Remove eaten food (high-to-low so swap_remove indices stay valid).
        for fi in (0..self.food.len()).rev() {
            if eaten_food[fi] {
                self.food.swap_remove(fi);
            }
        }

        // --- Carnivore eats herbivore ---
        let mut org_grid = SpatialGrid::new(cfg.width, cfg.height, cell);
        for (oi, o) in self.organisms.iter().enumerate() {
            org_grid.insert(oi, o.pos);
        }
        let mut eaten_org = vec![false; self.organisms.len()];
        let max_size = self.organisms.iter().map(|o| o.size).fold(0.0_f32, f32::max);
        for i in 0..self.organisms.len() {
            if self.organisms[i].diet != Diet::Carnivore {
                continue;
            }
            let (pos, reach, age) =
                (self.organisms[i].pos, self.organisms[i].size, self.organisms[i].age);
            let mut gain = 0.0;
            for pi in org_grid.candidates(pos, reach + max_size) {
                if pi == i || eaten_org[pi] {
                    continue;
                }
                if self.organisms[pi].diet != Diet::Herbivore {
                    continue;
                }
                let prey = &self.organisms[pi];
                if pos.sub(prey.pos).len() <= reach + prey.size {
                    eaten_org[pi] = true;
                    gain += prey.energy.max(0.0) * cfg.age_decay.powf(age);
                }
            }
            if gain > 0.0 {
                let o = &mut self.organisms[i];
                o.energy = (o.energy + gain).min(o.max_energy);
            }
        }
        // Collect eaten herbivores as deaths and remove them.
        let mut deaths = Vec::new();
        for pi in (0..self.organisms.len()).rev() {
            if eaten_org[pi] {
                let prey = self.organisms.swap_remove(pi);
                deaths.push(Death { species: prey.species, fitness: self.fitness_of(&prey) });
            }
        }
        deaths
    }
```

- [ ] **Step 3: Run the tests**

Run: `rtk cargo test -p evors-sandbox world`
Expected: all world tests including the three eating tests pass.

- [ ] **Step 4: Commit**

```bash
rtk git add evors-sandbox/src/world.rs
rtk git commit -m "evors-sandbox: eating — herbivore/food, carnivore/prey, age-decayed gain"
```

---

## Task 9: Wire vision into step + closest-approach fitness

**Files:**
- Modify: `evors-sandbox/src/world.rs`

This replaces the blank-senses placeholder with real vision (built from the grid), and adds the closest-approach fitness term updated each tick.

- [ ] **Step 1: Add the failing tests**

Append to the `tests` module in `evors-sandbox/src/world.rs`:

```rust
    /// A controller that steers toward the brightest (closest) ray, used to
    /// prove real senses reach the brain.
    struct SeekBright;
    impl Controller for SeekBright {
        fn sense_act(&mut self, s: &Senses) -> Motor {
            // Pick the strongest ray; if it's left of center turn left, etc.
            let n = s.rays.len();
            let (mut best_i, mut best) = (0usize, -1.0_f32);
            for (i, r) in s.rays.iter().enumerate() {
                if r.distance > best {
                    best = r.distance;
                    best_i = i;
                }
            }
            let center = (n as f32 - 1.0) / 2.0;
            let (mut tl, mut tr) = (0.0, 0.0);
            if (best_i as f32) < center {
                tl = 1.0; // target is to the left (rays fan left-to-right)
            } else if (best_i as f32) > center {
                tr = 1.0;
            }
            Motor { forward: 1.0, back: 0.0, turn_left: tl, turn_right: tr }
        }
    }

    #[test]
    fn closest_approach_improves_fitness_for_a_seeker() {
        // A seeker placed away from a single food, no replenishment, should end
        // with a finite best_approach (it got nearer than infinity) and thus
        // fitness above pure time-alive.
        let mut w = World::new(SimConfig { food_count: 0, seed: 3, ..SimConfig::default() });
        let mut o = Organism::new(
            0, Vec2::new(100.0, 500.0), 0.0, 5.0, 0.5, Diet::Herbivore, 100.0, 100.0,
            Box::new(SeekBright),
        );
        o.energy = 100.0;
        w.add_organism(o);
        w.food.push(Food { pos: Vec2::new(300.0, 500.0), size: 8.0, hue: 0.33 });
        for _ in 0..10 {
            w.step(0.1);
        }
        assert!(w.organisms[0].best_approach.is_finite());
    }

    #[test]
    fn fitness_includes_capped_approach_bonus() {
        // Construct an organism that died having approached to surface distance 0
        // (touching). Its bonus should equal one meal's energy.
        let mut w = World::new(SimConfig { food_count: 0, seed: 1, ..SimConfig::default() });
        let mut o = herbivore(Vec2::new(100.0, 100.0), 0.0, Box::new(Idle));
        o.best_approach = 0.0;
        o.time_alive = 5.0;
        let f = w.fitness_of(&o);
        assert!((f - (5.0 + w.cfg.food_energy as f64)).abs() < 1e-3);
    }

    #[test]
    fn approach_bonus_is_zero_when_never_approached() {
        let w = World::new(SimConfig { food_count: 0, seed: 1, ..SimConfig::default() });
        let mut o = herbivore(Vec2::new(0.0, 0.0), 0.0, Box::new(Idle));
        o.best_approach = f32::INFINITY;
        o.time_alive = 3.0;
        assert!((w.fitness_of(&o) - 3.0).abs() < 1e-9);
    }
```

- [ ] **Step 2: Implement vision wiring + approach tracking + fitness term**

Replace `update_organisms` in `evors-sandbox/src/world.rs` with a version that builds grids, computes real senses, and records nearest-target surface distance. Add a `nearest_target_dist` helper and update `fitness_of`.

```rust
    fn update_organisms(&mut self, dt: f32) {
        use crate::grid::SpatialGrid;
        let cfg = self.cfg.clone();
        let cell = cfg.vision_length.max(1.0);

        // Build grids over the pre-move frame.
        let mut food_grid = SpatialGrid::new(cfg.width, cfg.height, cell);
        for (fi, f) in self.food.iter().enumerate() {
            food_grid.insert(fi, f.pos);
        }
        let mut org_grid = SpatialGrid::new(cfg.width, cfg.height, cell);
        for (oi, o) in self.organisms.iter().enumerate() {
            org_grid.insert(oi, o.pos);
        }

        // Snapshot food + organisms so every organism senses the same frame.
        let food_snapshot = self.food.clone();
        let org_snapshot: Vec<(Vec2, f32, f32, Diet)> = self
            .organisms
            .iter()
            .map(|o| (o.pos, o.size, o.hue, o.diet))
            .collect();

        for i in 0..self.organisms.len() {
            let pos = self.organisms[i].pos;
            let food_cands = food_grid.candidates(pos, cfg.vision_length);
            let org_cands = org_grid.candidates(pos, cfg.vision_length);

            let senses = crate::vision::sense(
                &self.organisms[i],
                i,
                &cfg,
                &food_snapshot,
                &food_cands,
                &self.organisms,
                &org_cands,
            );
            let m = self.organisms[i].controller.sense_act(&senses);

            // Record closest approach to this organism's target type.
            let target_dist = Self::nearest_target_dist(
                pos,
                self.organisms[i].size,
                self.organisms[i].diet,
                &food_snapshot,
                &org_snapshot,
                i,
            );
            if target_dist < self.organisms[i].best_approach {
                self.organisms[i].best_approach = target_dist;
            }

            let thrust = (m.forward - m.back).clamp(-1.0, 1.0);
            let turn = (m.turn_right - m.turn_left).clamp(-1.0, 1.0);

            let o = &mut self.organisms[i];
            o.heading += turn * cfg.max_turn_rate * dt;
            let speed = thrust * cfg.max_speed;
            let dir = Vec2::new(o.heading.cos(), o.heading.sin());
            o.pos = o.pos.add(dir.scale(speed * dt));
            o.pos.x = o.pos.x.clamp(0.0, cfg.width);
            o.pos.y = o.pos.y.clamp(0.0, cfg.height);

            let move_cost = speed.abs() * cfg.move_cost * dt;
            o.energy -= cfg.passive_drain * dt + move_cost;
            o.age += dt;
            o.time_alive += dt;
        }
    }

    /// Surface distance from `pos` (radius `size`) to the nearest valid target:
    /// food for herbivores, other herbivores for carnivores. `+inf` if none.
    fn nearest_target_dist(
        pos: Vec2,
        size: f32,
        diet: Diet,
        food: &[Food],
        organisms: &[(Vec2, f32, f32, Diet)],
        self_idx: usize,
    ) -> f32 {
        let mut best = f32::INFINITY;
        match diet {
            Diet::Herbivore => {
                for f in food {
                    let d = (pos.sub(f.pos).len() - size - f.size).max(0.0);
                    best = best.min(d);
                }
            }
            Diet::Carnivore => {
                for (j, (p, sz, _hue, d)) in organisms.iter().enumerate() {
                    if j == self_idx || *d != Diet::Herbivore {
                        continue;
                    }
                    let dist = (pos.sub(*p).len() - size - sz).max(0.0);
                    best = best.min(dist);
                }
            }
        }
        best
    }
```

Update `fitness_of` to add the capped approach bonus:

```rust
    fn fitness_of(&self, o: &Organism) -> f64 {
        // Closest-approach bonus: max = one meal's energy, reached at surface
        // distance 0; decays linearly to 0 at vision_length; 0 if never approached.
        let bonus = if o.best_approach.is_finite() {
            let closeness = (1.0 - o.best_approach / self.cfg.vision_length).clamp(0.0, 1.0);
            (self.cfg.food_energy * closeness) as f64
        } else {
            0.0
        };
        o.time_alive as f64 + bonus
    }
```

- [ ] **Step 3: Run the tests**

Run: `rtk cargo test -p evors-sandbox world`
Expected: all world tests pass, including the three fitness/approach tests.

- [ ] **Step 4: Commit**

```bash
rtk git add evors-sandbox/src/world.rs
rtk git commit -m "evors-sandbox: wire vision into step + closest-approach fitness"
```

---

## Task 10: Integration tests — determinism + controller smoke test

**Files:**
- Create: `evors-sandbox/tests/sim.rs`

- [ ] **Step 1: Write the integration tests**

`evors-sandbox/tests/sim.rs`:

```rust
//! Integration tests for the headless sim core: determinism and a hand-coded
//! controller outperforming a do-nothing one.

use evors_sandbox::{
    Controller, Diet, Food, Motor, Organism, Senses, SimConfig, SpeciesConfig, World,
};
use evors_sandbox::geom::Vec2;

struct Idle;
impl Controller for Idle {
    fn sense_act(&mut self, _: &Senses) -> Motor {
        Motor { forward: 0.0, back: 0.0, turn_left: 0.0, turn_right: 0.0 }
    }
}

/// Forward-only, turn toward the strongest ray.
struct SeekBright;
impl Controller for SeekBright {
    fn sense_act(&mut self, s: &Senses) -> Motor {
        let n = s.rays.len();
        let (mut best_i, mut best) = (0usize, -1.0_f32);
        for (i, r) in s.rays.iter().enumerate() {
            if r.distance > best {
                best = r.distance;
                best_i = i;
            }
        }
        let center = (n as f32 - 1.0) / 2.0;
        let (mut tl, mut tr) = (0.0, 0.0);
        if best < 1e-6 {
            // Nothing seen: wander straight.
        } else if (best_i as f32) < center {
            tl = 1.0;
        } else if (best_i as f32) > center {
            tr = 1.0;
        }
        Motor { forward: 1.0, back: 0.0, turn_left: tl, turn_right: tr }
    }
}

fn base_cfg() -> SimConfig {
    SimConfig {
        food_count: 80,
        seed: 99,
        species: vec![SpeciesConfig {
            diet: Diet::Herbivore,
            count: 1,
            size: 5.0,
            hue: 0.6,
            max_energy: 100.0,
            start_energy: 40.0,
        }],
        ..SimConfig::default()
    }
}

/// Run an idle herbivore in a fixed world; return the ticks survived.
fn ticks_survived<C: Controller + 'static>(make: impl Fn() -> C, seed: u64) -> usize {
    let mut cfg = base_cfg();
    cfg.seed = seed;
    let mut w = World::new(cfg);
    let pos = Vec2::new(500.0, 500.0);
    w.add_organism(Organism::new(
        0, pos, 0.0, 5.0, 0.6, Diet::Herbivore, 100.0, 40.0, Box::new(make()),
    ));
    let mut ticks = 0;
    while !w.organisms.is_empty() && ticks < 100_000 {
        w.step(0.1);
        ticks += 1;
    }
    ticks
}

#[test]
fn deterministic_runs_match() {
    // Two worlds, same seed + same scripted organism -> identical food layout
    // and identical organism count/energy trajectory.
    let build = || {
        let mut w = World::new(base_cfg());
        w.add_organism(Organism::new(
            0, Vec2::new(250.0, 250.0), 0.0, 5.0, 0.6, Diet::Herbivore, 100.0, 80.0,
            Box::new(SeekBright),
        ));
        w
    };
    let mut a = build();
    let mut b = build();
    for _ in 0..200 {
        a.step(0.1);
        b.step(0.1);
        assert_eq!(a.food.len(), b.food.len());
        assert_eq!(a.organisms.len(), b.organisms.len());
        if let (Some(oa), Some(ob)) = (a.organisms.first(), b.organisms.first()) {
            assert_eq!(oa.pos, ob.pos);
            assert_eq!(oa.energy, ob.energy);
        }
    }
}

#[test]
fn seeker_outlives_idler() {
    // Averaged over a few seeds, a food-seeking herbivore survives longer than
    // one that never moves, in a food-rich world.
    let seeds = [1u64, 2, 3, 4, 5];
    let idle: usize = seeds.iter().map(|&s| ticks_survived(|| Idle, s)).sum();
    let seek: usize = seeds.iter().map(|&s| ticks_survived(|| SeekBright, s)).sum();
    assert!(seek > idle, "seeker {seek} should outlive idler {idle}");
}
```

- [ ] **Step 2: Expose `geom` for the test**

The integration test uses `evors_sandbox::geom::Vec2`. Confirm `geom` is `pub mod geom;` in `lib.rs` (it is, from Task 1). No change needed if already public.

- [ ] **Step 3: Run the integration tests**

Run: `rtk cargo test -p evors-sandbox --test sim`
Expected: `deterministic_runs_match` and `seeker_outlives_idler` pass.

- [ ] **Step 4: Run the whole crate test suite**

Run: `rtk cargo test -p evors-sandbox`
Expected: all unit + integration tests pass.

- [ ] **Step 5: Commit**

```bash
rtk git add evors-sandbox/tests/sim.rs
rtk git commit -m "evors-sandbox: integration tests — determinism + seeker smoke test"
```

---

## Task 11: Workspace check + README mention

**Files:**
- Modify: `README.md` (Workspace section)

- [ ] **Step 1: Run the full workspace test suite**

Run: `rtk cargo test --workspace`
Expected: every crate (evors, evors-neat, evors-cppn, evors-sandbox) passes.

- [ ] **Step 2: Add evors-sandbox to the README workspace list**

In `README.md`, under the "## Workspace" bullet list, after the `evors-neat` bullet, add:

```markdown
- **[`evors-cppn`](evors-cppn/)** — CPPN-NEAT image evolution: a NEAT genome
  rendered per-pixel on the GPU and scored against a target image.
- **[`evors-sandbox`](evors-sandbox/)** — a headless artificial-life simulation:
  organisms with pluggable brains sense food and each other via cone-vision ray
  tracing, spend energy to move and eat, age, and die. `World::step` advances the
  sim and returns deaths for an evolution loop to report on (sub-project A; the
  per-species evolution loop and egui viewer follow).
```

(If an `evors-cppn` bullet already exists, only add the `evors-sandbox` one.)

- [ ] **Step 3: Commit**

```bash
rtk git add README.md
rtk git commit -m "Document evors-sandbox in workspace README"
```

---

## Self-Review Notes (spec coverage)

- **Workspace member, deps** → Task 0.
- **Deterministic fixed-timestep `step(dt) -> Vec<Death>`** → Tasks 6-9; determinism test Task 10.
- **Controller trait** → Task 3; exercised by hand-coded controllers in Tasks 5/7/9/10.
- **Cone vision (R rays, length, arc), normalized distance + hue, sees food and organisms** → Task 5.
- **`2R+1` inputs / 4-output motor (opposing pairs)** → motor consumed in Task 7; senses shape in Task 5.
- **Energy: passive drain + movement cost; energy≤0 death** → Task 7.
- **Organism aging decay on eat gain** → Task 8.
- **Eating: herbivore/food, carnivore/herbivore; predation deaths routed to prey's species** → Task 8.
- **Uniform spatial grid; brute-force parity** → Task 4; used in Tasks 8-9.
- **Food target density top-up** → Task 6 (`replenish_food`), called each `step`.
- **Fitness = time_alive + capped closest-approach bonus (max = one meal)** → Task 9.
- **Clamp-at-walls bounds** → Task 7.
- **Deferred inputs (random/sin/memory) + recurrence** → intentionally out of scope (spec "Deferred inputs"); no task.
- **Physical genes as evolvable genes** → fixed per species in config for A (spec defers encoding to C); no task.

Deferred-by-design items are flagged so a reader does not mistake them for gaps.
