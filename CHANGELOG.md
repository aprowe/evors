# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-06

First public release.

### `evors`
- Core traits: `Genetic` (genome instruction set on a descriptor struct),
  `EvolutionStrategy` (ask/tell), and the opt-in score traits `Fitness` and
  `Behavior`.
- Runner: `run` / `run_observed` with `RunConfig` and `Stop` (generations,
  evaluations, or an `Until` predicate, always capped by `max_trials`).
- Progress observers: `Observer`, `NoObserver`, `RecordingObserver`,
  `CsvProgressObserver`.
- Strategies, each built from genetics plus a `Default`-able CreateInfo struct:
  - `ClassicPopulation`: elitist, tournament-selection steady-state GA.
  - `SoftStochastic` (SSEA): soft, probabilistic culling that blends fitness
    and genetic novelty.
  - `SpeciatedSoftStochastic`: SSEA plus assortative mating, behavioral
    novelty, a novelty archive, and speciation pressure.
  - `ResourceSoftStochastic`: SSEA soft cull on shared fitness (niche carrying
    capacity), with `balanced`, `even_spread` and `coverage` presets.
- `parallel` feature: batch fitness evaluation via `rayon`.

### Workspace crates
- `evors-neat`: NEAT genome, innovation registry and structural operators as an
  `evors::Genetic`; solves XOR on SSEA.
- `evors-cppn`: CPPN-NEAT image evolution, with GPU evaluation (OpenGL 4.3)
  and the `cppn-gui` egui app.
- `evors-bench` (unpublished): benchmark harness on classic 2-D test functions,
  with a differential-evolution baseline.

[0.1.0]: https://github.com/aprowe/evors/releases/tag/v0.1.0
