# evors-cppn

Evolve images with **CPPN-NEAT**, evaluated on the GPU. A CPPN genome (a NEAT
network whose nodes carry activation functions) is rendered per-pixel by a
fragment shader and scored against a target image.

Built in three sub-projects, all done:
- **A** — headless GPU evaluation core (encode → interpreter shader → fitness).
- **B** — `ImageTrial`: evolve toward a target with SSEA using genetic **and**
  solution-distance novelty.
- **C** — the egui app (`cppn-gui`): live convergence controls; the champion is
  rendered **directly to the window with GL** and EMA-accumulated so it converges
  smoothly.

GL plumbing is adapted from `soap-in-a-tub` (OpenGL 4.3 core).

```bash
cargo run --release -p evors-cppn --bin cppn-gui   # the interactive app
```

## Sub-project A — CPPN → GPU → fitness

- **`encode`** — topologically slots a `NeatGenome` and flattens it into two
  buffers (`nodes`, `conns`). Inputs occupy fixed slots `x, y, d, bias`.
- **interpreter shader** (`shaders/cppn.frag`) — one fixed fragment shader walks
  the uploaded SSBOs per pixel; evolving a genome = re-upload two buffers, zero
  recompiles. Activations: Identity, Sigmoid, Tanh, Sin, Gauss, Abs (a stable id
  mapping shared with the CPU evaluator).
- **`GpuEvaluator`** — owns a GL 4.3 context on a private thread;
  `evaluate(genome) -> Eval { fitness, signature }` renders to a small offscreen
  FBO, reads back once, and the CPU computes both the image-distance fitness and
  a downsampled **behavior signature** (the solution-distance descriptor B will
  feed into novelty).
- **`CppnGenetics`** — `impl evors::Genetic` (4 inputs, 3 outputs, varied
  activations), reusing `evors-neat`'s genome / registry / crossover / gene_dist.
- **`cpu_eval`** — a CPU reference evaluator used to validate the GPU path.

### Validation

`cargo test -p evors-cppn` runs CPU unit tests; `--test gpu` (needs a real GL 4.3
context) asserts the GPU render matches the CPU reference within `1e-3`, that
Sin/Gauss activations map identically, and that fitness behaves (self-match
≈ 1.0, different genome lower).

```bash
cargo test -p evors-cppn --workspace
cargo run --release -p evors-cppn --example dump_ppm  # writes data/cppn_render.ppm
```

## Scope (A)

Headless; feed-forward CPPNs; single-genome `evaluate` (batching later); fitness
computed on the CPU from one readback (GPU reduction deferred).
