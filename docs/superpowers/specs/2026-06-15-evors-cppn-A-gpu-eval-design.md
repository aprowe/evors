# evors-cppn (sub-project A) — CPPN→GPU→fitness, headless

**Date:** 2026-06-15
**Status:** Approved (design), implementing
**Parent project:** CPPN image evolution (egui + glow). This spec covers
**sub-project A** only: the headless GPU evaluation core. Sub-projects B
(evolution loop + novelty/fitness metric) and C (egui app) follow in their own
cycles.

## Summary

`evors-cppn` evolves images with CPPN-NEAT: a NEAT genome whose nodes carry
activation functions is rendered per-pixel by a fragment shader to produce an
image, scored against a target image. Sub-project A builds the headless core: the
genome→GPU encoding, a generic interpreter shader, and a GL eval engine that
returns `(fitness, behavior signature)` for a genome. No evolution loop, no GUI.

The boilerplate at `~/projects/soap-in-a-tub` supplies the reusable patterns:
glutin/winit GL 4.3 context creation (`window.rs::create_display`), the
dedicated-GL-thread request/response harness (`tests/support/mod.rs`), and the
fullscreen-triangle shader-canvas approach (`shader_canvas.rs`).

## Workspace

Add `evors-cppn` as a workspace member. It depends on `evors-neat` (genome +
registry), `glow`, `glutin`, `glutin-winit`, `winit` (versions matched to
soap-in-a-tub: glow 0.17, glutin 0.32, glutin-winit 0.5, winit via egui-winit
0.34 — pin the same).

## evors-neat change: activation genes

CPPN-NEAT is canonical NEAT with per-node activation functions. Generalize
`evors-neat`:

```rust
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Activation { Identity, Sigmoid, Tanh, Sin, Gauss, Abs }

pub struct NodeGene { pub id: u64, pub kind: NodeKind, pub activation: Activation }
```

- `Activation` has a **stable integer mapping** (`as u32`) shared by the CPU
  evaluator and the GPU shader.
- Existing genome construction defaults nodes to `Sigmoid`, so evors-neat's XOR
  test is unchanged; `network::activate` applies each node's `activation` instead
  of a hardcoded sigmoid.
- `evors-cppn` reuses `NeatGenome`, the innovation registry, crossover, and
  `gene_dist` wholesale; it only adds its own `CppnGenetics` and GPU machinery.

## CppnGenetics

`impl evors::Genetic<Genome = NeatGenome>` with 4 inputs (`x, y, d, bias`) and 3
outputs (`r, g, b`). Construction: `CppnGenetics::new(CppnParams { .. })`
(CreateInfo style), where `CppnParams` extends NEAT params with an
`activation_mutate_prob` and the allowed activation set. Operators mirror
evors-neat's, plus:

- new hidden nodes get a **random activation** from the allowed set;
- `mutate` may, with `activation_mutate_prob`, re-roll a random hidden/output
  node's activation;
- inputs are `Identity`; outputs default to `Sigmoid` (bounded `[0,1]` color).

(Sub-project A only needs `random`/`mutate` to produce valid genomes for testing;
the full evolution use is sub-project B.)

## GPU encoding

CPU prepares a genome for the GPU:

1. **Topological sort** nodes into slots `0..N`. Inputs occupy fixed slots: `x=0,
   y=1, d=2, bias=3`. Record the three output slots.
2. Build two buffers:
   - **nodes** — per slot: `{ activation_id: u32, conn_start: u32, conn_count: u32 }`
   - **conns** — `{ source_slot: u32, weight: f32 }`, grouped by target node so a
     node's incoming edges are contiguous; feed-forward ordering guarantees every
     `source_slot < target slot`.
3. Reject genomes with `N > MAX_NODES` (compile-time cap, 256) before upload.

Buffers are uploaded as SSBOs (GL 4.3). Uniforms: `node_count`, the three output
slots, and `resolution`.

## Interpreter shader

A fixed fragment shader over a vertex-pulling fullscreen triangle:

- Per pixel compute inputs: `x, y ∈ [-1, 1]` from `gl_FragCoord/resolution`,
  `d = length(vec2(x, y))`, `bias = 1`.
- `float value[MAX_NODES];` set slots 0–3 to the inputs.
- For slot `s` in `4..node_count`: `acc = Σ_{c in node[s] conns} value[c.source] *
  c.weight`; `value[s] = activation(node[s].activation_id, acc)`.
- Output `vec4(value[out_r], value[out_g], value[out_b], 1.0)`.
- `activation(id, x)` is a `switch` over the stable `Activation` mapping:
  Identity `x`, Sigmoid `1/(1+e^-x)`, Tanh, Sin `sin(x)`, Gauss `e^(-x²)`, Abs.

## Eval engine + threading

A `GpuEvaluator` owns one GL 4.3 context on a dedicated thread (soap-tub's
pattern: a `CtxGrabber` + `pump_app_events`, requests over an `mpsc` channel,
replies over `SyncSender`). The target image is uploaded once at construction,
downscaled to the eval resolution.

```rust
pub struct GpuEvaluator { /* channel to GL thread */ }
pub struct EvalConfig { pub eval_res: u32 /* e.g. 64 */, pub sig_res: u32 /* e.g. 8 */ }

pub struct Eval { pub fitness: f64, pub signature: Vec<f32> /* sig_res² × 3 */ }

impl GpuEvaluator {
    pub fn new(target_rgb: &[f32], target_w: u32, target_h: u32, cfg: EvalConfig) -> Self;
    pub fn evaluate(&self, genome: &NeatGenome) -> Eval; // blocks on the GL thread
}
```

Per `evaluate`: encode genome → upload SSBOs → render interpreter to an
`eval_res × eval_res` offscreen FBO → read back RGB → CPU computes:

- `fitness = (1 - mse(render, target) / mse_max).clamp(0, 1)` where `mse_max` is
  the worst-case per-channel error (so identical → 1.0).
- `signature` = render box-downsampled to `sig_res × sig_res × 3` (the
  behavior/solution descriptor sub-project B feeds into novelty).

## CPU reference evaluator

A `cpu_eval(genome, x, y) -> [f32; 3]` that evaluates the CPPN on the CPU using
the same activation mapping. Used only to validate the GPU path (the production
path is always GPU). Lives behind `#[cfg(test)]` or a debug module.

## Validation (headless GPU milestone)

GPU tests gated like soap-tub's `tests/gpu.rs`: a real GL 4.3 context on a
serialized dedicated thread; skip/fail explicitly if unavailable.

1. **Encoding correctness** — a hand-built CPPN (`r=x, g=y, b=d`) rendered on the
   GPU matches `cpu_eval` at every pixel within `1e-3`.
2. **Activation coverage** — a CPPN using `Sin` and `Gauss` matches `cpu_eval`
   within `1e-3`, proving each activation id maps identically GPU vs CPU.
3. **Fitness behavior** — target-matching genome → fitness ≈ 1.0; an unrelated
   genome scores strictly lower; a genome whose render equals the target → 1.0.
4. **PNG dump helper** — `dump_png(genome, path)` to eyeball a render (not an
   assertion; behind a debug helper).

Non-GPU unit tests (no context needed): topological slot assignment is valid
(every connection's source slot < target slot); `MAX_NODES` rejection; signature
length = `sig_res² × 3`.

## Error handling

- Shader compile/link failure panics with the GL info log (soap-tub convention).
- `N > MAX_NODES` returns/produces a clear error before any GL call.
- Missing GL 4.3 context fails the GPU tests explicitly (no hang).

## File structure

```
evors-cppn/
  Cargo.toml
  src/
    lib.rs
    activation.rs   # (re-exports evors_neat::Activation + shared id mapping helpers)
    cppn.rs         # CppnGenetics + CppnParams
    encode.rs       # genome -> (nodes, conns) SSBO data + topological slotting
    gpu.rs          # GpuEvaluator: GL thread, interpreter program, render+readback
    cpu_eval.rs     # CPU reference CPPN evaluator (validation)
    shaders/
      cppn.vert     # fullscreen triangle
      cppn.frag     # interpreter
  tests/
    gpu.rs          # gated GPU validation
    support/mod.rs  # dedicated GL-thread harness (adapted from soap-in-a-tub)
```

## Scope (sub-project A)

- Headless only — no evolution loop (B), no egui (C).
- Feed-forward CPPNs (inherited from evors-neat's acyclic invariant).
- Single-genome `evaluate`; batching is a later optimization.
- GPU fitness reduction deferred — CPU-on-readback per the design decision.
