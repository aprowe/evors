//! evors-cppn — evolve images with CPPN-NEAT, evaluated on the GPU.
//!
//! Sub-project A (this milestone) is the headless evaluation core: encode a
//! CPPN genome ([`encode`]) into buffers a generic interpreter fragment shader
//! walks per pixel, render it on the GPU, and return `(fitness, behavior
//! signature)` for a target image ([`gpu::GpuEvaluator`]). The CPU reference
//! evaluator ([`cpu_eval`]) validates the GPU path.

pub mod context;
pub mod cppn;
pub mod cpu_eval;
pub mod encode;
pub mod evolve;
pub mod gpu;
pub mod score;

pub use cppn::{CppnGenetics, CppnParams};
pub use encode::{EncodeError, Encoded, MAX_NODES, encode};
pub use evolve::{ImageTrial, demo_target};
pub use gpu::{Eval, EvalConfig, GpuEvaluator};
pub use score::ImgScore;
