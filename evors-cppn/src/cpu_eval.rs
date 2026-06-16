//! CPU reference evaluator for an [`Encoded`] CPPN. Uses the **same** slot
//! layout and `f32` activation math as the interpreter shader, so it can
//! validate the GPU path pixel-for-pixel. The production path is always the GPU.

use crate::encode::Encoded;

/// Activation by stable id (must match `evors_neat::Activation` and the shader).
pub fn activation(id: u32, x: f32) -> f32 {
    match id {
        0 => x,                              // Identity
        1 => 1.0 / (1.0 + (-4.9 * x).exp()), // Sigmoid
        2 => x.tanh(),                       // Tanh
        3 => x.sin(),                        // Sin
        4 => (-x * x).exp(),                 // Gauss
        5 => x.abs(),                        // Abs
        _ => x,
    }
}

/// Evaluate the CPPN at `(x, y)` (both in `[-1, 1]`). Slots 0..4 are the fixed
/// inputs `x, y, d, bias`. Returns `(r, g, b)` from the output slots.
pub fn eval_encoded(enc: &Encoded, x: f32, y: f32) -> [f32; 3] {
    let n = enc.node_count as usize;
    let mut value = vec![0.0f32; n];
    let d = (x * x + y * y).sqrt();
    if n > 0 {
        value[0] = x;
    }
    if n > 1 {
        value[1] = y;
    }
    if n > 2 {
        value[2] = d;
    }
    if n > 3 {
        value[3] = 1.0;
    }
    for slot in 4..n {
        let node = enc.nodes[slot];
        let mut acc = 0.0f32;
        for k in 0..node.conn_count {
            let c = enc.conns[(node.conn_start + k) as usize];
            acc += value[c.source as usize] * c.weight;
        }
        value[slot] = activation(node.activation, acc);
    }
    [
        value[enc.output_slots[0] as usize],
        value[enc.output_slots[1] as usize],
        value[enc.output_slots[2] as usize],
    ]
}

/// Map pixel `(px, py)` in a `res × res` image to CPPN coordinates in `[-1, 1]`.
/// Row 0 is the bottom (matches `glReadPixels` origin).
pub fn pixel_to_xy(px: u32, py: u32, res: u32) -> (f32, f32) {
    let x = 2.0 * (px as f32 + 0.5) / res as f32 - 1.0;
    let y = 2.0 * (py as f32 + 0.5) / res as f32 - 1.0;
    (x, y)
}

/// Render the CPPN to a flat `res*res*3` RGB buffer on the CPU (reference / PNG).
pub fn render_cpu(enc: &Encoded, res: u32) -> Vec<f32> {
    let mut out = vec![0.0f32; (res * res * 3) as usize];
    for py in 0..res {
        for px in 0..res {
            let (x, y) = pixel_to_xy(px, py, res);
            let rgb = eval_encoded(enc, x, y);
            let i = ((py * res + px) * 3) as usize;
            out[i] = rgb[0].clamp(0.0, 1.0);
            out[i + 1] = rgb[1].clamp(0.0, 1.0);
            out[i + 2] = rgb[2].clamp(0.0, 1.0);
        }
    }
    out
}
