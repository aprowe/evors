//! Render a random CPPN genome to a PPM image (no image-crate dependency) so the
//! GPU pipeline can be eyeballed. Run:
//! `cargo run --release -p evors-cppn --example dump_ppm`

use evors::Genetic;
use evors_cppn::{CppnGenetics, CppnParams, EvalConfig, GpuEvaluator};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};

fn main() {
    let res = 256u32;
    let cfg = EvalConfig {
        eval_res: res,
        sig_res: 8,
    };
    // Target is unused for a bare render; pass mid-gray.
    let target = vec![0.5f32; (res * res * 3) as usize];
    let eval = GpuEvaluator::new(&target, res, res, cfg);

    let g = CppnGenetics::new(CppnParams::default());
    let mut rng = ChaCha8Rng::seed_from_u64(20);
    // Grow the CPPN a bit so it produces structure.
    let mut genome = g.random(&mut rng);
    for _ in 0..60 {
        genome = g.mutate(&genome, &mut rng);
    }

    let rgb = eval.render_rgb(&genome); // row 0 = bottom
    create_dir_all("data").unwrap();
    let path = "data/cppn_render.ppm";
    let f = File::create(path).unwrap();
    let mut w = BufWriter::new(f);
    write!(w, "P6\n{res} {res}\n255\n").unwrap();
    // PPM is top-to-bottom; our render is bottom-up, so flip rows.
    for py in (0..res).rev() {
        for px in 0..res {
            let i = ((py * res + px) * 3) as usize;
            let bytes = [
                (rgb[i] * 255.0) as u8,
                (rgb[i + 1] * 255.0) as u8,
                (rgb[i + 2] * 255.0) as u8,
            ];
            w.write_all(&bytes).unwrap();
        }
    }
    println!(
        "wrote {path} ({res}x{res}), hidden nodes = {}",
        genome.hidden_count()
    );
}
