//! Headless CPPN image evolution toward a demo target, scored by fitness +
//! (genetic & solution) novelty. Dumps the champion render to PPM.
//! Run: `cargo run --release -p evors-cppn --example image_evolve`

use evors::strategies::SoftStochasticInfo;
use evors_cppn::{CppnParams, EvalConfig, ImageTrial, demo_target};
use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};

fn dump_ppm(path: &str, rgb: &[f32], res: u32) {
    create_dir_all("data").unwrap();
    let f = File::create(path).unwrap();
    let mut w = BufWriter::new(f);
    write!(w, "P6\n{res} {res}\n255\n").unwrap();
    for py in (0..res).rev() {
        for px in 0..res {
            let i = ((py * res + px) * 3) as usize;
            let _ = w.write_all(&[
                (rgb[i] * 255.0) as u8,
                (rgb[i + 1] * 255.0) as u8,
                (rgb[i + 2] * 255.0) as u8,
            ]);
        }
    }
}

fn main() {
    let res = 64u32;
    let target = demo_target(res);
    let mut trial = ImageTrial::new(
        &target,
        res,
        res,
        EvalConfig {
            eval_res: res,
            sig_res: 8,
        },
        CppnParams::default(),
        SoftStochasticInfo {
            target: 200,
            temperature: 0.1,
            novelty_weight: 0.3,
            behavior_weight: 0.3,
            knn: 5,
            seed: 7,
        },
    );

    dump_ppm("data/img_target.ppm", &target, res);
    let total = 6000;
    for chunk in 0..12 {
        trial.run(total / 12);
        println!(
            "trials {:5}  best_fitness {:.4}  pool {}",
            trial.trials(),
            trial.best_fitness(),
            trial.pool_size()
        );
        if chunk == 5 {
            dump_ppm("data/img_mid.ppm", &trial.best_render(), res);
        }
    }
    dump_ppm("data/img_best.ppm", &trial.best_render(), res);
    println!("wrote data/img_target.ppm, data/img_mid.ppm, data/img_best.ppm");
}
