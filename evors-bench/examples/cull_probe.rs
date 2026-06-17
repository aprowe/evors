//! Measure how much each cull removes and how close the pool stays to target.
//! `cargo run --release -p evors-bench --example cull_probe`

use evors::strategies::{
    ResourceSoftStochastic, ResourceSoftStochasticInfo, SoftStochastic, SoftStochasticInfo,
};
use evors::strategy::EvolutionStrategy;
use evors_bench::VecGenetics;
use evors_bench::two_circles::{bounds, objective};

const N: usize = 40_000;
const TARGET: usize = 300;

fn fitness(x: &Vec<f64>) -> f64 {
    -objective(x)
}

fn probe(label: &str, mut size_after_report: impl FnMut() -> usize) {
    let mut prev = 0usize;
    let mut culls: Vec<usize> = Vec::new();
    let mut sizes: Vec<usize> = Vec::new();
    for _ in 0..N {
        let sz = size_after_report();
        if sz <= prev {
            culls.push(prev + 1 - sz);
        }
        sizes.push(sz);
        prev = sz;
    }
    let back = &sizes[N / 2..];
    let bmin = *back.iter().min().unwrap();
    let bmax = *back.iter().max().unwrap();
    let bmean = back.iter().sum::<usize>() as f64 / back.len() as f64;
    let n_culls = culls.len();
    let removed_mean = if n_culls > 0 {
        culls.iter().sum::<usize>() as f64 / n_culls as f64
    } else {
        0.0
    };
    let removed_max = culls.iter().copied().max().unwrap_or(0);
    println!("\n=== {label} (target={TARGET}) ===");
    println!(
        "  cull events: {n_culls} ({:.0}% of reports)   removed/cull: mean {removed_mean:.1} ({:.0}% of target)  max {removed_max}",
        100.0 * n_culls as f64 / N as f64,
        100.0 * removed_mean / TARGET as f64
    );
    println!("  pool size (back half): min {bmin}  mean {bmean:.0}  max {bmax}");
}

fn main() {
    let mut ssea = SoftStochastic::new(
        VecGenetics::new(bounds(), 0.3),
        SoftStochasticInfo {
            target: TARGET,
            novelty_weight: 0.6,
            temperature: 0.2,
            behavior_weight: 0.0,
            knn: 15,
            seed: 1,
        },
    );
    probe("SSEA (old cull)", || {
        let g = ssea.next();
        let s = fitness(&g);
        ssea.report(g, s);
        ssea.pool_size()
    });

    let mut res = ResourceSoftStochastic::new(
        VecGenetics::new(bounds(), 0.3),
        ResourceSoftStochasticInfo {
            target: TARGET,
            temperature: 0.2,
            relative_width: 0.03,
            mate_locality: 2.0,
            seed: 1,
            ..Default::default()
        },
    );
    probe("Resource (soft-equilibrium cull)", || {
        let g = res.next();
        let s = fitness(&g);
        res.report(g, s);
        res.pool_size()
    });
}
