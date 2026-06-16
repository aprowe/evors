//! Two-circles demo: evolve plain SSEA vs SpeciatedSSEA on a landscape that
//! rewards being on either of two unit rings (centred at x = ±5). Dumps the
//! final pools to CSV for the scatter chart. Run:
//! `cargo run --release -p evors-bench --example two_circles`

use evors::population::PopulationView;
use evors::strategies::{
    SoftStochastic, SoftStochasticInfo, SpeciatedSoftStochastic, SpeciatedSoftStochasticInfo,
};
use evors::{RunConfig, Stop, run};
use evors_bench::VecGenetics;
use evors_bench::two_circles::{between_fraction, bounds, objective};
use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};

const BUDGET: usize = 40_000;

fn fitness(x: &Vec<f64>) -> f64 {
    -objective(x)
}

fn dump(path: &str, pool: &[Vec<f64>]) {
    let f = File::create(path).unwrap();
    let mut w = BufWriter::new(f);
    writeln!(w, "x,y,fit").unwrap();
    for p in pool {
        writeln!(w, "{},{},{}", p[0], p[1], -objective(p)).unwrap();
    }
}

fn main() {
    create_dir_all("data/bench").unwrap();
    let cfg = || RunConfig {
        stop: Stop::Trials(BUDGET),
        batch_size: 1,
        max_trials: None,
    };

    let mut ssea = SoftStochastic::new(
        VecGenetics::new(bounds(), 0.3),
        SoftStochasticInfo {
            target: 300,
            novelty_weight: 0.6,
            temperature: 0.2,
            behavior_weight: 0.0,
            knn: 5,
            seed: 1,
        },
    );
    let _ = run(&cfg(), &mut ssea, fitness);
    let ssea_pool = ssea.members();

    let mut spec = SpeciatedSoftStochastic::new(
        VecGenetics::new(bounds(), 0.3),
        SpeciatedSoftStochasticInfo {
            target: 300,
            novelty_weight: 0.6,
            temperature: 0.2,
            behavior_weight: 0.0,
            knn: 5,
            seed: 1,
            mate_locality: 5.0,
            archive_prob: 0.05,
            speciation_pressure: 0.6,
        },
    );
    let _ = run(&cfg(), &mut spec, fitness);
    let spec_pool = spec.members();

    dump("data/bench/two_circles_ssea.csv", &ssea_pool);
    dump("data/bench/two_circles_speciated.csv", &spec_pool);
    println!(
        "between-fraction  ssea={:.3}  speciated={:.3}  (pool {} / {})",
        between_fraction(&ssea_pool),
        between_fraction(&spec_pool),
        ssea_pool.len(),
        spec_pool.len(),
    );
    println!("wrote data/bench/two_circles_{{ssea,speciated}}.csv");
}
