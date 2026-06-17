//! Two-circles demo: evolve plain SSEA, Speciated SSEA, and Resource (fitness
//! sharing) on a landscape that rewards being on either of two unit rings
//! (centred at x = ±5). Dumps the final pools for the scatter chart, and an
//! occupancy-over-time CSV (SSEA vs Resource) for the equilibrium chart. Run:
//! `cargo run --release -p evors-bench --example two_circles`

use evors::population::PopulationView;
use evors::strategies::{
    ResourceSoftStochastic, ResourceSoftStochasticInfo, SoftStochastic, SoftStochasticInfo,
    SpeciatedSoftStochastic, SpeciatedSoftStochasticInfo,
};
use evors::strategy::EvolutionStrategy;
use evors::{RunConfig, Stop, run};
use evors_bench::VecGenetics;
use evors_bench::two_circles::{between_fraction, bounds, objective, occupancy};
use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};

const CHUNK: usize = 1_000;
const CHUNKS: usize = 40; // 40k evaluations total

fn fitness(x: &Vec<f64>) -> f64 {
    -objective(x)
}

/// Drive `strat` for CHUNKS×CHUNK evals, recording per-chunk niche occupancy.
/// Returns (final_pool, occupancy_rows) where each row is (trials, left, right, gap).
fn run_chunked<St>(strat: &mut St) -> (Vec<Vec<f64>>, Vec<(usize, usize, usize, usize)>)
where
    St: EvolutionStrategy<Score = f64> + PopulationView<Genome = Vec<f64>>,
    St::Genetic: evors::genetic::Genetic<Genome = Vec<f64>>,
{
    let mut rows = Vec::with_capacity(CHUNKS);
    for c in 0..CHUNKS {
        let cfg = RunConfig {
            stop: Stop::Trials(CHUNK),
            batch_size: 1,
            max_trials: None,
        };
        let _ = run(&cfg, strat, fitness);
        let (l, r, g) = occupancy(&strat.members());
        rows.push(((c + 1) * CHUNK, l, r, g));
    }
    (strat.members(), rows)
}

fn dump_pool(path: &str, pool: &[Vec<f64>]) {
    let f = File::create(path).unwrap();
    let mut w = BufWriter::new(f);
    writeln!(w, "x,y,fit").unwrap();
    for p in pool {
        writeln!(w, "{},{},{}", p[0], p[1], -objective(p)).unwrap();
    }
}

fn main() {
    create_dir_all("data/bench").unwrap();

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
    let (ssea_pool, ssea_occ) = run_chunked(&mut ssea);

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
    let (spec_pool, _spec_occ) = run_chunked(&mut spec);

    let mut res = ResourceSoftStochastic::new(
        VecGenetics::new(bounds(), 0.3),
        ResourceSoftStochasticInfo {
            target: 300,
            temperature: 0.2,
            relative_width: 0.03,
            mate_locality: 2.0,
            seed: 1,
        },
    );
    let (res_pool, res_occ) = run_chunked(&mut res);

    dump_pool("data/bench/two_circles_ssea.csv", &ssea_pool);
    dump_pool("data/bench/two_circles_speciated.csv", &spec_pool);
    dump_pool("data/bench/two_circles_resource.csv", &res_pool);

    // Occupancy over time: SSEA vs Resource (the equilibrium comparison).
    let f = File::create("data/bench/two_circles_occupancy.csv").unwrap();
    let mut w = BufWriter::new(f);
    writeln!(
        w,
        "trials,ssea_left,ssea_right,ssea_gap,resource_left,resource_right,resource_gap"
    )
    .unwrap();
    for (a, b) in ssea_occ.iter().zip(res_occ.iter()) {
        writeln!(w, "{},{},{},{},{},{},{}", a.0, a.1, a.2, a.3, b.1, b.2, b.3).unwrap();
    }

    println!(
        "between-fraction  ssea={:.3}  speciated={:.3}  resource={:.3}",
        between_fraction(&ssea_pool),
        between_fraction(&spec_pool),
        between_fraction(&res_pool),
    );
    println!("wrote data/bench/two_circles_{{ssea,speciated,resource}}.csv + _occupancy.csv");
}
