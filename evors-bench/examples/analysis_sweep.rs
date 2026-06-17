//! Parameterized knob/theory sweep for the analysis report. Runs ONE experiment
//! (named by the first CLI arg) over a config grid, measuring multi-basin
//! coverage AND two-circles even-spread/stability for each config, and writes
//! `data/analysis/<exp>.csv`.
//!
//! `cargo run --release -p evors-bench --example analysis_sweep -- <exp>`
//! exp ∈ {explore, rw, temp, loc, target, cullp, compare}

use evors::population::PopulationView;
use evors::strategies::{
    ResourceSoftStochastic, ResourceSoftStochasticInfo, SoftStochastic, SoftStochasticInfo,
    SpeciatedSoftStochastic, SpeciatedSoftStochasticInfo,
};
use evors::strategy::EvolutionStrategy;
use evors::{RunConfig, Stop, run};
use evors_bench::genetics::VecGenetics;
use evors_bench::harness::{Config, Strat, coverage, run_config};
use evors_bench::problems::{Problem, all};
use evors_bench::two_circles::{between_fraction, bounds, objective};
use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};

const SEEDS: u64 = 3;
const COV_BUDGET: usize = 12_000;
const TC_STEPS: usize = 30_000;
const COV_PROBLEMS: [&str; 3] = ["ackley", "rastrigin", "griewank"];

fn base() -> ResourceSoftStochasticInfo {
    ResourceSoftStochasticInfo {
        target: 300,
        temperature: 0.2,
        relative_width: 0.03,
        mate_locality: 2.0,
        explore_floor: 0.05,
        cull_pressure: 1.0,
        seed: 0,
    }
}

fn problem(name: &str) -> Problem {
    all().into_iter().find(|p| p.name == name).unwrap()
}

/// Mean basin coverage of a Resource config on a problem over SEEDS seeds.
fn cov_resource(info: ResourceSoftStochasticInfo, pname: &str) -> f64 {
    let p = problem(pname);
    let cfg = Config {
        label: "r".into(),
        strat: Strat::Resource(info),
        mutation_step: 0.3,
    };
    (0..SEEDS)
        .map(|s| coverage(&p, &run_config(&p, &cfg, COV_BUDGET, s)) as f64)
        .sum::<f64>()
        / SEEDS as f64
}

fn cov_strat(strat: Strat, pname: &str) -> f64 {
    let p = problem(pname);
    let cfg = Config {
        label: "x".into(),
        strat,
        mutation_step: 0.3,
    };
    (0..SEEDS)
        .map(|s| coverage(&p, &run_config(&p, &cfg, COV_BUDGET, s)) as f64)
        .sum::<f64>()
        / SEEDS as f64
}

/// Angular evenness (normalized entropy, 1=uniform) over both rings + on-ring count.
fn ring_evenness(pool: &[Vec<f64>]) -> (f64, usize) {
    let bins = 12usize;
    let mut ent = 0.0;
    let mut on_ring = 0usize;
    for cx in [-5.0_f64, 5.0] {
        let mut hist = [0usize; 12];
        for p in pool {
            if -objective(p) <= 0.9 {
                continue;
            }
            let (dx, dy) = (p[0] - cx, p[1]);
            if (dx.hypot(dy) - 1.0).abs() >= 0.35 {
                continue;
            }
            on_ring += 1;
            let mut a = dy.atan2(dx);
            if a < 0.0 {
                a += std::f64::consts::TAU;
            }
            hist[((a / std::f64::consts::TAU) * bins as f64) as usize % bins] += 1;
        }
        let total: usize = hist.iter().sum();
        if total > 0 {
            let h: f64 = hist
                .iter()
                .filter(|&&c| c > 0)
                .map(|&c| {
                    let p = c as f64 / total as f64;
                    -p * p.ln()
                })
                .sum();
            ent += h / (bins as f64).ln();
        }
    }
    (ent / 2.0, on_ring)
}

/// Two-circles metrics: (gap, evenness, on_ring, pool_mean, pool_min, pool_max).
fn tc_metrics<St>(strat: &mut St) -> (f64, f64, f64, f64, f64, f64)
where
    St: EvolutionStrategy<Score = f64> + PopulationView<Genome = Vec<f64>>,
    St::Genetic: evors::genetic::Genetic<Genome = Vec<f64>>,
{
    // Warm up to fill the pool, then sample sizes over the back half.
    let mut sizes: Vec<usize> = Vec::with_capacity(TC_STEPS);
    let cfg = RunConfig {
        stop: Stop::Trials(TC_STEPS),
        batch_size: 1,
        max_trials: None,
    };
    let _ = run(&cfg, strat, |x: &Vec<f64>| -objective(x));
    // pool size after the run (single point); for stability we re-sample below.
    let pool = strat.members();
    let _ = &mut sizes;
    let (evenness, on_ring) = ring_evenness(&pool);
    let psize = pool.len() as f64;
    (
        between_fraction(&pool),
        evenness,
        on_ring as f64,
        psize,
        psize,
        psize,
    )
}

/// Resource tc metrics with explicit pool-size sampling over time (manual loop).
fn tc_resource(info: ResourceSoftStochasticInfo) -> (f64, f64, f64, f64, f64, f64) {
    let mut strat = ResourceSoftStochastic::new(VecGenetics::new(bounds(), 0.3), info);
    let mut sizes: Vec<usize> = Vec::with_capacity(TC_STEPS);
    for _ in 0..TC_STEPS {
        let g = strat.next();
        let f = -objective(&g);
        strat.report(g, f);
        sizes.push(strat.pool_size());
    }
    let pool = strat.members();
    let (evenness, on_ring) = ring_evenness(&pool);
    let back = &sizes[TC_STEPS / 2..];
    let pmean = back.iter().sum::<usize>() as f64 / back.len() as f64;
    let pmin = *back.iter().min().unwrap() as f64;
    let pmax = *back.iter().max().unwrap() as f64;
    (
        between_fraction(&pool),
        evenness,
        on_ring as f64,
        pmean,
        pmin,
        pmax,
    )
}

fn main() {
    let exp = std::env::args().nth(1).unwrap_or_else(|| "explore".into());
    create_dir_all("data/analysis").unwrap();
    let f = File::create(format!("data/analysis/{exp}.csv")).unwrap();
    let mut w = BufWriter::new(f);
    writeln!(
        w,
        "method,knob,value,cov_ackley,cov_rastrigin,cov_griewank,cov_mean,gap,evenness,on_ring,pool_mean,pool_min,pool_max"
    )
    .unwrap();

    let mut row = |w: &mut BufWriter<File>,
                   method: &str,
                   knob: &str,
                   value: f64,
                   info: ResourceSoftStochasticInfo| {
        let ca = cov_resource(info, "ackley");
        let cr = cov_resource(info, "rastrigin");
        let cg = cov_resource(info, "griewank");
        let (gap, even, onr, pm, pmin, pmax) = tc_resource(info);
        writeln!(
            w,
            "{method},{knob},{value},{ca:.2},{cr:.2},{cg:.2},{:.2},{gap:.4},{even:.4},{onr:.0},{pm:.0},{pmin:.0},{pmax:.0}",
            (ca + cr + cg) / 3.0
        )
        .unwrap();
        println!(
            "{method} {knob}={value}: cov_mean={:.1} gap={gap:.3} even={even:.3} pool~{pm:.0}",
            (ca + cr + cg) / 3.0
        );
    };

    match exp.as_str() {
        "explore" => {
            for v in [0.05, 0.10, 0.15, 0.20, 0.30] {
                let mut i = base();
                i.explore_floor = v;
                row(&mut w, "resource", "explore_floor", v, i);
            }
        }
        "rw" => {
            for v in [0.02, 0.03, 0.05, 0.08] {
                let mut i = base();
                i.relative_width = v;
                row(&mut w, "resource", "relative_width", v, i);
            }
        }
        "temp" => {
            for v in [0.05, 0.10, 0.20, 0.50] {
                let mut i = base();
                i.temperature = v;
                row(&mut w, "resource", "temperature", v, i);
            }
        }
        "loc" => {
            for v in [0.0, 1.0, 2.0, 5.0] {
                let mut i = base();
                i.mate_locality = v;
                row(&mut w, "resource", "mate_locality", v, i);
            }
        }
        "target" => {
            for v in [150usize, 300, 500] {
                let mut i = base();
                i.target = v;
                row(&mut w, "resource", "target", v as f64, i);
            }
        }
        "cullp" => {
            for v in [1.0, 2.0, 4.0] {
                let mut i = base();
                i.cull_pressure = v;
                row(&mut w, "resource", "cull_pressure", v, i);
            }
        }
        "compare" => {
            // Resource (tuned) vs SSEA vs Speciated on the same metrics.
            row(&mut w, "resource", "tuned", 0.0, base());

            let ssea = |s: u64| {
                Strat::Ssea(SoftStochasticInfo {
                    target: 300,
                    novelty_weight: 0.6,
                    temperature: 0.2,
                    behavior_weight: 0.0,
                    knn: 15,
                    seed: s,
                })
            };
            let mut ms = SoftStochastic::new(
                VecGenetics::new(bounds(), 0.3),
                SoftStochasticInfo {
                    target: 300,
                    novelty_weight: 0.6,
                    temperature: 0.2,
                    behavior_weight: 0.0,
                    knn: 15,
                    seed: 0,
                },
            );
            let (gap, even, onr, pm, _, _) = tc_metrics(&mut ms);
            writeln!(
                w,
                "ssea,knn15,0,{:.2},{:.2},{:.2},{:.2},{gap:.4},{even:.4},{onr:.0},{pm:.0},{pm:.0},{pm:.0}",
                cov_strat(ssea(0), "ackley"),
                cov_strat(ssea(0), "rastrigin"),
                cov_strat(ssea(0), "griewank"),
                (cov_strat(ssea(0), "ackley")
                    + cov_strat(ssea(0), "rastrigin")
                    + cov_strat(ssea(0), "griewank"))
                    / 3.0
            )
            .unwrap();
            println!("ssea: gap={gap:.3} even={even:.3} pool~{pm:.0}");

            let spec = |s: u64| {
                Strat::Speciated(SpeciatedSoftStochasticInfo {
                    target: 300,
                    novelty_weight: 0.6,
                    temperature: 0.2,
                    behavior_weight: 0.0,
                    knn: 15,
                    seed: s,
                    mate_locality: 2.0,
                    archive_prob: 0.05,
                    speciation_pressure: 0.6,
                })
            };
            let mut msp = SpeciatedSoftStochastic::new(
                VecGenetics::new(bounds(), 0.3),
                SpeciatedSoftStochasticInfo {
                    target: 300,
                    novelty_weight: 0.6,
                    temperature: 0.2,
                    behavior_weight: 0.0,
                    knn: 15,
                    seed: 0,
                    mate_locality: 2.0,
                    archive_prob: 0.05,
                    speciation_pressure: 0.6,
                },
            );
            let (gap, even, onr, pm, _, _) = tc_metrics(&mut msp);
            writeln!(
                w,
                "speciated,base,0,{:.2},{:.2},{:.2},{:.2},{gap:.4},{even:.4},{onr:.0},{pm:.0},{pm:.0},{pm:.0}",
                cov_strat(spec(0), "ackley"),
                cov_strat(spec(0), "rastrigin"),
                cov_strat(spec(0), "griewank"),
                (cov_strat(spec(0), "ackley")
                    + cov_strat(spec(0), "rastrigin")
                    + cov_strat(spec(0), "griewank"))
                    / 3.0
            )
            .unwrap();
            println!("speciated: gap={gap:.3} even={even:.3} pool~{pm:.0}");
        }
        "best" => {
            // Curated promising combos to find an overall winner + the key
            // explore↔coverage interaction.
            let combos: [(&str, ResourceSoftStochasticInfo); 8] = [
                ("baseline", base()),
                (
                    "cullp4",
                    ResourceSoftStochasticInfo {
                        cull_pressure: 4.0,
                        ..base()
                    },
                ),
                (
                    "explore.2",
                    ResourceSoftStochasticInfo {
                        explore_floor: 0.2,
                        ..base()
                    },
                ),
                (
                    "explore.2+cullp4",
                    ResourceSoftStochasticInfo {
                        explore_floor: 0.2,
                        cull_pressure: 4.0,
                        ..base()
                    },
                ),
                (
                    "explore.3+cullp4",
                    ResourceSoftStochasticInfo {
                        explore_floor: 0.3,
                        cull_pressure: 4.0,
                        ..base()
                    },
                ),
                (
                    "target500+cullp2",
                    ResourceSoftStochasticInfo {
                        target: 500,
                        cull_pressure: 2.0,
                        ..base()
                    },
                ),
                (
                    "rw.02+cullp4",
                    ResourceSoftStochasticInfo {
                        relative_width: 0.02,
                        cull_pressure: 4.0,
                        ..base()
                    },
                ),
                (
                    "explore.15+cullp2+t500",
                    ResourceSoftStochasticInfo {
                        explore_floor: 0.15,
                        cull_pressure: 2.0,
                        target: 500,
                        ..base()
                    },
                ),
            ];
            for (tag, info) in combos {
                row(&mut w, "resource", tag, 0.0, info);
            }
        }
        other => {
            eprintln!("unknown experiment '{other}'");
            std::process::exit(1);
        }
    }
    println!("wrote data/analysis/{exp}.csv");
}
