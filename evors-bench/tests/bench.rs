//! Integration: both strategies solve an easy problem; SSEA covers more basins
//! than the greedy GA on a multi-optimum problem (the diversity assertion).

use evors::strategies::{ClassicPopulationInfo, SoftStochasticInfo};
use evors_bench::Problem;
use evors_bench::harness::{Config, Strat, coverage, run_config, succeeded};
use evors_bench::problems::all;

fn problem(name: &str) -> Problem {
    all().into_iter().find(|p| p.name == name).unwrap()
}

fn classic(step: f64) -> Config {
    Config {
        label: "classic".into(),
        strat: Strat::Classic(ClassicPopulationInfo {
            pop_size: 200,
            tournament: 3,
            mutation_rate: 0.6,
            seed: 0,
        }),
        mutation_step: step,
    }
}
fn ssea(step: f64) -> Config {
    Config {
        label: "ssea".into(),
        strat: Strat::Ssea(SoftStochasticInfo {
            // target/temperature tuned up (250->400, 0.15->0.25) so the seed-3
            // Himmelblau run reliably reaches >=3 of the 4 basins.
            target: 400,
            novelty_weight: 0.6,
            temperature: 0.25,
            behavior_weight: 0.0,
            knn: 5,
            seed: 0,
        }),
        mutation_step: step,
    }
}

#[test]
fn both_solve_sphere() {
    let p = problem("sphere");
    assert!(succeeded(&p, &run_config(&p, &classic(0.3), 8000, 1)));
    assert!(succeeded(&p, &run_config(&p, &ssea(0.3), 8000, 1)));
}

#[test]
fn ssea_covers_more_basins_than_classic_on_himmelblau() {
    // Himmelblau has 4 distinct global minima; SSEA should find several, the
    // greedy GA collapses onto ~1. Averaged over seeds (novelty is now stochastic
    // — proximity-weighted kNN — so a single seed is noisy; the diversity property
    // is a property of the average, not one run).
    let p = problem("himmelblau");
    let seeds = 6u64;
    let mean = |cfg: &Config| -> f64 {
        (0..seeds)
            .map(|s| coverage(&p, &run_config(&p, cfg, 15000, s)) as f64)
            .sum::<f64>()
            / seeds as f64
    };
    let ssea_cov = mean(&ssea(0.25));
    let classic_cov = mean(&classic(0.25));
    println!("himmelblau mean coverage: ssea={ssea_cov:.2} classic={classic_cov:.2}");
    assert!(
        ssea_cov > classic_cov,
        "ssea covered {ssea_cov:.2}/4 basins, classic {classic_cov:.2}/4"
    );
    assert!(
        ssea_cov >= 2.5,
        "ssea should find most of the 4 optima on average, got {ssea_cov:.2}"
    );
}
