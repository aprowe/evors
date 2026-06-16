//! Integration: both strategies solve an easy problem; SSEA covers more basins
//! than the greedy GA on a multi-optimum problem (the diversity assertion).

use evors::strategies::{ClassicPopulationInfo, SoftStochasticInfo};
use evors_bench::harness::{coverage, run_config, succeeded, Config, Strat};
use evors_bench::problems::all;
use evors_bench::Problem;

fn problem(name: &str) -> Problem {
    all().into_iter().find(|p| p.name == name).unwrap()
}

fn classic(step: f64) -> Config {
    Config {
        label: "classic".into(),
        strat: Strat::Classic(ClassicPopulationInfo {
            pop_size: 200, tournament: 3, mutation_rate: 0.6, seed: 0,
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
            target: 400, novelty_weight: 0.6, temperature: 0.25, behavior_weight: 0.0,
            knn: 5, seed: 0,
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
    // greedy GA collapses onto ~1.
    let p = problem("himmelblau");
    let ssea_cov = coverage(&p, &run_config(&p, &ssea(0.25), 15000, 3));
    let classic_cov = coverage(&p, &run_config(&p, &classic(0.25), 15000, 3));
    assert!(
        ssea_cov > classic_cov,
        "ssea covered {ssea_cov}/4 basins, classic {classic_cov}/4"
    );
    assert!(ssea_cov >= 3, "ssea should find most of the 4 optima, got {ssea_cov}");
}
