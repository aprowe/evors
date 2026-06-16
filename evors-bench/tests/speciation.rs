//! Speciation property + sanity tests.

use evors::population::PopulationView;
use evors::strategies::{
    SoftStochastic, SoftStochasticInfo, SpeciatedSoftStochastic, SpeciatedSoftStochasticInfo,
};
use evors::{RunConfig, Stop, run};
use evors_bench::VecGenetics;
use evors_bench::two_circles::{between_fraction, bounds, covers_both, objective};

fn fitness(x: &Vec<f64>) -> f64 {
    -objective(x)
}

#[test]
fn speciation_reduces_between_population() {
    let budget = 30_000;
    let cfg = || RunConfig { stop: Stop::Trials(budget), batch_size: 1, max_trials: None };

    let mut ssea = SoftStochastic::new(
        VecGenetics::new(bounds(), 0.3),
        SoftStochasticInfo {
            target: 300, novelty_weight: 0.6, temperature: 0.2, behavior_weight: 0.0, knn: 5, seed: 1,
        },
    );
    let _ = run(&cfg(), &mut ssea, fitness);
    let ssea_pool = ssea.members();

    let mut spec = SpeciatedSoftStochastic::new(
        VecGenetics::new(bounds(), 0.3),
        SpeciatedSoftStochasticInfo {
            target: 300, novelty_weight: 0.6, temperature: 0.2, behavior_weight: 0.0, knn: 5, seed: 1,
            mate_locality: 5.0,
        },
    );
    let _ = run(&cfg(), &mut spec, fitness);
    let spec_pool = spec.members();

    assert!(covers_both(&ssea_pool), "plain SSEA should cover both rings");
    assert!(covers_both(&spec_pool), "speciated should cover both rings");

    let bf_ssea = between_fraction(&ssea_pool);
    let bf_spec = between_fraction(&spec_pool);
    println!("between-fraction: ssea={bf_ssea:.3} speciated={bf_spec:.3}");
    assert!(
        bf_spec < bf_ssea,
        "speciation did not thin the between-population: speciated {bf_spec:.3} >= ssea {bf_ssea:.3}"
    );
}

#[test]
fn speciated_solves_sphere() {
    use evors_bench::harness::{Config, Strat, run_config, succeeded};
    use evors_bench::problems::all;
    let p = all().into_iter().find(|p| p.name == "sphere").unwrap();
    let cfg = Config {
        label: "speciated".into(),
        strat: Strat::Speciated(SpeciatedSoftStochasticInfo {
            target: 100, novelty_weight: 0.2, temperature: 0.1, behavior_weight: 0.0, knn: 5, seed: 0,
            mate_locality: 2.0,
        }),
        mutation_step: 0.3,
    };
    let r = run_config(&p, &cfg, 8000, 1);
    assert!(succeeded(&p, &r), "speciated should solve sphere");
}
