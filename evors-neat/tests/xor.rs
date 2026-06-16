//! Integration test: NEAT (as an evors Genetic) solves XOR on the SSEA strategy.

use evors::strategies::{SoftStochastic, SoftStochasticInfo};
use evors::trial::{Outcome, RunConfig, Stop, run};
use evors_neat::{NeatGenetics, NeatGenome, NeatParams, activate};

const CASES: [([f64; 2], f64); 4] = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
];

fn xor_fitness(g: &NeatGenome) -> f64 {
    let mse: f64 = CASES
        .iter()
        .map(|(inp, t)| (activate(g, inp)[0] - t).powi(2))
        .sum::<f64>()
        / 4.0;
    1.0 - mse
}

fn classifies(g: &NeatGenome) -> bool {
    CASES
        .iter()
        .all(|(inp, t)| (activate(g, inp)[0] > 0.5) == (*t > 0.5))
}

#[test]
fn neat_solves_xor_on_ssea() {
    let params = NeatParams {
        add_node_prob: 0.02,
        ..NeatParams::default()
    };
    let mut ssea = SoftStochastic::new(
        NeatGenetics::new(2, 1, params),
        SoftStochasticInfo {
            target: 150,
            temperature: 0.1,
            novelty_weight: 0.1,
            behavior_weight: 0.0,
            knn: 5,
            seed: 7,
        },
    );
    let cfg = RunConfig {
        stop: Stop::Until(Box::new(|s: &f64| *s > 0.98)),
        batch_size: 1,
        max_trials: Some(40_000),
    };
    let out: Outcome<NeatGenome, f64> = run(&cfg, &mut ssea, xor_fitness);
    assert!(
        out.best_score.unwrap() > 0.98,
        "best fitness {:?}",
        out.best_score
    );
    let champ = out.best_genome.unwrap();
    assert!(
        classifies(&champ),
        "champion does not classify all four XOR cases"
    );
    // NEAT should reach XOR with a compact topology (it can represent XOR with a
    // single hidden node); allow a little slack.
    assert!(
        champ.hidden_count() <= 4,
        "topology bloated: {} hidden",
        champ.hidden_count()
    );
}
