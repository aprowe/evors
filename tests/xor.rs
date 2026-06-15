use evors::genetic::Genetic;
use evors::strategies::{
    ClassicPopulation, ClassicPopulationInfo, SoftStochastic, SoftStochasticInfo,
};
use evors::trial::{run, Outcome, RunConfig, Stop};
use rand::Rng;

/// Genome = weights of a 2-2-1 net: 6 weights + 3 biases = 9 params.
#[derive(Clone)]
pub struct Net(pub Vec<f64>); // length 9

pub struct NetGenetics;
impl Genetic for NetGenetics {
    type Genome = Net;
    fn random(&self, rng: &mut impl Rng) -> Net {
        Net((0..9).map(|_| rng.random_range(-5.0..5.0)).collect())
    }
    fn cross(&self, a: &Net, b: &Net, rng: &mut impl Rng) -> Net {
        Net(a.0.iter().zip(&b.0).map(|(x, y)| if rng.random_bool(0.5) { *x } else { *y }).collect())
    }
    fn mutate(&self, g: &Net, rng: &mut impl Rng) -> Net {
        let mut w = g.0.clone();
        // Perturb a couple of weights with a larger step so the search can reach
        // the sharp, large-magnitude weights a clean XOR solution needs.
        for _ in 0..2 {
            let i = rng.random_range(0..9);
            w[i] = (w[i] + rng.random_range(-2.5..2.5)).clamp(-15.0, 15.0);
        }
        Net(w)
    }
    fn gene_dist(&self, a: &Net, b: &Net) -> f64 {
        a.0.iter().zip(&b.0).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
    }
}

fn sigmoid(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }

pub fn forward(w: &[f64], x0: f64, x1: f64) -> f64 {
    let h0 = sigmoid(w[0] * x0 + w[1] * x1 + w[2]);
    let h1 = sigmoid(w[3] * x0 + w[4] * x1 + w[5]);
    sigmoid(w[6] * h0 + w[7] * h1 + w[8])
}

pub fn xor_fitness(net: &Net) -> f64 {
    let cases = [((0.0, 0.0), 0.0), ((0.0, 1.0), 1.0), ((1.0, 0.0), 1.0), ((1.0, 1.0), 0.0)];
    let mse: f64 = cases.iter().map(|((a, b), t)| (forward(&net.0, *a, *b) - t).powi(2)).sum::<f64>() / 4.0;
    1.0 - mse
}

fn classifies_xor(net: &Net) -> bool {
    forward(&net.0, 0.0, 0.0) < 0.5
        && forward(&net.0, 0.0, 1.0) > 0.5
        && forward(&net.0, 1.0, 0.0) > 0.5
        && forward(&net.0, 1.0, 1.0) < 0.5
}

#[test]
fn classic_population_solves_xor() {
    // Low selection pressure (tournament 2) + high mutation avoids the premature
    // convergence that traps a greedy GA at the 3/4-case (~0.875) plateau.
    let mut strat = ClassicPopulation::new(
        NetGenetics,
        ClassicPopulationInfo { pop_size: 200, tournament: 2, mutation_rate: 0.9, seed: 7 },
    );
    let cfg = RunConfig {
        stop: Stop::Until(Box::new(|s: &f64| *s > 0.98)),
        batch_size: 1,
        max_trials: Some(60_000),
    };
    let out: Outcome<Net, f64> = run(&cfg, &mut strat, xor_fitness);
    assert!(out.best_score.unwrap() > 0.98, "best fitness {:?}", out.best_score);
    assert!(classifies_xor(&out.best_genome.unwrap()));
}

#[test]
fn soft_stochastic_solves_xor() {
    // SSEA keeps a diverse soft pool; on a single-objective problem it pays for
    // that diversity in speed but still escapes the plateau and solves XOR.
    let mut strat = SoftStochastic::new(
        NetGenetics,
        SoftStochasticInfo { target: 120, temperature: 0.05, novelty_weight: 0.05, knn: 5, seed: 7 },
    );
    let cfg = RunConfig {
        stop: Stop::Until(Box::new(|s: &f64| *s > 0.98)),
        batch_size: 1,
        max_trials: Some(60_000),
    };
    let out: Outcome<Net, f64> = run(&cfg, &mut strat, xor_fitness);
    assert!(out.best_score.unwrap() > 0.98, "best fitness {:?}", out.best_score);
    assert!(classifies_xor(&out.best_genome.unwrap()));
}
