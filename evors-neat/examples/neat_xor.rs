//! Evolve a NEAT network that solves XOR, using evors' SSEA strategy with the
//! compatibility distance as the novelty metric. Records convergence + champion
//! topology growth into ./data. Run: `cargo run --release -p evors-neat --example neat_xor`.

use evors::strategies::{SoftStochastic, SoftStochasticInfo};
use evors::strategy::EvolutionStrategy;
use evors_neat::{NeatGenetics, NeatGenome, NeatParams, activate};
use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};

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

fn main() {
    create_dir_all("data").unwrap();

    let params = NeatParams {
        add_node_prob: 0.02,
        ..NeatParams::default()
    };
    let genetics = NeatGenetics::new(2, 1, params);
    let mut ssea = SoftStochastic::new(
        genetics,
        SoftStochasticInfo {
            target: 150,
            temperature: 0.1,
            novelty_weight: 0.1,
            knn: 5,
            seed: 7,
        },
    );

    let cap = 40_000usize;
    let thr = 0.98;
    let mut best = f64::MIN;
    let mut best_genome: Option<NeatGenome> = None;
    let mut hit: Option<usize> = None;
    let mut rows: Vec<(usize, f64, usize)> = Vec::with_capacity(cap);

    for t in 0..cap {
        let g = ssea.next();
        let f = xor_fitness(&g);
        if f > best {
            best = f;
            best_genome = Some(g.clone());
        }
        ssea.report(g, f);
        if hit.is_none() && best >= thr {
            hit = Some(t);
        }
        let hidden = best_genome.as_ref().map(|g| g.hidden_count()).unwrap_or(0);
        rows.push((t, best, hidden));
        if best >= thr {
            break;
        }
    }

    let champ = best_genome.unwrap();
    println!(
        "NEAT XOR: best={:.4} hit@{:?} hidden={} conns={} classifies={}",
        best,
        hit,
        champ.hidden_count(),
        champ.conns.iter().filter(|c| c.enabled).count(),
        classifies(&champ),
    );

    let f = File::create("data/neat_xor.csv").unwrap();
    let mut w = BufWriter::new(f);
    writeln!(w, "trial,best,hidden").unwrap();
    for (t, b, h) in &rows {
        writeln!(w, "{t},{b},{h}").unwrap();
    }
}
