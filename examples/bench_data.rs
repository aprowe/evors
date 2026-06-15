//! Data-collection harness. Drives the strategies manually (so we can record
//! per-trial history with a hard cap), writes CSVs consumed by a plotting
//! script. Run: `cargo run --release --example bench_data`.

use evors::genetic::Genetic;
use evors::strategies::{
    ClassicPopulation, ClassicPopulationInfo, SoftStochastic, SoftStochasticInfo,
};
use evors::strategy::EvolutionStrategy;
use rand::Rng;
use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};

// ---------------- XOR problem ----------------

#[derive(Clone)]
struct Net(Vec<f64>);

struct NetGenetics;
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
        // Perturb a random subset of weights with a larger Gaussian-ish step so
        // the search can reach the sharp, large-magnitude weights XOR needs.
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
fn forward(w: &[f64], x0: f64, x1: f64) -> f64 {
    let h0 = sigmoid(w[0] * x0 + w[1] * x1 + w[2]);
    let h1 = sigmoid(w[3] * x0 + w[4] * x1 + w[5]);
    sigmoid(w[6] * h0 + w[7] * h1 + w[8])
}
fn xor_fitness(net: &Net) -> f64 {
    let cases = [((0.0, 0.0), 0.0), ((0.0, 1.0), 1.0), ((1.0, 0.0), 1.0), ((1.0, 1.0), 0.0)];
    let mse: f64 = cases.iter().map(|((a, b), t)| (forward(&net.0, *a, *b) - t).powi(2)).sum::<f64>() / 4.0;
    1.0 - mse
}

// ---------------- unit circle problem ----------------

#[derive(Clone)]
struct Point { x: f64, y: f64 }

struct PointGenetics;
impl Genetic for PointGenetics {
    type Genome = Point;
    fn random(&self, rng: &mut impl Rng) -> Point {
        Point { x: rng.random_range(-2.0..2.0), y: rng.random_range(-2.0..2.0) }
    }
    fn cross(&self, a: &Point, b: &Point, _rng: &mut impl Rng) -> Point {
        Point { x: (a.x + b.x) / 2.0, y: (a.y + b.y) / 2.0 }
    }
    fn mutate(&self, g: &Point, rng: &mut impl Rng) -> Point {
        Point { x: g.x + rng.random_range(-0.2..0.2), y: g.y + rng.random_range(-0.2..0.2) }
    }
    fn gene_dist(&self, a: &Point, b: &Point) -> f64 {
        ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt()
    }
}
fn ring_fitness(p: &Point) -> f64 {
    let r2 = p.x * p.x + p.y * p.y;
    1.0 - (r2 - 1.0).powi(2)
}

// ---------------- manual driver with history ----------------

/// Step a strategy `cap` times, recording (trial, best_fitness) each step.
/// Returns the history and the first trial index where best >= threshold.
fn run_history<St, F>(strat: &mut St, fitness: F, cap: usize, threshold: f64)
    -> (Vec<(usize, f64)>, Option<usize>)
where
    St: EvolutionStrategy<Score = f64>,
    F: Fn(&<St::Genetic as Genetic>::Genome) -> f64,
{
    let mut hist = Vec::with_capacity(cap);
    let mut best = f64::MIN;
    let mut hit = None;
    for t in 0..cap {
        let g = strat.next();
        let s = fitness(&g);
        strat.report(g, s);
        if s > best { best = s; }
        if hit.is_none() && best >= threshold { hit = Some(t); }
        hist.push((t, best));
    }
    (hist, hit)
}

fn write_convergence(path: &str, classic: &[(usize, f64)], ssea: &[(usize, f64)]) {
    let f = File::create(path).unwrap();
    let mut w = BufWriter::new(f);
    writeln!(w, "trial,classic_best,ssea_best").unwrap();
    let n = classic.len().max(ssea.len());
    for i in 0..n {
        let c = classic.get(i).map(|x| x.1).unwrap_or(f64::NAN);
        let s = ssea.get(i).map(|x| x.1).unwrap_or(f64::NAN);
        let t = classic.get(i).or_else(|| ssea.get(i)).map(|x| x.0).unwrap_or(i);
        writeln!(w, "{t},{c},{s}").unwrap();
    }
}

fn write_points(path: &str, pts: &[Point]) {
    let f = File::create(path).unwrap();
    let mut w = BufWriter::new(f);
    writeln!(w, "x,y,fitness").unwrap();
    for p in pts {
        writeln!(w, "{},{},{}", p.x, p.y, ring_fitness(p)).unwrap();
    }
}

fn main() {
    create_dir_all("data").unwrap();

    // ---- XOR convergence ----
    let cap = 60_000usize;
    let thr = 0.98;

    let mut classic = ClassicPopulation::new(
        NetGenetics,
        ClassicPopulationInfo { pop_size: 200, tournament: 2, mutation_rate: 0.9, seed: 7 },
    );
    let (c_hist, c_hit) = run_history(&mut classic, xor_fitness, cap, thr);

    let mut ssea = SoftStochastic::new(
        NetGenetics,
        SoftStochasticInfo { target: 120, temperature: 0.05, novelty_weight: 0.05, knn: 5, seed: 7 },
    );
    let (s_hist, s_hit) = run_history(&mut ssea, xor_fitness, cap, thr);

    write_convergence("data/xor_convergence.csv", &c_hist, &s_hist);
    println!("XOR classic: best={:.4} hit@{:?}", c_hist.last().unwrap().1, c_hit);
    println!("XOR ssea:    best={:.4} hit@{:?}", s_hist.last().unwrap().1, s_hit);

    // ---- unit circle: final populations ----
    let circle_cap = 20_000usize;

    let mut c_circle = ClassicPopulation::new(
        PointGenetics,
        ClassicPopulationInfo { pop_size: 300, tournament: 3, mutation_rate: 0.2, seed: 13 },
    );
    let _ = run_history(&mut c_circle, ring_fitness, circle_cap, 2.0); // threshold unreachable -> just run
    let c_pts = c_circle.population();
    write_points("data/circle_classic.csv", &c_pts);

    let mut s_circle = SoftStochastic::new(
        PointGenetics,
        SoftStochasticInfo { target: 300, temperature: 0.2, novelty_weight: 0.6, knn: 5, seed: 13 },
    );
    let _ = run_history(&mut s_circle, ring_fitness, circle_cap, 2.0);
    let s_pts = s_circle.pool_genomes();
    write_points("data/circle_ssea.csv", &s_pts);

    let cov = |pts: &[Point]| {
        let mut hit = [false; 12];
        for p in pts {
            if ring_fitness(p) >= 0.95 {
                let mut a = p.y.atan2(p.x);
                if a < 0.0 { a += std::f64::consts::TAU; }
                hit[((a / std::f64::consts::TAU) * 12.0) as usize % 12] = true;
            }
        }
        hit.iter().filter(|b| **b).count()
    };
    println!("circle classic sector coverage: {}/12", cov(&c_pts));
    println!("circle ssea    sector coverage: {}/12", cov(&s_pts));
}
