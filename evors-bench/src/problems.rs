//! Standard continuous **minimization** test functions (2-D), each with its
//! bounds, known global minimum, global optima, and notable local minima (for
//! basin coverage). `value_eps` is the success tolerance on objective value;
//! `basin_eps` is the L2 distance a pool member must be within to "occupy" a
//! local minimum.

use std::f64::consts::PI;

pub struct Problem {
    pub name: &'static str,
    pub dim: usize,
    pub bounds: Vec<(f64, f64)>,
    pub objective: fn(&[f64]) -> f64,
    pub global_min: f64,
    pub optima: Vec<Vec<f64>>,
    pub local_minima: Vec<Vec<f64>>,
    pub value_eps: f64,
    pub basin_eps: f64,
}

// ---- objective functions ----

fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum()
}
fn rosenbrock(x: &[f64]) -> f64 {
    let (a, b) = (x[0], x[1]);
    (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2)
}
fn rastrigin(x: &[f64]) -> f64 {
    10.0 * x.len() as f64
        + x.iter().map(|v| v * v - 10.0 * (2.0 * PI * v).cos()).sum::<f64>()
}
fn eggcrate(x: &[f64]) -> f64 {
    let (a, b) = (x[0], x[1]);
    a * a + b * b + 25.0 * (a.sin().powi(2) + b.sin().powi(2))
}
fn ackley(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let s1 = x.iter().map(|v| v * v).sum::<f64>() / n;
    let s2 = x.iter().map(|v| (2.0 * PI * v).cos()).sum::<f64>() / n;
    -20.0 * (-0.2 * s1.sqrt()).exp() - s2.exp() + 20.0 + std::f64::consts::E
}
fn griewank(x: &[f64]) -> f64 {
    let s = x.iter().map(|v| v * v).sum::<f64>() / 4000.0;
    let p: f64 = x.iter().enumerate().map(|(i, v)| (v / ((i + 1) as f64).sqrt()).cos()).product();
    s - p + 1.0
}
fn schwefel(x: &[f64]) -> f64 {
    418.982_887_272_433_8 * x.len() as f64
        - x.iter().map(|v| v * v.abs().sqrt().sin()).sum::<f64>()
}
fn dropwave(x: &[f64]) -> f64 {
    let r2 = x[0] * x[0] + x[1] * x[1];
    -(1.0 + (12.0 * r2.sqrt()).cos()) / (0.5 * r2 + 2.0)
}
fn himmelblau(x: &[f64]) -> f64 {
    let (a, b) = (x[0], x[1]);
    (a * a + b - 11.0).powi(2) + (a + b * b - 7.0).powi(2)
}
fn cross_in_tray(x: &[f64]) -> f64 {
    let (a, b) = (x[0], x[1]);
    let inner = (100.0 - (a * a + b * b).sqrt() / PI).abs();
    -0.0001 * ((a.sin() * b.sin() * inner.exp()).abs() + 1.0).powf(0.1)
}

// ---- helpers ----

fn b2(lo: f64, hi: f64) -> Vec<(f64, f64)> {
    vec![(lo, hi), (lo, hi)]
}

/// Integer-lattice points within `[lo, hi]^2` (used for grid-structured minima).
fn int_grid(lo: f64, hi: f64) -> Vec<Vec<f64>> {
    let mut v = Vec::new();
    let a = lo.ceil() as i64;
    let b = hi.floor() as i64;
    for i in a..=b {
        for j in a..=b {
            v.push(vec![i as f64, j as f64]);
        }
    }
    v
}

/// Multiples-of-π lattice within `[lo, hi]^2` (Egg Crate local minima).
fn pi_grid(lo: f64, hi: f64) -> Vec<Vec<f64>> {
    let mut v = Vec::new();
    let k = (lo / PI).ceil() as i64;
    let m = (hi / PI).floor() as i64;
    for i in k..=m {
        for j in k..=m {
            v.push(vec![i as f64 * PI, j as f64 * PI]);
        }
    }
    v
}

/// The full benchmark set.
pub fn all() -> Vec<Problem> {
    let himmel = vec![
        vec![3.0, 2.0],
        vec![-2.805118, 3.131312],
        vec![-3.779310, -3.283186],
        vec![3.584428, -1.848126],
    ];
    let cross = vec![
        vec![1.349406, 1.349406],
        vec![1.349406, -1.349406],
        vec![-1.349406, 1.349406],
        vec![-1.349406, -1.349406],
    ];
    vec![
        Problem { name: "sphere", dim: 2, bounds: b2(-5.0, 5.0), objective: sphere,
            global_min: 0.0, optima: vec![vec![0.0, 0.0]], local_minima: vec![vec![0.0, 0.0]],
            value_eps: 1e-3, basin_eps: 0.3 },
        Problem { name: "rosenbrock", dim: 2, bounds: b2(-2.0, 2.0), objective: rosenbrock,
            global_min: 0.0, optima: vec![vec![1.0, 1.0]], local_minima: vec![vec![1.0, 1.0]],
            value_eps: 1e-2, basin_eps: 0.2 },
        Problem { name: "rastrigin", dim: 2, bounds: b2(-5.12, 5.12), objective: rastrigin,
            global_min: 0.0, optima: vec![vec![0.0, 0.0]], local_minima: int_grid(-5.0, 5.0),
            value_eps: 1e-2, basin_eps: 0.3 },
        Problem { name: "eggcrate", dim: 2, bounds: b2(-5.0, 5.0), objective: eggcrate,
            global_min: 0.0, optima: vec![vec![0.0, 0.0]], local_minima: pi_grid(-5.0, 5.0),
            value_eps: 1e-2, basin_eps: 0.5 },
        Problem { name: "ackley", dim: 2, bounds: b2(-5.0, 5.0), objective: ackley,
            global_min: 0.0, optima: vec![vec![0.0, 0.0]], local_minima: int_grid(-4.0, 4.0),
            value_eps: 1e-2, basin_eps: 0.3 },
        Problem { name: "griewank", dim: 2, bounds: b2(-10.0, 10.0), objective: griewank,
            global_min: 0.0, optima: vec![vec![0.0, 0.0]], local_minima: int_grid(-9.0, 9.0),
            value_eps: 1e-2, basin_eps: 0.5 },
        Problem { name: "schwefel", dim: 2, bounds: b2(-500.0, 500.0), objective: schwefel,
            global_min: 0.0, optima: vec![vec![420.9687, 420.9687]],
            local_minima: vec![vec![420.9687, 420.9687], vec![420.9687, -302.5249],
                vec![-302.5249, 420.9687], vec![-302.5249, -302.5249]],
            value_eps: 1.0, basin_eps: 20.0 },
        Problem { name: "dropwave", dim: 2, bounds: b2(-5.12, 5.12), objective: dropwave,
            global_min: -1.0, optima: vec![vec![0.0, 0.0]], local_minima: vec![vec![0.0, 0.0]],
            value_eps: 1e-2, basin_eps: 0.3 },
        Problem { name: "himmelblau", dim: 2, bounds: b2(-5.0, 5.0), objective: himmelblau,
            global_min: 0.0, optima: himmel.clone(), local_minima: himmel,
            value_eps: 1e-2, basin_eps: 0.4 },
        Problem { name: "cross_in_tray", dim: 2, bounds: b2(-10.0, 10.0), objective: cross_in_tray,
            global_min: -2.062612, optima: cross.clone(), local_minima: cross,
            value_eps: 1e-2, basin_eps: 0.4 },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn objective_matches_global_min_at_each_optimum() {
        for p in all() {
            for opt in &p.optima {
                let v = (p.objective)(opt);
                assert!(
                    (v - p.global_min).abs() < p.value_eps.max(1e-2),
                    "{}: objective({:?}) = {} != global_min {}",
                    p.name, opt, v, p.global_min
                );
            }
        }
    }

    #[test]
    fn local_minima_lie_within_bounds() {
        for p in all() {
            for m in &p.local_minima {
                for (v, (lo, hi)) in m.iter().zip(&p.bounds) {
                    assert!(*v >= *lo - 1e-9 && *v <= *hi + 1e-9, "{}: {:?} out of bounds", p.name, m);
                }
            }
        }
    }
}
