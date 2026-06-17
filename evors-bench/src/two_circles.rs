//! A two-rings landscape: fitness rewards being on EITHER unit ring, centered at
//! x = ±C. Used to test that soft speciation thins the "between" population.

pub const C: f64 = 5.0;
pub const GAP: f64 = 3.0;

fn ring(u: f64, v: f64) -> f64 {
    1.0 - (u * u + v * v - 1.0).powi(2)
}

/// Minimized objective: `-max(ring_left, ring_right)`; global min is -1 (on a ring).
pub fn objective(x: &[f64]) -> f64 {
    let l = ring(x[0] + C, x[1]);
    let r = ring(x[0] - C, x[1]);
    -(l.max(r))
}

pub fn bounds() -> Vec<(f64, f64)> {
    vec![(-7.0, 7.0), (-2.0, 2.0)]
}

/// Fraction of the pool stranded in the central dead zone `|x| < GAP`.
pub fn between_fraction(pool: &[Vec<f64>]) -> f64 {
    if pool.is_empty() {
        return 0.0;
    }
    pool.iter().filter(|x| x[0].abs() < GAP).count() as f64 / pool.len() as f64
}

/// Both rings covered: a high-fitness member near each ring centre.
pub fn covers_both(pool: &[Vec<f64>]) -> bool {
    let on = |left: bool| {
        pool.iter().any(|x| {
            let near = if left {
                x[0] < -(C - 1.5)
            } else {
                x[0] > C - 1.5
            };
            near && -objective(x) > 0.9
        })
    };
    on(true) && on(false)
}

/// Counts of pool members in the left ring (`x < -(C-1.5)`), the right ring
/// (`x > C-1.5`), and the central dead zone (`|x| < GAP`). Positional only — no
/// fitness gate — so it tracks where the population actually sits over time.
pub fn occupancy(pool: &[Vec<f64>]) -> (usize, usize, usize) {
    let left = pool.iter().filter(|x| x[0] < -(C - 1.5)).count();
    let right = pool.iter().filter(|x| x[0] > C - 1.5).count();
    let gap = pool.iter().filter(|x| x[0].abs() < GAP).count();
    (left, right, gap)
}
