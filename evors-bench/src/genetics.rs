//! Real-vector genetics for continuous optimization. `Genome = Vec<f64>`; here
//! the genome *is* the solution, so `gene_dist` is the solution distance that
//! drives SSEA's basin spreading.

use evors::Genetic;
use rand::Rng;

pub struct VecGenetics {
    bounds: Vec<(f64, f64)>,
    /// Per-coordinate uniform mutation half-range.
    step: f64,
}

impl VecGenetics {
    pub fn new(bounds: Vec<(f64, f64)>, step: f64) -> Self {
        Self { bounds, step: step.max(1e-9) }
    }
}

impl Genetic for VecGenetics {
    type Genome = Vec<f64>;

    fn random(&self, rng: &mut impl Rng) -> Vec<f64> {
        self.bounds.iter().map(|(lo, hi)| rng.random_range(*lo..*hi)).collect()
    }

    fn cross(&self, a: &Vec<f64>, b: &Vec<f64>, rng: &mut impl Rng) -> Vec<f64> {
        a.iter().zip(b).map(|(x, y)| if rng.random_bool(0.5) { *x } else { *y }).collect()
    }

    fn mutate(&self, g: &Vec<f64>, rng: &mut impl Rng) -> Vec<f64> {
        g.iter()
            .zip(&self.bounds)
            .map(|(v, (lo, hi))| (v + rng.random_range(-self.step..self.step)).clamp(*lo, *hi))
            .collect()
    }

    fn gene_dist(&self, a: &Vec<f64>, b: &Vec<f64>) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn random_and_mutate_stay_in_bounds() {
        let g = VecGenetics::new(vec![(-2.0, 2.0), (0.0, 1.0)], 0.5);
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let mut genome = g.random(&mut rng);
        for _ in 0..200 {
            genome = g.mutate(&genome, &mut rng);
            assert!(genome[0] >= -2.0 && genome[0] <= 2.0);
            assert!(genome[1] >= 0.0 && genome[1] <= 1.0);
        }
    }

    #[test]
    fn gene_dist_is_euclidean() {
        let g = VecGenetics::new(vec![(0.0, 1.0), (0.0, 1.0)], 0.1);
        assert!((g.gene_dist(&vec![0.0, 0.0], &vec![3.0, 4.0]) - 5.0).abs() < 1e-9);
    }
}
