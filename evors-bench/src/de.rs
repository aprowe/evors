//! Differential Evolution (DE/rand/1/bin) as an evors `EvolutionStrategy`,
//! specialized to `Vec<f64>`. A strong continuous-optimization baseline; unlike
//! the generic `Genetic`-based strategies it does vector arithmetic on the
//! genome. Assumes the runner steps one `next()`/`report()` at a time
//! (batch_size 1), which the harness uses.

use crate::genetics::VecGenetics;
use evors::population::PopulationView;
use evors::strategy::EvolutionStrategy;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[derive(Clone, Copy)]
pub struct DeInfo {
    pub pop: usize,
    pub f: f64,  // differential weight
    pub cr: f64, // crossover rate
    pub seed: u64,
}

impl Default for DeInfo {
    fn default() -> Self {
        Self {
            pop: 60,
            f: 0.6,
            cr: 0.9,
            seed: 0,
        }
    }
}

struct Indiv {
    x: Vec<f64>,
    fit: f64,
}

pub struct Differential {
    bounds: Vec<(f64, f64)>,
    rng: ChaCha8Rng,
    f: f64,
    cr: f64,
    pop_target: usize,
    pop: Vec<Indiv>,
    cursor: usize,
    pending_target: Option<usize>,
    best: Option<(Vec<f64>, f64)>,
}

impl Differential {
    pub fn new(bounds: Vec<(f64, f64)>, info: DeInfo) -> Self {
        Self {
            bounds,
            rng: ChaCha8Rng::seed_from_u64(info.seed),
            f: info.f,
            cr: info.cr.clamp(0.0, 1.0),
            pop_target: info.pop.max(4),
            pop: Vec::new(),
            cursor: 0,
            pending_target: None,
            best: None,
        }
    }

    fn rand_vec(&mut self) -> Vec<f64> {
        self.bounds
            .iter()
            .map(|(lo, hi)| self.rng.random_range(*lo..*hi))
            .collect()
    }

    /// Three distinct indices in `0..n`, all different from `exclude`.
    fn distinct3(&mut self, exclude: usize, n: usize) -> (usize, usize, usize) {
        let mut a = self.rng.random_range(0..n);
        while a == exclude {
            a = self.rng.random_range(0..n);
        }
        let mut b = self.rng.random_range(0..n);
        while b == exclude || b == a {
            b = self.rng.random_range(0..n);
        }
        let mut c = self.rng.random_range(0..n);
        while c == exclude || c == a || c == b {
            c = self.rng.random_range(0..n);
        }
        (a, b, c)
    }
}

impl EvolutionStrategy for Differential {
    type Genetic = VecGenetics;
    type Score = f64;

    fn next(&mut self) -> Vec<f64> {
        // Initialization: fill the population with random vectors.
        if self.pop.len() < self.pop_target {
            self.pending_target = None;
            return self.rand_vec();
        }
        // DE/rand/1/bin: build a trial for the current target index.
        let i = self.cursor;
        let n = self.pop.len();
        let (a, b, c) = self.distinct3(i, n);
        let dim = self.bounds.len();
        let r = self.rng.random_range(0..dim); // forced-crossover dimension
        let mut trial = self.pop[i].x.clone();
        for d in 0..dim {
            if d == r || self.rng.random_bool(self.cr) {
                let v = self.pop[a].x[d] + self.f * (self.pop[b].x[d] - self.pop[c].x[d]);
                let (lo, hi) = self.bounds[d];
                trial[d] = v.clamp(lo, hi);
            }
        }
        self.pending_target = Some(i);
        trial
    }

    fn report(&mut self, genome: Vec<f64>, score: f64) {
        if self.best.as_ref().map_or(true, |(_, b)| score > *b) {
            self.best = Some((genome.clone(), score));
        }
        match self.pending_target {
            None => self.pop.push(Indiv {
                x: genome,
                fit: score,
            }),
            Some(i) => {
                // Greedy selection (we maximize, score = -objective): keep the
                // trial if it is no worse than the target it challenged.
                if score >= self.pop[i].fit {
                    self.pop[i] = Indiv {
                        x: genome,
                        fit: score,
                    };
                }
                self.cursor = (i + 1) % self.pop.len();
            }
        }
        self.pending_target = None;
    }

    fn best(&self) -> Option<(&Vec<f64>, &f64)> {
        self.best.as_ref().map(|(g, s)| (g, s))
    }
}

impl PopulationView for Differential {
    type Genome = Vec<f64>;
    fn members(&self) -> Vec<Vec<f64>> {
        self.pop.iter().map(|m| m.x.clone()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::{Config, Strat, run_config, succeeded};
    use crate::problems::all;

    #[test]
    fn de_solves_sphere() {
        let p = all().into_iter().find(|p| p.name == "sphere").unwrap();
        let cfg = Config {
            label: "de".into(),
            strat: Strat::De(DeInfo {
                pop: 60,
                f: 0.6,
                cr: 0.9,
                seed: 0,
            }),
            mutation_step: 0.0,
        };
        let r = run_config(&p, &cfg, 6000, 1);
        assert!(succeeded(&p, &r), "DE error too large on sphere");
    }
}
