use evors::genetic::Genetic;
use evors::strategies::{ClassicPopulation, ClassicPopulationInfo};
use evors::trial::{RunConfig, Stop, run};
use rand::Rng;

const TARGET: &str = "HELLO WORLD";
const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ ";

struct StringGenetics {
    len: usize,
}

impl Genetic for StringGenetics {
    type Genome = Vec<u8>;
    fn random(&self, rng: &mut impl Rng) -> Vec<u8> {
        (0..self.len)
            .map(|_| ALPHABET[rng.random_range(0..ALPHABET.len())])
            .collect()
    }
    fn cross(&self, a: &Vec<u8>, b: &Vec<u8>, rng: &mut impl Rng) -> Vec<u8> {
        let cut = rng.random_range(0..self.len);
        a[..cut].iter().chain(b[cut..].iter()).copied().collect()
    }
    fn mutate(&self, g: &Vec<u8>, rng: &mut impl Rng) -> Vec<u8> {
        let mut g = g.clone();
        let i = rng.random_range(0..self.len);
        g[i] = ALPHABET[rng.random_range(0..ALPHABET.len())];
        g
    }
    fn gene_dist(&self, a: &Vec<u8>, b: &Vec<u8>) -> f64 {
        a.iter().zip(b).filter(|(x, y)| x != y).count() as f64
    }
}

fn fitness(target: &[u8], g: &Vec<u8>) -> f64 {
    g.iter().zip(target).filter(|(x, y)| x == y).count() as f64 / target.len() as f64
}

#[test]
fn classic_population_evolves_target_string() {
    let target = TARGET.as_bytes().to_vec();
    let mut strat = ClassicPopulation::new(
        StringGenetics { len: TARGET.len() },
        ClassicPopulationInfo {
            pop_size: 200,
            mutation_rate: 0.3,
            seed: 42,
            ..Default::default()
        },
    );
    let cfg = RunConfig {
        stop: Stop::Until(Box::new(|s: &f64| *s >= 1.0)),
        batch_size: 200,
        max_trials: Some(200_000),
    };
    let out = run(&cfg, &mut strat, |g: &Vec<u8>| fitness(&target, g));
    assert_eq!(out.best_score, Some(1.0));
    assert_eq!(out.best_genome.unwrap(), target);
}
