use evors::genetic::Genetic;
use evors::strategies::{
    ClassicPopulation, ClassicPopulationInfo, SoftStochastic, SoftStochasticInfo,
};
use evors::trial::{RunConfig, Stop, run};
use rand::Rng;

#[derive(Clone)]
struct Point {
    x: f64,
    y: f64,
}

struct PointGenetics;
impl Genetic for PointGenetics {
    type Genome = Point;
    fn random(&self, rng: &mut impl Rng) -> Point {
        Point {
            x: rng.random_range(-2.0..2.0),
            y: rng.random_range(-2.0..2.0),
        }
    }
    fn cross(&self, a: &Point, b: &Point, _rng: &mut impl Rng) -> Point {
        Point {
            x: (a.x + b.x) / 2.0,
            y: (a.y + b.y) / 2.0,
        }
    }
    fn mutate(&self, g: &Point, rng: &mut impl Rng) -> Point {
        Point {
            x: g.x + rng.random_range(-0.2..0.2),
            y: g.y + rng.random_range(-0.2..0.2),
        }
    }
    fn gene_dist(&self, a: &Point, b: &Point) -> f64 {
        ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt()
    }
}

/// Whole unit circle is optimal: fitness = 1 - (x^2 + y^2 - 1)^2.
fn ring_fitness(p: &Point) -> f64 {
    let r2 = p.x * p.x + p.y * p.y;
    1.0 - (r2 - 1.0).powi(2)
}

/// Number of populated angle sectors (out of `sectors`) among high-fitness points.
fn sector_coverage(points: &[Point], sectors: usize, fit_threshold: f64) -> usize {
    let mut hit = vec![false; sectors];
    for p in points {
        if ring_fitness(p) >= fit_threshold {
            let mut ang = p.y.atan2(p.x); // -pi..pi
            if ang < 0.0 {
                ang += std::f64::consts::TAU;
            }
            let idx = ((ang / std::f64::consts::TAU) * sectors as f64) as usize % sectors;
            hit[idx] = true;
        }
    }
    hit.iter().filter(|b| **b).count()
}

#[test]
fn ssea_spreads_around_the_ring() {
    let mut strat = SoftStochastic::new(
        PointGenetics,
        SoftStochasticInfo {
            target: 150,
            temperature: 0.2,
            novelty_weight: 0.6,
            behavior_weight: 0.0,
            knn: 5,
            seed: 13,
        },
    );
    let cfg = RunConfig {
        stop: Stop::Trials(16_000),
        batch_size: 1,
        max_trials: None,
    };
    let _ = run(&cfg, &mut strat, ring_fitness);

    // Measure the maintained pool's angular coverage of the ring.
    let covered = sector_coverage(&strat.pool_genomes(), 12, 0.95);
    assert!(covered >= 9, "SSEA only covered {covered}/12 sectors");
}

#[test]
fn classic_population_collapses_relative_to_ssea() {
    let mut classic = ClassicPopulation::new(
        PointGenetics,
        ClassicPopulationInfo {
            pop_size: 150,
            tournament: 3,
            mutation_rate: 0.2,
            seed: 13,
        },
    );
    let cfg = RunConfig {
        stop: Stop::Trials(16_000),
        batch_size: 1,
        max_trials: None,
    };
    let _ = run(&cfg, &mut classic, ring_fitness);
    let classic_cov = sector_coverage(&classic.population(), 12, 0.95);

    let mut ssea = SoftStochastic::new(
        PointGenetics,
        SoftStochasticInfo {
            target: 150,
            temperature: 0.2,
            novelty_weight: 0.6,
            behavior_weight: 0.0,
            knn: 5,
            seed: 13,
        },
    );
    let cfg2 = RunConfig {
        stop: Stop::Trials(16_000),
        batch_size: 1,
        max_trials: None,
    };
    let _ = run(&cfg2, &mut ssea, ring_fitness);
    let ssea_cov = sector_coverage(&ssea.pool_genomes(), 12, 0.95);

    // SSEA preserves diversity; the greedy GA collapses onto one arc.
    assert!(
        ssea_cov > classic_cov,
        "ssea {ssea_cov} vs classic {classic_cov}"
    );
}
