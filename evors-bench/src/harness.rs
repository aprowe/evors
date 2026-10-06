//! Run a strategy on a problem and compute metrics.

use crate::de::{DeInfo, Differential};
use crate::genetics::VecGenetics;
use crate::problems::Problem;
use evors::genetic::Genetic;
use evors::population::PopulationView;
use evors::strategies::{
    ClassicPopulation, ClassicPopulationInfo, ResourceSoftStochastic, ResourceSoftStochasticInfo,
    SoftStochastic, SoftStochasticInfo, SpeciatedSoftStochastic, SpeciatedSoftStochasticInfo,
};
use evors::strategy::EvolutionStrategy;
use evors::{RecordingObserver, RunConfig, Stop, run_observed};

/// The result of one (problem, config, seed) run.
pub struct RunResult {
    /// best-so-far *objective value* (minimized) per evaluation.
    pub convergence: Vec<f64>,
    pub final_pool: Vec<Vec<f64>>,
    pub best_x: Vec<f64>,
    pub best_value: f64,
}

/// Which strategy + its parameters (seed is injected by the harness).
#[derive(Clone)]
pub enum Strat {
    Classic(ClassicPopulationInfo),
    Ssea(SoftStochasticInfo),
    Speciated(SpeciatedSoftStochasticInfo),
    Resource(ResourceSoftStochasticInfo),
    De(DeInfo),
}

/// A full config: a strategy spec, the genetics mutation step, and a label.
#[derive(Clone)]
pub struct Config {
    pub label: String,
    pub strat: Strat,
    pub mutation_step: f64,
}

fn run_one<St>(p: &Problem, strat: &mut St, budget: usize) -> RunResult
where
    St: EvolutionStrategy<Score = f64> + PopulationView<Genome = Vec<f64>>,
    St::Genetic: Genetic<Genome = Vec<f64>>,
{
    let cfg = RunConfig {
        stop: Stop::Trials(budget),
        batch_size: 1,
        max_trials: None,
    };
    let mut rec = RecordingObserver::new();
    let objective = p.objective;
    let out = run_observed(&cfg, strat, move |x: &Vec<f64>| -objective(x), &mut rec);

    // Recorded fitness is -objective; convert to minimized objective value.
    let convergence: Vec<f64> = rec.best_fitness.iter().map(|f| -f).collect();
    let best_value = out.best_score.map(|f| -f).unwrap_or(f64::INFINITY);
    let best_x = out.best_genome.unwrap_or_default();
    RunResult {
        convergence,
        final_pool: strat.members(),
        best_x,
        best_value,
    }
}

/// Run a config on a problem at a given seed.
pub fn run_config(p: &Problem, cfg: &Config, budget: usize, seed: u64) -> RunResult {
    match &cfg.strat {
        Strat::Classic(info) => {
            let genetics = VecGenetics::new(p.bounds.clone(), cfg.mutation_step);
            let mut info = *info;
            info.seed = seed;
            let mut strat = ClassicPopulation::new(genetics, info);
            run_one(p, &mut strat, budget)
        }
        Strat::Ssea(info) => {
            let genetics = VecGenetics::new(p.bounds.clone(), cfg.mutation_step);
            let mut info = *info;
            info.seed = seed;
            let mut strat = SoftStochastic::new(genetics, info);
            run_one(p, &mut strat, budget)
        }
        Strat::Speciated(info) => {
            let mut info = *info;
            info.seed = seed;
            let mut strat = SpeciatedSoftStochastic::new(
                VecGenetics::new(p.bounds.clone(), cfg.mutation_step),
                info,
            );
            run_one(p, &mut strat, budget)
        }
        Strat::Resource(info) => {
            let mut info = *info;
            info.seed = seed;
            let mut strat = ResourceSoftStochastic::new(
                VecGenetics::new(p.bounds.clone(), cfg.mutation_step),
                info,
            );
            run_one(p, &mut strat, budget)
        }
        Strat::De(info) => {
            let mut info = *info;
            info.seed = seed;
            let mut strat = Differential::new(p.bounds.clone(), info);
            run_one(p, &mut strat, budget)
        }
    }
}

// ---- metrics over a single RunResult ----

pub fn error(p: &Problem, r: &RunResult) -> f64 {
    r.best_value - p.global_min
}

pub fn succeeded(p: &Problem, r: &RunResult) -> bool {
    error(p, r) < p.value_eps
}

/// First eval index reaching within `value_eps` of the optimum (`None` if never).
pub fn evals_to_target(p: &Problem, r: &RunResult) -> Option<usize> {
    r.convergence
        .iter()
        .position(|v| v - p.global_min < p.value_eps)
}

/// Distinct local minima with at least one final-pool member within `basin_eps`.
pub fn coverage(p: &Problem, r: &RunResult) -> usize {
    p.local_minima
        .iter()
        .filter(|m| {
            r.final_pool.iter().any(|x| {
                x.iter()
                    .zip(m.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt()
                    < p.basin_eps
            })
        })
        .count()
}

/// Aggregated metrics for one (problem, config) over many seeds.
pub struct Summary {
    pub problem: String,
    pub config: String,
    pub mean_best: f64,
    pub mean_error: f64,
    pub success_rate: f64,
    pub median_evals: Option<usize>,
    pub mean_coverage: f64,
    /// Mean best-so-far value per eval, across seeds (for charts).
    pub mean_convergence: Vec<f64>,
    /// Final pool of the first seed (for the scatter chart).
    pub sample_pool: Vec<Vec<f64>>,
}

fn median(mut xs: Vec<usize>) -> Option<usize> {
    if xs.is_empty() {
        return None;
    }
    xs.sort_unstable();
    Some(xs[xs.len() / 2])
}

/// Run `seeds` repetitions of a config on a problem and aggregate.
pub fn summarize(p: &Problem, cfg: &Config, budget: usize, seeds: u64) -> Summary {
    let runs: Vec<RunResult> = (0..seeds).map(|s| run_config(p, cfg, budget, s)).collect();
    let n = seeds as f64;

    let mean_best = runs.iter().map(|r| r.best_value).sum::<f64>() / n;
    let mean_error = runs.iter().map(|r| error(p, r)).sum::<f64>() / n;
    let success_rate = runs.iter().filter(|r| succeeded(p, r)).count() as f64 / n;
    let median_evals = median(runs.iter().filter_map(|r| evals_to_target(p, r)).collect());
    let mean_coverage = runs.iter().map(|r| coverage(p, r) as f64).sum::<f64>() / n;

    let len = runs.iter().map(|r| r.convergence.len()).min().unwrap_or(0);
    let mean_convergence: Vec<f64> = (0..len)
        .map(|i| runs.iter().map(|r| r.convergence[i]).sum::<f64>() / n)
        .collect();

    Summary {
        problem: p.name.to_string(),
        config: cfg.label.clone(),
        mean_best,
        mean_error,
        success_rate,
        median_evals,
        mean_coverage,
        mean_convergence,
        sample_pool: runs
            .into_iter()
            .next()
            .map(|r| r.final_pool)
            .unwrap_or_default(),
    }
}

/// The default config grid: ClassicPopulation vs SoftStochastic, a few params each.
pub fn default_configs() -> Vec<Config> {
    let mut v = Vec::new();
    for &(pop, tour, mr) in &[(100usize, 2usize, 0.6f64), (200, 3, 0.8)] {
        v.push(Config {
            label: format!("classic(pop={pop},t={tour},mr={mr})"),
            strat: Strat::Classic(ClassicPopulationInfo {
                pop_size: pop,
                tournament: tour,
                mutation_rate: mr,
                seed: 0,
            }),
            mutation_step: 0.3,
        });
    }
    for &(target, nw, temp) in &[
        (150usize, 0.2f64, 0.1f64),
        (200, 0.5, 0.15),
        (300, 0.7, 0.2),
    ] {
        v.push(Config {
            label: format!("ssea(target={target},nw={nw},T={temp})"),
            strat: Strat::Ssea(SoftStochasticInfo {
                target,
                novelty_weight: nw,
                temperature: temp,
                behavior_weight: 0.0,
                // knn=15: a larger novelty neighbourhood reduces kNN saturation,
                // so SSEA spreads more evenly instead of clumping.
                knn: 15,
                seed: 0,
            }),
            mutation_step: 0.3,
        });
    }
    for &(target, nw, loc) in &[(300usize, 0.6f64, 1.0f64), (300, 0.6, 3.0)] {
        v.push(Config {
            label: format!("speciated(target={target},nw={nw},loc={loc})"),
            strat: Strat::Speciated(SpeciatedSoftStochasticInfo {
                target,
                novelty_weight: nw,
                temperature: 0.2,
                behavior_weight: 0.0,
                knn: 5,
                seed: 0,
                mate_locality: loc,
                archive_prob: 0.05,
                speciation_pressure: 0.6,
            }),
            mutation_step: 0.3,
        });
    }
    for &(target, rw, loc) in &[(300usize, 0.05f64, 2.0f64), (300, 0.1, 2.0)] {
        v.push(Config {
            label: format!("resource(target={target},rw={rw},loc={loc})"),
            strat: Strat::Resource(ResourceSoftStochasticInfo {
                target,
                temperature: 0.2,
                relative_width: rw,
                mate_locality: loc,
                seed: 0,
                ..Default::default()
            }),
            mutation_step: 0.3,
        });
    }
    for &(pop, f, cr) in &[(60usize, 0.6f64, 0.9f64), (100, 0.5, 0.9)] {
        v.push(Config {
            label: format!("de(pop={pop},F={f},CR={cr})"),
            strat: Strat::De(DeInfo {
                pop,
                f,
                cr,
                seed: 0,
            }),
            mutation_step: 0.0,
        });
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problems::all;

    fn problem(name: &str) -> Problem {
        all().into_iter().find(|p| p.name == name).unwrap()
    }

    #[test]
    fn ssea_solves_sphere() {
        let p = problem("sphere");
        let cfg = Config {
            label: "ssea".into(),
            strat: Strat::Ssea(SoftStochasticInfo {
                target: 80,
                novelty_weight: 0.2,
                ..Default::default()
            }),
            mutation_step: 0.3,
        };
        let r = run_config(&p, &cfg, 6000, 7);
        assert!(
            succeeded(&p, &r),
            "error {} not < {}",
            error(&p, &r),
            p.value_eps
        );
        assert!(coverage(&p, &r) >= 1);
    }
}
