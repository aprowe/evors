//! Run a strategy on a problem and compute metrics.

use crate::genetics::VecGenetics;
use crate::problems::Problem;
use evors::genetic::Genetic;
use evors::population::PopulationView;
use evors::strategies::{
    ClassicPopulation, ClassicPopulationInfo, SoftStochastic, SoftStochasticInfo,
};
use evors::strategy::EvolutionStrategy;
use evors::{run_observed, RecordingObserver, RunConfig, Stop};

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
    let cfg = RunConfig { stop: Stop::Trials(budget), batch_size: 1, max_trials: None };
    let mut rec = RecordingObserver::new();
    let objective = p.objective;
    let out = run_observed(&cfg, strat, move |x: &Vec<f64>| -objective(x), &mut rec);

    // Recorded fitness is -objective; convert to minimized objective value.
    let convergence: Vec<f64> = rec.best_fitness.iter().map(|f| -f).collect();
    let best_value = out.best_score.map(|f| -f).unwrap_or(f64::INFINITY);
    let best_x = out.best_genome.unwrap_or_default();
    RunResult { convergence, final_pool: strat.members(), best_x, best_value }
}

/// Run a config on a problem at a given seed.
pub fn run_config(p: &Problem, cfg: &Config, budget: usize, seed: u64) -> RunResult {
    match &cfg.strat {
        Strat::Classic(info) => {
            let genetics = VecGenetics::new(p.bounds.clone(), cfg.mutation_step);
            let mut info = info.clone();
            info.seed = seed;
            let mut strat = ClassicPopulation::new(genetics, info);
            run_one(p, &mut strat, budget)
        }
        Strat::Ssea(info) => {
            let genetics = VecGenetics::new(p.bounds.clone(), cfg.mutation_step);
            let mut info = info.clone();
            info.seed = seed;
            let mut strat = SoftStochastic::new(genetics, info);
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
    r.convergence.iter().position(|v| v - p.global_min < p.value_eps)
}

/// Distinct local minima with at least one final-pool member within `basin_eps`.
pub fn coverage(p: &Problem, r: &RunResult) -> usize {
    p.local_minima
        .iter()
        .filter(|m| {
            r.final_pool.iter().any(|x| {
                x.iter().zip(m.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt() < p.basin_eps
            })
        })
        .count()
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
                target: 80, novelty_weight: 0.2, ..Default::default()
            }),
            mutation_step: 0.3,
        };
        let r = run_config(&p, &cfg, 6000, 7);
        assert!(succeeded(&p, &r), "error {} not < {}", error(&p, &r), p.value_eps);
        assert!(coverage(&p, &r) >= 1);
    }
}
