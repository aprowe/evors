//! `ImageTrial` — drive CPPN-NEAT toward a target image with evors' SSEA, using
//! a score that blends genetic novelty (`gene_dist`) and solution novelty
//! (image-signature distance). Steppable so a GUI can run it in the background.

use crate::cppn::{CppnGenetics, CppnParams};
use crate::gpu::{EvalConfig, GpuEvaluator};
use crate::score::ImgScore;
use evors::strategies::{SoftStochastic, SoftStochasticInfo};
use evors::strategy::EvolutionStrategy;
use evors_neat::NeatGenome;

pub struct ImageTrial {
    evaluator: GpuEvaluator,
    strategy: SoftStochastic<CppnGenetics, ImgScore>,
    best_fitness: f64,
    best_genome: Option<NeatGenome>,
    trials: usize,
}

impl ImageTrial {
    /// Build a trial for `target_rgb` (`w*h*3`, row 0 = bottom).
    pub fn new(
        target_rgb: &[f32],
        w: u32,
        h: u32,
        eval_cfg: EvalConfig,
        cppn: CppnParams,
        ssea: SoftStochasticInfo,
    ) -> Self {
        let evaluator = GpuEvaluator::new(target_rgb, w, h, eval_cfg);
        let strategy = SoftStochastic::new(CppnGenetics::new(cppn), ssea);
        Self {
            evaluator,
            strategy,
            best_fitness: f64::MIN,
            best_genome: None,
            trials: 0,
        }
    }

    /// One ask/tell step: render and score a genome, feed it back.
    pub fn step(&mut self) {
        let genome = self.strategy.next();
        let eval = self.evaluator.evaluate(&genome);
        if eval.fitness > self.best_fitness {
            self.best_fitness = eval.fitness;
            self.best_genome = Some(genome.clone());
        }
        let score = ImgScore {
            fitness: eval.fitness,
            signature: eval.signature,
        };
        self.strategy.report(genome, score);
        self.trials += 1;
    }

    /// Run `n` steps.
    pub fn run(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }

    pub fn best_fitness(&self) -> f64 {
        self.best_fitness
    }
    pub fn trials(&self) -> usize {
        self.trials
    }
    pub fn pool_size(&self) -> usize {
        self.strategy.pool_size()
    }
    pub fn eval_res(&self) -> u32 {
        self.evaluator.eval_res()
    }
    pub fn best_genome(&self) -> Option<&NeatGenome> {
        self.best_genome.as_ref()
    }

    /// Render the current champion to `eval_res² × 3` RGB (row 0 = bottom).
    /// Returns black until at least one genome has been evaluated.
    pub fn best_render(&self) -> Vec<f32> {
        match &self.best_genome {
            Some(g) => self.evaluator.render_rgb(g),
            None => vec![0.0; (self.eval_res() * self.eval_res() * 3) as usize],
        }
    }
}

/// A colorful, CPPN-friendly analytic target image at `res × res` (row 0 =
/// bottom). Useful for headless demos/tests without an image-file dependency.
pub fn demo_target(res: u32) -> Vec<f32> {
    use std::f32::consts::PI;
    let mut out = vec![0.0f32; (res * res * 3) as usize];
    for py in 0..res {
        for px in 0..res {
            let x = 2.0 * (px as f32 + 0.5) / res as f32 - 1.0;
            let y = 2.0 * (py as f32 + 0.5) / res as f32 - 1.0;
            let d = (x * x + y * y).sqrt();
            let i = ((py * res + px) * 3) as usize;
            out[i] = 0.5 + 0.5 * (3.0 * PI * x).sin();
            out[i + 1] = 0.5 + 0.5 * (3.0 * PI * y).cos();
            out[i + 2] = 0.5 + 0.5 * (2.0 * PI * d).sin();
        }
    }
    out
}
