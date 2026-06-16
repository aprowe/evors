//! Progress observation for `run_observed`. The `Progress` event is kept
//! scalar-friendly so sinks can serialize it cheaply (e.g. stream over UDP to a
//! topic, or to CSV). Sinks implement `Observer`.

use crate::strategy::Fitness;
use std::io::Write;

/// One evaluation's worth of progress.
pub struct Progress<'a, G, S> {
    pub trials: usize,
    pub generation: usize,
    pub genome: &'a G,
    pub score: &'a S,
    /// The strategy's best *before* this eval was folded in (may be `None`
    /// early). Sinks that want best-so-far including the current eval should
    /// track it from `score` themselves (see `RecordingObserver`).
    pub best_score: Option<&'a S>,
}

/// A sink for progress events.
pub trait Observer<G, S> {
    fn on_eval(&mut self, p: Progress<'_, G, S>);
}

/// Does nothing; the default for plain `run`.
pub struct NoObserver;
impl<G, S> Observer<G, S> for NoObserver {
    fn on_eval(&mut self, _p: Progress<'_, G, S>) {}
}

/// Records best-so-far fitness per evaluation (drives convergence curves).
#[derive(Default)]
pub struct RecordingObserver {
    pub best_fitness: Vec<f64>,
    cur_best: f64,
    started: bool,
}

impl RecordingObserver {
    pub fn new() -> Self {
        Self {
            best_fitness: Vec::new(),
            cur_best: f64::MIN,
            started: false,
        }
    }
}

impl<G, S: Fitness> Observer<G, S> for RecordingObserver {
    fn on_eval(&mut self, p: Progress<'_, G, S>) {
        let f = p.score.fitness();
        if !self.started || f > self.cur_best {
            self.cur_best = f;
            self.started = true;
        }
        self.best_fitness.push(self.cur_best);
    }
}

/// Streams `trials,fitness,best` rows to any writer. The seed of a future
/// UDP/topic sink — same trait, different transport.
pub struct CsvProgressObserver<W> {
    w: W,
    header: bool,
    cur_best: f64,
}

impl<W: Write> CsvProgressObserver<W> {
    pub fn new(w: W) -> Self {
        Self {
            w,
            header: false,
            cur_best: f64::MIN,
        }
    }
}

impl<W: Write, G, S: Fitness> Observer<G, S> for CsvProgressObserver<W> {
    fn on_eval(&mut self, p: Progress<'_, G, S>) {
        if !self.header {
            let _ = writeln!(self.w, "trials,fitness,best");
            self.header = true;
        }
        let f = p.score.fitness();
        if f > self.cur_best {
            self.cur_best = f;
        }
        let _ = writeln!(self.w, "{},{},{}", p.trials, f, self.cur_best);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genetic::Genetic;
    use crate::strategies::{SoftStochastic, SoftStochasticInfo};
    use crate::trial::{RunConfig, Stop, run_observed};
    use rand::Rng;

    struct UnitGenetics;
    impl Genetic for UnitGenetics {
        type Genome = f64;
        fn random(&self, rng: &mut impl Rng) -> f64 {
            rng.random::<f64>()
        }
        fn cross(&self, a: &f64, b: &f64, _r: &mut impl Rng) -> f64 {
            (a + b) / 2.0
        }
        fn mutate(&self, g: &f64, rng: &mut impl Rng) -> f64 {
            (g + rng.random_range(-0.05..0.05)).clamp(0.0, 1.0)
        }
        fn gene_dist(&self, a: &f64, b: &f64) -> f64 {
            (a - b).abs()
        }
    }

    #[test]
    fn recording_observer_tracks_monotone_best() {
        let mut strat: SoftStochastic<UnitGenetics, f64> = SoftStochastic::new(
            UnitGenetics,
            SoftStochasticInfo {
                target: 20,
                seed: 1,
                ..Default::default()
            },
        );
        let cfg = RunConfig {
            stop: Stop::Trials(300),
            batch_size: 1,
            max_trials: None,
        };
        let mut rec = RecordingObserver::new();
        let _ = run_observed(&cfg, &mut strat, |x: &f64| *x, &mut rec);
        assert_eq!(rec.best_fitness.len(), 300, "one record per eval");
        // best-so-far must be non-decreasing.
        for w in rec.best_fitness.windows(2) {
            assert!(w[1] >= w[0], "best-so-far regressed: {} -> {}", w[0], w[1]);
        }
    }
}
