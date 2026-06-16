//! Gated GPU integration test for sub-project B: CPPN image evolution improves
//! fitness toward a target using SSEA with genetic + solution novelty.

use evors::strategies::SoftStochasticInfo;
use evors_cppn::{CppnParams, EvalConfig, ImageTrial, demo_target};

#[test]
fn image_trial_improves_toward_target() {
    let res = 64u32;
    let target = demo_target(res);
    let mut trial = ImageTrial::new(
        &target,
        res,
        res,
        EvalConfig {
            eval_res: res,
            sig_res: 8,
        },
        CppnParams::default(),
        SoftStochasticInfo {
            target: 200,
            temperature: 0.1,
            novelty_weight: 0.3,
            behavior_weight: 0.3,
            knn: 5,
            seed: 7,
        },
    );

    trial.run(300);
    let early = trial.best_fitness();
    trial.run(2500);
    let late = trial.best_fitness();

    assert!(late >= early, "fitness regressed: {early} -> {late}");
    // Substantial convergence above the ~0.6 random-CPPN floor on this
    // high-frequency target. (The default CppnParams use bold weights, which
    // trade a little match accuracy for richer patterns, so the converged level
    // here sits around 0.77 rather than the old gentle-weight ~0.83.)
    assert!(late > 0.72, "did not converge enough: best {late}");
    assert!(
        late > early + 0.02,
        "did not improve meaningfully: {early} -> {late}"
    );

    // The champion render is a real, non-trivial image (not all black).
    let render = trial.best_render();
    let bright: f32 = render.iter().sum::<f32>() / render.len() as f32;
    assert!(bright > 0.05, "champion render looks blank (mean {bright})");

    // Diversity preserved: the soft pool stays populated, not collapsed to 1.
    assert!(
        trial.pool_size() > 20,
        "pool collapsed to {}",
        trial.pool_size()
    );
}
