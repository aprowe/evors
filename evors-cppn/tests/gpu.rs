//! Gated GPU validation: requires a real OpenGL 4.3 context. One test owns a
//! single `GpuEvaluator` (one GL thread / event loop per process).

use evors_cppn::cpu_eval::render_cpu;
use evors_cppn::encode::encode;
use evors_cppn::{CppnGenetics, EvalConfig, GpuEvaluator};
use evors_neat::{Activation, ConnGene, NeatGenome, NodeGene, NodeKind};

fn node(id: u64, kind: NodeKind, activation: Activation) -> NodeGene {
    NodeGene {
        id,
        kind,
        activation,
    }
}
fn conn(innov: u64, from: u64, to: u64, weight: f64) -> ConnGene {
    ConnGene {
        innov,
        from,
        to,
        weight,
        enabled: true,
    }
}

/// CPPN with identity outputs: r = x, g = y, b = d.
fn gradient_genome() -> NeatGenome {
    NeatGenome {
        nodes: vec![
            node(0, NodeKind::Input, Activation::Identity),
            node(1, NodeKind::Input, Activation::Identity),
            node(2, NodeKind::Input, Activation::Identity),
            node(3, NodeKind::Bias, Activation::Identity),
            node(4, NodeKind::Output, Activation::Identity),
            node(5, NodeKind::Output, Activation::Identity),
            node(6, NodeKind::Output, Activation::Identity),
        ],
        conns: vec![conn(0, 0, 4, 1.0), conn(1, 1, 5, 1.0), conn(2, 2, 6, 1.0)],
    }
}

/// CPPN exercising Sin and Gauss hidden nodes.
fn activation_genome() -> NeatGenome {
    NeatGenome {
        nodes: vec![
            node(0, NodeKind::Input, Activation::Identity),
            node(1, NodeKind::Input, Activation::Identity),
            node(2, NodeKind::Input, Activation::Identity),
            node(3, NodeKind::Bias, Activation::Identity),
            node(7, NodeKind::Hidden, Activation::Sin),
            node(8, NodeKind::Hidden, Activation::Gauss),
            node(4, NodeKind::Output, Activation::Identity),
            node(5, NodeKind::Output, Activation::Identity),
            node(6, NodeKind::Output, Activation::Identity),
        ],
        conns: vec![
            conn(10, 0, 7, 1.5), // x -> sin hidden
            conn(11, 7, 4, 1.0), // -> r
            conn(12, 1, 8, 1.0), // y -> gauss hidden
            conn(13, 8, 5, 1.0), // -> g
            conn(14, 2, 6, 1.0), // d -> b
        ],
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn gpu_matches_cpu_and_fitness_behaves() {
    let cfg = EvalConfig {
        eval_res: 64,
        sig_res: 8,
    };
    let res = cfg.eval_res;

    let g = gradient_genome();
    let enc_g = encode(&g, &CppnGenetics::INPUT_IDS, &CppnGenetics::OUTPUT_IDS).unwrap();
    let cpu_g = render_cpu(&enc_g, res);

    // Target = the CPU render of g; since GPU matches CPU, evaluating g should
    // score ~1.0.
    let eval = GpuEvaluator::new(&cpu_g, res, res, cfg);

    // 1. Encoding correctness: GPU render of g matches the CPU reference.
    let gpu_g = eval.render_rgb(&g);
    let d1 = max_abs_diff(&gpu_g, &cpu_g);
    assert!(d1 < 1e-3, "gradient GPU vs CPU max diff {d1}");

    // 2. Activation coverage: Sin/Gauss CPPN matches the CPU reference.
    let s = activation_genome();
    let enc_s = encode(&s, &CppnGenetics::INPUT_IDS, &CppnGenetics::OUTPUT_IDS).unwrap();
    let cpu_s = render_cpu(&enc_s, res);
    let gpu_s = eval.render_rgb(&s);
    let d2 = max_abs_diff(&gpu_s, &cpu_s);
    assert!(d2 < 1e-3, "sin/gauss GPU vs CPU max diff {d2}");

    // 3. Fitness: g matches the target ~perfectly; s renders differently -> lower.
    let fit_g = eval.evaluate(&g).fitness;
    let fit_s = eval.evaluate(&s).fitness;
    assert!(fit_g > 0.99, "self-match fitness {fit_g}");
    assert!(
        fit_s < fit_g,
        "different genome should score lower: s={fit_s} g={fit_g}"
    );

    // Signature length sanity.
    assert_eq!(
        eval.evaluate(&g).signature.len(),
        (cfg.sig_res * cfg.sig_res * 3) as usize
    );
}
