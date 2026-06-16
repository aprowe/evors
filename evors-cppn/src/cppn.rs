//! `CppnGenetics` — CPPN-NEAT genetics: a NEAT genome with 3 inputs (x, y, d) +
//! bias and 3 outputs (r, g, b), whose hidden/output nodes carry varied
//! activation functions. Thin wrapper over `evors_neat::NeatGenetics` configured
//! with a full activation set.

use evors::Genetic;
use evors_neat::{Activation, NeatGenetics, NeatGenome, NeatParams};
use rand::Rng;

/// CreateInfo for `CppnGenetics`. Wraps the NEAT structural params; the default
/// enables the full CPPN activation palette and activation mutation.
pub struct CppnParams {
    pub neat: NeatParams,
}

impl Default for CppnParams {
    fn default() -> Self {
        let neat = NeatParams {
            add_node_prob: 0.04,
            // CPPN patterns need bold weights, else everything is mid-gray gradients.
            weight_init_range: 3.0,
            weight_perturb: 1.2,
            activation_mutate_prob: 0.1,
            output_activation: Activation::Sigmoid,
            hidden_activations: vec![
                Activation::Identity,
                Activation::Sigmoid,
                Activation::Tanh,
                Activation::Sin,
                Activation::Gauss,
                Activation::Abs,
            ],
            ..NeatParams::default()
        };
        Self { neat }
    }
}

/// CPPN-NEAT genetics. `Genome = NeatGenome`.
pub struct CppnGenetics {
    inner: NeatGenetics,
}

impl CppnGenetics {
    /// Input node ids in slot order: x, y, d, bias.
    pub const INPUT_IDS: [u64; 4] = [0, 1, 2, 3];
    /// Output node ids: r, g, b.
    pub const OUTPUT_IDS: [u64; 3] = [4, 5, 6];

    pub fn new(params: CppnParams) -> Self {
        // 3 explicit inputs (x, y, d) + an automatic bias node = 4 input slots;
        // 3 outputs (r, g, b).
        Self {
            inner: NeatGenetics::new(3, 3, params.neat),
        }
    }

    /// Live tuning of structural growth (e.g. from a GUI).
    pub fn set_add_node_prob(&mut self, p: f64) {
        self.inner.set_add_node_prob(p);
    }
    pub fn set_add_conn_prob(&mut self, p: f64) {
        self.inner.set_add_conn_prob(p);
    }
    pub fn set_weight_strength(&mut self, s: f64) {
        self.inner.set_weight_strength(s);
    }
}

impl Genetic for CppnGenetics {
    type Genome = NeatGenome;

    fn random(&self, rng: &mut impl Rng) -> NeatGenome {
        self.inner.random(rng)
    }
    fn cross(&self, a: &NeatGenome, b: &NeatGenome, rng: &mut impl Rng) -> NeatGenome {
        self.inner.cross(a, b, rng)
    }
    fn mutate(&self, g: &NeatGenome, rng: &mut impl Rng) -> NeatGenome {
        self.inner.mutate(g, rng)
    }
    fn gene_dist(&self, a: &NeatGenome, b: &NeatGenome) -> f64 {
        self.inner.gene_dist(a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn random_has_four_inputs_three_outputs() {
        let g = CppnGenetics::new(CppnParams::default());
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let genome = g.random(&mut rng);
        // ids 0,1,2 inputs; 3 bias; 4,5,6 outputs.
        for id in CppnGenetics::INPUT_IDS {
            assert!(genome.node(id).is_some(), "missing input/bias {id}");
        }
        for id in CppnGenetics::OUTPUT_IDS {
            assert!(genome.node(id).is_some(), "missing output {id}");
        }
        // (3 inputs + bias) x 3 outputs = 12 initial connections.
        assert_eq!(genome.conns.len(), 12);
    }

    #[test]
    fn mutation_introduces_varied_activations() {
        let g = CppnGenetics::new(CppnParams::default());
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let mut genome = g.random(&mut rng);
        for _ in 0..200 {
            genome = g.mutate(&genome, &mut rng);
        }
        let acts: std::collections::HashSet<_> =
            genome.nodes.iter().map(|n| n.activation).collect();
        assert!(acts.len() > 1, "expected varied activations, got {acts:?}");
    }
}
