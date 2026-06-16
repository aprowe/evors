//! `NeatGenetics` — the NEAT instruction set, implementing [`evors::Genetic`].

use crate::genome::{ConnGene, NeatGenome, NodeGene, NodeKind};
use crate::innovation::Innovations;
use evors::Genetic;
use rand::Rng;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

/// Tunable NEAT parameters (CreateInfo style).
#[derive(Clone)]
pub struct NeatParams {
    /// Probability a genome undergoes weight perturbation on mutate.
    pub weight_mutate_prob: f64,
    /// Uniform half-range of a weight perturbation step.
    pub weight_perturb: f64,
    /// Probability a perturbed weight is reset to a fresh random value.
    pub weight_reset_prob: f64,
    /// Half-range for initial / reset weights.
    pub weight_init_range: f64,
    /// Probability of an add-connection structural mutation.
    pub add_conn_prob: f64,
    /// Probability of an add-node structural mutation.
    pub add_node_prob: f64,
    /// Compatibility coefficients (excess, disjoint, mean weight diff).
    pub c_excess: f64,
    pub c_disjoint: f64,
    pub c_weight: f64,
}

impl Default for NeatParams {
    fn default() -> Self {
        Self {
            weight_mutate_prob: 0.8,
            weight_perturb: 0.5,
            weight_reset_prob: 0.1,
            weight_init_range: 1.0,
            add_conn_prob: 0.08,
            add_node_prob: 0.03,
            c_excess: 1.0,
            c_disjoint: 1.0,
            c_weight: 0.4,
        }
    }
}

/// NEAT genetics over a fixed number of inputs/outputs. Owns the shared
/// innovation registry behind a `RefCell` (strategy ops are sequential).
pub struct NeatGenetics {
    inputs: usize,
    outputs: usize,
    params: NeatParams,
    innov: RefCell<Innovations>,
}

impl NeatGenetics {
    pub fn new(inputs: usize, outputs: usize, params: NeatParams) -> Self {
        let first_hidden = (inputs + outputs + 1) as u64; // +1 for the bias node
        Self {
            inputs,
            outputs,
            params,
            innov: RefCell::new(Innovations::new(first_hidden)),
        }
    }

    fn bias_id(&self) -> u64 {
        self.inputs as u64
    }
    fn input_ids(&self) -> impl Iterator<Item = u64> {
        0..self.inputs as u64
    }
    fn output_ids(&self) -> impl Iterator<Item = u64> + use<> {
        let start = self.inputs as u64 + 1;
        start..start + self.outputs as u64
    }

    /// The fixed (always-present) node genes: inputs, bias, outputs.
    fn fixed_nodes(&self) -> Vec<NodeGene> {
        let mut v: Vec<NodeGene> = self
            .input_ids()
            .map(|id| NodeGene {
                id,
                kind: NodeKind::Input,
            })
            .collect();
        v.push(NodeGene {
            id: self.bias_id(),
            kind: NodeKind::Bias,
        });
        v.extend(self.output_ids().map(|id| NodeGene {
            id,
            kind: NodeKind::Output,
        }));
        v
    }

    fn rand_weight(&self, rng: &mut impl Rng) -> f64 {
        rng.random_range(-self.params.weight_init_range..self.params.weight_init_range)
    }

    fn perturb_weights(&self, g: &mut NeatGenome, rng: &mut impl Rng) {
        for c in g.conns.iter_mut() {
            if rng.random_bool(self.params.weight_reset_prob) {
                c.weight = self.rand_weight(rng);
            } else {
                c.weight = (c.weight
                    + rng.random_range(-self.params.weight_perturb..self.params.weight_perturb))
                .clamp(-30.0, 30.0);
            }
        }
    }

    fn add_connection(&self, g: &mut NeatGenome, rng: &mut impl Rng) {
        // Source: any non-output node. Target: any non-input, non-bias node.
        let sources: Vec<u64> = g
            .nodes
            .iter()
            .filter(|n| n.kind != NodeKind::Output)
            .map(|n| n.id)
            .collect();
        let targets: Vec<u64> = g
            .nodes
            .iter()
            .filter(|n| matches!(n.kind, NodeKind::Hidden | NodeKind::Output))
            .map(|n| n.id)
            .collect();
        if sources.is_empty() || targets.is_empty() {
            return;
        }
        for _ in 0..20 {
            let from = sources[rng.random_range(0..sources.len())];
            let to = targets[rng.random_range(0..targets.len())];
            if from == to || g.has_conn(from, to) {
                continue;
            }
            // Feed-forward: reject if `to` can already reach `from` (would cycle).
            if reachable(to, from, &g.conns) {
                continue;
            }
            let innov = self.innov.borrow_mut().conn_innov(from, to);
            let weight = self.rand_weight(rng);
            g.conns.push(ConnGene {
                innov,
                from,
                to,
                weight,
                enabled: true,
            });
            return;
        }
    }

    fn add_node(&self, g: &mut NeatGenome, rng: &mut impl Rng) {
        let enabled: Vec<usize> = g
            .conns
            .iter()
            .enumerate()
            .filter(|(_, c)| c.enabled)
            .map(|(i, _)| i)
            .collect();
        if enabled.is_empty() {
            return;
        }
        let idx = enabled[rng.random_range(0..enabled.len())];
        let (from, to, weight) = {
            let c = &mut g.conns[idx];
            c.enabled = false;
            (c.from, c.to, c.weight)
        };
        let (node, innov_in, innov_out) = self.innov.borrow_mut().split(from, to);
        if g.node_kind(node).is_none() {
            g.nodes.push(NodeGene {
                id: node,
                kind: NodeKind::Hidden,
            });
        }
        g.conns.push(ConnGene {
            innov: innov_in,
            from,
            to: node,
            weight: 1.0,
            enabled: true,
        });
        g.conns.push(ConnGene {
            innov: innov_out,
            from: node,
            to,
            weight,
            enabled: true,
        });
    }
}

impl Genetic for NeatGenetics {
    type Genome = NeatGenome;

    fn random(&self, rng: &mut impl Rng) -> NeatGenome {
        let nodes = self.fixed_nodes();
        let mut conns = Vec::new();
        let sources: Vec<u64> = self
            .input_ids()
            .chain(std::iter::once(self.bias_id()))
            .collect();
        for from in sources {
            for to in self.output_ids() {
                let innov = self.innov.borrow_mut().conn_innov(from, to);
                conns.push(ConnGene {
                    innov,
                    from,
                    to,
                    weight: self.rand_weight(rng),
                    enabled: true,
                });
            }
        }
        NeatGenome { nodes, conns }
    }

    fn mutate(&self, g: &NeatGenome, rng: &mut impl Rng) -> NeatGenome {
        let mut child = g.clone();
        if rng.random_bool(self.params.weight_mutate_prob) {
            self.perturb_weights(&mut child, rng);
        }
        if rng.random_bool(self.params.add_conn_prob) {
            self.add_connection(&mut child, rng);
        }
        if rng.random_bool(self.params.add_node_prob) {
            self.add_node(&mut child, rng);
        }
        child
    }

    fn cross(&self, a: &NeatGenome, b: &NeatGenome, rng: &mut impl Rng) -> NeatGenome {
        // `a` is the fitter parent. Matching genes come from a random parent;
        // disjoint/excess genes come from `a` only.
        let b_by_innov: HashMap<u64, &ConnGene> = b.conns.iter().map(|c| (c.innov, c)).collect();
        let mut conns: Vec<ConnGene> = Vec::with_capacity(a.conns.len());
        for ca in &a.conns {
            let chosen = match b_by_innov.get(&ca.innov) {
                Some(cb) => {
                    // Matching: take from a random parent; re-enable with 0.75
                    // if disabled in either.
                    let mut gene = if rng.random_bool(0.5) { *ca } else { **cb };
                    if (!ca.enabled || !cb.enabled) && rng.random_bool(0.75) {
                        gene.enabled = true;
                    }
                    gene
                }
                None => *ca, // disjoint/excess from fitter parent
            };
            conns.push(chosen);
        }

        // Child nodes: the fixed I/O/bias nodes plus any hidden node referenced.
        let mut nodes = self.fixed_nodes();
        let present: HashSet<u64> = nodes.iter().map(|n| n.id).collect();
        let mut added: HashSet<u64> = HashSet::new();
        for c in &conns {
            for id in [c.from, c.to] {
                if !present.contains(&id) && added.insert(id) {
                    let kind = a
                        .node_kind(id)
                        .or_else(|| b.node_kind(id))
                        .unwrap_or(NodeKind::Hidden);
                    nodes.push(NodeGene { id, kind });
                }
            }
        }
        NeatGenome { nodes, conns }
    }

    fn gene_dist(&self, a: &NeatGenome, b: &NeatGenome) -> f64 {
        let a_by: HashMap<u64, &ConnGene> = a.conns.iter().map(|c| (c.innov, c)).collect();
        let b_by: HashMap<u64, &ConnGene> = b.conns.iter().map(|c| (c.innov, c)).collect();
        let max_a = a.conns.iter().map(|c| c.innov).max().unwrap_or(0);
        let max_b = b.conns.iter().map(|c| c.innov).max().unwrap_or(0);

        let mut excess = 0.0;
        let mut disjoint = 0.0;
        let mut wdiff_sum = 0.0;
        let mut matching = 0.0;

        for (innov, ca) in &a_by {
            match b_by.get(innov) {
                Some(cb) => {
                    matching += 1.0;
                    wdiff_sum += (ca.weight - cb.weight).abs();
                }
                None => {
                    if *innov > max_b {
                        excess += 1.0;
                    } else {
                        disjoint += 1.0;
                    }
                }
            }
        }
        for innov in b_by.keys() {
            if !a_by.contains_key(innov) {
                if *innov > max_a {
                    excess += 1.0;
                } else {
                    disjoint += 1.0;
                }
            }
        }

        let n = a.conns.len().max(b.conns.len()).max(1) as f64;
        let wbar = if matching > 0.0 {
            wdiff_sum / matching
        } else {
            0.0
        };
        self.params.c_excess * excess / n
            + self.params.c_disjoint * disjoint / n
            + self.params.c_weight * wbar
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn has_cycle(g: &NeatGenome) -> bool {
        // A node reachable from itself over enabled edges => cycle.
        g.nodes.iter().any(|n| {
            g.conns
                .iter()
                .filter(|c| c.enabled && c.from == n.id)
                .any(|c| reachable(c.to, n.id, &g.conns))
        })
    }

    #[test]
    fn random_is_minimal_and_well_formed() {
        let ng = NeatGenetics::new(2, 1, NeatParams::default());
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let g = ng.random(&mut rng);
        assert_eq!(g.hidden_count(), 0, "no hidden nodes initially");
        // (2 inputs + bias) x 1 output = 3 connections.
        assert_eq!(g.conns.len(), 3);
        assert_eq!(g.nodes.len(), 4); // 2 inputs + bias + output
    }

    #[test]
    fn mutation_stays_feed_forward() {
        let ng = NeatGenetics::new(
            3,
            2,
            NeatParams {
                add_conn_prob: 0.9,
                add_node_prob: 0.9,
                ..NeatParams::default()
            },
        );
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut g = ng.random(&mut rng);
        for _ in 0..50 {
            g = ng.mutate(&g, &mut rng);
            assert!(!has_cycle(&g), "mutation introduced a cycle");
        }
        assert!(
            g.hidden_count() > 0,
            "structural mutation should add hidden nodes"
        );
    }

    #[test]
    fn gene_dist_zero_for_identical_grows_with_structure() {
        let ng = NeatGenetics::new(
            2,
            1,
            NeatParams {
                add_node_prob: 1.0,
                ..NeatParams::default()
            },
        );
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let g = ng.random(&mut rng);
        assert_eq!(ng.gene_dist(&g, &g), 0.0);
        let mutated = ng.mutate(&g, &mut rng);
        assert!(
            ng.gene_dist(&g, &mutated) > 0.0,
            "structural change should raise distance"
        );
    }

    #[test]
    fn cross_child_is_well_formed() {
        let ng = NeatGenetics::new(
            2,
            1,
            NeatParams {
                add_node_prob: 0.5,
                add_conn_prob: 0.5,
                ..NeatParams::default()
            },
        );
        let mut rng = ChaCha8Rng::seed_from_u64(4);
        let base = ng.random(&mut rng);
        let a = ng.mutate(&ng.mutate(&base, &mut rng), &mut rng);
        let b = ng.mutate(&base, &mut rng);
        let child = ng.cross(&a, &b, &mut rng);
        // Every connection's endpoints must have a node gene.
        for c in &child.conns {
            assert!(
                child.node_kind(c.from).is_some(),
                "missing from-node {}",
                c.from
            );
            assert!(child.node_kind(c.to).is_some(), "missing to-node {}", c.to);
        }
        assert!(!has_cycle(&child), "crossover produced a cycle");
    }
}

/// Whether `target` is reachable from `start` over enabled connections.
fn reachable(start: u64, target: u64, conns: &[ConnGene]) -> bool {
    let mut stack = vec![start];
    let mut seen = HashSet::new();
    while let Some(node) = stack.pop() {
        if node == target {
            return true;
        }
        if !seen.insert(node) {
            continue;
        }
        for c in conns.iter().filter(|c| c.enabled && c.from == node) {
            stack.push(c.to);
        }
    }
    false
}
