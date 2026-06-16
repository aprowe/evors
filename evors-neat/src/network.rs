//! Phenotype: evaluate a [`NeatGenome`] as a feed-forward network.
//!
//! The genome's acyclic invariant (guaranteed by the mutation operators) lets us
//! activate nodes in a single topological pass.

use crate::genome::{Activation, NeatGenome, NodeKind};
use std::collections::HashMap;

/// Activate `genome` on `inputs` (length must equal the network's input count)
/// and return the output node values in output-id order.
pub fn activate(genome: &NeatGenome, inputs: &[f64]) -> Vec<f64> {
    // Initialise node values: inputs from the argument, bias = 1.0, rest = 0.
    let mut value: HashMap<u64, f64> = HashMap::new();
    let mut input_ids: Vec<u64> = genome
        .nodes
        .iter()
        .filter(|n| n.kind == NodeKind::Input)
        .map(|n| n.id)
        .collect();
    input_ids.sort_unstable();
    for (slot, id) in input_ids.iter().enumerate() {
        value.insert(*id, inputs.get(slot).copied().unwrap_or(0.0));
    }
    for n in &genome.nodes {
        match n.kind {
            NodeKind::Bias => {
                value.insert(n.id, 1.0);
            }
            NodeKind::Input => {}
            _ => {
                value.entry(n.id).or_insert(0.0);
            }
        }
    }

    // Incoming enabled connections per node.
    let mut incoming: HashMap<u64, Vec<(u64, f64)>> = HashMap::new();
    for c in genome.conns.iter().filter(|c| c.enabled) {
        incoming.entry(c.to).or_default().push((c.from, c.weight));
    }
    let activation: HashMap<u64, Activation> =
        genome.nodes.iter().map(|n| (n.id, n.activation)).collect();

    // Topological order over enabled edges (Kahn's algorithm).
    let order = topo_order(genome);

    for id in order {
        let kind = genome.node_kind(id);
        if matches!(kind, Some(NodeKind::Input) | Some(NodeKind::Bias)) {
            continue;
        }
        if let Some(ins) = incoming.get(&id) {
            let sum: f64 = ins
                .iter()
                .map(|(from, w)| value.get(from).copied().unwrap_or(0.0) * w)
                .sum();
            let act = activation.get(&id).copied().unwrap_or(Activation::Sigmoid);
            value.insert(id, act.apply(sum));
        }
    }

    let mut out_ids: Vec<u64> = genome
        .nodes
        .iter()
        .filter(|n| n.kind == NodeKind::Output)
        .map(|n| n.id)
        .collect();
    out_ids.sort_unstable();
    out_ids
        .iter()
        .map(|id| value.get(id).copied().unwrap_or(0.0))
        .collect()
}

/// Topological order of node ids over the genome's enabled connections.
/// Falls back to appending any nodes not reached (defensive; the graph is acyclic).
fn topo_order(genome: &NeatGenome) -> Vec<u64> {
    let mut indeg: HashMap<u64, usize> = genome.nodes.iter().map(|n| (n.id, 0)).collect();
    let mut adj: HashMap<u64, Vec<u64>> = HashMap::new();
    for c in genome.conns.iter().filter(|c| c.enabled) {
        *indeg.entry(c.to).or_insert(0) += 1;
        adj.entry(c.from).or_default().push(c.to);
    }
    let mut queue: Vec<u64> = indeg
        .iter()
        .filter(|(_, d)| **d == 0)
        .map(|(id, _)| *id)
        .collect();
    queue.sort_unstable();
    let mut order = Vec::with_capacity(genome.nodes.len());
    let mut i = 0;
    while i < queue.len() {
        let id = queue[i];
        i += 1;
        order.push(id);
        if let Some(outs) = adj.get(&id) {
            let mut outs = outs.clone();
            outs.sort_unstable();
            for to in outs {
                let d = indeg.get_mut(&to).unwrap();
                *d -= 1;
                if *d == 0 {
                    queue.push(to);
                }
            }
        }
    }
    order
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::{ConnGene, NodeGene};

    fn node(id: u64, kind: NodeKind) -> NodeGene {
        NodeGene {
            id,
            kind,
            activation: Activation::Sigmoid,
        }
    }

    #[test]
    fn activates_a_hand_built_or_gate() {
        // 2 inputs (0,1) + bias (2) -> output (3). Weights make an OR-ish gate.
        let genome = NeatGenome {
            nodes: vec![
                node(0, NodeKind::Input),
                node(1, NodeKind::Input),
                node(2, NodeKind::Bias),
                node(3, NodeKind::Output),
            ],
            conns: vec![
                ConnGene {
                    innov: 0,
                    from: 0,
                    to: 3,
                    weight: 10.0,
                    enabled: true,
                },
                ConnGene {
                    innov: 1,
                    from: 1,
                    to: 3,
                    weight: 10.0,
                    enabled: true,
                },
                ConnGene {
                    innov: 2,
                    from: 2,
                    to: 3,
                    weight: -5.0,
                    enabled: true,
                },
            ],
        };
        let off = activate(&genome, &[0.0, 0.0])[0];
        let on = activate(&genome, &[1.0, 0.0])[0];
        assert!(off < 0.5, "0,0 -> {off}");
        assert!(on > 0.5, "1,0 -> {on}");
    }
}
