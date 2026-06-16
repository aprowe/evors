//! Encode a `NeatGenome` into flat buffers the interpreter shader walks.
//!
//! Nodes are assigned slots `0..node_count`: the given input ids take slots
//! `0..inputs.len()` in order (so the shader knows which slot is x/y/d/bias),
//! then the rest are topologically ordered so every connection's source slot
//! precedes its target slot.

use evors_neat::NeatGenome;
use std::collections::HashMap;

/// Maximum number of nodes the interpreter shader's local array supports.
pub const MAX_NODES: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodeError {
    TooManyNodes(usize),
    /// A cycle was found (should never happen for feed-forward genomes).
    Cycle,
}

/// Per-node record: activation id + a slice into the connection list.
#[derive(Clone, Copy, Debug)]
pub struct NodeSlot {
    pub activation: u32,
    pub conn_start: u32,
    pub conn_count: u32,
}

/// One incoming connection: the source slot and its weight.
#[derive(Clone, Copy, Debug)]
pub struct ConnSlot {
    pub source: u32,
    pub weight: f32,
}

/// A genome flattened for GPU upload.
#[derive(Debug)]
pub struct Encoded {
    pub node_count: u32,
    pub nodes: Vec<NodeSlot>,
    pub conns: Vec<ConnSlot>,
    pub output_slots: [u32; 3],
}

/// Encode `genome`. `input_ids` are placed at slots `0..n` in the given order
/// (e.g. `[x, y, d, bias]`); `output_ids` are looked up after slotting.
pub fn encode(
    genome: &NeatGenome,
    input_ids: &[u64],
    output_ids: &[u64; 3],
) -> Result<Encoded, EncodeError> {
    if genome.nodes.len() > MAX_NODES {
        return Err(EncodeError::TooManyNodes(genome.nodes.len()));
    }

    // In-degree over enabled connections.
    let mut indeg: HashMap<u64, usize> = genome.nodes.iter().map(|n| (n.id, 0)).collect();
    let mut out_adj: HashMap<u64, Vec<u64>> = HashMap::new();
    for c in genome.conns.iter().filter(|c| c.enabled) {
        *indeg.entry(c.to).or_insert(0) += 1;
        out_adj.entry(c.from).or_default().push(c.to);
    }

    // Slot order: inputs first (fixed order), then Kahn topological order.
    let mut order: Vec<u64> = Vec::with_capacity(genome.nodes.len());
    let mut assigned: HashMap<u64, ()> = HashMap::new();
    let relax = |id: u64, indeg: &mut HashMap<u64, usize>, out_adj: &HashMap<u64, Vec<u64>>| {
        if let Some(outs) = out_adj.get(&id) {
            for to in outs {
                if let Some(d) = indeg.get_mut(to) {
                    *d -= 1;
                }
            }
        }
    };
    for &id in input_ids {
        if genome.node(id).is_some() && assigned.insert(id, ()).is_none() {
            order.push(id);
            relax(id, &mut indeg, &out_adj);
        }
    }
    loop {
        // Pick the lowest-id unassigned node with in-degree 0 (deterministic).
        let next = genome
            .nodes
            .iter()
            .map(|n| n.id)
            .filter(|id| !assigned.contains_key(id) && indeg.get(id).copied().unwrap_or(0) == 0)
            .min();
        match next {
            Some(id) => {
                assigned.insert(id, ());
                order.push(id);
                relax(id, &mut indeg, &out_adj);
            }
            None => break,
        }
    }
    if order.len() != genome.nodes.len() {
        return Err(EncodeError::Cycle);
    }

    let slot_of: HashMap<u64, u32> = order
        .iter()
        .enumerate()
        .map(|(slot, id)| (*id, slot as u32))
        .collect();

    // Incoming connections grouped by target node, in slot order.
    let mut nodes = Vec::with_capacity(order.len());
    let mut conns = Vec::new();
    for &id in &order {
        let activation = genome.node(id).map(|n| n.activation as u32).unwrap_or(0);
        let conn_start = conns.len() as u32;
        for c in genome.conns.iter().filter(|c| c.enabled && c.to == id) {
            let source = slot_of[&c.from];
            debug_assert!(
                source < slot_of[&id],
                "non-feed-forward edge after topo sort"
            );
            conns.push(ConnSlot {
                source,
                weight: c.weight as f32,
            });
        }
        let conn_count = conns.len() as u32 - conn_start;
        nodes.push(NodeSlot {
            activation,
            conn_start,
            conn_count,
        });
    }

    let output_slots = [
        slot_of[&output_ids[0]],
        slot_of[&output_ids[1]],
        slot_of[&output_ids[2]],
    ];

    Ok(Encoded {
        node_count: order.len() as u32,
        nodes,
        conns,
        output_slots,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use evors::Genetic;
    use evors_neat::{NeatGenetics, NeatParams};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn every_connection_points_backward() {
        // A grown genome; after encoding, sources must precede targets.
        let g = NeatGenetics::new(
            3,
            3,
            NeatParams {
                add_node_prob: 0.8,
                add_conn_prob: 0.8,
                ..NeatParams::default()
            },
        );
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let mut genome = g.random(&mut rng);
        for _ in 0..40 {
            genome = g.mutate(&genome, &mut rng);
        }
        let enc = encode(&genome, &[0, 1, 2, 3], &[4, 5, 6]).unwrap();
        for (slot, node) in enc.nodes.iter().enumerate() {
            for k in 0..node.conn_count {
                let c = enc.conns[(node.conn_start + k) as usize];
                assert!(
                    (c.source as usize) < slot,
                    "edge {} -> {slot} not backward",
                    c.source
                );
            }
        }
        assert_eq!(enc.node_count as usize, genome.nodes.len());
    }

    #[test]
    fn rejects_oversized_genome() {
        let mut genome = NeatGenome {
            nodes: Vec::new(),
            conns: Vec::new(),
        };
        for id in 0..(MAX_NODES as u64 + 1) {
            genome.nodes.push(evors_neat::NodeGene {
                id,
                kind: evors_neat::NodeKind::Hidden,
                activation: evors_neat::Activation::Identity,
            });
        }
        assert_eq!(
            encode(&genome, &[], &[0, 1, 2]).unwrap_err(),
            EncodeError::TooManyNodes(MAX_NODES + 1)
        );
    }
}
