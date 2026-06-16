//! The NEAT genome: a list of node genes and connection genes.

/// Role of a node in the network.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeKind {
    Input,
    Bias,
    Output,
    Hidden,
}

/// A node gene.
#[derive(Clone, Copy, Debug)]
pub struct NodeGene {
    pub id: u64,
    pub kind: NodeKind,
}

/// A connection gene. `innov` is the historical marking used to align genes
/// across genomes during crossover and compatibility distance.
#[derive(Clone, Copy, Debug)]
pub struct ConnGene {
    pub innov: u64,
    pub from: u64,
    pub to: u64,
    pub weight: f64,
    pub enabled: bool,
}

/// A NEAT genome: nodes plus connections between them.
#[derive(Clone, Debug)]
pub struct NeatGenome {
    pub nodes: Vec<NodeGene>,
    pub conns: Vec<ConnGene>,
}

impl NeatGenome {
    /// Number of hidden nodes — a simple measure of evolved topology size.
    pub fn hidden_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| n.kind == NodeKind::Hidden)
            .count()
    }

    /// Whether a connection between `from` and `to` already exists (either
    /// direction-sensitive: exact `from -> to`).
    pub fn has_conn(&self, from: u64, to: u64) -> bool {
        self.conns.iter().any(|c| c.from == from && c.to == to)
    }

    pub fn node_kind(&self, id: u64) -> Option<NodeKind> {
        self.nodes.iter().find(|n| n.id == id).map(|n| n.kind)
    }
}
