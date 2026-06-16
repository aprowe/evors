//! Historical markings (innovation numbers) shared across the population.
//!
//! NEAT assigns each structural change a global, stable id so the same change
//! appearing in different genomes lines up during crossover and compatibility
//! distance. The registry lives behind a `RefCell` inside `NeatGenetics`.

use std::collections::HashMap;

pub struct Innovations {
    next_innov: u64,
    next_node: u64,
    /// (from, to) -> innovation id, so identical new edges align.
    conn_ids: HashMap<(u64, u64), u64>,
    /// split of connection (from, to) -> (new_node, innov_in, innov_out), so the
    /// same split in different genomes produces the same node id + innovations.
    split_ids: HashMap<(u64, u64), (u64, u64, u64)>,
}

impl Innovations {
    /// `first_hidden` is the id reserved for the first hidden node (above the
    /// fixed input/bias/output ids).
    pub fn new(first_hidden: u64) -> Self {
        Self {
            next_innov: 0,
            next_node: first_hidden,
            conn_ids: HashMap::new(),
            split_ids: HashMap::new(),
        }
    }

    /// Stable innovation id for an edge `from -> to`.
    pub fn conn_innov(&mut self, from: u64, to: u64) -> u64 {
        if let Some(&id) = self.conn_ids.get(&(from, to)) {
            id
        } else {
            let id = self.next_innov;
            self.next_innov += 1;
            self.conn_ids.insert((from, to), id);
            id
        }
    }

    /// Stable `(new_node, innov_in, innov_out)` for splitting edge `from -> to`.
    pub fn split(&mut self, from: u64, to: u64) -> (u64, u64, u64) {
        if let Some(&v) = self.split_ids.get(&(from, to)) {
            v
        } else {
            let node = self.next_node;
            self.next_node += 1;
            let innov_in = self.next_innov;
            let innov_out = self.next_innov + 1;
            self.next_innov += 2;
            let v = (node, innov_in, innov_out);
            self.split_ids.insert((from, to), v);
            v
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conn_innov_is_stable() {
        let mut inv = Innovations::new(10);
        let a = inv.conn_innov(0, 5);
        let b = inv.conn_innov(1, 5);
        let a2 = inv.conn_innov(0, 5);
        assert_eq!(a, a2, "same edge -> same innovation");
        assert_ne!(a, b, "different edges -> different innovations");
    }

    #[test]
    fn split_is_stable_and_allocates_node() {
        let mut inv = Innovations::new(10);
        let (n1, i1, o1) = inv.split(0, 5);
        let (n2, i2, o2) = inv.split(0, 5);
        assert_eq!((n1, i1, o1), (n2, i2, o2), "same split -> same ids");
        assert_eq!(n1, 10, "first hidden node id");
        let (n3, _, _) = inv.split(1, 5);
        assert_eq!(n3, 11, "next split allocates a fresh node");
    }
}
