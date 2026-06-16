//! The CPPN image score: scalar fitness (image distance) plus a behavior
//! descriptor (downsampled render) so SSEA can blend **genetic** novelty
//! (`gene_dist`) with **solution** novelty (`behavior_dist`).

use evors::{Behavior, Fitness};

#[derive(Clone, Debug)]
pub struct ImgScore {
    /// `1 - MSE` vs the target image, in `[0, 1]`.
    pub fitness: f64,
    /// Downsampled render (`sig_res²×3`) — the solution-space descriptor.
    pub signature: Vec<f32>,
}

impl Fitness for ImgScore {
    fn fitness(&self) -> f64 {
        self.fitness
    }
}

impl Behavior for ImgScore {
    /// Euclidean distance between the two renders' signatures.
    fn behavior_dist(&self, other: &Self) -> f64 {
        let n = self.signature.len().min(other.signature.len());
        let sum: f64 = (0..n)
            .map(|i| (self.signature[i] - other.signature[i]) as f64)
            .map(|d| d * d)
            .sum();
        sum.sqrt()
    }
}
