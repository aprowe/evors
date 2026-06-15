use rand::Rng;

/// An *instruction set* describing how to create and manipulate genomes of a
/// given type. Implemented on a descriptor struct, NOT on the genome itself,
/// so multiple `Genetic` impls can target the same `Genome` type.
pub trait Genetic {
    type Genome: Clone;

    fn random(&self, rng: &mut impl Rng) -> Self::Genome;
    fn cross(&self, a: &Self::Genome, b: &Self::Genome, rng: &mut impl Rng) -> Self::Genome;
    fn mutate(&self, g: &Self::Genome, rng: &mut impl Rng) -> Self::Genome;
    /// Genotypic distance between two genomes.
    fn gene_dist(&self, a: &Self::Genome, b: &Self::Genome) -> f64;
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    /// Minimal genetics over a single f64 in [0, 1) for testing the trait shape.
    struct UnitGenetics;
    impl Genetic for UnitGenetics {
        type Genome = f64;
        fn random(&self, rng: &mut impl Rng) -> f64 { rng.random::<f64>() }
        fn cross(&self, a: &f64, b: &f64, _rng: &mut impl Rng) -> f64 { (a + b) / 2.0 }
        fn mutate(&self, g: &f64, rng: &mut impl Rng) -> f64 {
            (g + rng.random_range(-0.1..0.1)).clamp(0.0, 1.0)
        }
        fn gene_dist(&self, a: &f64, b: &f64) -> f64 { (a - b).abs() }
    }

    #[test]
    fn ops_are_reproducible_with_same_seed() {
        let g = UnitGenetics;
        let mut r1 = ChaCha8Rng::seed_from_u64(7);
        let mut r2 = ChaCha8Rng::seed_from_u64(7);
        assert_eq!(g.random(&mut r1), g.random(&mut r2));
    }

    #[test]
    fn cross_is_midpoint_and_dist_symmetric() {
        let g = UnitGenetics;
        let mut r = ChaCha8Rng::seed_from_u64(1);
        assert_eq!(g.cross(&0.0, &1.0, &mut r), 0.5);
        assert_eq!(g.gene_dist(&0.2, &0.5), g.gene_dist(&0.5, &0.2));
    }
}
