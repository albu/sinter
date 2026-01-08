// Cost model for kernel fusion decisions
//
// Helps determine whether fusing two kernels is beneficial based on
// the computational cost of the fused vs sequential approaches.

use super::fusion::KernelData;

/// Cost model for kernel fusion decisions
///
/// Provides static methods to evaluate whether fusing kernels is beneficial
/// based on the number of operations required.
pub struct KernelCostModel;

impl KernelCostModel {
    /// Calculate the cost of applying kernels sequentially
    ///
    /// Returns the approximate number of operations per pixel.
    ///
    /// # Cost Model
    /// - Size2D kernels: width * height operations per pixel
    /// - Separable kernels: 2 * kernel_length operations per pixel (horizontal + vertical)
    pub fn sequential_cost(k1: &KernelData, k2: &KernelData) -> usize {
        match (k1, k2) {
            (KernelData::Size2D { width: w1, height: h1, .. },
             KernelData::Size2D { width: w2, height: h2, .. }) => {
                w1 * h1 + w2 * h2
            }
            (KernelData::Size2D { width: w, height: h, .. },
             KernelData::Separable { horizontal: h_sep, .. }) => {
                w * h + 2 * h_sep.len()
            }
            (KernelData::Separable { horizontal: h1, .. },
             KernelData::Size2D { width: w, height: h, .. }) => {
                2 * h1.len() + w * h
            }
            (KernelData::Separable { horizontal: h1, .. },
             KernelData::Separable { horizontal: h2, .. }) => {
                2 * h1.len() + 2 * h2.len()
            }
        }
    }

    /// Calculate the cost of applying a fused kernel
    ///
    /// Returns the approximate number of operations per pixel if fused.
    ///
    /// # Cost Model
    /// - Size2D + Size2D: (w1+w2-1) * (h1+h2-1) operations per pixel (2D convolution result)
    /// - Separable + Separable: 2 * (len1 + len2 - 1) operations per pixel (composed 1D kernels)
    /// - Mixed: usize::MAX (not recommended to fuse)
    pub fn fused_cost(k1: &KernelData, k2: &KernelData) -> usize {
        match (k1, k2) {
            (KernelData::Size2D { width: w1, height: h1, .. },
             KernelData::Size2D { width: w2, height: h2, .. }) => {
                // Fused size: (w1+w2-1) x (h1+h2-1)
                (w1 + w2 - 1) * (h1 + h2 - 1)
            }
            (KernelData::Separable { horizontal: h1, .. },
             KernelData::Separable { horizontal: h2, .. }) => {
                // Fused separable: 2 * (len1 + len2 - 1)
                2 * (h1.len() + h2.len() - 1)
            }
            _ => {
                // Mixed 2D + separable fusion is not efficient
                // Converting separable to 2D is expensive
                usize::MAX
            }
        }
    }

    /// Decide whether to fuse based on cost
    ///
    /// Returns true if fusion reduces operations or has equal cost.
    /// Equal cost still provides benefits (fewer allocations, better cache locality).
    pub fn should_fuse(k1: &KernelData, k2: &KernelData) -> bool {
        let seq_cost = Self::sequential_cost(k1, k2);
        let fused_cost = Self::fused_cost(k1, k2);

        // Fuse if it reduces operations or has equal cost
        fused_cost <= seq_cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_cost_2d_2d() {
        let k1 = KernelData::Size2D {
            kernel: vec![0; 9],
            width: 3,
            height: 3,
            scale: 1,
            offset: 0,
        };
        let k2 = KernelData::Size2D {
            kernel: vec![0; 9],
            width: 3,
            height: 3,
            scale: 1,
            offset: 0,
        };

        assert_eq!(KernelCostModel::sequential_cost(&k1, &k2), 18); // 9 + 9
    }

    #[test]
    fn test_fused_cost_2d_2d() {
        let k1 = KernelData::Size2D {
            kernel: vec![0; 9],
            width: 3,
            height: 3,
            scale: 1,
            offset: 0,
        };
        let k2 = KernelData::Size2D {
            kernel: vec![0; 9],
            width: 3,
            height: 3,
            scale: 1,
            offset: 0,
        };

        // 3x3 * 3x3 = 5x5 = 25 ops
        assert_eq!(KernelCostModel::fused_cost(&k1, &k2), 25);
    }

    #[test]
    fn test_should_fuse_2d_2d() {
        let k1 = KernelData::Size2D {
            kernel: vec![0; 9],
            width: 3,
            height: 3,
            scale: 1,
            offset: 0,
        };
        let k2 = KernelData::Size2D {
            kernel: vec![0; 9],
            width: 3,
            height: 3,
            scale: 1,
            offset: 0,
        };

        // Sequential: 18 ops, Fused: 25 ops
        // Should NOT fuse (fused is more expensive)
        assert!(!KernelCostModel::should_fuse(&k1, &k2));
    }

    #[test]
    fn test_sequential_cost_separable_separable() {
        let k1 = KernelData::Separable {
            horizontal: vec![0; 7],
            vertical: vec![0; 7],
            scale: 64,
            offset: 0,
        };
        let k2 = KernelData::Separable {
            horizontal: vec![0; 7],
            vertical: vec![0; 7],
            scale: 64,
            offset: 0,
        };

        assert_eq!(KernelCostModel::sequential_cost(&k1, &k2), 28); // 14 + 14
    }

    #[test]
    fn test_fused_cost_separable_separable() {
        let k1 = KernelData::Separable {
            horizontal: vec![0; 7],
            vertical: vec![0; 7],
            scale: 64,
            offset: 0,
        };
        let k2 = KernelData::Separable {
            horizontal: vec![0; 7],
            vertical: vec![0; 7],
            scale: 64,
            offset: 0,
        };

        // 7x7 + 7x7 separable = 13x13 separable = 2 * 13 = 26 ops
        assert_eq!(KernelCostModel::fused_cost(&k1, &k2), 26);
    }

    #[test]
    fn test_should_fuse_separable_separable() {
        let k1 = KernelData::Separable {
            horizontal: vec![0; 7],
            vertical: vec![0; 7],
            scale: 64,
            offset: 0,
        };
        let k2 = KernelData::Separable {
            horizontal: vec![0; 7],
            vertical: vec![0; 7],
            scale: 64,
            offset: 0,
        };

        // Sequential: 28 ops, Fused: 26 ops
        // Should fuse (fused is cheaper)
        assert!(KernelCostModel::should_fuse(&k1, &k2));
    }

    #[test]
    fn test_should_not_fuse_mixed() {
        let k1 = KernelData::Size2D {
            kernel: vec![0; 9],
            width: 3,
            height: 3,
            scale: 1,
            offset: 0,
        };
        let k2 = KernelData::Separable {
            horizontal: vec![0; 7],
            vertical: vec![0; 7],
            scale: 64,
            offset: 0,
        };

        // Mixed fusion is not efficient
        assert!(!KernelCostModel::should_fuse(&k1, &k2));
    }

    #[test]
    fn test_cost_example_gaussian_3x3_twice() {
        // Two 3x3 Gaussian blurs
        let k1 = KernelData::Size2D {
            kernel: vec![1, 2, 1, 2, 4, 2, 1, 2, 1],
            width: 3,
            height: 3,
            scale: 16,
            offset: 0,
        };
        let k2 = k1.clone();

        // Sequential: 9 + 9 = 18 ops
        // Fused: 5x5 = 25 ops
        // Don't fuse (fused is more expensive)
        assert_eq!(KernelCostModel::sequential_cost(&k1, &k2), 18);
        assert_eq!(KernelCostModel::fused_cost(&k1, &k2), 25);
        assert!(!KernelCostModel::should_fuse(&k1, &k2));
    }

    #[test]
    fn test_cost_example_gaussian_7x7_twice() {
        // Two 7x7 Gaussian blurs
        let k1 = KernelData::Separable {
            horizontal: vec![1, 6, 15, 20, 15, 6, 1],
            vertical: vec![1, 6, 15, 20, 15, 6, 1],
            scale: 64,
            offset: 0,
        };
        let k2 = k1.clone();

        // Sequential: 14 + 14 = 28 ops
        // Fused: 2 * 13 = 26 ops
        // Fuse! (fused is cheaper)
        assert_eq!(KernelCostModel::sequential_cost(&k1, &k2), 28);
        assert_eq!(KernelCostModel::fused_cost(&k1, &k2), 26);
        assert!(KernelCostModel::should_fuse(&k1, &k2));
    }
}
