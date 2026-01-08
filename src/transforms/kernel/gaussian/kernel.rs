// Gaussian kernel generation
//
// Generates 1D Gaussian kernels for separable convolution.
// Kernels are symmetric, normalized to power-of-two scale for fixed-point arithmetic.

/// Generate a 1D Gaussian kernel with the given sigma
///
/// Returns (kernel_coeffs, scale) where:
/// - kernel_coeffs: symmetric kernel as [center, k1, k2, ..., k_radius]
/// - scale: power-of-two divisor (e.g., 256 for Q8.8, 16384 for Q14)
///
/// # Algorithm
/// 1. Radius = ceil(3 * sigma) (covers 99.7% of distribution)
/// 2. Generate Gaussian coefficients: exp(-x² / (2σ²))
/// 3. Normalize to power-of-two scale
/// 4. Enforce symmetry
///
///
/// # Example
///
/// ```text
/// let (kernel, scale) = gaussian_kernel_1d(1.0);
/// // kernel = [center, k1, k2, k3] for 7-tap (center + 3 symmetric pairs)
/// // scale = 16384 (Q14 fixed-point)
/// ```
pub fn gaussian_kernel_1d(sigma: f32) -> (Vec<i32>, i32) {
    // Match OpenCV's kernel size calculation: round(6*sigma) to nearest odd
    // This ensures apples-to-apples comparison
    let ksize = (6.0 * sigma).round() as usize;
    let ksize = if ksize % 2 == 0 { ksize + 1 } else { ksize };
    let ksize = ksize.max(3); // Minimum 3x3

    let radius = ksize / 2;

    // Clamp to reasonable max to prevent overflow
    let radius = radius.min(31); // Max 63-tap kernel

    let size = 2 * radius + 1;

    // Generate raw Gaussian coefficients
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut raw = Vec::with_capacity(size);

    for i in -(radius as isize)..=(radius as isize) {
        let x = i as f32;
        let g = (-x * x / two_sigma_sq).exp();
        raw.push(g);
    }

    // Normalize and scale to Q14 fixed-point (scale = 16384)
    // Q14 gives good precision while staying in i32 range
    const SCALE: f32 = 16384.0;
    let sum: f32 = raw.iter().sum();

    // Return symmetric half + center: [center, k1, k2, ..., k_radius]
    // raw is indexed from -radius to +radius, so:
    // - raw[0] = coefficient at x = -radius
    // - raw[radius] = coefficient at x = 0 (center)
    // - raw[2*radius] = coefficient at x = +radius
    let mut kernel = Vec::with_capacity(radius + 1);
    for i in radius..=(2 * radius) {
        let scaled = (raw[i] * SCALE / sum).round() as i32;
        kernel.push(scaled);
    }

    (kernel, SCALE as i32)
}

/// Calculate the box blur size for approximating a given sigma
///
/// Uses the 3-pass box blur approximation:
/// σ² ≈ (n/12) * (size² - 1) where n=3 passes
///
/// Solves for size:
/// size = sqrt(12σ²/n + 1)
///
/// Returns odd size >= 3
pub fn box_size_for_sigma(sigma: f32, passes: usize) -> usize {
    let n = passes as f32;
    let ideal = ((12.0 * sigma * sigma / n) + 1.0).sqrt();
    let size = ideal.floor() as usize;

    // Force odd and minimum size of 3
    if size < 3 {
        3
    } else if size % 2 == 0 {
        size + 1
    } else {
        size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_kernel_sigma_1_0() {
        let (kernel, scale) = gaussian_kernel_1d(1.0);

        // sigma=1.0 -> radius=ceil(3)=3 -> 7-tap kernel
        assert_eq!(kernel.len(), 4); // center + 3 symmetric pairs
        assert_eq!(scale, 16384);

        // Kernel should be symmetric and decreasing
        assert!(kernel[0] > kernel[1]);
        assert!(kernel[1] > kernel[2]);
        assert!(kernel[2] > kernel[3]);
    }

    #[test]
    fn test_gaussian_kernel_sigma_0_5() {
        let (kernel, scale) = gaussian_kernel_1d(0.5);

        // sigma=0.5 -> ksize=round(6*0.5)=3 -> radius=1 -> 3-tap kernel
        assert_eq!(kernel.len(), 2); // center + 1 symmetric pair
        assert_eq!(scale, 16384);
    }

    #[test]
    fn test_gaussian_kernel_sigma_2_0() {
        let (kernel, scale) = gaussian_kernel_1d(2.0);

        // sigma=2.0 -> ksize=round(12)=12 -> 13 (odd) -> radius=6 -> 13-tap kernel
        assert_eq!(kernel.len(), 7); // center + 6 symmetric pairs
        assert_eq!(scale, 16384);
    }

    #[test]
    fn test_gaussian_kernel_sum() {
        // Kernel should approximately sum to scale
        for sigma in [0.5f32, 1.0, 1.5, 2.0, 3.0] {
            let (kernel, scale) = gaussian_kernel_1d(sigma);

            // Full kernel sum (double the half, subtract center once)
            let full_sum: i32 = kernel.iter().enumerate().map(|(i, &k)| {
                if i == 0 {
                    k // center counted once
                } else {
                    2 * k // symmetric pairs counted twice
                }
            }).sum();

            // Should be very close to scale (within 0.1%)
            let error = (full_sum - scale).abs() as f64 / scale as f64;
            assert!(error < 0.001, "sigma={}: error={}", sigma, error);
        }
    }

    #[test]
    fn test_box_size_for_sigma() {
        // Test common sigma values
        assert_eq!(box_size_for_sigma(1.0, 3), 3);
        assert_eq!(box_size_for_sigma(2.0, 3), 5);
        assert_eq!(box_size_for_sigma(3.0, 3), 7);

        // All sizes should be odd
        for sigma in [0.5f32, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0] {
            let size = box_size_for_sigma(sigma, 3);
            assert!(size >= 3);
            assert_eq!(size % 2, 1, "size should be odd, got {}", size);
        }
    }
}
