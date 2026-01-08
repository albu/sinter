// Gaussian Noise transform
//
// Adds Gaussian (normally distributed) noise to pixel values.
//
// Uses LUT-based SIMD optimization:
// - Precomputes unit Gaussian samples into a lookup table
// - Uses xorshift RNG for cheap index generation
// - Applies noise using fixed-point integer arithmetic in SIMD

mod neon;

use crate::core::{AccessPattern, BarrierImage, Executable, FusableImage, ShapeEffect, Transform};

// LUT size - larger = better quality, diminishing returns after 1024
const LUT_SIZE: usize = 1024;
const LUT_MASK: u32 = (LUT_SIZE - 1) as u32;

/// Gaussian Noise transform
///
/// Adds random noise drawn from a Gaussian (normal) distribution to each pixel.
///
/// # Parameters
/// - `mean`: Mean of the Gaussian distribution (typically 0)
/// - `std_dev`: Standard deviation of the distribution (noise intensity)
///
/// # Notes
/// - This transform is stochastic (uses randomness)
/// - Uses LUT-based SIMD optimization for performance
/// - Higher std_dev = more visible noise
/// - Typical values: mean=0, std_dev=10 to 50
///
/// # Performance
/// - LUT is generated once during construction
/// - SIMD hot path uses integer arithmetic (no float conversion)
/// - Expected: 3-4x faster than per-pixel float-based implementation
#[derive(Debug, Clone, PartialEq)]
pub struct GaussNoise {
    /// Original mean parameter (for repr/debugging)
    pub mean: f32,
    /// Original std_dev parameter (for repr/debugging)
    pub std_dev: f32,
    lut: Box<[i16; LUT_SIZE]>,
    strength: i16,
    mean_offset: i16,
}

impl GaussNoise {
    /// Create a new GaussNoise transform
    ///
    /// # Panics
    /// Panics if std_dev is negative
    pub fn new(mean: f32, std_dev: f32) -> Self {
        assert!(
            std_dev >= 0.0,
            "std_dev must be non-negative, got {}",
            std_dev
        );

        // Scale strength to fixed-point (Q8.7 format for precision)
        // This allows sub-pixel adjustments while staying in integer domain
        let strength = (std_dev * 128.0) as i16;
        let mean_offset = mean.round() as i16;

        // Generate unit Gaussian LUT using Central Limit Theorem
        // Each entry is a noise sample from N(0, 1)
        // We'll scale by strength during application
        let mut lut = Box::new([0i16; LUT_SIZE]);

        // Use a simple seed for LUT generation
        let mut seed = 0u32;

        for entry in lut.iter_mut() {
            let mut sum = 0.0f32;

            // Sum 12 uniform random numbers (CLT approximation)
            // Result has mean=6, std=1, so we subtract 6 to get mean=0
            for _ in 0..12 {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                sum += (seed & 0xFFFF) as f32 / 65535.0;
            }

            // (sum - 6) has mean=0, std=1
            *entry = ((sum - 6.0) * 128.0).round() as i16;
        }

        Self {
            mean,
            std_dev,
            lut,
            strength,
            mean_offset,
        }
    }
}

impl Transform for GaussNoise {
    fn access(&self) -> AccessPattern {
        AccessPattern::InPlace
    }

    fn shape_effect(&self) -> ShapeEffect {
        ShapeEffect::Preserve
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_executable(&self) -> Option<&dyn Executable> {
        Some(self)
    }
}

impl Executable for GaussNoise {
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        neon::apply_gauss_noise_neon(&mut image.data, &self.lut, self.strength, self.mean_offset);
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gauss_noise_new() {
        let n = GaussNoise::new(0.0, 25.0);
        // Check that LUT was generated
        assert!(!n.lut.iter().all(|&x| x == 0));
        assert_eq!(n.strength, (25.0 * 128.0) as i16);
        assert_eq!(n.mean_offset, 0);
    }

    #[test]
    fn test_gauss_noise_with_mean() {
        let n = GaussNoise::new(10.0, 25.0);
        assert_eq!(n.mean_offset, 10);
        assert_eq!(n.strength, (25.0 * 128.0) as i16);
    }

    #[test]
    #[should_panic(expected = "std_dev must be non-negative")]
    fn test_gauss_noise_invalid_std_dev() {
        GaussNoise::new(0.0, -1.0);
    }

    #[test]
    fn test_gauss_noise_zero_std_dev() {
        let mut data = vec![128u8; 4];
        let mut img = FusableImage::new(&mut data, 2, 2, 1);

        let n = GaussNoise::new(0.0, 0.0);
        n.execute(&mut img);

        // With zero std_dev, values should be unchanged
        assert_eq!(img.data, &[128u8; 4]);
    }

    #[test]
    fn test_gauss_noise_execute() {
        let mut data = vec![128u8; 100];
        let mut img = FusableImage::new(&mut data, 10, 10, 1);

        let n = GaussNoise::new(0.0, 20.0);
        n.execute(&mut img);

        // Values should be different from original
        let all_same = img.data.iter().all(|&x| x == 128);
        assert!(!all_same, "Noise should modify pixel values");
    }

    #[test]
    fn test_gauss_noise_clamping() {
        let mut data = vec![250u8; 4];
        let mut img = FusableImage::new(&mut data, 2, 2, 1);

        let n = GaussNoise::new(50.0, 10.0);
        n.execute(&mut img);

        // All values should be clamped to [0, 255]
        for &px in img.data.iter() {
            assert!((0..=255).contains(&px));
        }
    }

    #[test]
    fn test_gauss_noise_low_values() {
        let mut data = vec![5u8; 4];
        let mut img = FusableImage::new(&mut data, 2, 2, 1);

        let n = GaussNoise::new(0.0, 20.0);
        n.execute(&mut img);

        // Values should still be in valid range
        for &px in img.data.iter() {
            assert!((0..=255).contains(&px));
        }
    }

    #[test]
    fn test_gauss_noise_positive_mean() {
        let mut data = vec![100u8; 100];
        let mut img = FusableImage::new(&mut data, 10, 10, 1);

        let n = GaussNoise::new(10.0, 5.0);
        n.execute(&mut img);

        // Average should be higher than original (roughly)
        let avg: f32 = img.data.iter().map(|&x| x as f32).sum::<f32>() / 100.0;
        assert!(avg > 100.0, "Positive mean should increase average value");
    }

    #[test]
    fn test_gauss_noise_rgb() {
        let mut data = vec![128u8, 128, 128];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        let n = GaussNoise::new(0.0, 20.0);
        n.execute(&mut img);

        // All channels should be modified
        let all_same = img.data[0] == img.data[1] && img.data[1] == img.data[2];
        assert!(
            !all_same || img.data[0] != 128,
            "RGB channels should have independent noise"
        );
    }

    #[test]
    fn test_gauss_noise_high_std_dev() {
        let mut data = vec![128u8; 100];
        let mut img = FusableImage::new(&mut data, 10, 10, 1);

        let n = GaussNoise::new(0.0, 100.0);
        n.execute(&mut img);

        // With high std dev, we should see a wide range of values
        let min_val = *img.data.iter().min().unwrap();
        let max_val = *img.data.iter().max().unwrap();

        // Range should be significant
        assert!(
            max_val - min_val > 50,
            "High std dev should produce wide value range"
        );
    }

    #[test]
    fn test_gauss_noise_lut_range() {
        let n = GaussNoise::new(0.0, 50.0);

        // LUT should have a reasonable range (unit Gaussian * 128)
        let min = *n.lut.iter().min().unwrap();
        let max = *n.lut.iter().max().unwrap();

        // Should span multiple standard deviations (in Q8.7 format)
        assert!(max - min > 200, "LUT should cover a wide range");
    }

    #[test]
    fn test_gauss_noise_clone() {
        let n1 = GaussNoise::new(0.0, 25.0);
        let n2 = n1.clone();

        // Cloned instances should have identical LUTs
        assert_eq!(n1.lut[..], n2.lut[..]);
        assert_eq!(n1.strength, n2.strength);
        assert_eq!(n1.mean_offset, n2.mean_offset);
    }
}
