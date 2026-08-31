// Salt and Pepper noise transform
//
// Randomly sets pixels to either 0 (pepper) or 255 (salt).

use crate::core::{AccessPattern, BarrierImage, Executable, FusableImage, ShapeEffect, Transform};

/// Salt and Pepper noise transform
///
/// Randomly sets pixels to either 0 (black/pepper) or 255 (white/salt).
///
/// # Parameters
/// - `amount`: Total proportion of pixels to affect [0.0, 1.0]
///   - e.g., 0.01 means 1% of pixels will be affected
/// - `salt_ratio`: Ratio of salt (255) vs pepper (0) [0.0, 1.0]
///   - 0.5 means equal amounts of salt and pepper
///   - 1.0 means only salt (white pixels)
///   - 0.0 means only pepper (black pixels)
///
/// # Notes
/// - This transform is stochastic (uses randomness)
/// - "Salt" = white pixels (255), "Pepper" = black pixels (0)
/// - Total affected pixels = amount * total_pixels
/// - Of affected pixels: salt_ratio become salt, (1-salt_ratio) become pepper
/// - Typical values: amount=0.01 to 0.05, salt_ratio=0.5
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SaltAndPepper {
    pub amount: f32,
    pub salt_ratio: f32,
    /// Per-pipeline seed so different images get different noise.
    pub seed: u64,
}

impl SaltAndPepper {
    /// Create a new SaltAndPepper transform
    ///
    /// # Panics
    /// Panics if:
    /// - amount is outside [0.0, 1.0]
    /// - salt_ratio is outside [0.0, 1.0]
    pub fn new(amount: f32, salt_ratio: f32) -> Self {
        Self::with_seed(amount, salt_ratio, 0)
    }

    /// Create a new SaltAndPepper transform with an explicit per-pipeline seed.
    ///
    /// # Panics
    /// Panics if:
    /// - amount is outside [0.0, 1.0]
    /// - salt_ratio is outside [0.0, 1.0]
    pub fn with_seed(amount: f32, salt_ratio: f32, seed: u64) -> Self {
        assert!(
            (0.0..=1.0).contains(&amount),
            "amount must be in [0.0, 1.0], got {}",
            amount
        );
        assert!(
            (0.0..=1.0).contains(&salt_ratio),
            "salt_ratio must be in [0.0, 1.0], got {}",
            salt_ratio
        );
        Self {
            amount,
            salt_ratio,
            seed,
        }
    }

    /// Simple hash function for reproducible pseudo-randomness
    #[inline]
    fn hash(&self, index: usize, seed: u64) -> f32 {
        let mut state = (index as u64).wrapping_add(seed);
        // Simple mixing function
        state = state.wrapping_mul(0x517cc1b727220a95);
        state ^= state >> 33;
        state = state.wrapping_mul(0xff51afd7ed558ccd);
        state ^= state >> 33;
        // Convert to [0, 1) range
        (state & 0xFFFFFF) as f32 / 0xFFFFFF as f32
    }
}

impl Transform for SaltAndPepper {
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

impl Executable for SaltAndPepper {
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        let total_pixels = image.data.len();

        for (i, px) in image.data.iter_mut().enumerate() {
            // First check if this pixel should be affected
            let r1 = self.hash(i, self.seed);
            if r1 < self.amount {
                // Pixel is affected - decide salt or pepper
                let r2 = self.hash(i + total_pixels, self.seed);
                *px = if r2 < self.salt_ratio { 255 } else { 0 };
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_salt_pepper_new() {
        let n = SaltAndPepper::new(0.1, 0.5);
        assert_eq!(n.amount, 0.1);
        assert_eq!(n.salt_ratio, 0.5);
    }

    #[test]
    #[should_panic(expected = "amount must be in")]
    fn test_salt_pepper_invalid_amount() {
        SaltAndPepper::new(1.5, 0.5);
    }

    #[test]
    #[should_panic(expected = "salt_ratio must be in")]
    fn test_salt_pepper_invalid_ratio() {
        SaltAndPepper::new(0.1, 1.5);
    }

    #[test]
    fn test_salt_pepper_zero_amount() {
        let mut data = vec![128u8; 100];
        let mut img = FusableImage::new(&mut data, 10, 10, 1);

        let n = SaltAndPepper::new(0.0, 0.5);
        n.execute(&mut img);

        // With zero amount, values should be unchanged
        assert!(img.data.iter().all(|&x| x == 128));
    }

    #[test]
    fn test_salt_pepper_full_amount() {
        let mut data = vec![128u8; 100];
        let mut img = FusableImage::new(&mut data, 10, 10, 1);

        let n = SaltAndPepper::new(1.0, 0.5);
        n.execute(&mut img);

        // All pixels should be either 0 or 255
        for &px in img.data.iter() {
            assert!(px == 0 || px == 255, "All pixels should be 0 or 255");
        }
    }

    #[test]
    fn test_salt_pepper_only_salt() {
        let mut data = vec![128u8; 100];
        let mut img = FusableImage::new(&mut data, 10, 10, 1);

        let n = SaltAndPepper::new(1.0, 1.0);
        n.execute(&mut img);

        // All pixels should be 255 (salt only)
        assert!(img.data.iter().all(|&x| x == 255));
    }

    #[test]
    fn test_salt_pepper_only_pepper() {
        let mut data = vec![128u8; 100];
        let mut img = FusableImage::new(&mut data, 10, 10, 1);

        let n = SaltAndPepper::new(1.0, 0.0);
        n.execute(&mut img);

        // All pixels should be 0 (pepper only)
        assert!(img.data.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_salt_pepper_ratio() {
        let mut data = vec![128u8; 10000];
        let mut img = FusableImage::new(&mut data, 10, 10, 100);

        let n = SaltAndPepper::new(1.0, 0.3);
        n.execute(&mut img);

        let salt_count = img.data.iter().filter(|&&x| x == 255).count();
        let pepper_count = img.data.iter().filter(|&&x| x == 0).count();

        // Should have approximately 30% salt, 70% pepper
        let salt_ratio = salt_count as f32 / (salt_count + pepper_count) as f32;
        assert!((salt_ratio - 0.3).abs() < 0.05, "Salt ratio should be ~0.3");
    }

    #[test]
    fn test_salt_pepper_amount_proportion() {
        let mut data = vec![128u8; 10000];
        let mut img = FusableImage::new(&mut data, 10, 10, 100);

        let amount = 0.1;
        let n = SaltAndPepper::new(amount, 0.5);
        n.execute(&mut img);

        let affected_count = img.data.iter().filter(|&&x| x == 0 || x == 255).count();
        let actual_ratio = affected_count as f32 / img.data.len() as f32;

        // Should have approximately 10% affected pixels
        assert!(
            (actual_ratio - amount).abs() < 0.02,
            "Affected ratio should be ~{}",
            amount
        );
    }

    #[test]
    fn test_salt_pepper_reproducibility() {
        let mut data1 = vec![128u8; 100];
        let mut img1 = FusableImage::new(&mut data1, 10, 10, 1);

        let mut data2 = vec![128u8; 100];
        let mut img2 = FusableImage::new(&mut data2, 10, 10, 1);

        let n = SaltAndPepper::new(0.1, 0.5);
        n.execute(&mut img1);
        n.execute(&mut img2);

        // Same input should produce same output (reproducible)
        assert_eq!(img1.data, img2.data);
    }

    #[test]
    fn test_salt_pepper_rgb() {
        let mut data = vec![128u8; 300]; // 100 RGB pixels
        let mut img = FusableImage::new(&mut data, 10, 10, 3);

        let n = SaltAndPepper::new(0.2, 0.5);
        n.execute(&mut img);

        // Some pixels should be affected
        let affected = img.data.iter().filter(|&&x| x == 0 || x == 255).count();
        assert!(affected > 0, "Some pixels should be affected");
        // But not all
        assert!(affected < 300, "Not all pixels should be affected");
    }

    #[test]
    fn test_salt_pepper_access_pattern() {
        let n = SaltAndPepper::new(0.05, 0.5);
        assert_eq!(n.access(), AccessPattern::InPlace);
        assert_eq!(n.shape_effect(), ShapeEffect::Preserve);
    }
}
