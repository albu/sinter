// CoarseDropout transform
//
// Randomly drops out rectangular regions by setting them to zero.

use crate::core::{AccessPattern, BarrierImage, Executable, FusableImage, ShapeEffect, Transform};

/// CoarseDropout transform
///
/// Randomly sets rectangular regions in the image to zero.
/// This is a form of regularization that forces the model to learn
/// from partial information.
///
/// # Parameters
/// - `num_holes`: Number of rectangular regions to drop out
/// - `max_hole_size`: Maximum size of each hole as a fraction of image dimensions (0.0 to 1.0)
/// - `fill_value`: Value to fill holes with (default: 0)
///
/// # Notes
/// - Operates in-place (no allocation)
/// - Preserves image shape
/// - Position-dependent operation (breaks pure per-pixel fusion)
/// - Uses deterministic hashing for reproducible results
#[derive(Debug, Clone, PartialEq)]
pub struct CoarseDropout {
    /// Number of holes to create
    pub num_holes: u32,
    /// Maximum hole size as fraction of image dimensions
    pub max_hole_size: (f32, f32), // (width_fraction, height_fraction)
    /// Value to fill holes with
    pub fill_value: u8,
    /// Per-pipeline seed so different images get different hole layouts.
    pub seed: u64,
}

impl CoarseDropout {
    /// Create a new CoarseDropout transform
    ///
    /// # Arguments
    /// * `num_holes` - Number of rectangular regions to drop out
    /// * `max_hole_size` - Maximum size of each hole as (width_fraction, height_fraction) of image
    /// * `fill_value` - Value to fill holes with (default: 0)
    ///
    /// # Panics
    /// Panics if max_hole_size values are outside (0.0, 1.0]
    pub fn new(num_holes: u32, max_hole_size: (f32, f32), fill_value: u8) -> Self {
        Self::with_seed(num_holes, max_hole_size, fill_value, 0)
    }

    /// Create a new CoarseDropout transform with an explicit per-pipeline seed.
    ///
    /// # Panics
    /// Panics if max_hole_size values are outside (0.0, 1.0]
    pub fn with_seed(num_holes: u32, max_hole_size: (f32, f32), fill_value: u8, seed: u64) -> Self {
        assert!(
            max_hole_size.0 > 0.0 && max_hole_size.0 <= 1.0,
            "max_hole_size width must be in (0.0, 1.0], got {}",
            max_hole_size.0
        );
        assert!(
            max_hole_size.1 > 0.0 && max_hole_size.1 <= 1.0,
            "max_hole_size height must be in (0.0, 1.0], got {}",
            max_hole_size.1
        );
        Self {
            num_holes,
            max_hole_size,
            fill_value,
            seed,
        }
    }

    /// Create a default CoarseDropout with common parameters
    ///
    /// - 8 holes
    /// - Max hole size: 8% of width, 8% of height
    /// - Fill with 0
    pub fn default_params() -> Self {
        Self::new(8, (0.08, 0.08), 0)
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

    /// Generate random hole coordinates
    fn generate_holes(&self, width: usize, height: usize) -> Vec<(usize, usize, usize, usize)> {
        let max_w = ((width as f32) * self.max_hole_size.0).ceil() as usize;
        let max_h = ((height as f32) * self.max_hole_size.1).ceil() as usize;

        let mut holes = Vec::with_capacity(self.num_holes as usize);

        for i in 0..self.num_holes as usize {
            // Random hole size (at least 1x1)
            let hole_w = if max_w > 1 {
                let r = self.hash(i, self.seed);
                ((r * (max_w - 1) as f32) as usize) + 1
            } else {
                1
            };
            let hole_h = if max_h > 1 {
                let r = self.hash(i + 1000, self.seed);
                ((r * (max_h - 1) as f32) as usize) + 1
            } else {
                1
            };

            // Random position
            let x = if width > hole_w {
                let r = self.hash(i + 2000, self.seed);
                (r * (width - hole_w) as f32) as usize
            } else {
                0
            };
            let y = if height > hole_h {
                let r = self.hash(i + 3000, self.seed);
                (r * (height - hole_h) as f32) as usize
            } else {
                0
            };

            holes.push((x, y, hole_w, hole_h));
        }

        holes
    }

    /// Apply holes to image data
    fn apply_holes(&self, image: &mut FusableImage, holes: &[(usize, usize, usize, usize)]) {
        let width = image.width;
        let channels = image.channels;
        let row_stride = width * channels;

        for &(x, y, hole_w, hole_h) in holes {
            let x_end = (x + hole_w).min(width);
            let y_end = (y + hole_h).min(image.height);

            for row in y..y_end {
                let row_start = row * row_stride + x * channels;
                let row_end = row * row_stride + x_end * channels;

                for px in row_start..row_end {
                    image.data[px] = self.fill_value;
                }
            }
        }
    }
}

impl Transform for CoarseDropout {
    fn access(&self) -> AccessPattern {
        // InPlace - we modify the existing buffer
        AccessPattern::InPlace
    }

    fn shape_effect(&self) -> ShapeEffect {
        // Preserves dimensions
        ShapeEffect::Preserve
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_executable(&self) -> Option<&dyn Executable> {
        Some(self)
    }
}

impl Executable for CoarseDropout {
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        let holes = self.generate_holes(image.width, image.height);
        self.apply_holes(image, &holes);
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coarse_dropout_new() {
        let cd = CoarseDropout::new(8, (0.1, 0.1), 0);
        assert_eq!(cd.num_holes, 8);
        assert_eq!(cd.max_hole_size, (0.1, 0.1));
        assert_eq!(cd.fill_value, 0);
    }

    #[test]
    fn test_coarse_dropout_default_params() {
        let cd = CoarseDropout::default_params();
        assert_eq!(cd.num_holes, 8);
        assert_eq!(cd.max_hole_size, (0.08, 0.08));
        assert_eq!(cd.fill_value, 0);
    }

    #[test]
    #[should_panic(expected = "max_hole_size width must be in")]
    fn test_coarse_dropout_invalid_max_size() {
        CoarseDropout::new(8, (0.0, 0.1), 0);
    }

    #[test]
    #[should_panic(expected = "max_hole_size height must be in")]
    fn test_coarse_dropout_invalid_max_size_height() {
        CoarseDropout::new(8, (0.1, 1.5), 0);
    }

    #[test]
    fn test_coarse_dropout_generate_holes() {
        let cd = CoarseDropout::new(4, (0.5, 0.5), 0);
        let holes = cd.generate_holes(100, 100);

        assert_eq!(holes.len(), 4);

        // Check all holes are within bounds
        for &(x, y, w, h) in &holes {
            assert!(x + w <= 100);
            assert!(y + h <= 100);
            assert!(w >= 1);
            assert!(h >= 1);
        }
    }

    #[test]
    fn test_coarse_dropout_execute_single_channel() {
        // Create a 10x10 image with all pixels = 128
        let mut data = vec![128u8; 100];
        let mut img = FusableImage::new(&mut data, 10, 10, 1);

        // Fixed seed: manually set holes for testing
        let cd = CoarseDropout::new(2, (0.3, 0.3), 0);

        // Manually apply known holes for deterministic test
        cd.apply_holes(&mut img, &[(2usize, 2, 3, 3), (5, 5, 2, 2)][..]);

        // Check that holes were filled
        // Hole 1: (2,2) to (4,4) - 3x3 region
        for y in 2..5 {
            for x in 2..5 {
                let idx = y * 10 + x;
                assert_eq!(
                    img.data[idx as usize], 0,
                    "Pixel at ({}, {}) should be 0",
                    x, y
                );
            }
        }

        // Hole 2: (5,5) to (6,6) - 2x2 region
        for y in 5..7 {
            for x in 5..7 {
                let idx = y * 10 + x;
                assert_eq!(
                    img.data[idx as usize], 0,
                    "Pixel at ({}, {}) should be 0",
                    x, y
                );
            }
        }

        // Check a pixel outside holes is still 128
        assert_eq!(img.data[0], 128); // Corner
        assert_eq!(img.data[50], 128); // Middle-ish
    }

    #[test]
    fn test_coarse_dropout_execute_rgb() {
        // Create a 5x5 RGB image with all pixels = (100, 150, 200)
        let mut data = vec![0u8; 5 * 5 * 3];
        for i in 0..25 {
            data[i * 3] = 100;
            data[i * 3 + 1] = 150;
            data[i * 3 + 2] = 200;
        }
        let mut img = FusableImage::new(&mut data, 5, 5, 3);

        let cd = CoarseDropout::new(1, (0.4, 0.4), 42);

        // Manually apply known hole
        cd.apply_holes(&mut img, &[(1usize, 1, 2, 2)][..]);

        // Check hole region is filled with 42
        // Hole: (1,1) to (2,2) - 2x2 region
        for y in 1..3 {
            for x in 1..3 {
                let idx = (y * 5 + x) * 3;
                assert_eq!(img.data[idx], 42, "R at ({}, {}) should be 42", x, y);
                assert_eq!(img.data[idx + 1], 42, "G at ({}, {}) should be 42", x, y);
                assert_eq!(img.data[idx + 2], 42, "B at ({}, {}) should be 42", x, y);
            }
        }

        // Check pixel outside holes is unchanged
        let idx = 0; // (0, 0)
        assert_eq!(img.data[idx], 100);
        assert_eq!(img.data[idx + 1], 150);
        assert_eq!(img.data[idx + 2], 200);
    }

    #[test]
    fn test_coarse_dropout_hole_at_boundary() {
        // Test holes at image boundaries
        let mut data = vec![255u8; 100];
        let mut img = FusableImage::new(&mut data, 10, 10, 1);

        let cd = CoarseDropout::new(1, (0.5, 0.5), 0);

        // Hole that extends beyond boundary should be clamped
        cd.apply_holes(&mut img, &[(8usize, 8, 5, 5)][..]);

        // Check the actual filled region (8,8) to (9,9) - clamped
        for y in 8..10 {
            for x in 8..10 {
                let idx = y * 10 + x;
                assert_eq!(img.data[idx as usize], 0);
            }
        }
    }

    #[test]
    fn test_coarse_dropout_access_pattern() {
        let cd = CoarseDropout::new(8, (0.1, 0.1), 0);
        assert_eq!(cd.access(), AccessPattern::InPlace);
        assert_eq!(cd.shape_effect(), ShapeEffect::Preserve);
    }

    #[test]
    fn test_coarse_dropout_custom_fill_value() {
        let mut data = vec![128u8; 100];
        let mut img = FusableImage::new(&mut data, 10, 10, 1);

        let cd = CoarseDropout::new(1, (0.2, 0.2), 99);
        cd.apply_holes(&mut img, &[(0usize, 0, 2, 2)][..]);

        // Check filled with 99
        assert_eq!(img.data[0], 99);
        assert_eq!(img.data[1], 99);
        assert_eq!(img.data[10], 99);
        assert_eq!(img.data[11], 99);
    }

    #[test]
    fn test_coarse_dropout_zero_holes() {
        let mut data = vec![128u8; 100];
        let mut img = FusableImage::new(&mut data, 10, 10, 1);

        let cd = CoarseDropout::new(0, (0.1, 0.1), 0);
        cd.execute(&mut img);

        // Image should be unchanged
        assert!(img.data.iter().all(|&p| p == 128));
    }

    #[test]
    fn test_coarse_dropout_reproducibility() {
        let mut data1 = vec![128u8; 100];
        let mut img1 = FusableImage::new(&mut data1, 10, 10, 1);

        let mut data2 = vec![128u8; 100];
        let mut img2 = FusableImage::new(&mut data2, 10, 10, 1);

        let cd = CoarseDropout::new(4, (0.3, 0.3), 0);
        cd.execute(&mut img1);
        cd.execute(&mut img2);

        // Same input should produce same output (reproducible)
        assert_eq!(img1.data, img2.data);
    }
}
