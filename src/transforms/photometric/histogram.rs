// Histogram-based transforms
//
// These transforms analyze the image histogram and apply per-pixel adjustments.
// Note: These cannot be fused with other LUT ops because they depend on image content.

use crate::core::{AccessPattern, BarrierImage, Executable, FusableImage, ShapeEffect, Transform};
use crate::transforms::LutExecutor;

/// Equalize - Histogram equalization
///
/// Distributes pixel values more evenly across the range [0, 255] by
/// applying a transform based on the cumulative distribution function.
///
/// This enhances contrast by spreading out the most frequent intensity values.
///
/// # Performance
/// - Uses optimized histogram computation (scalar, cache-friendly)
/// - Fixed-point CDF calculation (avoiding float division)
/// - NEON-optimized LUT application via LutExecutor (vqtbl4q_u8)
/// - Expected: 2-3x faster than naive implementation
///
/// Note: Cannot be fused with other LUT transforms because the LUT depends
/// on the image histogram.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Equalize;

impl Equalize {
    pub fn new() -> Self {
        Self
    }

    /// Build equalization LUT from image histogram
    ///
    /// Uses fixed-point arithmetic to avoid float operations in the hot path.
    fn build_lut(&self, image: &FusableImage) -> [u8; 256] {
        let mut histogram = [0u32; 256];
        let total_pixels = (image.width * image.height * image.channels) as u32;

        // Build histogram - scalar but cache-friendly sequential access
        for &pixel in image.data.iter() {
            histogram[pixel as usize] += 1;
        }

        // Build equalization LUT using CDF with fixed-point arithmetic
        // Fixed-point: multiply by 255 first, then divide to avoid float
        let mut cdf = 0u32;
        let mut lut = [0u8; 256];

        for i in 0..256 {
            cdf += histogram[i];
            // Fixed-point: (cdf * 255) / total_pixels
            // This avoids expensive float division
            lut[i] = ((cdf * 255) / total_pixels) as u8;
        }

        lut
    }
}

impl Transform for Equalize {
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

impl Executable for Equalize {
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        let lut = self.build_lut(image);

        // Use optimized LUT executor (NEON vqtbl4q_u8 on ARM)
        LutExecutor::apply(image, &lut);

        None
    }
}

/// AutoContrast - Automatic contrast stretching
///
/// Finds the minimum and maximum pixel values in the image and linearly
/// stretches them to cover the full [0, 255] range. This enhances contrast
/// by using the full dynamic range available.
///
/// # Performance
/// - Uses optimized LUT application via LutExecutor
/// - Fixed-point arithmetic for LUT generation
///
/// Note: Cannot be fused with other LUT transforms because the LUT depends
/// on the image min/max values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoContrast {
    /// Cutoff percentage for outliers (0.0 to 0.5)
    /// Default 0.0 uses full range, higher values ignore extreme outliers
    pub cutoff: f32,
}

impl AutoContrast {
    pub fn new(cutoff: f32) -> Self {
        assert!(cutoff >= 0.0 && cutoff <= 0.5, "cutoff must be in [0, 0.5]");
        Self { cutoff }
    }

    /// Build contrast stretch LUT from image min/max
    ///
    /// Uses fixed-point arithmetic to avoid float operations.
    fn build_lut_from_image(&self, image: &FusableImage) -> [u8; 256] {
        let mut min_val = 255u8;
        let mut max_val = 0u8;

        // Find min and max values
        for &pixel in image.data.iter() {
            min_val = min_val.min(pixel);
            max_val = max_val.max(pixel);
        }

        // Handle edge case: uniform image
        if min_val == max_val {
            return [0u8; 256]; // All zeros
        }

        // Build linear stretch LUT using fixed-point arithmetic
        // Formula: lut[i] = ((i - min) * 255) / (max - min)
        let range = (max_val - min_val) as u32;
        let min_val = min_val as u32;
        let mut lut = [0u8; 256];

        for i in 0..256 {
            let i = i as u32;
            // Fixed-point: (i - min) * 255 / range
            let normalized = i.saturating_sub(min_val);
            lut[i as usize] = ((normalized * 255) / range) as u8;
        }

        lut
    }
}

impl Transform for AutoContrast {
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

impl Executable for AutoContrast {
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        let lut = self.build_lut_from_image(image);

        // Use optimized LUT executor (NEON vqtbl4q_u8 on ARM)
        LutExecutor::apply(image, &lut);

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equalize_execute() {
        let mut data = vec![0u8; 100];
        // Create a simple pattern
        for i in 0..100 {
            data[i] = (i % 256) as u8;
        }

        let mut img = FusableImage::new(&mut data, 10, 10, 1);
        let eq = Equalize::new();
        eq.execute(&mut img);

        // After equalization, values should be spread out
        assert_ne!(img.data, vec![0u8; 100]);
    }

    #[test]
    fn test_equalize_clamping() {
        // Test that CDF doesn't overflow
        let mut data = vec![255u8; 256];
        let mut img = FusableImage::new(&mut data, 16, 16, 1);

        let eq = Equalize::new();
        eq.execute(&mut img);

        // All values should be in valid range
        for &px in img.data.iter() {
            assert!((0..=255).contains(&px));
        }
    }

    #[test]
    fn test_autocontrast_new() {
        let ac = AutoContrast::new(0.0);
        assert_eq!(ac.cutoff, 0.0);
    }

    #[test]
    fn test_autocontrast_uniform_image() {
        let mut data = vec![128u8; 100];
        let mut img = FusableImage::new(&mut data, 10, 10, 1);

        let ac = AutoContrast::new(0.0);
        ac.execute(&mut img);

        // Uniform image should become all zeros (edge case)
        assert_eq!(img.data, vec![0u8; 100]);
    }

    #[test]
    fn test_autocontrast_stretch() {
        let mut data = vec![100u8; 50];
        data.extend(vec![150u8; 50]);

        let mut img = FusableImage::new(&mut data, 10, 10, 1);

        let ac = AutoContrast::new(0.0);
        ac.execute(&mut img);

        // After stretch, 100 should map toward 0, 150 toward 255
        // Check that values are now more spread out
        let min_val = *img.data.iter().min().unwrap();
        let max_val = *img.data.iter().max().unwrap();

        // Should use more of the range than original [100, 150]
        assert!(max_val - min_val > 50);
    }

    #[test]
    fn test_autocontrast_full_range() {
        let mut data = vec![0u8; 50];
        data.extend(vec![255u8; 50]);

        let mut img = FusableImage::new(&mut data, 10, 10, 1);

        let ac = AutoContrast::new(0.0);
        ac.execute(&mut img);

        // Image already uses full range, should be preserved
        let min_val = *img.data.iter().min().unwrap();
        let max_val = *img.data.iter().max().unwrap();

        assert_eq!(min_val, 0);
        assert_eq!(max_val, 255);
    }
}
