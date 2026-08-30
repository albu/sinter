// Sharpen transform
//
// Applies a high-performance vectorized sharpening convolution kernel to enhance edges.

use super::convolve::convolve_3x3;
use crate::core::{AccessPattern, Executable, FusableImage, ShapeEffect, Transform};

/// Sharpen transform
///
/// Enhances edges by applying a sharpening convolution kernel.
/// The standard sharpen kernel is:
///
///   0  -1   0
///  -1   5  -1
///   0  -1   0
///
/// # Parameters
/// - `strength`: Sharpening strength multiplier (default 1.0)
///   - Higher values increase sharpening effect
///   - Negative values produce blur effect
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sharpen {
    pub strength: f32,
}

impl Sharpen {
    /// Create a new Sharpen transform with default strength (1.0)
    pub fn new() -> Self {
        Self { strength: 1.0 }
    }

    /// Create a new Sharpen transform with custom strength
    ///
    /// # Panics
    /// Panics if strength is outside [-5.0, 5.0]
    pub fn with_strength(strength: f32) -> Self {
        assert!(
            (-5.0..=5.0).contains(&strength),
            "strength must be in [-5.0, 5.0], got {}",
            strength
        );
        Self { strength }
    }
}

impl Default for Sharpen {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform for Sharpen {
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

impl Executable for Sharpen {
    fn execute(&self, image: &mut FusableImage) -> Option<crate::core::BarrierImage> {
        super::convolve_2d::apply_sharpen(image, self.strength);
        None
    }
}

impl Sharpen {
    /// Pure Rust implementation (used as fallback or when opencv feature is disabled)
    fn execute_rust(&self, image: &mut FusableImage) {
        // Standard sharpen kernel
        //  0  -1   0
        // -1   5  -1
        //  0  -1   0

        // Adjust kernel based on strength
        // For strength s: center = 1 + 4s, neighbors = -s
        let s = self.strength;
        let center = (1.0 + 4.0 * s) as i32;
        let neighbor = (-s) as i32;

        let kernel = [0, neighbor, 0, neighbor, center, neighbor, 0, neighbor, 0];

        convolve_3x3(image, &kernel, 1, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sharpen_new() {
        let s = Sharpen::new();
        assert_eq!(s.strength, 1.0);
    }

    #[test]
    fn test_sharpen_default() {
        let s = Sharpen::default();
        assert_eq!(s.strength, 1.0);
    }

    #[test]
    fn test_sharpen_with_strength() {
        let s = Sharpen::with_strength(2.0);
        assert_eq!(s.strength, 2.0);
    }

    #[test]
    fn test_sharpen_invalid_strength() {
        let result = std::panic::catch_unwind(|| {
            Sharpen::with_strength(10.0);
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_sharpen_apply_constant() {
        // Constant image should remain mostly constant
        let mut data = vec![128u8; 9]; // 3x3 constant
        let mut img = FusableImage::new(&mut data, 3, 3, 1);

        Sharpen::new().execute(&mut img);

        // All pixels should still be 128 (no edges to sharpen)
        assert!(img.data.iter().all(|&p| p == 128));
    }

    #[test]
    fn test_sharpen_apply_edge() {
        // Image with a sharp edge
        // 0 0 0
        // 0 255 255
        // 0 255 255
        let mut data = vec![0u8, 0u8, 0u8, 0u8, 255u8, 255u8, 0u8, 255u8, 255u8];
        let mut img = FusableImage::new(&mut data, 3, 3, 1);

        Sharpen::new().execute(&mut img);

        // Edge pixels should be enhanced (higher contrast)
        // Center pixel (1,1) is on the edge
        // Before: 0, 0, 0, 0, 255, 255, 0, 255, 255
        // After sharpen: center = -0 -0 -0 -0 + 5*255 -255 = 1020 - 255 = 765 -> 255
        assert_eq!(img.data[4], 255);
    }

    #[test]
    fn test_sharpen_rgb() {
        // Test RGB image
        let mut data = vec![
            100u8, 100u8, 100u8, 128u8, 128u8, 128u8, 150u8, 150u8, 150u8,
        ];
        let mut img = FusableImage::new(&mut data, 3, 1, 3);

        Sharpen::new().execute(&mut img);

        // Each channel should be processed independently
        // With a gradient like this, sharpening should enhance differences
        assert!(img.data[3] > 100); // R of middle pixel
        assert!(img.data[4] > 100); // G of middle pixel
        assert!(img.data[5] > 100); // B of middle pixel
    }
}
