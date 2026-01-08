// Color Balance transform
//
// Adjusts the balance between RGB channels by scaling each channel independently.
// This is useful for color grading and correction.
//
// This is a 3x3 RGB matrix operation that can be fused with other MatrixOp transforms.

use crate::core::{AccessPattern, BarrierImage, Executable, FusableImage, ShapeEffect, Transform};
use crate::transforms::runtime::matrix::{apply_matrix, MatrixOp};

/// ColorBalance transform
///
/// Adjusts the balance between RGB channels by scaling each independently.
///
/// # Parameters
/// - `r_scale`: Red channel multiplier (0.0 = no red, 1.0 = unchanged, 2.0 = doubled)
/// - `g_scale`: Green channel multiplier
/// - `b_scale`: Blue channel multiplier
///
/// # Algorithm
/// Uses a 3x3 diagonal matrix for independent channel scaling:
/// ```text
/// R' = R * r_scale
/// G' = G * g_scale
/// B' = B * b_scale
/// ```
///
/// # Example
/// ```text
/// ColorBalance(1.2, 1.0, 0.8): Boost red, reduce blue (warm look)
/// ColorBalance(0.8, 1.0, 1.2): Reduce red, boost blue (cool look)
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorBalance {
    /// Red channel scale factor
    pub r_scale: f32,
    /// Green channel scale factor
    pub g_scale: f32,
    /// Blue channel scale factor
    pub b_scale: f32,
}

impl ColorBalance {
    /// Create a new ColorBalance transform
    ///
    /// # Arguments
    /// * `r_scale` - Red channel multiplier (typical range: 0.0 to 2.0)
    /// * `g_scale` - Green channel multiplier (typical range: 0.0 to 2.0)
    /// * `b_scale` - Blue channel multiplier (typical range: 0.0 to 2.0)
    pub fn new(r_scale: f32, g_scale: f32, b_scale: f32) -> Self {
        Self {
            r_scale,
            g_scale,
            b_scale,
        }
    }

    /// Create a warm color balance (boost red/green, reduce blue)
    pub fn warm() -> Self {
        Self {
            r_scale: 1.2,
            g_scale: 1.05,
            b_scale: 0.8,
        }
    }

    /// Create a cool color balance (reduce red/green, boost blue)
    pub fn cool() -> Self {
        Self {
            r_scale: 0.8,
            g_scale: 1.0,
            b_scale: 1.2,
        }
    }

    /// Create a red boost (for sunset/fiery effects)
    pub fn red_boost() -> Self {
        Self {
            r_scale: 1.5,
            g_scale: 0.9,
            b_scale: 0.7,
        }
    }

    /// Create a green boost (for nature/vibrant effects)
    pub fn green_boost() -> Self {
        Self {
            r_scale: 0.9,
            g_scale: 1.5,
            b_scale: 0.9,
        }
    }

    /// Create a blue boost (for ocean/sky effects)
    pub fn blue_boost() -> Self {
        Self {
            r_scale: 0.7,
            g_scale: 0.9,
            b_scale: 1.5,
        }
    }
}

impl Default for ColorBalance {
    fn default() -> Self {
        Self {
            r_scale: 1.0,
            g_scale: 1.0,
            b_scale: 1.0,
        }
    }
}

impl Transform for ColorBalance {
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

impl MatrixOp for ColorBalance {
    fn get_matrix(&self) -> [[f32; 3]; 3] {
        // Diagonal matrix for independent channel scaling
        [
            [self.r_scale, 0.0, 0.0],
            [0.0, self.g_scale, 0.0],
            [0.0, 0.0, self.b_scale],
        ]
    }
}

impl Executable for ColorBalance {
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        if image.channels != 3 {
            return None;
        }
        apply_matrix(image, &self.get_matrix());
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_balance_new() {
        let cb = ColorBalance::new(1.2, 1.0, 0.8);
        assert_eq!(cb.r_scale, 1.2);
        assert_eq!(cb.g_scale, 1.0);
        assert_eq!(cb.b_scale, 0.8);
    }

    #[test]
    fn test_color_balance_default() {
        let cb = ColorBalance::default();
        assert_eq!(cb.r_scale, 1.0);
        assert_eq!(cb.g_scale, 1.0);
        assert_eq!(cb.b_scale, 1.0);
    }

    #[test]
    fn test_color_balance_matrix_identity() {
        let cb = ColorBalance::new(1.0, 1.0, 1.0);
        let matrix = cb.get_matrix();

        assert_eq!(matrix[0][0], 1.0);
        assert_eq!(matrix[1][1], 1.0);
        assert_eq!(matrix[2][2], 1.0);
        assert_eq!(matrix[0][1], 0.0);
        assert_eq!(matrix[0][2], 0.0);
    }

    #[test]
    fn test_color_balance_execute_boost_red() {
        let mut data = vec![100u8, 100, 100];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        ColorBalance::new(2.0, 1.0, 1.0).execute(&mut img);

        // Red should be doubled, G and B unchanged
        assert_eq!(img.data[0], 200); // R
        assert_eq!(img.data[1], 100); // G
        assert_eq!(img.data[2], 100); // B
    }

    #[test]
    fn test_color_balance_execute_warm() {
        let mut data = vec![100u8, 100, 100];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        ColorBalance::warm().execute(&mut img);

        // Warm: R boosted, G slightly boosted, B reduced
        assert!(img.data[0] > 100); // R
        assert!(img.data[1] >= 100); // G
        assert!(img.data[2] < 100); // B
    }

    #[test]
    fn test_color_balance_execute_cool() {
        let mut data = vec![100u8, 100, 100];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        ColorBalance::cool().execute(&mut img);

        // Cool: R reduced, B boosted
        assert!(img.data[0] < 100); // R
        assert!(img.data[2] > 100); // B
    }

    #[test]
    fn test_color_balance_clamping() {
        // Test clamping at 255
        let mut data = vec![200u8, 200, 200];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        ColorBalance::new(2.0, 2.0, 2.0).execute(&mut img);

        // All should be clamped to 255
        assert_eq!(img.data[0], 255);
        assert_eq!(img.data[1], 255);
        assert_eq!(img.data[2], 255);
    }

    #[test]
    fn test_color_balance_grayscale_passthrough() {
        let mut data = vec![128u8];
        let mut img = FusableImage::new(&mut data, 1, 1, 1);

        ColorBalance::warm().execute(&mut img);

        assert_eq!(img.data[0], 128);
    }

    #[test]
    fn test_color_balance_access_pattern() {
        let cb = ColorBalance::warm();
        assert_eq!(cb.access(), AccessPattern::InPlace);
        assert_eq!(cb.shape_effect(), ShapeEffect::Preserve);
    }

    #[test]
    fn test_color_balance_red_boost() {
        let mut data = vec![100u8, 100, 100];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        ColorBalance::red_boost().execute(&mut img);

        // Red should be significantly boosted
        assert!(img.data[0] > img.data[1]);
        assert!(img.data[0] > img.data[2]);
    }

    #[test]
    fn test_color_balance_green_boost() {
        let mut data = vec![100u8, 100, 100];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        ColorBalance::green_boost().execute(&mut img);

        // Green should be significantly boosted
        assert!(img.data[1] > img.data[0]);
        assert!(img.data[1] > img.data[2]);
    }

    #[test]
    fn test_color_balance_blue_boost() {
        let mut data = vec![100u8, 100, 100];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        ColorBalance::blue_boost().execute(&mut img);

        // Blue should be significantly boosted
        assert!(img.data[2] > img.data[0]);
        assert!(img.data[2] > img.data[1]);
    }
}
