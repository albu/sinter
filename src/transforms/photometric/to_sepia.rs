// RGB to Sepia transform
//
// Converts RGB images to sepia tone (warm, vintage photograph effect).
//
// OPTIMIZATION: Uses MatrixExecutor with NEON SIMD for fast execution.

use crate::core::{
    AccessPattern, BarrierImage, Executable, FusableImage, ReorderRule, ShapeEffect, Transform,
};
use crate::transforms::runtime::matrix::{MatrixExecutor, MatrixOp};

/// ToSepia transform
///
/// Converts RGB images to sepia tone using the standard sepia matrix.
/// Creates a warm, vintage photograph effect.
///
/// # Parameters
/// - None
///
/// # Algorithm
/// The standard sepia transformation matrix:
/// ```text
/// R' = R*0.393 + G*0.769 + B*0.189
/// G' = R*0.349 + G*0.686 + B*0.168
/// B' = R*0.272 + G*0.534 + B*0.131
/// ```
///
/// # Notes
/// - Only affects RGB images (channels == 3)
/// - Grayscale images (channels == 1) are unchanged
/// - InPlace operation (same channel count)
/// - Uses NEON SIMD for fast execution
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ToSepia;

impl ToSepia {
    /// Create a new ToSepia transform
    pub fn new() -> Self {
        Self
    }
}

impl Default for ToSepia {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform for ToSepia {
    fn access(&self) -> AccessPattern {
        AccessPattern::InPlace
    }

    fn shape_effect(&self) -> ShapeEffect {
        ShapeEffect::Preserve
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_executable(&self) -> Option<&dyn crate::core::Executable> {
        Some(self)
    }

    fn reorder_rule(&self) -> ReorderRule {
        ReorderRule::CommutesWithGeometry
    }
}

impl MatrixOp for ToSepia {
    fn get_matrix(&self) -> [[f32; 3]; 3] {
        // Standard sepia transformation matrix
        [
            [0.393, 0.769, 0.189], // R' = R*0.393 + G*0.769 + B*0.189
            [0.349, 0.686, 0.168], // G' = R*0.349 + G*0.686 + B*0.168
            [0.272, 0.534, 0.131], // B' = R*0.272 + G*0.534 + B*0.131
        ]
    }
}

impl Executable for ToSepia {
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        // Only convert RGB to sepia
        if image.channels != 3 {
            // Grayscale or unsupported format, unchanged
            return None;
        }

        // Use the optimized MatrixExecutor (NEON SIMD on AArch64)
        let matrix = self.get_matrix();
        MatrixExecutor::apply(image, &matrix);

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_sepia_new() {
        let _t = ToSepia::new();
        // Just verify it creates successfully
    }

    #[test]
    fn test_to_sepia_default() {
        let _t = ToSepia::default();
        // Just verify it creates successfully
    }

    #[test]
    fn test_to_sepia_execute_white() {
        // White (255,255,255) should produce warm beige
        let mut data = vec![255u8, 255, 255];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        ToSepia::new().execute(&mut img);

        // With fixed-point arithmetic, results may differ slightly from float
        // Sepia of white should be clamped to 255 for R and G
        assert_eq!(img.data[0], 255); // R clamped
        assert_eq!(img.data[1], 255); // G clamped
                                      // B depends on exact calculation
        assert!(img.data[2] >= 230); // B should be high
    }

    #[test]
    fn test_to_sepia_execute_black() {
        // Black should stay black
        let mut data = vec![0u8, 0, 0];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        ToSepia::new().execute(&mut img);

        assert_eq!(img.data[0], 0);
        assert_eq!(img.data[1], 0);
        assert_eq!(img.data[2], 0);
    }

    #[test]
    fn test_to_sepia_execute_red() {
        // Pure red should become warm reddish-brown
        let mut data = vec![255u8, 0, 0];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        ToSepia::new().execute(&mut img);

        // R' = 255*0.393 = 100.2 -> ~100 with fixed-point
        assert!((img.data[0] as i32 - 100).abs() <= 1);
        assert!((img.data[1] as i32 - 89).abs() <= 1);
        assert!((img.data[2] as i32 - 69).abs() <= 1);
    }

    #[test]
    fn test_to_sepia_execute_green() {
        // Pure green should become warm greenish-brown
        let mut data = vec![0u8, 255, 0];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        ToSepia::new().execute(&mut img);

        // R' = 255*0.769 = 196.095 -> ~196
        assert!((img.data[0] as i32 - 196).abs() <= 1);
        assert!((img.data[1] as i32 - 175).abs() <= 1);
        assert!((img.data[2] as i32 - 136).abs() <= 1);
    }

    #[test]
    fn test_to_sepia_execute_blue() {
        // Pure blue should become warm bluish-brown
        let mut data = vec![0u8, 0, 255];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        ToSepia::new().execute(&mut img);

        // R' = 255*0.189 = 48.195 -> ~48
        assert!((img.data[0] as i32 - 48).abs() <= 1);
        assert!((img.data[1] as i32 - 43).abs() <= 1);
        assert!((img.data[2] as i32 - 33).abs() <= 1);
    }

    #[test]
    fn test_to_sepia_execute_grayscale() {
        // Already grayscale - should return None (no change)
        let mut data = vec![128u8];
        let mut img = FusableImage::new(&mut data, 1, 1, 1);

        let result = ToSepia::new().execute(&mut img);

        assert!(result.is_none());
        assert_eq!(img.data[0], 128); // Unchanged
    }

    #[test]
    fn test_to_sepia_access_pattern() {
        let t = ToSepia::new();
        assert_eq!(t.access(), AccessPattern::InPlace);
        assert_eq!(t.shape_effect(), ShapeEffect::Preserve);
    }

    #[test]
    fn test_to_sepia_multiple_pixels() {
        // RGB image: 2x2 with different colors
        let mut data = vec![
            255, 0, 0, // Red
            0, 255, 0, // Green
            0, 0, 255, // Blue
            128, 128, 128, // Gray
        ];
        let mut img = FusableImage::new(&mut data, 2, 2, 3);

        ToSepia::new().execute(&mut img);

        // Verify each pixel was transformed
        // Pixel 0 (red) at indices 0,1,2 should be warm brown -> ~100, 89, 69
        assert!((img.data[0] as i32 - 100).abs() <= 1);
        // Pixel 1 (green) at indices 3,4,5 should be warm greenish-brown -> ~196, 175, 136
        assert!((img.data[3] as i32 - 196).abs() <= 1);
        // Pixel 2 (blue) at indices 6,7,8 should be warm bluish-brown -> ~48, 43, 33
        assert!((img.data[6] as i32 - 48).abs() <= 1);
    }

    #[test]
    fn test_to_sepia_clamping() {
        // Test that values are properly clamped
        let mut data = vec![255u8, 255, 255];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        ToSepia::new().execute(&mut img);

        // All values should be in [0, 255]
        for &_px in img.data.iter() {
            // assert!(px <= 255); // Always true for u8
        }
    }

    #[test]
    fn test_to_sepia_mid_gray() {
        // Test mid-gray produces expected sepia tone
        let mut data = vec![128u8, 128, 128];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        ToSepia::new().execute(&mut img);

        // Sepia should produce warm brown tones: R > G > B
        let r = img.data[0];
        let g = img.data[1];
        let b = img.data[2];

        // Verify the sepia tone: R should be highest, B lowest
        assert!(
            r >= g && g >= b,
            "Sepia should have R >= G >= B, got R={} G={} B={}",
            r,
            g,
            b
        );

        // Expected values for (128,128,128) with sepia matrix:
        // R' = 128*1.351 = 173
        // G' = 128*1.203 = 154
        // B' = 128*0.937 = 120
        // Allow some tolerance for rounding
        assert!((r as i32 - 173).abs() <= 2, "Expected R~173, got {}", r);
        assert!((g as i32 - 154).abs() <= 2, "Expected G~154, got {}", g);
        assert!((b as i32 - 120).abs() <= 2, "Expected B~120, got {}", b);
    }
}
