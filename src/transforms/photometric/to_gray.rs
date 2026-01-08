// RGB to Grayscale transform
//
// Converts RGB images to grayscale using luminance weighting.
//
// OPTIMIZATION: Uses NEON SIMD for 8-16x speedup.

#[cfg(target_arch = "aarch64")]
mod neon;

use crate::core::{AccessPattern, BarrierImage, Executable, FusableImage, ShapeEffect, Transform};

/// ToGray transform
///
/// Converts RGB images to grayscale using standard luminance weighting.
/// Formula: gray = 0.299*R + 0.587*G + 0.114*B
///
/// # Parameters
/// - None
///
/// # Notes
/// - Only affects RGB images (channels == 3)
/// - Grayscale images (channels == 1) are unchanged
/// - Allocates a new buffer (OutOfPlace) since channel count changes
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ToGray;

impl ToGray {
    /// Create a new ToGray transform
    pub fn new() -> Self {
        Self
    }
}

impl Default for ToGray {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform for ToGray {
    fn access(&self) -> AccessPattern {
        // OutOfPlace because we change channel count (3 -> 1)
        AccessPattern::OutOfPlace
    }

    fn shape_effect(&self) -> ShapeEffect {
        // Preserves width/height but changes channels
        ShapeEffect::Preserve
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_executable(&self) -> Option<&dyn Executable> {
        Some(self)
    }
}

impl Executable for ToGray {
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        // Only convert RGB to grayscale
        if image.channels != 3 {
            // Already grayscale or unsupported format, return None
            return None;
        }

        let pixel_count = image.width * image.height;
        let mut gray_data = vec![0u8; pixel_count];

        // Use platform-specific SIMD for 8-16x speedup
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon::to_gray_neon(&image.data, &mut gray_data, pixel_count);
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            // Scalar fallback for other platforms
            for i in 0..pixel_count {
                let r = image.data[i * 3] as u32;
                let g = image.data[i * 3 + 1] as u32;
                let b = image.data[i * 3 + 2] as u32;
                gray_data[i] = ((77 * r + 150 * g + 29 * b + 128) >> 8) as u8;
            }
        }

        Some(BarrierImage::from_vec(
            gray_data,
            image.width,
            image.height,
            1,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_gray_new() {
        let _t = ToGray::new();
        // Just verify it creates successfully
    }

    #[test]
    fn test_to_gray_default() {
        let _t = ToGray::default();
        // Just verify it creates successfully
    }

    #[test]
    fn test_to_gray_execute_rgb() {
        // Pure red should map to ~76
        let mut data = vec![255u8, 0, 0];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        let result = ToGray::new().execute(&mut img);

        assert!(result.is_some());
        let gray_img = result.unwrap();
        assert_eq!(gray_img.width, 1);
        assert_eq!(gray_img.height, 1);
        assert_eq!(gray_img.channels, 1);
        // Fixed-point: (77*255 + 128) >> 8 = 76
        assert!((gray_img.data[0] as f32 - 76.0).abs() < 1.5);
    }

    #[test]
    fn test_to_gray_execute_green() {
        // Pure green should map to ~150
        let mut data = vec![0u8, 255, 0];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        let result = ToGray::new().execute(&mut img);

        assert!(result.is_some());
        let gray_img = result.unwrap();
        // Fixed-point: (150*255 + 128) >> 8 = 149
        assert!((gray_img.data[0] as f32 - 149.0).abs() < 1.5);
    }

    #[test]
    fn test_to_gray_execute_blue() {
        // Pure blue should map to ~29
        let mut data = vec![0u8, 0, 255];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        let result = ToGray::new().execute(&mut img);

        assert!(result.is_some());
        let gray_img = result.unwrap();
        // Fixed-point: (29*255 + 128) >> 8 = 29
        assert!((gray_img.data[0] as f32 - 29.0).abs() < 1.5);
    }

    #[test]
    fn test_to_gray_execute_white() {
        // White should stay 255
        let mut data = vec![255u8, 255, 255];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        let result = ToGray::new().execute(&mut img);

        assert!(result.is_some());
        let gray_img = result.unwrap();
        assert_eq!(gray_img.data[0], 255);
    }

    #[test]
    fn test_to_gray_execute_grayscale() {
        // Already grayscale - should return None
        let mut data = vec![128u8];
        let mut img = FusableImage::new(&mut data, 1, 1, 1);

        let result = ToGray::new().execute(&mut img);

        assert!(result.is_none());
    }

    #[test]
    fn test_to_gray_execute_multiple_pixels() {
        // RGB image: 2x2
        // (255,0,0) (0,255,0)
        // (0,0,255) (128,128,128)
        let mut data = vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128];
        let mut img = FusableImage::new(&mut data, 2, 2, 3);

        let result = ToGray::new().execute(&mut img);

        assert!(result.is_some());
        let gray_img = result.unwrap();
        assert_eq!(gray_img.width, 2);
        assert_eq!(gray_img.height, 2);
        assert_eq!(gray_img.channels, 1);
        assert_eq!(gray_img.data.len(), 4);
    }

    #[test]
    fn test_to_gray_access_pattern() {
        let _t = ToGray::new();
        assert_eq!(_t.access(), AccessPattern::OutOfPlace);
        assert_eq!(_t.shape_effect(), ShapeEffect::Preserve);
    }

    #[test]
    fn test_to_gray_gray_color() {
        // Gray color RGB(128, 128, 128) should map to 128
        let mut data = vec![128u8, 128, 128];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        let result = ToGray::new().execute(&mut img);

        assert!(result.is_some());
        let gray_img = result.unwrap();
        // Fixed-point: (77+150+29)*128 >> 8 = 256*128 >> 8 = 128
        assert_eq!(gray_img.data[0], 128);
    }

    #[test]
    fn test_to_gray_neon_alignment() {
        // Test that NEON works with non-multiple-of-8 pixel counts
        let mut data = vec![
            255u8, 0, 0, // Pixel 0
            0, 255, 0, // Pixel 1
            0, 0, 255, // Pixel 2
            128, 128, 128, // Pixel 3
            64, 64, 64, // Pixel 4
            200, 100, 50, // Pixel 5
            10, 20, 30, // Pixel 6
            255, 255, 255, // Pixel 7
            100, 150, 200, // Pixel 8 (extra pixel for alignment test)
        ];
        let mut img = FusableImage::new(&mut data, 3, 3, 3);

        let result = ToGray::new().execute(&mut img);

        assert!(result.is_some());
        let gray_img = result.unwrap();
        assert_eq!(gray_img.data.len(), 9);
        // Last pixel should be computed correctly
        // (77*100 + 150*150 + 29*200 + 128) >> 8
        // = (7700 + 22500 + 5800 + 128) >> 8
        // = 36128 >> 8 = 141
        assert_eq!(gray_img.data[8], 141);
    }
}
