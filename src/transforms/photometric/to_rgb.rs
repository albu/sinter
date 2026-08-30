// Grayscale to RGB transform
//
// Converts grayscale images to RGB by replicating the luminance value.
//
// OPTIMIZATION: Uses NEON SIMD for 3-5x speedup.

#[cfg(target_arch = "aarch64")]
mod neon;

use crate::core::{AccessPattern, BarrierImage, Executable, FusableImage, ShapeEffect, Transform};

/// ToRGB transform
///
/// Converts grayscale images to RGB by replicating the luminance value to all three channels.
/// Formula: R = G = B = gray
///
/// # Parameters
/// - None
///
/// # Notes
/// - Only affects grayscale images (channels == 1)
/// - RGB images (channels == 3) are unchanged
/// - Allocates a new buffer (OutOfPlace) since channel count changes
/// - Uses NEON SIMD on ARM64 for 3-5x speedup
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ToRGB;

impl ToRGB {
    /// Create a new ToRGB transform
    pub fn new() -> Self {
        Self
    }
}

impl Default for ToRGB {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform for ToRGB {
    fn access(&self) -> AccessPattern {
        // OutOfPlace because we change channel count (1 -> 3)
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

impl Executable for ToRGB {
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        // Only convert grayscale to RGB
        if image.channels != 1 {
            // Already RGB or unsupported format, return None
            return None;
        }

        let pixel_count = image.width * image.height;
        let mut rgb_data = Vec::with_capacity(pixel_count * 3);
        unsafe {
            rgb_data.set_len(pixel_count * 3);
        }

        // Use platform-specific SIMD for 3-5x speedup
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon::to_rgb_neon(&image.data, &mut rgb_data, pixel_count);
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            // Scalar fallback for other platforms
            for i in 0..pixel_count {
                let gray = image.data[i];
                rgb_data[i * 3] = gray;
                rgb_data[i * 3 + 1] = gray;
                rgb_data[i * 3 + 2] = gray;
            }
        }

        Some(BarrierImage::from_vec(
            rgb_data,
            image.width,
            image.height,
            3,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_rgb_new() {
        let _t = ToRGB::new();
        // Just verify it creates successfully
    }

    #[test]
    fn test_to_rgb_default() {
        let _t = ToRGB::default();
        // Just verify it creates successfully
    }

    #[test]
    fn test_to_rgb_execute_grayscale() {
        // Single grayscale pixel
        let mut data = vec![128u8];
        let mut img = FusableImage::new(&mut data, 1, 1, 1);

        let result = ToRGB::new().execute(&mut img);

        assert!(result.is_some());
        let rgb_img = result.unwrap();
        assert_eq!(rgb_img.width, 1);
        assert_eq!(rgb_img.height, 1);
        assert_eq!(rgb_img.channels, 3);
        assert_eq!(rgb_img.data[0], 128); // R
        assert_eq!(rgb_img.data[1], 128); // G
        assert_eq!(rgb_img.data[2], 128); // B
    }

    #[test]
    fn test_to_rgb_execute_rgb() {
        // Already RGB - should return None
        let mut data = vec![255u8, 128, 64];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        let result = ToRGB::new().execute(&mut img);

        assert!(result.is_none());
    }

    #[test]
    fn test_to_rgb_execute_multiple_pixels() {
        // Grayscale image: 2x2
        // [0, 128]
        // [255, 64]
        let mut data = vec![0u8, 128, 255, 64];
        let mut img = FusableImage::new(&mut data, 2, 2, 1);

        let result = ToRGB::new().execute(&mut img);

        assert!(result.is_some());
        let rgb_img = result.unwrap();
        assert_eq!(rgb_img.width, 2);
        assert_eq!(rgb_img.height, 2);
        assert_eq!(rgb_img.channels, 3);
        assert_eq!(rgb_img.data.len(), 12); // 4 pixels * 3 channels

        // Check first pixel: gray=0 -> (0, 0, 0)
        assert_eq!(rgb_img.data[0], 0);
        assert_eq!(rgb_img.data[1], 0);
        assert_eq!(rgb_img.data[2], 0);

        // Check second pixel: gray=128 -> (128, 128, 128)
        assert_eq!(rgb_img.data[3], 128);
        assert_eq!(rgb_img.data[4], 128);
        assert_eq!(rgb_img.data[5], 128);

        // Check third pixel: gray=255 -> (255, 255, 255)
        assert_eq!(rgb_img.data[6], 255);
        assert_eq!(rgb_img.data[7], 255);
        assert_eq!(rgb_img.data[8], 255);

        // Check fourth pixel: gray=64 -> (64, 64, 64)
        assert_eq!(rgb_img.data[9], 64);
        assert_eq!(rgb_img.data[10], 64);
        assert_eq!(rgb_img.data[11], 64);
    }

    #[test]
    fn test_to_rgb_access_pattern() {
        let t = ToRGB::new();
        assert_eq!(t.access(), AccessPattern::OutOfPlace);
        assert_eq!(t.shape_effect(), ShapeEffect::Preserve);
    }

    #[test]
    fn test_to_rgb_full_range() {
        // Test all gray values
        for gray in 0u8..=255 {
            let mut data = vec![gray];
            let mut img = FusableImage::new(&mut data, 1, 1, 1);

            let result = ToRGB::new().execute(&mut img);
            assert!(result.is_some());

            let rgb_img = result.unwrap();
            assert_eq!(rgb_img.data[0], gray);
            assert_eq!(rgb_img.data[1], gray);
            assert_eq!(rgb_img.data[2], gray);
        }
    }

    #[test]
    fn test_to_rgb_neon_alignment() {
        // Test that NEON works with non-multiple-of-8 pixel counts
        let mut data = vec![
            0u8, // Pixel 0
            64,  // Pixel 1
            128, // Pixel 2
            192, // Pixel 3
            255, // Pixel 4
            32,  // Pixel 5
            96,  // Pixel 6
            160, // Pixel 7
            224, // Pixel 8 (extra pixel for alignment test)
        ];
        let mut img = FusableImage::new(&mut data, 3, 3, 1);

        let result = ToRGB::new().execute(&mut img);

        assert!(result.is_some());
        let rgb_img = result.unwrap();
        assert_eq!(rgb_img.data.len(), 27); // 9 pixels * 3 channels

        // Check first 8 pixels (processed by NEON)
        for i in 0..8 {
            let gray = data[i];
            assert_eq!(rgb_img.data[i * 3], gray);
            assert_eq!(rgb_img.data[i * 3 + 1], gray);
            assert_eq!(rgb_img.data[i * 3 + 2], gray);
        }

        // Check last pixel (handled by scalar fallback)
        let gray = data[8];
        assert_eq!(rgb_img.data[24], gray);
        assert_eq!(rgb_img.data[25], gray);
        assert_eq!(rgb_img.data[26], gray);
    }

    #[test]
    fn test_to_rgb_exact_multiple() {
        // Test with exactly 8 pixels (boundary case for NEON)
        let mut data = vec![128u8; 8];
        let mut img = FusableImage::new(&mut data, 4, 2, 1);

        let result = ToRGB::new().execute(&mut img);

        assert!(result.is_some());
        let rgb_img = result.unwrap();
        assert_eq!(rgb_img.data.len(), 24); // 8 pixels * 3 channels

        // All pixels should be (128, 128, 128)
        for i in 0..8 {
            assert_eq!(rgb_img.data[i * 3], 128);
            assert_eq!(rgb_img.data[i * 3 + 1], 128);
            assert_eq!(rgb_img.data[i * 3 + 2], 128);
        }
    }
}
