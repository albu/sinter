// Image inversion
//
// Inverts all pixel values (creates a negative image).

#[cfg(target_arch = "aarch64")]
mod neon;

use crate::core::{
    AccessPattern, BarrierImage, Executable, FusableImage, ReorderRule, ShapeEffect, Transform,
};
use crate::transforms::runtime::lut::LutOp;

/// Image inversion
///
/// Inverts all pixel values to create a negative image.
///
/// Each pixel is transformed as: `pixel = 255 - pixel`
///
/// # Example
/// ```text
/// Input:  [0,   128, 255]
/// Output: [255, 127, 0]
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Invert;

impl Invert {
    /// Create a new Invert transform
    pub fn new() -> Self {
        Self
    }
}

impl Default for Invert {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform for Invert {
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

// LUT implementation for fast execution
impl LutOp for Invert {
    fn build_lut(&self) -> [u8; 256] {
        let mut lut = [0u8; 256];
        for i in 0u8..=255 {
            lut[i as usize] = 255 - i;
        }
        lut
    }

    fn get_lut(&self) -> [u8; 256] {
        // Invert LUT is constant - use const for zero-cost
        const INVERT_LUT: [u8; 256] = {
            let mut lut = [0u8; 256];
            let mut i = 0;
            while i < 256 {
                lut[i] = 255 - i as u8;
                i += 1;
            }
            lut
        };
        INVERT_LUT
    }

    fn execute_with_lut(&self, image: &mut crate::core::FusableImage) {
        #[cfg(target_arch = "aarch64")]
        {
            neon::apply_invert_neon(image);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            for pixel in &mut image.data[..] {
                *pixel = 255 - *pixel;
            }
        }
    }
}

impl Executable for Invert {
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        // Direct SIMD implementation - ~x is equivalent to 255-x for u8
        // This is MUCH faster than LUT for this simple operation
        #[cfg(target_arch = "aarch64")]
        {
            neon::apply_invert_neon(image);
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            // Scalar fallback for other platforms
            for pixel in &mut image.data[..] {
                *pixel = 255 - *pixel;
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invert_new() {
        let _i = Invert::new();
        // Invert is a unit struct, just check it exists
    }

    #[test]
    fn test_invert_default() {
        let _i = Invert::default();
        // Invert is a unit struct, just check it exists
    }

    #[test]
    fn test_invert_execute() {
        let mut data = vec![0u8, 128u8, 255u8];
        let mut img = FusableImage::new(&mut data, 3, 1, 1);

        Invert::new().execute(&mut img);

        assert_eq!(img.data[0], 255);
        assert_eq!(img.data[1], 127);
        assert_eq!(img.data[2], 0);
    }

    #[test]
    fn test_invert_rgb() {
        let mut data = vec![
            255u8, 0u8, 128u8, // Red pixel (255, 0, 128) becomes (0, 255, 127)
        ];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        Invert::new().execute(&mut img);

        assert_eq!(img.data[0], 0); // R: 255 -> 0
        assert_eq!(img.data[1], 255); // G: 0 -> 255
        assert_eq!(img.data[2], 127); // B: 128 -> 127
    }

    #[test]
    fn test_invert_double_invert() {
        // Inverting twice should give original image
        let mut data = vec![42u8, 128u8, 200u8];
        let mut img = FusableImage::new(&mut data, 3, 1, 1);

        Invert::new().execute(&mut img);
        // After first invert: [213, 127, 55]

        Invert::new().execute(&mut img);

        // Should be back to original
        assert_eq!(img.data[0], 42);
        assert_eq!(img.data[1], 128);
        assert_eq!(img.data[2], 200);
    }

    #[test]
    fn test_invert_constant() {
        // Constant value 128 stays 128 when inverted
        let mut data = vec![128u8; 9];
        let mut img = FusableImage::new(&mut data, 3, 3, 1);

        Invert::new().execute(&mut img);

        assert!(img.data.iter().all(|&p| p == 127));
    }

    #[test]
    fn test_invert_lut_const() {
        // Verify that get_lut returns a const (compile-time)
        let lut = Invert::new().get_lut();
        assert_eq!(lut[0], 255);
        assert_eq!(lut[128], 127);
        assert_eq!(lut[255], 0);
    }
}
