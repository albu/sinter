// Gamma correction transform
//
// Applies gamma correction to pixel values.

use crate::core::{AccessPattern, Executable, FusableImage, BarrierImage, ShapeEffect, Transform, ReorderRule};
use crate::transforms::runtime::lut::LutOp;
use std::sync::OnceLock;

/// Gamma correction transform
///
/// Applies non-linear gamma correction to pixel values.
/// Formula: output = 255 * (input/255)^gamma
///
/// # Parameters
/// - `gamma`: Gamma correction value
///   - gamma < 1.0: brightens the image (adds light to shadows)
///   - gamma = 1.0: no change (linear)
///   - gamma > 1.0: darkens the image (increases contrast)
///
/// # Notes
/// - Typical values: 0.5 to 2.0
/// - Gamma correction is commonly used to match display characteristics
/// - This transform implements the LutOp trait for fast LUT-based execution
#[derive(Debug, Clone, PartialEq)]
pub struct Gamma {
    pub gamma: f32,
    /// Cached LUT - built once on first access
    lut: OnceLock<[u8; 256]>,
}

impl Gamma {
    /// Create a new Gamma transform
    ///
    /// # Panics
    /// Panics if gamma is not positive
    pub fn new(gamma: f32) -> Self {
        assert!(gamma > 0.0, "gamma must be positive, got {}", gamma);
        Self {
            gamma,
            lut: OnceLock::new(),
        }
    }
}

impl Transform for Gamma {
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

impl Executable for Gamma {
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        // Use LUT execution - much faster than PixelOp for single transform
        self.execute_with_lut(image);
        None
    }
}

/// Implement LutOp for Gamma
///
/// LUT: lut[i] = clamp(255 * (i/255)^gamma, 0, 255)
impl LutOp for Gamma {
    fn build_lut(&self) -> [u8; 256] {
        let mut lut = [0u8; 256];
        let gamma = self.gamma;
        for i in 0u8..=255 {
            let normalized = i as f32 / 255.0;
            let corrected = normalized.powf(gamma);
            lut[i as usize] = (corrected * 255.0).clamp(0.0, 255.0) as u8;
        }
        lut
    }

    fn get_lut(&self) -> [u8; 256] {
        *self.lut.get_or_init(|| self.build_lut())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamma_new() {
        let g = Gamma::new(2.2);
        assert_eq!(g.gamma, 2.2);
    }

    #[test]
    #[should_panic(expected = "gamma must be positive")]
    fn test_gamma_invalid() {
        Gamma::new(0.0);
    }

    #[test]
    fn test_gamma_lut_linear() {
        let g = Gamma::new(1.0);
        let lut = g.build_lut();
        // Gamma 1.0 should be approximately a no-op
        for i in 0..=255u8 {
            assert!((lut[i as usize] as i16 - i as i16).abs() <= 1);
        }
    }

    #[test]
    fn test_gamma_lut_brighten() {
        let g = Gamma::new(0.5);
        let lut = g.build_lut();
        // Gamma < 1 brightens the image
        // Dark values should become much brighter
        assert!(lut[64] > 100); // 64 -> should be > 100
        assert!(lut[128] >= 180); // 128 -> should be >= 180 (exact value is 180 due to rounding)
    }

    #[test]
    fn test_gamma_lut_darken() {
        let g = Gamma::new(2.0);
        let lut = g.build_lut();
        // Gamma > 1 darkens the image
        // Light values should become darker
        assert!(lut[200] < 160); // 200 -> should be < 160
        assert!(lut[255] == 255); // 255 should stay 255
    }

    #[test]
    fn test_gamma_lut_zeros() {
        let g = Gamma::new(2.2);
        let lut = g.build_lut();
        // Zero should stay zero
        assert_eq!(lut[0], 0);
    }

    #[test]
    fn test_gamma_execute() {
        let mut data = vec![128u8; 9];
        let mut img = FusableImage::new(&mut data, 3, 3, 1);

        let g = Gamma::new(2.0);
        g.execute(&mut img);

        // All pixels should be darkened
        // 128 -> should be less than 128
        for &px in img.data.iter() {
            assert!(px < 128);
        }
    }

    #[test]
    fn test_gamma_lut_caching() {
        // Since get_lut() returns by value, we verify the LUT is correctly cached
        // by checking that multiple calls return the same values (OnceLock ensures caching)
        let g = Gamma::new(2.0);
        let lut1 = g.get_lut();
        let lut2 = g.get_lut();
        // Values should be identical (cached in OnceLock)
        assert_eq!(lut1, lut2);
    }

    #[test]
    fn test_gamma_typical_values() {
        // Test typical gamma values used in practice
        let g1 = Gamma::new(0.5);  // Strong brightening
        let g2 = Gamma::new(2.2);  // Standard sRGB gamma
        let g3 = Gamma::new(0.8);  // Subtle brightening

        // Verify LUTs are generated correctly
        let lut1 = g1.build_lut();
        let lut2 = g2.build_lut();
        let lut3 = g3.build_lut();

        // Low gamma brightens midtones
        assert!(lut1[128] >= 180);
        // High gamma darkens midtones
        assert!(lut2[128] < 80);
        // Slightly low gamma brightens a bit
        assert!(lut3[128] > 128);
    }
}
