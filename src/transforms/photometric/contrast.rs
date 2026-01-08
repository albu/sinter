// Contrast adjustment
//
// Scales pixel values around a midpoint (128).

use crate::core::{AccessPattern, Executable, FusableImage, BarrierImage, ShapeEffect, Transform, ReorderRule};
use crate::transforms::runtime::lut::LutOp;
use std::sync::OnceLock;

/// Contrast adjustment
///
/// Scales pixel values around a midpoint (128).
///
/// # Parameters
/// - `factor`: Contrast multiplier
///   - 1.0 = no change
///   - > 1.0 = increase contrast
///   - < 1.0 = decrease contrast
#[derive(Debug, Clone, PartialEq)]
pub struct Contrast {
    pub factor: f32,
    /// Cached LUT - built once on first access
    lut: OnceLock<[u8; 256]>,
}

impl Contrast {
    /// Create a new Contrast transform
    ///
    /// # Panics
    /// Panics if factor is negative
    pub fn new(factor: f32) -> Self {
        assert!(factor >= 0.0, "factor must be non-negative, got {}", factor);
        Self {
            factor,
            lut: OnceLock::new(),
        }
    }
}

impl Transform for Contrast {
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

impl Executable for Contrast {
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        // Use LUT execution - much faster than PixelOp for single transform
        self.execute_with_lut(image);
        None
    }
}

/// Implement LutOp for Contrast
///
/// LUT: lut[i] = clamp((i - 128) * factor + 128, 0, 255)
impl LutOp for Contrast {
    fn build_lut(&self) -> [u8; 256] {
        let mut lut = [0u8; 256];
        let midpoint = 128.0;
        for i in 0..256 {
            let x = i as f32;
            let y = (x - midpoint) * self.factor + midpoint;
            lut[i] = y.clamp(0.0, 255.0) as u8;
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
    fn test_contrast_new() {
        let c = Contrast::new(1.5);
        assert_eq!(c.factor, 1.5);
    }

    #[test]
    #[should_panic(expected = "factor must be non-negative")]
    fn test_contrast_invalid_factor() {
        Contrast::new(-1.0);
    }

    #[test]
    fn test_lut_caching() {
        // Since get_lut() returns by value, verify values are identical
        let c = Contrast::new(1.5);
        let lut1 = c.get_lut();
        let lut2 = c.get_lut();
        assert_eq!(lut1, lut2);
    }
}
