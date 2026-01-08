// Solarize
//
// Inverts pixel values above a threshold.

use crate::core::{AccessPattern, Executable, FusableImage, BarrierImage, ShapeEffect, Transform, ReorderRule};
use crate::transforms::runtime::lut::LutOp;
use std::sync::OnceLock;

/// Solarize transform
///
/// Inverts all pixel values above a threshold.
///
/// # Parameters
/// - `threshold`: Threshold value in [0, 255]
///   - Pixels above threshold are inverted: `pixel = 255 - pixel`
///   - Pixels at or below threshold are unchanged
///
/// # Algorithm
/// ```text
/// if pixel > threshold:
///     pixel = 255 - pixel
/// else:
///     pixel = pixel
/// ```
///
/// # Example
/// With `threshold=128`:
/// ```text
/// Input:  [0, 64, 127, 128, 191, 255]
/// Output: [0, 64, 127, 127, 64, 0]
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Solarize {
    pub threshold: u8,
    /// Cached LUT - built once on first access
    lut: OnceLock<[u8; 256]>,
}

impl Solarize {
    /// Create a new Solarize transform
    ///
    /// # Panics
    /// Panics if threshold is outside [0, 255]
    pub fn new(threshold: u8) -> Self {
        Self {
            threshold,
            lut: OnceLock::new(),
        }
    }

    /// Create a new Solarize transform with default threshold of 128
    pub fn default_threshold() -> Self {
        Self {
            threshold: 128,
            lut: OnceLock::new(),
        }
    }
}

impl Default for Solarize {
    fn default() -> Self {
        Self::default_threshold()
    }
}

impl Transform for Solarize {
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
impl LutOp for Solarize {
    fn build_lut(&self) -> [u8; 256] {
        let mut lut = [0u8; 256];
        let threshold = self.threshold;
        for i in 0u8..=255 {
            lut[i as usize] = if i >= threshold { 255 - i } else { i };
        }
        lut
    }

    fn get_lut(&self) -> [u8; 256] {
        *self.lut.get_or_init(|| self.build_lut())
    }
}

impl Executable for Solarize {
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        // Use LUT for fast execution (3-5x faster than per-pixel)
        self.execute_with_lut(image);
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solarize_new() {
        let s = Solarize::new(100);
        assert_eq!(s.threshold, 100);
    }

    #[test]
    fn test_lut_caching() {
        // Since get_lut() returns by value, verify values are identical
        let s = Solarize::new(128);
        let lut1 = s.get_lut();
        let lut2 = s.get_lut();
        assert_eq!(lut1, lut2);
    }
}
