// Posterize (color level reduction)
//
// Reduces the number of color levels in the image.

use crate::core::{AccessPattern, Executable, FusableImage, BarrierImage, ShapeEffect, Transform, ReorderRule};
use crate::transforms::runtime::lut::LutOp;
use std::sync::OnceLock;

/// Posterize transform
///
/// Reduces the number of color levels by decreasing the bit depth.
///
/// # Parameters
/// - `bits`: Number of bits to keep for each channel (1-7)
///   - `bits=1`: 2 levels (black/white)
///   - `bits=2`: 4 levels
///   - `bits=3`: 8 levels
///   - `bits=4`: 16 levels
///   - `bits=5`: 32 levels
///   - `bits=6`: 64 levels
///   - `bits=7`: 128 levels
///   - `bits=8`: 256 levels (no change)
///
/// # Algorithm
/// ```text
/// shifted = pixel >> (8 - bits)
/// result = shifted << (8 - bits)
/// ```
///
/// # Example
/// With `bits=2`:
/// ```text
/// Input:  [0, 63, 127, 191, 255]
/// Output: [0, 0,  128, 128, 192]
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Posterize {
    pub bits: u8,
    /// Cached LUT - built once on first access
    lut: OnceLock<[u8; 256]>,
}

impl Posterize {
    /// Create a new Posterize transform
    ///
    /// # Panics
    /// Panics if bits is outside [1, 8]
    pub fn new(bits: u8) -> Self {
        assert!((1..=8).contains(&bits), "bits must be in [1, 8], got {}", bits);
        Self {
            bits,
            lut: OnceLock::new(),
        }
    }

    /// Calculate the number of color levels
    ///
    /// Returns 2^bits
    pub fn levels(&self) -> u16 {
        1 << self.bits
    }
}

impl Transform for Posterize {
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
impl LutOp for Posterize {
    fn build_lut(&self) -> [u8; 256] {
        let mut lut = [0u8; 256];
        let bits_to_discard = 8 - self.bits;
        for i in 0u8..=255 {
            lut[i as usize] = (i >> bits_to_discard) << bits_to_discard;
        }
        lut
    }

    fn get_lut(&self) -> [u8; 256] {
        *self.lut.get_or_init(|| self.build_lut())
    }
}

impl Executable for Posterize {
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
    fn test_posterize_new() {
        let p = Posterize::new(4);
        assert_eq!(p.bits, 4);
        assert_eq!(p.levels(), 16);
    }

    #[test]
    fn test_lut_caching() {
        // Since get_lut() returns by value, verify values are identical
        let p = Posterize::new(4);
        let lut1 = p.get_lut();
        let lut2 = p.get_lut();
        assert_eq!(lut1, lut2);
    }
}
