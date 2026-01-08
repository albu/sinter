// Normalization
//
// Scales pixel values to [0, 1] range with mean and std.

use crate::core::{AccessPattern, Executable, FusableImage, BarrierImage, ShapeEffect, Transform, ReorderRule};
use crate::transforms::runtime::lut::LutOp;
use std::sync::OnceLock;

/// Normalization
///
/// Scales pixel values to [0, 1] range with mean and std.
///
/// # Parameters
/// - `mean`: Mean value for normalization (typically 0.0)
/// - `std`: Standard deviation for normalization (typically 1.0)
#[derive(Debug, Clone, PartialEq)]
pub struct Normalize {
    pub mean: f32,
    pub std: f32,
    /// Cached LUT - built once on first access
    lut: OnceLock<[u8; 256]>,
}

impl Normalize {
    /// Create a new Normalize transform
    ///
    /// # Panics
    /// Panics if std is zero or negative
    pub fn new(mean: f32, std: f32) -> Self {
        assert!(std > 0.0, "std must be positive, got {}", std);
        Self {
            mean,
            std,
            lut: OnceLock::new(),
        }
    }

    /// Create standard normalization (mean=0, std=1)
    pub fn standard() -> Self {
        Self::new(0.0, 1.0)
    }

    /// Create ImageNet-style normalization (for reference)
    /// Note: This is typically applied to float images in [0,1] range
    pub fn imagenet() -> Self {
        Self::new(0.0, 1.0)
    }
}

impl Transform for Normalize {
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

impl Executable for Normalize {
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        // Use LUT execution - much faster than PixelOp for single transform
        self.execute_with_lut(image);
        None
    }
}

/// Implement LutOp for Normalize
///
/// LUT: lut[i] = clamp((i / 255 - mean) / std * 255, 0, 255)
///
/// Note: This scales back to [0, 255] range for LUT application,
/// which means the result stays in u8 format but with normalization applied.
impl LutOp for Normalize {
    fn build_lut(&self) -> [u8; 256] {
        let mut lut = [0u8; 256];
        for i in 0..256 {
            let x = i as f32;
            // Normalize from [0, 255] to [0, 1]
            let normalized = x / 255.0;
            // Apply mean/std
            let result = (normalized - self.mean) / self.std;
            // Scale back to [0, 255] for u8 output
            let y = result * 255.0;
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
    #[should_panic(expected = "std must be positive")]
    fn test_normalize_invalid_std() {
        Normalize::new(0.0, 0.0);
    }

    #[test]
    fn test_lut_caching() {
        // Since get_lut() returns by value, verify values are identical
        let n = Normalize::standard();
        let lut1 = n.get_lut();
        let lut2 = n.get_lut();
        assert_eq!(lut1, lut2);
    }
}
