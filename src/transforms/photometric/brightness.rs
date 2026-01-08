// Brightness adjustment
//
// Adds a constant delta to pixel values.

use crate::core::{AccessPattern, Executable, FusableImage, BarrierImage, ShapeEffect, Transform, ReorderRule};
use crate::transforms::runtime::lut::LutOp;

/// Brightness adjustment
///
/// Adds a constant delta to pixel values.
///
/// # Parameters
/// - `delta`: Brightness adjustment in range [-255, 255]
///   - Positive values increase brightness
///   - Negative values decrease brightness
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Brightness {
    pub delta: f32,
    // Note: LUT caching requires interior mutability, but Copy trait prevents this.
    // For single transforms, we cache at the call site. For fusion, build_lut() is
    // called once during planning, so caching is not needed there.
}

impl Brightness {
    /// Create a new Brightness transform
    ///
    /// # Panics
    /// Panics if delta is outside [-255, 255]
    pub fn new(delta: f32) -> Self {
        assert!(
            (-255.0..=255.0).contains(&delta),
            "delta must be in [-255, 255], got {}",
            delta
        );
        Self { delta }
    }
}

impl Transform for Brightness {
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

impl Executable for Brightness {
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        // Use LUT execution - benchmarked faster than scalar unrolling
        self.execute_with_lut(image);
        None
    }
}

/// Implement LutOp for Brightness
///
/// LUT: lut[i] = clamp(i + delta, 0, 255)
impl LutOp for Brightness {
    fn build_lut(&self) -> [u8; 256] {
        let mut lut = [0u8; 256];
        for i in 0..256 {
            let x = i as f32;
            let y = x + self.delta;
            lut[i] = y.clamp(0.0, 255.0) as u8;
        }
        lut
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_brightness_new() {
        let b = Brightness::new(10.0);
        assert_eq!(b.delta, 10.0);
    }

    #[test]
    #[should_panic(expected = "delta must be in")]
    fn test_brightness_invalid_delta() {
        Brightness::new(300.0);
    }

    #[test]
    fn bench_brightness_lut() {
        let mut data = vec![128u8; 512 * 512 * 3];
        let mut img = FusableImage::new(&mut data, 512, 512, 3);
        let brightness = Brightness::new(50.0);

        // Warmup
        for _ in 0..10 {
            brightness.execute_with_lut(&mut img);
        }

        let start = Instant::now();
        for _ in 0..1000 {
            brightness.execute_with_lut(&mut img);
        }
        let elapsed = start.elapsed();

        println!("LUT Brightness per call: {:?}", elapsed / 1000);
    }
}

