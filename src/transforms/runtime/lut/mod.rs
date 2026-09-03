// Look-Up Table (LUT) optimization for per-pixel transforms
//
// LUT is extremely fast for transforms that operate on uint8 values:
// - Pre-compute output for all 256 possible input values
// - Apply by simple table lookup (no per-pixel computation)
// - Cache-friendly and SIMD-able
//
// Performance benefits:
// - Solarize: ~3-5x faster than per-pixel computation
// - Posterize: ~3-5x faster than per-pixel computation
// - Invert: ~3-5x faster than per-pixel computation
//
// LUT FUSION:
// Multiple LUT transforms can be fused into a single LUT:
// - Compose LUTs: lut_fused[i] = lut3[lut2[lut1[i]]]
// - Apply composed LUT in single pass
// - Best of both worlds: LUT speed + fusion speed

use std::fmt;

mod executor;
mod fused;
#[cfg(test)]
mod tests;

// Re-export for convenience
pub use executor::{LutExecutor, FusedLutExecutor};
pub use fused::FusedLut;

/// Trait for transforms that can use LUT optimization
///
/// Any transform that:
/// - Takes u8 input (0-255)
/// - Produces u8 output (0-255)
/// - Is position-independent (same input always gives same output)
///
/// Can benefit from LUT optimization.
pub trait LutOp: fmt::Debug {
    /// Build the 256-entry LUT for this transform
    ///
    /// LUT[i] = output value for input value i
    fn build_lut(&self) -> [u8; 256];

    /// Get the LUT, with caching by default
    ///
    /// Transforms that implement caching should override this to return
    /// a cached LUT. The default implementation calls build_lut() every time.
    fn get_lut(&self) -> [u8; 256] {
        self.build_lut()
    }

    /// Build the 3-channel (RGB) 256-entry LUTs for this transform.
    /// Default implementation uses the single-channel LUT across all 3 channels.
    fn build_lut_3c(&self) -> [[u8; 256]; 3] {
        let lut = self.build_lut();
        [lut, lut, lut]
    }

    /// Whether this LUT operation produces different LUTs per channel.
    fn is_3c(&self) -> bool {
        false
    }

    /// Execute using LUT (default implementation)
    ///
    /// Transforms can override this for specialized behavior.
    fn execute_with_lut(&self, image: &mut crate::core::FusableImage) {
        if self.is_3c() && image.channels == 3 {
            let luts = self.build_lut_3c();
            LutExecutor::apply_rgb_luts(image, &luts);
        } else {
            let lut = self.get_lut();
            LutExecutor::apply(image, &lut);
        }
    }
}

