// Hue Saturation Value (HSV) transform
//
// Adjusts hue, saturation, and value (brightness) of RGB images.

// mod fast_hue;
mod fast_impl;
mod rust_impl;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(test)]
mod tests;

use crate::core::{AccessPattern, BarrierImage, Executable, FusableImage, ShapeEffect, Transform};

use fast_impl::execute_fast;

#[cfg(target_arch = "aarch64")]
use neon::execute_fast_simd;
#[cfg(not(target_arch = "aarch64"))]
use fast_impl::execute_fast as execute_fast_simd;

/// Hue Saturation Value transform
///
/// Adjusts the HSV components of an RGB image.
/// Only applies to RGB images (channels == 3).
///
/// # Parameters
/// - `hue_shift`: Hue rotation in degrees [-180, 180]
///   - Positive shifts clockwise, negative counter-clockwise
/// - `sat_scale`: Saturation scaling factor [0.0, inf)
///   - 1.0 = no change, < 1.0 = desaturate, > 1.0 = saturate
/// - `val_scale`: Value (brightness) scaling factor [0.0, inf)
///   - 1.0 = no change, < 1.0 = darken, > 1.0 = brighten
///
/// # Implementation
/// Uses proper RGB→HSV→RGB conversion with optimized SIMD:
/// - NEON SIMD on ARM (aarch64)
/// - Scalar fallback on other architectures
///
/// # Performance
/// - Produces mathematically accurate results
/// - SIMD-optimized for 16-pixel batches
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HueSaturationValue {
    pub hue_shift: f32,
    pub sat_scale: f32,
    pub val_scale: f32,
}

impl HueSaturationValue {
    /// Create a new HueSaturationValue transform
    ///
    /// # Panics
    /// Panics if:
    /// - hue_shift is outside [-180, 180]
    /// - sat_scale is negative
    /// - val_scale is negative
    pub fn new(hue_shift: f32, sat_scale: f32, val_scale: f32) -> Self {
        assert!(
            (-180.0..=180.0).contains(&hue_shift),
            "hue_shift must be in [-180, 180], got {}",
            hue_shift
        );
        assert!(
            sat_scale >= 0.0,
            "sat_scale must be >= 0, got {}",
            sat_scale
        );
        assert!(
            val_scale >= 0.0,
            "val_scale must be >= 0, got {}",
            val_scale
        );
        Self {
            hue_shift,
            sat_scale,
            val_scale,
        }
    }
}

impl Transform for HueSaturationValue {
    fn access(&self) -> AccessPattern {
        AccessPattern::InPlace
    }

    fn shape_effect(&self) -> ShapeEffect {
        ShapeEffect::Preserve
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_executable(&self) -> Option<&dyn Executable> {
        Some(self)
    }

    fn reorder_rule(&self) -> crate::core::ReorderRule {
        crate::core::ReorderRule::CommutesWithGeometry
    }
}

impl Executable for HueSaturationValue {
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        // Early exit for identity transform
        if self.hue_shift == 0.0 && self.sat_scale == 1.0 && self.val_scale == 1.0 {
            return None;
        }

        if self.hue_shift == 0.0 && image.channels == 3 {
            let s = self.sat_scale;
            let v = self.val_scale;
            let om_s = 1.0 - s;
            let r_r = (om_s * 0.299 + s) * v;
            let r_g = (om_s * 0.587) * v;
            let r_b = (om_s * 0.114) * v;

            let g_r = (om_s * 0.299) * v;
            let g_g = (om_s * 0.587 + s) * v;
            let g_b = (om_s * 0.114) * v;

            let b_r = (om_s * 0.299) * v;
            let b_g = (om_s * 0.587) * v;
            let b_b = (om_s * 0.114 + s) * v;

            let matrix = [
                [r_r, r_g, r_b],
                [g_r, g_g, g_b],
                [b_r, b_g, b_b],
            ];
            crate::transforms::runtime::matrix::MatrixExecutor::apply(image, &matrix);
            return None;
        }

        // Use SIMD implementation (with scalar fallback for non-RGB images)
        execute_fast_simd(self, image);
        None
    }
}
