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
            // Exact hexcone-model sat/val scaling (V = max), consistent with the
            // hue-shift SIMD path. Previously this branch used a luma-weighted
            // matrix (0.299/0.587/0.114) that disagreed with the hue path by up
            // to ~50 for sat_scale 1.3 — same params, different result depending
            // on whether hue_shift was 0.
            let s = self.sat_scale;
            let v = self.val_scale;
            #[cfg(target_arch = "aarch64")]
            {
                if s <= 1.0 && v <= 1.0 {
                    // No S/V clipping: closed-form RGB' = vs*ss*RGB + vs*(1-ss)*V
                    // (V = max) is exact and fast (Q8.8).
                    unsafe { neon::apply_satval_neon(image, s, v); }
                } else {
                    // ss > 1 or vs > 1 can clip S/V: use the exact hexcone path.
                    unsafe { neon::apply_satval_neon_exact(image, s, v); }
                }
                return None;
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                let data = &mut image.data;
                if s <= 1.0 && v <= 1.0 {
                    let m = s * v;
                    let k = v * (1.0 - s);
                    let mut i = 0;
                    while i + 3 <= data.len() {
                        let vmax = data[i].max(data[i + 1]).max(data[i + 2]) as f32;
                        for c in 0..3 {
                            let val = m * data[i + c] as f32 + k * vmax;
                            data[i + c] = val.clamp(0.0, 255.0).round() as u8;
                        }
                        i += 3;
                    }
                } else {
                    let mut i = 0;
                    while i + 3 <= data.len() {
                        let (r, g, b) = (data[i] as f32, data[i + 1] as f32, data[i + 2] as f32);
                        let mm = r.max(g).max(b);
                        let ll = r.min(g).min(b);
                        let uu = r + g + b - mm - ll;
                        let cc = mm - ll;
                        let vp = (mm * v).clamp(0.0, 255.0);
                        let cp = if mm > 0.0 { (vp * cc * s / mm).min(vp) } else { 0.0 };
                        let xp = if cc > 0.0 { cp * (uu - ll) / cc } else { 0.0 };
                        let mp = vp - cp;
                        let vals = [r, g, b];
                        for (c, &ch) in vals.iter().enumerate() {
                            let out = if ch == mm {
                                mp + cp
                            } else if ch == ll {
                                mp
                            } else {
                                mp + xp
                            };
                            data[i + c] = out.clamp(0.0, 255.0).round() as u8;
                        }
                        i += 3;
                    }
                }
                return None;
            }
            return None;
        }

        // Use SIMD implementation (with scalar fallback for non-RGB images)
        execute_fast_simd(self, image);
        None
    }
}
