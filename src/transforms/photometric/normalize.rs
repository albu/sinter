// Normalization
//
// True normalization: out = (v / 255 - mean) / std, produced as float32.
// This is a TERMINAL transform: it allocates a float32 buffer, so the
// engine cannot run further u8 nodes after it. Pipelines must end here
// (the Python binding validates this and raises a clean error).

use crate::core::{AccessPattern, BarrierImage, Executable, FusableImage, ShapeEffect, Transform, ReorderRule};

/// Normalization
///
/// Scales uint8 pixel values to float32:
///
/// ```text
/// out = (v / 255 - mean) / std
/// ```
///
/// This matches torchvision/albumentations semantics. Because the output
/// dtype is float32, Normalize must be the LAST transform in a pipeline.
///
/// # Parameters
/// - `mean`: Mean value for normalization (typically 0.0)
/// - `std`: Standard deviation for normalization (must be > 0.0)
#[derive(Debug, Clone, PartialEq)]
pub struct Normalize {
    pub mean: f32,
    pub std: f32,
}

impl Normalize {
    /// Create a new Normalize transform
    ///
    /// # Panics
    /// Panics if std is zero or negative
    pub fn new(mean: f32, std: f32) -> Self {
        assert!(std > 0.0, "std must be positive, got {}", std);
        Self { mean, std }
    }

    /// Create standard normalization (mean=0, std=1) -> v / 255
    pub fn standard() -> Self {
        Self::new(0.0, 1.0)
    }
}

impl Transform for Normalize {
    fn access(&self) -> AccessPattern {
        // Allocates the float32 output buffer.
        AccessPattern::OutOfPlace
    }

    fn shape_effect(&self) -> ShapeEffect {
        ShapeEffect::Preserve
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn reorder_rule(&self) -> ReorderRule {
        // Changing dtype breaks the u8 world: never move Normalize across
        // other ops or the optimizer would place non-float ops after it.
        ReorderRule::Barrier
    }

    fn as_executable(&self) -> Option<&dyn crate::core::Executable> {
        Some(self)
    }
}

impl Executable for Normalize {
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        // Fold the reference formula (v / 255 - mean) / std into one
        // multiply-add: out = v * scale + bias.
        let scale = 1.0 / (255.0 * self.std);
        let bias = -self.mean / self.std;

        let len = image.data.len();
        let mut out = vec![0f32; len];
        normalize_u8_to_f32(&image.data, &mut out, scale, bias);

        Some(BarrierImage::from_f32_vec(
            out,
            image.width,
            image.height,
            image.channels,
        ))
    }
}

/// Elementwise `out[i] = src[i] as f32 * scale + bias` (HWC layout preserved)
#[cfg(target_arch = "aarch64")]
fn normalize_u8_to_f32(src: &[u8], dst: &mut [f32], scale: f32, bias: f32) {
    use std::arch::aarch64::*;

    let len = src.len();
    assert!(dst.len() >= len);
    let vscale = unsafe { vdupq_n_f32(scale) };
    let vbias = unsafe { vdupq_n_f32(bias) };

    let mut i = 0;
    // 16 lanes per iteration
    unsafe {
        while i + 16 <= len {
            let v8 = vld1q_u8(src.as_ptr().add(i));
            let lo16 = vmovl_u8(vget_low_u8(v8));
            let hi16 = vmovl_u8(vget_high_u8(v8));
            let f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16)));
            let f1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo16)));
            let f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16)));
            let f3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi16)));
            vst1q_f32(dst.as_mut_ptr().add(i), vmlaq_f32(vbias, f0, vscale));
            vst1q_f32(dst.as_mut_ptr().add(i + 4), vmlaq_f32(vbias, f1, vscale));
            vst1q_f32(dst.as_mut_ptr().add(i + 8), vmlaq_f32(vbias, f2, vscale));
            vst1q_f32(dst.as_mut_ptr().add(i + 12), vmlaq_f32(vbias, f3, vscale));
            i += 16;
        }
        while i < len {
            dst[i] = src[i] as f32 * scale + bias;
            i += 1;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn normalize_u8_to_f32(src: &[u8], dst: &mut [f32], scale: f32, bias: f32) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d = s as f32 * scale + bias;
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
    fn test_normalize_matches_reference() {
        // out = (v / 255 - mean) / std, in f32
        let mut data: Vec<u8> = (0..=255u8).collect();
        let mut img = FusableImage::new(&mut data, 16, 16, 1);
        let n = Normalize::new(0.45, 0.22);

        let barrier = n.execute(&mut img).unwrap();
        let out = barrier.f32_data.unwrap();

        for i in 0..256 {
            let expected = (i as f32 / 255.0 - 0.45) / 0.22;
            assert!(
                (out[i] - expected).abs() < 1e-6,
                "v={}: got {}, expected {}",
                i,
                out[i],
                expected
            );
        }
        // Full dynamic range survives: min < 0 < max
        assert!(out.iter().cloned().fold(f32::MIN, f32::max) > 2.0);
        assert!(out.iter().cloned().fold(f32::MAX, f32::min) < -2.0);
    }

    #[test]
    fn test_normalize_preserves_dims_and_input() {
        let mut data = vec![10u8, 20, 30, 40, 50, 60]; // 2x1 RGB
        let before = data.clone();
        let mut img = FusableImage::new(&mut data, 2, 1, 3);

        let barrier = Normalize::standard().execute(&mut img).unwrap();
        assert_eq!((barrier.width, barrier.height, barrier.channels), (2, 1, 3));
        assert!(barrier.is_f32());
        assert_eq!(barrier.f32_data.as_ref().unwrap().len(), 6);
        // Input u8 buffer is untouched (OutOfPlace)
        assert_eq!(img.data, &before[..]);
    }

    #[test]
    fn test_normalize_access_pattern() {
        let n = Normalize::standard();
        assert_eq!(n.access(), AccessPattern::OutOfPlace);
        assert_eq!(n.shape_effect(), ShapeEffect::Preserve);
        assert_eq!(n.reorder_rule(), ReorderRule::Barrier);
    }
}
