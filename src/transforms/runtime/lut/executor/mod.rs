// LUT executors for single and fused operations
//
// Provides optimized execution strategies for LUT-based transforms.

mod scalar;
#[cfg(target_arch = "aarch64")]
mod neon;

use crate::core::FusableImage;
use super::LutOp;

/// Fused LUT executor for multiple LUT transforms
///
/// Composes multiple LUTs into a single LUT for single-pass execution.
///
/// # Algorithm
/// Given transforms T1, T2, T3 with LUTs L1, L2, L3:
/// ```text
/// lut_fused[i] = L3[L2[L1[i]]]
/// ```
///
/// Then apply lut_fused in a single pass.
pub struct FusedLutExecutor;

impl FusedLutExecutor {
    /// Execute multiple LUT transforms in a single pass
    ///
    /// # Parameters
    /// - `image`: The image to transform
    /// - `ops`: Slice of LUT operations to apply in order
    ///
    /// # Algorithm
    /// 1. Compose all LUTs into a single LUT (O(256 * n_ops))
    /// 2. Apply composed LUT to image (O(image_size))
    ///
    /// This is much faster than applying each LUT separately when there are
    /// multiple transforms.
    pub fn execute(image: &mut FusableImage, ops: &[Box<dyn LutOp>]) {
        if ops.is_empty() {
            return;
        }

        // Single transform: use fast path
        if ops.len() == 1 {
            ops[0].execute_with_lut(image);
            return;
        }

        // Multiple transforms: compose LUTs and apply with optimized executor
        let fused_lut = Self::compose_luts(ops);
        LutExecutor::apply(image, &fused_lut);
    }

    /// Compose multiple LUTs into a single LUT
    ///
    /// Given ops = [T1, T2, T3], computes:
    /// ```text
    /// lut_fused[i] = T3.build_lut()[T2.build_lut()[T1.build_lut()[i]]]
    /// ```
    ///
    /// This is O(256 * n_ops) regardless of image size.
    pub fn compose_luts(ops: &[Box<dyn LutOp>]) -> [u8; 256] {
        // Start with identity LUT
        let mut fused = [0u8; 256];
        for i in 0..256 {
            fused[i] = i as u8;
        }

        // Compose each LUT in sequence
        for op in ops {
            let lut = op.build_lut();
            // Apply this LUT to the composed LUT
            for i in 0..=255u8 {
                fused[i as usize] = lut[fused[i as usize] as usize];
            }
        }

        fused
    }
}

/// LUT executor with optimized application
///
/// Uses multiple strategies depending on image size:
/// 1. For large images: uses loop unrolling for better throughput
/// 2. For small images: simple loop (avoid overhead)
/// 3. Always cache-friendly sequential access
pub struct LutExecutor;

impl LutExecutor {
    /// Apply a LUT to an image
    ///
    /// # Algorithm
    /// ```text
    /// for each pixel in image:
    ///     pixel = lut[pixel]
    /// ```
    ///
    /// # Optimizations
    /// - NEON vqtbl4q_u8 on ARM (12-18 GB/s)
    /// - Sequential memory access (cache-friendly)
    /// - Loop unrolling for larger images
    /// - No branching in the hot loop
    pub fn apply(image: &mut FusableImage, lut: &[u8; 256]) {
        #[cfg(target_arch = "aarch64")]
        {
            // Apple M1/M2/M3 always supports NEON with vqtbl4q_u8
            // This gives ~12-18 GB/s vs ~3.3 GB/s scalar
            if image.data.len() >= 16 {
                unsafe { neon::apply_neon_vqtbl(image, lut); }
                return;
            }
        }

        const UNROLL_THRESHOLD: usize = 64;
        if image.data.len() >= UNROLL_THRESHOLD {
            scalar::LutExecutorScalar::apply_unrolled(image, lut);
        } else {
            scalar::LutExecutorScalar::apply_simple(image, lut);
        }
    }

    /// Apply a per-channel LUT to an interleaved RGB image (3 channels).
    ///
    /// Byte `i` of the image is remapped through `luts[i % 3]`. On ARM this
    /// uses a NEON vld3q/vst3q gather; elsewhere it falls back to scalar.
    pub fn apply_rgb_luts(image: &mut FusableImage, luts: &[[u8; 256]; 3]) {
        #[cfg(target_arch = "aarch64")]
        {
            if image.data.len() >= 48 {
                unsafe { neon::apply_neon_vqtbl3(image, luts); }
                return;
            }
        }

        let data = &mut image.data;
        for (i, px) in data.iter_mut().enumerate() {
            *px = luts[i % 3][*px as usize];
        }
    }
}

// Re-export the scalar implementation for external use
pub use scalar::LutExecutorScalar;
