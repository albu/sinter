// Pixel operations and fused execution
//
// This module provides the foundation for fusing multiple per-pixel
// transforms into a single-pass execution.
//
// SIMD-optimized implementation using std::simd (nightly only)
// Branchless clamping for improved performance

use crate::core::FusableImage;
use crate::transforms::runtime::utils::clamp::saturate_u8_branchless;
use std::fmt;

#[cfg(feature = "simd")]
use std::simd::prelude::*;

/// Per-pixel operation
///
/// This trait defines how a transform operates on a single pixel value.
/// Fused executors use this to apply multiple transforms in a single loop.
pub trait PixelOp: fmt::Debug {
    /// Apply the operation to a single pixel value (as f32 in [0, 255])
    fn apply(&self, value: f32) -> f32;

    /// Apply the operation to a SIMD vector of pixel values (nightly only)
    ///
    /// Default implementation applies the scalar operation to each lane.
    /// Transforms can override this for specialized SIMD implementations.
    #[cfg(feature = "simd")]
    fn apply_simd(&self, values: f32x8) -> f32x8 {
        let mut result = [0.0f32; 8];
        for i in 0..8 {
            result[i] = self.apply(values.as_array()[i]);
        }
        f32x8::from_array(result)
    }
}

/// Fused executor for photometric transforms
///
/// Executes multiple transforms in a single pass over the image.
/// This is the key innovation - zero intermediate buffers, one traversal.
///
/// Uses SIMD for vectorized processing when available (nightly feature).
pub struct FusedExecutor;

impl FusedExecutor {
    /// Execute a fused block of transforms on an image
    ///
    /// # Parameters
    /// - `image`: The image to transform (mutated in-place)
    /// - `ops`: Slice of operations to apply in order
    ///
    /// # Algorithm
    /// ```text
    /// for each pixel in image:
    ///     v = pixel as f32
    ///     for each op in ops:
    ///         v = op.apply(v)
    ///     pixel = clamp(v, 0, 255) as u8
    /// ```
    ///
    /// # Properties
    /// - Single traversal of image data
    /// - No intermediate allocations
    /// - Cache-friendly sequential access
    /// - SIMD-accelerated when built with nightly (portable_simd feature)
    pub fn execute(image: &mut FusableImage, ops: &[Box<dyn PixelOp>]) {
        #[cfg(feature = "simd")]
        {
            // Use SIMD for larger images, scalar for small images
            const SIMD_THRESHOLD: usize = 32;

            if image.data.len() >= SIMD_THRESHOLD && !ops.is_empty() {
                Self::execute_simd(image, ops);
            } else {
                Self::execute_scalar(image, ops);
            }
        }

        #[cfg(not(feature = "simd"))]
        {
            Self::execute_scalar(image, ops);
        }
    }

    /// Scalar fallback for small images or when SIMD isn't available
    fn execute_scalar(image: &mut FusableImage, ops: &[Box<dyn PixelOp>]) {
        for px in image.data.iter_mut() {
            let mut v = *px as f32;
            for op in ops {
                v = op.apply(v);
            }
            // Use branchless saturating cast for better performance
            *px = saturate_u8_branchless(v);
        }
    }

    /// SIMD-optimized execution using f32x8 vectors (nightly only)
    ///
    /// Processes 8 pixels at a time for better throughput.
    /// Falls back to scalar for the remainder.
    #[cfg(feature = "simd")]
    fn execute_simd(image: &mut FusableImage, ops: &[Box<dyn PixelOp>]) {
        use std::simd::prelude::*;

        let data_len = image.data.len();
        let simd_len = data_len - (data_len % 8); // Round down to multiple of 8

        // Process 8 pixels at a time
        for i in (0..simd_len).step_by(8) {
            // Load 8 pixels as f32
            let mut pixels = [0.0f32; 8];
            for j in 0..8 {
                pixels[j] = image.data[i + j] as f32;
            }
            let mut values = f32x8::from_array(pixels);

            // Apply all operations
            for op in ops {
                values = op.apply_simd(values);
            }

            // Clamp and store back - use branchless saturating cast
            let array = values.as_array();
            for j in 0..8 {
                image.data[i + j] = saturate_u8_branchless(array[j]);
            }
        }

        // Handle remaining pixels with scalar code
        for i in simd_len..data_len {
            let mut v = image.data[i] as f32;
            for op in ops {
                v = op.apply(v);
            }
            image.data[i] = saturate_u8_branchless(v);
        }
    }
}

/// Convenience: Execute transforms that implement PixelOp
pub fn execute_fused<T: PixelOp + Copy + 'static>(image: &mut FusableImage, ops: &[T]) {
    // Convert to trait objects for uniform handling
    let boxed_ops: Vec<Box<dyn PixelOp>> = ops
        .iter()
        .map(|op| -> Box<dyn PixelOp> { Box::new(*op) })
        .collect();
    FusedExecutor::execute(image, &boxed_ops);
}

// Note: PixelOp tests removed - LUT-capable transforms (Brightness, Contrast, Normalize,
// Invert, Posterize, Solarize) no longer implement PixelOp. PixelOp is now only used by
// Gamma (which doesn't have LUT support). Gamma tests are in gamma.rs.
