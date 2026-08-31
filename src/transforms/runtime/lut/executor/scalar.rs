// Scalar (non-SIMD) LUT executor implementations
//
// Provides simple and unrolled loops for LUT application.
#![allow(dead_code)]

use crate::core::FusableImage;

/// Scalar LUT executor - simple and unrolled implementations
pub struct LutExecutorScalar;

impl LutExecutorScalar {
    /// Simple application for small images
    #[inline]
    pub fn apply_simple(image: &mut FusableImage, lut: &[u8; 256]) {
        for px in image.data.iter_mut() {
            *px = lut[*px as usize];
        }
    }

    /// Unrolled application for larger images
    ///
    /// Process 8 pixels at a time to allow:
    /// - Better instruction pipelining
    /// - Reduced loop overhead
    /// - Potential SIMD auto-vectorization by compiler
    #[inline]
    pub fn apply_unrolled(image: &mut FusableImage, lut: &[u8; 256]) {
        let data = &mut image.data;
        let len = data.len();
        let unroll_len = len - (len % 8);

        // Process 8 pixels at a time
        let mut i = 0;
        while i < unroll_len {
            // Manual unrolling - helps compiler vectorize
            data[i] = lut[data[i] as usize];
            data[i + 1] = lut[data[i + 1] as usize];
            data[i + 2] = lut[data[i + 2] as usize];
            data[i + 3] = lut[data[i + 3] as usize];
            data[i + 4] = lut[data[i + 4] as usize];
            data[i + 5] = lut[data[i + 5] as usize];
            data[i + 6] = lut[data[i + 6] as usize];
            data[i + 7] = lut[data[i + 7] as usize];
            i += 8;
        }

        // Handle remaining pixels
        for i in unroll_len..len {
            data[i] = lut[data[i] as usize];
        }
    }

    /// Apply a LUT using the scalar implementation
    /// This is a convenience method that chooses the appropriate scalar implementation.
    #[inline]
    pub fn apply(image: &mut FusableImage, lut: &[u8; 256]) {
        const UNROLL_THRESHOLD: usize = 64;
        if image.data.len() >= UNROLL_THRESHOLD {
            Self::apply_unrolled(image, lut);
        } else {
            Self::apply_simple(image, lut);
        }
    }
}
