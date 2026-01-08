// NEON SIMD implementation for image inversion
//
// Uses vmvnq_u8 (bitwise NOT) for single-instruction inversion.

use crate::core::FusableImage;

/// Apply image inversion using NEON SIMD
///
/// NEON's bitwise NOT (~x) is equivalent to 255-x for u8.
/// This is a single instruction per 16 pixels!
#[cfg(target_arch = "aarch64")]
pub fn apply_invert_neon(image: &mut FusableImage) {
    use std::arch::aarch64::*;
    let data = &mut image.data;
    let len = data.len();

    unsafe {
        let chunks = len / 16;

        for i in 0..chunks {
            let offset = i * 16;

            // Load 16 pixels
            let pixels = vld1q_u8(data.as_ptr().add(offset));

            // Bitwise NOT = 255 - x for u8 (single instruction!)
            let inverted = vmvnq_u8(pixels);

            // Store result
            vst1q_u8(data.as_mut_ptr().add(offset), inverted);
        }

        // Handle remaining pixels
        for i in (chunks * 16)..len {
            data[i] = 255 - data[i];
        }
    }
}

/// Fallback for non-ARM64 platforms (unused in this module)
#[cfg(not(target_arch = "aarch64"))]
pub fn apply_invert_neon(_image: &mut FusableImage) {
    unreachable!("NEON path should not be called on non-ARM64");
}
