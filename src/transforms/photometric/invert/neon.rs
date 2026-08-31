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
        let chunks = len / 64;
        let mut ptr = data.as_mut_ptr();
        for _ in 0..chunks {
            let v0 = vld1q_u8(ptr);
            let v1 = vld1q_u8(ptr.add(16));
            let v2 = vld1q_u8(ptr.add(32));
            let v3 = vld1q_u8(ptr.add(48));

            vst1q_u8(ptr, vmvnq_u8(v0));
            vst1q_u8(ptr.add(16), vmvnq_u8(v1));
            vst1q_u8(ptr.add(32), vmvnq_u8(v2));
            vst1q_u8(ptr.add(48), vmvnq_u8(v3));

            ptr = ptr.add(64);
        }

        let rem_chunks = (len % 64) / 16;
        for _ in 0..rem_chunks {
            let v = vld1q_u8(ptr);
            vst1q_u8(ptr, vmvnq_u8(v));
            ptr = ptr.add(16);
        }

        let rem_start = chunks * 64 + rem_chunks * 16;
        for i in rem_start..len {
            data[i] = 255 - data[i];
        }
    }
}

/// Fallback for non-ARM64 platforms (unused in this module)
#[cfg(not(target_arch = "aarch64"))]
pub fn apply_invert_neon(_image: &mut FusableImage) {
    unreachable!("NEON path should not be called on non-ARM64");
}
