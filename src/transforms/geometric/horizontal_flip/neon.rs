// NEON SIMD implementation for horizontal flip (grayscale and RGB)
//
// Uses vrev64q_u8 and vcombine_u8 for efficient register reversal.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn reverse16(v: uint8x16_t) -> uint8x16_t {
    let rev = vrev64q_u8(v);
    vcombine_u8(vget_high_u8(rev), vget_low_u8(rev))
}

/// Reverse a grayscale row using NEON SIMD
#[cfg(target_arch = "aarch64")]
pub unsafe fn horizontal_flip_gray_neon(row: *mut u8, width: usize) {
    const CHUNK_SIZE: usize = 16;

    let mut left = 0usize;
    let mut right = width;

    while left + CHUNK_SIZE <= right {
        right -= CHUNK_SIZE;

        let left_chunk = vld1q_u8(row.add(left));
        let right_chunk = vld1q_u8(row.add(right));

        let left_final = reverse16(right_chunk);
        let right_final = reverse16(left_chunk);

        vst1q_u8(row.add(left), left_final);
        vst1q_u8(row.add(right), right_final);

        left += CHUNK_SIZE;
    }

    while left < right {
        right -= 1;
        let left_ptr = row.add(left);
        let right_ptr = row.add(right);
        let temp = *left_ptr;
        *left_ptr = *right_ptr;
        *right_ptr = temp;
        left += 1;
    }
}

/// Reverse an RGB row using NEON SIMD
#[cfg(target_arch = "aarch64")]
pub unsafe fn horizontal_flip_rgb_neon(row: *mut u8, width: usize) {
    const CHUNK_SIZE: usize = 16;

    let mut left = 0usize;
    let mut right = width;

    while left + CHUNK_SIZE <= right {
        right -= CHUNK_SIZE;

        let left_ptr = row.add(left * 3);
        let right_ptr = row.add(right * 3);

        let left_rgb = vld3q_u8(left_ptr);
        let right_rgb = vld3q_u8(right_ptr);

        let rev_r_left = reverse16(left_rgb.0);
        let rev_g_left = reverse16(left_rgb.1);
        let rev_b_left = reverse16(left_rgb.2);

        let rev_r_right = reverse16(right_rgb.0);
        let rev_g_right = reverse16(right_rgb.1);
        let rev_b_right = reverse16(right_rgb.2);

        vst3q_u8(left_ptr, uint8x16x3_t(rev_r_right, rev_g_right, rev_b_right));
        vst3q_u8(right_ptr, uint8x16x3_t(rev_r_left, rev_g_left, rev_b_left));

        left += CHUNK_SIZE;
    }

    while left < right {
        right -= 1;
        let left_ptr = row.add(left * 3);
        let right_ptr = row.add(right * 3);

        let (r0, g0, b0) = (*left_ptr, *left_ptr.add(1), *left_ptr.add(2));
        let (r1, g1, b1) = (*right_ptr, *right_ptr.add(1), *right_ptr.add(2));

        *left_ptr = r1;
        *left_ptr.add(1) = g1;
        *left_ptr.add(2) = b1;

        *right_ptr = r0;
        *right_ptr.add(1) = g0;
        *right_ptr.add(2) = b0;

        left += 1;
    }
}
