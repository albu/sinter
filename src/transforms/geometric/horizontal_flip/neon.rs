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

/// Reverse a grayscale row using NEON SIMD (32-byte unrolled)
#[cfg(target_arch = "aarch64")]
pub unsafe fn horizontal_flip_gray_neon(row: *mut u8, width: usize) {
    let mut left = 0usize;
    let mut right = width;

    while left + 32 <= right {
        right -= 32;

        let l0 = vld1q_u8(row.add(left));
        let l1 = vld1q_u8(row.add(left + 16));
        let r0 = vld1q_u8(row.add(right));
        let r1 = vld1q_u8(row.add(right + 16));

        vst1q_u8(row.add(left), reverse16(r1));
        vst1q_u8(row.add(left + 16), reverse16(r0));
        vst1q_u8(row.add(right), reverse16(l1));
        vst1q_u8(row.add(right + 16), reverse16(l0));

        left += 32;
    }

    while left + 16 <= right {
        right -= 16;

        let left_chunk = vld1q_u8(row.add(left));
        let right_chunk = vld1q_u8(row.add(right));

        vst1q_u8(row.add(left), reverse16(right_chunk));
        vst1q_u8(row.add(right), reverse16(left_chunk));

        left += 16;
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

/// Reverse an RGB row using NEON SIMD (32-pixel unrolled)
#[cfg(target_arch = "aarch64")]
pub unsafe fn horizontal_flip_rgb_neon(row: *mut u8, width: usize) {
    let mut left = 0usize;
    let mut right = width;

    while left + 32 <= right {
        right -= 32;

        let left_ptr0 = row.add(left * 3);
        let left_ptr1 = row.add((left + 16) * 3);
        let right_ptr0 = row.add(right * 3);
        let right_ptr1 = row.add((right + 16) * 3);

        let left_rgb0 = vld3q_u8(left_ptr0);
        let left_rgb1 = vld3q_u8(left_ptr1);
        let right_rgb0 = vld3q_u8(right_ptr0);
        let right_rgb1 = vld3q_u8(right_ptr1);

        let rev_r_r1 = reverse16(right_rgb1.0);
        let rev_g_r1 = reverse16(right_rgb1.1);
        let rev_b_r1 = reverse16(right_rgb1.2);

        let rev_r_r0 = reverse16(right_rgb0.0);
        let rev_g_r0 = reverse16(right_rgb0.1);
        let rev_b_r0 = reverse16(right_rgb0.2);

        let rev_r_l0 = reverse16(left_rgb0.0);
        let rev_g_l0 = reverse16(left_rgb0.1);
        let rev_b_l0 = reverse16(left_rgb0.2);

        let rev_r_l1 = reverse16(left_rgb1.0);
        let rev_g_l1 = reverse16(left_rgb1.1);
        let rev_b_l1 = reverse16(left_rgb1.2);

        vst3q_u8(left_ptr0, uint8x16x3_t(rev_r_r1, rev_g_r1, rev_b_r1));
        vst3q_u8(left_ptr1, uint8x16x3_t(rev_r_r0, rev_g_r0, rev_b_r0));
        vst3q_u8(right_ptr0, uint8x16x3_t(rev_r_l1, rev_g_l1, rev_b_l1));
        vst3q_u8(right_ptr1, uint8x16x3_t(rev_r_l0, rev_g_l0, rev_b_l0));

        left += 32;
    }

    while left + 16 <= right {
        right -= 16;

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

        left += 16;
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
