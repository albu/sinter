// NEON SIMD implementation for horizontal flip (grayscale only)
//
// Uses vrev64q_u8 for efficient 64-bit lane reversal.

use crate::core::FusableImage;

/// Reverse a grayscale row using NEON SIMD
///
/// Algorithm:
/// 1. Process 16-byte (128-bit) chunks from both ends
/// 2. Reverse each chunk using vrev64q_u8 (reverses 8-byte halves)
/// 3. Swap reversed chunks
/// 4. Handle remaining middle bytes if width is odd
#[cfg(target_arch = "aarch64")]
pub unsafe fn horizontal_flip_gray_neon(row: *mut u8, width: usize) {
    use std::arch::aarch64::*;

    const CHUNK_SIZE: usize = 16; // 128-bit NEON register

    let mut left = 0usize;
    let mut right = width;

    // Process 16-byte chunks from both ends
    while left + CHUNK_SIZE <= right {
        right -= CHUNK_SIZE;

        // Load 16 bytes from each end
        let left_chunk = vld1q_u8(row.add(left));
        let right_chunk = vld1q_u8(row.add(right));

        // Reverse each 8-byte half and swap them
        // vrev64q_u8 reverses bytes within each 64-bit lane
        let left_rev = vrev64q_u8(left_chunk);
        let right_rev = vrev64q_u8(right_chunk);

        // Combine: swap the 64-bit lanes
        // We need to extract the high and low 64-bit lanes and swap them
        let left_high = vget_high_u8(left_rev);
        let left_low = vget_low_u8(left_rev);
        let right_high = vget_high_u8(right_rev);
        let right_low = vget_low_u8(right_rev);

        // Reconstruct with swapped lanes (high lane first for true reversal)
        let left_final = vcombine_u8(right_high, right_low);
        let right_final = vcombine_u8(left_high, left_low);

        // Store back
        vst1q_u8(row.add(left), left_final);
        vst1q_u8(row.add(right), right_final);

        left += CHUNK_SIZE;
    }

    // Handle remaining middle bytes (odd width or < 16 bytes remaining)
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

/// Stub for non-ARM64 platforms
#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn horizontal_flip_gray_neon(_row: *mut u8, _width: usize) {
    unreachable!("NEON path should not be called on non-ARM64");
}
