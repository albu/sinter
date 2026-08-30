// NEON SIMD optimizations for pad operations
//
// This module contains ARM64 NEON implementations for padding operations.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ============================================================================
// NEON SIMD Implementation (AArch64 only)
// ============================================================================

/// Pad with constant value using NEON SIMD
///
/// Optimized to avoid double-write: only fills padding regions, then copies image data.
/// This is critical for cache-resident images (512x512 fits in L2 cache).
#[cfg(target_arch = "aarch64")]
pub unsafe fn pad_constant_neon(
    dst: &mut [u8],
    src: &[u8],
    new_width: usize,
    src_width: usize,
    src_height: usize,
    top: usize,
    left: usize,
    src_stride: usize,
    fill: u8,
    channels: usize,
) {
    let dst_stride = new_width * channels;
    let new_height = dst.len() / dst_stride;
    let bottom = top + src_height;
    let fill_vec = vdupq_n_u8(fill);

    // Fill top padding region (entire rows)
    for y in 0..top {
        let row_start = y * dst_stride;
        fill_row_neon(dst, row_start, dst_stride, fill_vec);
    }

    // Fill middle section: left padding + image + right padding
    for y in top..bottom {
        let row_start = y * dst_stride;
        let image_start = row_start + left * channels;

        // Fill left padding
        if left > 0 {
            fill_row_neon(dst, row_start, left * channels, fill_vec);
        }

        // Copy image data (this region is NOT pre-filled, avoiding double-write)
        let src_row_start = (y - top) * src_stride;
        copy_row_neon(dst, src, image_start, src_row_start, src_width, channels);

        // Fill right padding
        let image_end = image_start + src_stride;
        let row_end = row_start + dst_stride;
        if image_end < row_end {
            fill_row_neon(dst, image_end, row_end - image_end, fill_vec);
        }
    }

    // Fill bottom padding region (entire rows)
    for y in bottom..new_height {
        let row_start = y * dst_stride;
        fill_row_neon(dst, row_start, dst_stride, fill_vec);
    }
}

/// Fill a row range using NEON SIMD
#[cfg(target_arch = "aarch64")]
unsafe fn fill_row_neon(dst: &mut [u8], start: usize, len: usize, fill_vec: uint8x16_t) {
    let mut i = start;
    let end = start + len;
    while i + 16 <= end {
        vst1q_u8(dst.as_mut_ptr().add(i), fill_vec);
        i += 16;
    }
    // Handle remaining bytes
    while i < end {
        dst[i] = vgetq_lane_u8(fill_vec, 0);
        i += 1;
    }
}

/// Copy a row using optimized approach
///
/// For simple row copies without processing, memcpy (via copy_from_slice) is
/// faster than SIMD de-interleaving operations like vld3_u8/vst3_u8.
///
/// We use memcpy for all cases since:
/// 1. RGB: vld3_u8/vst3_u8 de-interleaves unnecessarily for simple copy
/// 2. Grayscale: memcpy is as fast as vld1q_u8/vst1q_u8 for contiguous data
#[cfg(target_arch = "aarch64")]
unsafe fn copy_row_neon(
    dst: &mut [u8],
    src: &[u8],
    dst_start: usize,
    src_start: usize,
    width: usize,
    channels: usize,
) {
    let byte_count = width * channels;
    dst[dst_start..dst_start + byte_count]
        .copy_from_slice(&src[src_start..src_start + byte_count]);
}

/// Reverse-copy `n` bytes: `dst[i] = src[n - 1 - i]`, using vrev64q on 16-byte
/// chunks and a scalar tail.
#[cfg(target_arch = "aarch64")]
unsafe fn reverse_bytes_neon(dst: *mut u8, src: *const u8, n: usize) {
    let mut i = 0usize;
    while i + 16 <= n {
        let chunk = vld1q_u8(src.add(i));
        let rev64 = vrev64q_u8(chunk);
        let rev = vcombine_u8(vget_high_u8(rev64), vget_low_u8(rev64));
        vst1q_u8(dst.add(n - i - 16), rev);
        i += 16;
    }
    while i < n {
        *dst.add(n - 1 - i) = *src.add(i);
        i += 1;
    }
}

/// Fill `n` bytes at `dst` with `val` using NEON.
#[cfg(target_arch = "aarch64")]
unsafe fn fill_run_neon_ptr(dst: *mut u8, n: usize, val: u8) {
    if n == 0 {
        return;
    }
    let fill_vec = vdupq_n_u8(val);
    let mut i = 0usize;
    while i + 16 <= n {
        vst1q_u8(dst.add(i), fill_vec);
        i += 16;
    }
    while i < n {
        *dst.add(i) = val;
        i += 1;
    }
}

/// Pad with reflection (NEON-accelerated for grayscale)
///
/// Per output row:
/// - Left edge: reversed copy of the source row's first `left` pixels
///   (edge-saturated when `left` exceeds the source width)
/// - Center: direct memcpy of the reflected source row
/// - Right edge: reversed copy of the source row's last `right` pixels
///   (edge-saturated when `right` exceeds the source width)
///
/// The horizontal reflect mapping is `dst[x] = src[reflect(x)]` where
/// reflect() mirrors around the border pixels (-1 maps to 0, matching
/// `pad_fast_slice`'s Reflect map_coord exactly). When the border is no larger
/// than the source dimension the edge is a pure byte reversal of a contiguous
/// run (vrev64q); when the border exceeds the dimension the overflow saturates
/// at the edge pixel (memset) followed by the reversed run.
///
/// Only channels == 1 is routed here today; any other channel count defers to
/// the scalar reference so semantics stay correct for all inputs.
#[cfg(target_arch = "aarch64")]
pub fn pad_reflect_neon(
    dst: &mut [u8],
    src: &[u8],
    new_width: usize,
    new_height: usize,
    src_width: usize,
    src_height: usize,
    top: usize,
    left: usize,
    channels: usize,
) {
    if channels != 1 {
        super::pad_reflect_scalar(
            dst, src, new_width, src_width, src_height, top, left, channels,
        );
        return;
    }

    let src_stride = src_width * channels;
    let dst_stride = new_width * channels;

    // Reflect coordinate in the vertical direction (row mapping).
    let reflect_row = |p: i32, border: usize, size: usize| -> usize {
        let val = p - border as i32;
        if val < 0 {
            (-val - 1).clamp(0, size as i32 - 1) as usize
        } else if val >= size as i32 {
            (2 * size as i32 - val - 1).clamp(0, size as i32 - 1) as usize
        } else {
            val as usize
        }
    };

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    let left_bytes = left * channels;
    let right_bytes = (new_width - left - src_width) * channels;

    for y in 0..new_height {
        let src_y = reflect_row(y as i32, top, src_height);
        let s_row = unsafe { src_ptr.add(src_y * src_stride) };
        let d_row = unsafe { dst_ptr.add(y * dst_stride) };

        // Left edge: dst[0..left] = reflect(src[0..left])
        if left_bytes > 0 {
            if left <= src_width {
                unsafe { reverse_bytes_neon(d_row, s_row, left_bytes) };
            } else {
                // Saturate: [src[iw-1]] * (left - iw), then the reversed full row.
                let sat_px = left - src_width;
                unsafe {
                    fill_run_neon_ptr(
                        d_row,
                        sat_px * channels,
                        *s_row.add((src_width - 1) * channels),
                    );
                    reverse_bytes_neon(d_row.add(sat_px * channels), s_row, src_stride);
                }
            }
        }

        // Center: copy the reflected source row.
        unsafe {
            std::ptr::copy_nonoverlapping(s_row, d_row.add(left_bytes), src_stride);
        }

        // Right edge: dst[left + iw ..] = reflect(src[iw - right .. iw])
        if right_bytes > 0 {
            let d_edge = unsafe { d_row.add(left_bytes + src_stride) };
            if right_bytes <= src_stride {
                unsafe {
                    reverse_bytes_neon(
                        d_edge,
                        s_row.add(src_stride - right_bytes),
                        right_bytes,
                    );
                }
            } else {
                // Saturate: reversed full row, then [src[0]] * (right - iw).
                let sat_px = right_bytes - src_stride;
                unsafe {
                    reverse_bytes_neon(d_edge, s_row, src_stride);
                    fill_run_neon_ptr(d_edge.add(src_stride), sat_px, *s_row);
                }
            }
        }
    }
}

// ============================================================================
// Fallback implementations for non-ARM64 platforms
// ============================================================================

#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn pad_constant_neon(
    dst: &mut [u8],
    src: &[u8],
    new_width: usize,
    src_width: usize,
    src_height: usize,
    top: usize,
    left: usize,
    src_stride: usize,
    fill: u8,
    channels: usize,
) {
    // Fallback to scalar implementation
    super::pad_constant_scalar(
        dst, src, new_width, src_width, src_height, top, left, src_stride, fill, channels,
    );
}

#[cfg(not(target_arch = "aarch64"))]
pub fn pad_reflect_neon(
    dst: &mut [u8],
    src: &[u8],
    new_width: usize,
    new_height: usize,
    src_width: usize,
    src_height: usize,
    top: usize,
    left: usize,
    channels: usize,
) {
    // Fallback to scalar implementation
    super::pad_reflect_scalar(
        dst, src, new_width, src_width, src_height, top, left, channels,
    );
}
