// NEON SIMD implementation for crop transform
//
// For simple row copies without processing, memcpy (via copy_from_slice) is
// faster than SIMD operations like vld3_u8/vst3_u8 which de-interleave data.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Crop using optimized memcpy approach
///
/// For simple cropping operations, memcpy is optimal since we're just copying
/// contiguous row data without any processing. SIMD de-interleaving operations
/// like vld3_u8/vst3_u8 add unnecessary overhead for simple copies.
#[cfg(target_arch = "aarch64")]
pub unsafe fn crop_neon_simd(
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    x_offset: usize,
    y_start: usize,
    cropped_width: usize,
    cropped_height: usize,
    channels: usize,
) {
    let row_bytes = cropped_width * channels;

    for row in 0..cropped_height {
        let src_row_start = (y_start + row) * src_stride + x_offset;
        let dst_row_start = row * row_bytes;
        let src_row_end = src_row_start + row_bytes;

        dst[dst_row_start..dst_row_start + row_bytes]
            .copy_from_slice(&src[src_row_start..src_row_end]);
    }
}

/// Fallback implementation for non-AArch64 platforms
#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn crop_neon_simd(
    _src: &[u8],
    _dst: &mut [u8],
    _src_stride: usize,
    _x_offset: usize,
    _y_start: usize,
    _cropped_width: usize,
    _cropped_height: usize,
    _channels: usize,
) {
    // This should never be called on non-AArch64 platforms
    unreachable!("crop_neon_simd should not be called on non-AArch64 platforms")
}
