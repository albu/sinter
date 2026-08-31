// NEON SIMD implementation for transpose transform
//
// This module contains optimized ARM NEON implementations for transposing images.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ============================================================================
// NEON SIMD Implementation (AArch64 only)
// ============================================================================

#[cfg(target_arch = "aarch64")]
/// Transpose an 8x8 block of u8 values using NEON SIMD
#[inline(always)]
unsafe fn transpose_8x8_u8(rows: &mut [uint8x8_t; 8]) {
    // Stage 1: 8-bit transpose (swap within 2x2 blocks)
    let (t0, t1) = (vtrn1_u8(rows[0], rows[1]), vtrn2_u8(rows[0], rows[1]));
    let (t2, t3) = (vtrn1_u8(rows[2], rows[3]), vtrn2_u8(rows[2], rows[3]));
    let (t4, t5) = (vtrn1_u8(rows[4], rows[5]), vtrn2_u8(rows[4], rows[5]));
    let (t6, t7) = (vtrn1_u8(rows[6], rows[7]), vtrn2_u8(rows[6], rows[7]));

    // Stage 2: 16-bit transpose (swap 2x2 blocks within 4x4)
    let (s0, s2) = (
        vreinterpret_u8_u16(vtrn1_u16(vreinterpret_u16_u8(t0), vreinterpret_u16_u8(t2))),
        vreinterpret_u8_u16(vtrn2_u16(vreinterpret_u16_u8(t0), vreinterpret_u16_u8(t2))),
    );
    let (s1, s3) = (
        vreinterpret_u8_u16(vtrn1_u16(vreinterpret_u16_u8(t1), vreinterpret_u16_u8(t3))),
        vreinterpret_u8_u16(vtrn2_u16(vreinterpret_u16_u8(t1), vreinterpret_u16_u8(t3))),
    );
    let (s4, s6) = (
        vreinterpret_u8_u16(vtrn1_u16(vreinterpret_u16_u8(t4), vreinterpret_u16_u8(t6))),
        vreinterpret_u8_u16(vtrn2_u16(vreinterpret_u16_u8(t4), vreinterpret_u16_u8(t6))),
    );
    let (s5, s7) = (
        vreinterpret_u8_u16(vtrn1_u16(vreinterpret_u16_u8(t5), vreinterpret_u16_u8(t7))),
        vreinterpret_u8_u16(vtrn2_u16(vreinterpret_u16_u8(t5), vreinterpret_u16_u8(t7))),
    );

    // Stage 3: 32-bit transpose (swap 4x4 blocks within 8x8)
    rows[0] = vreinterpret_u8_u32(vtrn1_u32(vreinterpret_u32_u8(s0), vreinterpret_u32_u8(s4)));
    rows[4] = vreinterpret_u8_u32(vtrn2_u32(vreinterpret_u32_u8(s0), vreinterpret_u32_u8(s4)));
    rows[1] = vreinterpret_u8_u32(vtrn1_u32(vreinterpret_u32_u8(s1), vreinterpret_u32_u8(s5)));
    rows[5] = vreinterpret_u8_u32(vtrn2_u32(vreinterpret_u32_u8(s1), vreinterpret_u32_u8(s5)));
    rows[2] = vreinterpret_u8_u32(vtrn1_u32(vreinterpret_u32_u8(s2), vreinterpret_u32_u8(s6)));
    rows[6] = vreinterpret_u8_u32(vtrn2_u32(vreinterpret_u32_u8(s2), vreinterpret_u32_u8(s6)));
    rows[3] = vreinterpret_u8_u32(vtrn1_u32(vreinterpret_u32_u8(s3), vreinterpret_u32_u8(s7)));
    rows[7] = vreinterpret_u8_u32(vtrn2_u32(vreinterpret_u32_u8(s3), vreinterpret_u32_u8(s7)));
}

/// Process a single 8x8 RGB tile for transpose
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn transpose_rgb_8x8_tile(
    src: &[u8],
    dst: &mut [u8],
    x_tile: usize,
    y_tile: usize,
    src_stride: usize,
    dst_stride: usize,
) {
    // Step 1: Load 8x8 RGB tile (deinterleaved)
    let row0 = vld3_u8(src.as_ptr().add((y_tile + 0) * src_stride + x_tile * 3));
    let row1 = vld3_u8(src.as_ptr().add((y_tile + 1) * src_stride + x_tile * 3));
    let row2 = vld3_u8(src.as_ptr().add((y_tile + 2) * src_stride + x_tile * 3));
    let row3 = vld3_u8(src.as_ptr().add((y_tile + 3) * src_stride + x_tile * 3));
    let row4 = vld3_u8(src.as_ptr().add((y_tile + 4) * src_stride + x_tile * 3));
    let row5 = vld3_u8(src.as_ptr().add((y_tile + 5) * src_stride + x_tile * 3));
    let row6 = vld3_u8(src.as_ptr().add((y_tile + 6) * src_stride + x_tile * 3));
    let row7 = vld3_u8(src.as_ptr().add((y_tile + 7) * src_stride + x_tile * 3));

    // Step 2: Build per-channel 8x8 matrices
    let mut r = [
        row0.0, row1.0, row2.0, row3.0, row4.0, row5.0, row6.0, row7.0,
    ];
    let mut g = [
        row0.1, row1.1, row2.1, row3.1, row4.1, row5.1, row6.1, row7.1,
    ];
    let mut b = [
        row0.2, row1.2, row2.2, row3.2, row4.2, row5.2, row6.2, row7.2,
    ];

    // Step 3: Transpose each channel
    transpose_8x8_u8(&mut r);
    transpose_8x8_u8(&mut g);
    transpose_8x8_u8(&mut b);

    // Step 4: Store transposed columns as rows (no flip - just transpose!)
    for i in 0..8 {
        let dst_x = y_tile;
        let dst_y = x_tile + i;

        // Build interleaved output
        let out = uint8x8x3_t(r[i], g[i], b[i]);

        vst3_u8(dst.as_mut_ptr().add(dst_y * dst_stride + dst_x * 3), out);
    }
}

/// Process a single 8x8 grayscale tile for transpose
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(dead_code)]
unsafe fn transpose_gray_8x8_tile(
    src: &[u8],
    dst: &mut [u8],
    x_tile: usize,
    y_tile: usize,
    src_stride: usize,
    dst_stride: usize,
) {
    // Step 1: Load 8 rows (8 pixels each)
    let mut rows = [
        vld1_u8(src.as_ptr().add((y_tile + 0) * src_stride + x_tile)),
        vld1_u8(src.as_ptr().add((y_tile + 1) * src_stride + x_tile)),
        vld1_u8(src.as_ptr().add((y_tile + 2) * src_stride + x_tile)),
        vld1_u8(src.as_ptr().add((y_tile + 3) * src_stride + x_tile)),
        vld1_u8(src.as_ptr().add((y_tile + 4) * src_stride + x_tile)),
        vld1_u8(src.as_ptr().add((y_tile + 5) * src_stride + x_tile)),
        vld1_u8(src.as_ptr().add((y_tile + 6) * src_stride + x_tile)),
        vld1_u8(src.as_ptr().add((y_tile + 7) * src_stride + x_tile)),
    ];

    // Step 2: Transpose the 8x8 block (SIMD)
    transpose_8x8_u8(&mut rows);

    // Step 3: Store transposed columns as rows
    for i in 0..8 {
        let dst_x = y_tile;
        let dst_y = x_tile + i;
        vst1_u8(dst.as_mut_ptr().add(dst_y * dst_stride + dst_x), rows[i]);
    }
}

/// RGB transpose using 8x8 tiles with deinterleave -> transpose -> reinterleave
///
/// Pipeline:
/// 1. Load 8x8 RGB tile with vld3_u8 (deinterleaves into R,G,B planes)
/// 2. Build per-channel 8x8 matrices
/// 3. Transpose each channel (using transpose_8x8_u8)
/// 4. Reinterleave with vst3_u8 and store
#[cfg(target_arch = "aarch64")]
pub unsafe fn transpose_rgb_tiled(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) {
    const TILE_SIZE: usize = 8;

    // Process 8x8 tiles
    for y_tile in (0..height).step_by(TILE_SIZE) {
        for x_tile in (0..width).step_by(TILE_SIZE) {
            let y_max = (y_tile + TILE_SIZE).min(height);
            let x_max = (x_tile + TILE_SIZE).min(width);

            // Process full 8x8 tiles with SIMD
            if y_tile + TILE_SIZE <= height && x_tile + TILE_SIZE <= width {
                transpose_rgb_8x8_tile(src, dst, x_tile, y_tile, src_stride, dst_stride);
            } else {
                // Fallback for borders
                for y in y_tile..y_max {
                    for x in x_tile..x_max {
                        let src_idx = y * src_stride + x * 3;
                        let dst_idx = x * dst_stride + y * 3;

                        std::ptr::copy_nonoverlapping(
                            src.as_ptr().add(src_idx),
                            dst.as_mut_ptr().add(dst_idx),
                            3,
                        );
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn transpose_16x16_u8(r: &mut [uint8x16_t; 16]) {
    // Stage 1: 8-bit transpose
    let (t0, t1) = (vtrn1q_u8(r[0], r[1]), vtrn2q_u8(r[0], r[1]));
    let (t2, t3) = (vtrn1q_u8(r[2], r[3]), vtrn2q_u8(r[2], r[3]));
    let (t4, t5) = (vtrn1q_u8(r[4], r[5]), vtrn2q_u8(r[4], r[5]));
    let (t6, t7) = (vtrn1q_u8(r[6], r[7]), vtrn2q_u8(r[6], r[7]));
    let (t8, t9) = (vtrn1q_u8(r[8], r[9]), vtrn2q_u8(r[8], r[9]));
    let (t10, t11) = (vtrn1q_u8(r[10], r[11]), vtrn2q_u8(r[10], r[11]));
    let (t12, t13) = (vtrn1q_u8(r[12], r[13]), vtrn2q_u8(r[12], r[13]));
    let (t14, t15) = (vtrn1q_u8(r[14], r[15]), vtrn2q_u8(r[14], r[15]));

    // Stage 2: 16-bit transpose
    let (s0, s2) = (vreinterpretq_u8_u16(vtrn1q_u16(vreinterpretq_u16_u8(t0), vreinterpretq_u16_u8(t2))), vreinterpretq_u8_u16(vtrn2q_u16(vreinterpretq_u16_u8(t0), vreinterpretq_u16_u8(t2))));
    let (s1, s3) = (vreinterpretq_u8_u16(vtrn1q_u16(vreinterpretq_u16_u8(t1), vreinterpretq_u16_u8(t3))), vreinterpretq_u8_u16(vtrn2q_u16(vreinterpretq_u16_u8(t1), vreinterpretq_u16_u8(t3))));
    let (s4, s6) = (vreinterpretq_u8_u16(vtrn1q_u16(vreinterpretq_u16_u8(t4), vreinterpretq_u16_u8(t6))), vreinterpretq_u8_u16(vtrn2q_u16(vreinterpretq_u16_u8(t4), vreinterpretq_u16_u8(t6))));
    let (s5, s7) = (vreinterpretq_u8_u16(vtrn1q_u16(vreinterpretq_u16_u8(t5), vreinterpretq_u16_u8(t7))), vreinterpretq_u8_u16(vtrn2q_u16(vreinterpretq_u16_u8(t5), vreinterpretq_u16_u8(t7))));
    let (s8, s10) = (vreinterpretq_u8_u16(vtrn1q_u16(vreinterpretq_u16_u8(t8), vreinterpretq_u16_u8(t10))), vreinterpretq_u8_u16(vtrn2q_u16(vreinterpretq_u16_u8(t8), vreinterpretq_u16_u8(t10))));
    let (s9, s11) = (vreinterpretq_u8_u16(vtrn1q_u16(vreinterpretq_u16_u8(t9), vreinterpretq_u16_u8(t11))), vreinterpretq_u8_u16(vtrn2q_u16(vreinterpretq_u16_u8(t9), vreinterpretq_u16_u8(t11))));
    let (s12, s14) = (vreinterpretq_u8_u16(vtrn1q_u16(vreinterpretq_u16_u8(t12), vreinterpretq_u16_u8(t14))), vreinterpretq_u8_u16(vtrn2q_u16(vreinterpretq_u16_u8(t12), vreinterpretq_u16_u8(t14))));
    let (s13, s15) = (vreinterpretq_u8_u16(vtrn1q_u16(vreinterpretq_u16_u8(t13), vreinterpretq_u16_u8(t15))), vreinterpretq_u8_u16(vtrn2q_u16(vreinterpretq_u16_u8(t13), vreinterpretq_u16_u8(t15))));

    // Stage 3: 32-bit transpose
    let (u0, u4) = (vreinterpretq_u8_u32(vtrn1q_u32(vreinterpretq_u32_u8(s0), vreinterpretq_u32_u8(s4))), vreinterpretq_u8_u32(vtrn2q_u32(vreinterpretq_u32_u8(s0), vreinterpretq_u32_u8(s4))));
    let (u1, u5) = (vreinterpretq_u8_u32(vtrn1q_u32(vreinterpretq_u32_u8(s1), vreinterpretq_u32_u8(s5))), vreinterpretq_u8_u32(vtrn2q_u32(vreinterpretq_u32_u8(s1), vreinterpretq_u32_u8(s5))));
    let (u2, u6) = (vreinterpretq_u8_u32(vtrn1q_u32(vreinterpretq_u32_u8(s2), vreinterpretq_u32_u8(s6))), vreinterpretq_u8_u32(vtrn2q_u32(vreinterpretq_u32_u8(s2), vreinterpretq_u32_u8(s6))));
    let (u3, u7) = (vreinterpretq_u8_u32(vtrn1q_u32(vreinterpretq_u32_u8(s3), vreinterpretq_u32_u8(s7))), vreinterpretq_u8_u32(vtrn2q_u32(vreinterpretq_u32_u8(s3), vreinterpretq_u32_u8(s7))));
    let (u8, u12) = (vreinterpretq_u8_u32(vtrn1q_u32(vreinterpretq_u32_u8(s8), vreinterpretq_u32_u8(s12))), vreinterpretq_u8_u32(vtrn2q_u32(vreinterpretq_u32_u8(s8), vreinterpretq_u32_u8(s12))));
    let (u9, u13) = (vreinterpretq_u8_u32(vtrn1q_u32(vreinterpretq_u32_u8(s9), vreinterpretq_u32_u8(s13))), vreinterpretq_u8_u32(vtrn2q_u32(vreinterpretq_u32_u8(s9), vreinterpretq_u32_u8(s13))));
    let (u10, u14) = (vreinterpretq_u8_u32(vtrn1q_u32(vreinterpretq_u32_u8(s10), vreinterpretq_u32_u8(s14))), vreinterpretq_u8_u32(vtrn2q_u32(vreinterpretq_u32_u8(s10), vreinterpretq_u32_u8(s14))));
    let (u11, u15) = (vreinterpretq_u8_u32(vtrn1q_u32(vreinterpretq_u32_u8(s11), vreinterpretq_u32_u8(s15))), vreinterpretq_u8_u32(vtrn2q_u32(vreinterpretq_u32_u8(s11), vreinterpretq_u32_u8(s15))));

    // Stage 4: 64-bit transpose
    r[0] = vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(u0), vreinterpretq_u64_u8(u8)));
    r[8] = vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(u0), vreinterpretq_u64_u8(u8)));
    r[1] = vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(u1), vreinterpretq_u64_u8(u9)));
    r[9] = vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(u1), vreinterpretq_u64_u8(u9)));
    r[2] = vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(u2), vreinterpretq_u64_u8(u10)));
    r[10] = vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(u2), vreinterpretq_u64_u8(u10)));
    r[3] = vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(u3), vreinterpretq_u64_u8(u11)));
    r[11] = vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(u3), vreinterpretq_u64_u8(u11)));
    r[4] = vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(u4), vreinterpretq_u64_u8(u12)));
    r[12] = vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(u4), vreinterpretq_u64_u8(u12)));
    r[5] = vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(u5), vreinterpretq_u64_u8(u13)));
    r[13] = vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(u5), vreinterpretq_u64_u8(u13)));
    r[6] = vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(u6), vreinterpretq_u64_u8(u14)));
    r[14] = vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(u6), vreinterpretq_u64_u8(u14)));
    r[7] = vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(u7), vreinterpretq_u64_u8(u15)));
    r[15] = vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(u7), vreinterpretq_u64_u8(u15)));
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn transpose_gray_16x16_tile(
    src: &[u8],
    dst: &mut [u8],
    x_tile: usize,
    y_tile: usize,
    src_stride: usize,
    dst_stride: usize,
) {
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    let mut rows = [
        vld1q_u8(src_ptr.add((y_tile + 0) * src_stride + x_tile)),
        vld1q_u8(src_ptr.add((y_tile + 1) * src_stride + x_tile)),
        vld1q_u8(src_ptr.add((y_tile + 2) * src_stride + x_tile)),
        vld1q_u8(src_ptr.add((y_tile + 3) * src_stride + x_tile)),
        vld1q_u8(src_ptr.add((y_tile + 4) * src_stride + x_tile)),
        vld1q_u8(src_ptr.add((y_tile + 5) * src_stride + x_tile)),
        vld1q_u8(src_ptr.add((y_tile + 6) * src_stride + x_tile)),
        vld1q_u8(src_ptr.add((y_tile + 7) * src_stride + x_tile)),
        vld1q_u8(src_ptr.add((y_tile + 8) * src_stride + x_tile)),
        vld1q_u8(src_ptr.add((y_tile + 9) * src_stride + x_tile)),
        vld1q_u8(src_ptr.add((y_tile + 10) * src_stride + x_tile)),
        vld1q_u8(src_ptr.add((y_tile + 11) * src_stride + x_tile)),
        vld1q_u8(src_ptr.add((y_tile + 12) * src_stride + x_tile)),
        vld1q_u8(src_ptr.add((y_tile + 13) * src_stride + x_tile)),
        vld1q_u8(src_ptr.add((y_tile + 14) * src_stride + x_tile)),
        vld1q_u8(src_ptr.add((y_tile + 15) * src_stride + x_tile)),
    ];

    transpose_16x16_u8(&mut rows);

    for i in 0..16 {
        vst1q_u8(dst_ptr.add((x_tile + i) * dst_stride + y_tile), rows[i]);
    }
}

/// Grayscale transpose using 16x16 tiles with NEON SIMD
#[cfg(target_arch = "aarch64")]
pub unsafe fn transpose_gray_tiled(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) {
    const TILE_SIZE: usize = 16;

    for y_tile in (0..height).step_by(TILE_SIZE) {
        for x_tile in (0..width).step_by(TILE_SIZE) {
            let y_max = (y_tile + TILE_SIZE).min(height);
            let x_max = (x_tile + TILE_SIZE).min(width);

            if y_tile + TILE_SIZE <= height && x_tile + TILE_SIZE <= width {
                transpose_gray_16x16_tile(src, dst, x_tile, y_tile, src_stride, dst_stride);
            } else {
                for y in y_tile..y_max {
                    for x in x_tile..x_max {
                        let src_idx = y * src_stride + x;
                        let dst_idx = x * dst_stride + y;
                        dst[dst_idx] = src[src_idx];
                    }
                }
            }
        }
    }
}

// Fallback implementations for non-AArch64 platforms

#[cfg(not(target_arch = "aarch64"))]
/// Transpose an 8x8 block of u8 values (fallback for non-ARM)
#[inline(always)]
unsafe fn transpose_8x8_u8(_rows: &mut [u8; 8]) {
    // No-op stub - not used in fallback path
}

#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn transpose_rgb_tiled(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) {
    for y in 0..height {
        for x in 0..width {
            let src_idx = y * src_stride + x * 3;
            let dst_idx = x * dst_stride + y * 3;

            std::ptr::copy_nonoverlapping(
                src.as_ptr().add(src_idx),
                dst.as_mut_ptr().add(dst_idx),
                3,
            );
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn transpose_gray_tiled(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    src_stride: usize,
    dst_stride: usize,
) {
    for y in 0..height {
        for x in 0..width {
            let src_idx = y * src_stride + x;
            let dst_idx = x * dst_stride + y;
            dst[dst_idx] = src[src_idx];
        }
    }
}
