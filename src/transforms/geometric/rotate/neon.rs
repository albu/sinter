// NEON SIMD Transpose Kernels for Rotation
//
// Provides optimized rotation implementations using tiled NEON SIMD.
//
// NOTE: This file contains duplicate function definitions via conditional compilation:
// - Functions with #[cfg(target_arch = "aarch64")] use optimized NEON SIMD instructions
// - Functions with #[cfg(not(target_arch = "aarch64"))] provide scalar fallbacks
//
// Only one version of each function is compiled, depending on the target architecture.
// This is intentional: the same function name provides architecture-specific implementations.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ============================================================================
// NEON SIMD Transpose Kernels (AArch64 only)
// ============================================================================

#[cfg(target_arch = "aarch64")]
/// Transpose an 8x8 block of u8 values using NEON SIMD
///
/// Uses divide-and-conquer vtrn (transpose) instructions:
/// - Stage 1: Swap 1x1 elements within 2x2 blocks (vtrn_u8)
/// - Stage 2: Swap 2x2 blocks within 4x4 blocks (vtrn_u16)
/// - Stage 3: Swap 4x4 blocks within 8x8 matrix (vtrn_u32)
///
/// This is done entirely in registers - no memory access during transpose.
#[inline(always)]
pub(crate) unsafe fn transpose_8x8_u8_neon(rows: &mut [uint8x8_t; 8]) {
    // Stage 1: 8-bit transpose (swap within 2x2 blocks)
    // vtrn1_u8 swaps odd elements, vtrn2_u8 swaps even elements
    let (t0, t1) = (vtrn1_u8(rows[0], rows[1]), vtrn2_u8(rows[0], rows[1]));
    let (t2, t3) = (vtrn1_u8(rows[2], rows[3]), vtrn2_u8(rows[2], rows[3]));
    let (t4, t5) = (vtrn1_u8(rows[4], rows[5]), vtrn2_u8(rows[4], rows[5]));
    let (t6, t7) = (vtrn1_u8(rows[6], rows[7]), vtrn2_u8(rows[6], rows[7]));

    // Stage 2: 16-bit transpose (swap 2x2 blocks within 4x4)
    // Reinterpret as u16 for 16-bit transpose
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

#[cfg(target_arch = "aarch64")]
/// Rotate 90° clockwise using tiled NEON SIMD
///
/// For RGB images, uses the optimal pipeline:
/// deinterleave (vld3) → transpose per-channel → reinterleave (vst3)
///
/// For grayscale images, uses a simpler pipeline:
/// load rows → transpose → store reversed rows
///
/// This is algorithmically correct for SIMD hardware:
/// - 8x8 tiles fit entirely in registers
/// - Contiguous stores (no strided writes)
/// - Full SIMD utilization
pub unsafe fn rotate_90_cw_neon(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    channels: usize,
) {
    let src_stride = width * channels;
    let dst_stride = height * channels;

    // Use optimized SIMD path for RGB and grayscale
    if channels == 3 {
        rotate_90_rgb_tiled(src, dst, width, height, src_stride, dst_stride);
    } else if channels == 1 {
        rotate_90_gray_tiled(src, dst, width, height, src_stride, dst_stride);
    } else {
        // Fallback for other channel counts
        rotate_90_scalar(src, dst, width, height, channels, src_stride, dst_stride);
    }
}

#[cfg(target_arch = "aarch64")]
/// RGB 90° rotation using 8x8 tiles with deinterleave → transpose → reinterleave
///
/// Uses row-wise store strategy: after transpose, each transposed row is stored contiguously.
/// This avoids scatter and is the correct SIMD approach.
unsafe fn rotate_90_rgb_tiled(
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
            // Only process full 8x8 tiles with SIMD
            if y_tile + TILE_SIZE <= height && x_tile + TILE_SIZE <= width {
                rotate_90_rgb_8x8_tile(src, dst, x_tile, y_tile, src_stride, dst_stride, height);
            } else {
                // Fallback for partial tiles at borders
                let y_max = (y_tile + TILE_SIZE).min(height);
                let x_max = (x_tile + TILE_SIZE).min(width);
                for y in y_tile..y_max {
                    for x in x_tile..x_max {
                        let src_idx = y * src_stride + x * 3;
                        let dst_x = height - 1 - y;
                        let dst_y = x;
                        let dst_idx = dst_y * dst_stride + dst_x * 3;

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
/// Process a single 8x8 RGB tile for 90° CW rotation
///
/// Strategy:
/// 1. Load 8 rows with vld3_u8 (deinterleaves into R,G,B planes)
/// 2. Transpose each channel (SIMD)
/// 3. Store row-by-row: each transposed row maps to a contiguous run in dst
///
/// For rotate90: src(x,y) -> dst(y, height-1-x)
/// After transpose: r[row] = original column (x_tile + row)
/// We store each transposed row reversed, starting from the rightmost x position.
#[inline(always)]
unsafe fn rotate_90_rgb_8x8_tile(
    src: &[u8],
    dst: &mut [u8],
    x_tile: usize,
    y_tile: usize,
    src_stride: usize,
    dst_stride: usize,
    height: usize,
) {
    // Step 1: Load + Deinterleave (8 rows)
    // After: r[i] = RRRRRRRR (row i of tile), g[i] = GGGGGGGG, b[i] = BBBBBBBB
    let row0 = vld3_u8(src.as_ptr().add((y_tile + 0) * src_stride + x_tile * 3));
    let row1 = vld3_u8(src.as_ptr().add((y_tile + 1) * src_stride + x_tile * 3));
    let row2 = vld3_u8(src.as_ptr().add((y_tile + 2) * src_stride + x_tile * 3));
    let row3 = vld3_u8(src.as_ptr().add((y_tile + 3) * src_stride + x_tile * 3));
    let row4 = vld3_u8(src.as_ptr().add((y_tile + 4) * src_stride + x_tile * 3));
    let row5 = vld3_u8(src.as_ptr().add((y_tile + 5) * src_stride + x_tile * 3));
    let row6 = vld3_u8(src.as_ptr().add((y_tile + 6) * src_stride + x_tile * 3));
    let row7 = vld3_u8(src.as_ptr().add((y_tile + 7) * src_stride + x_tile * 3));

    let mut r = [row0.0, row1.0, row2.0, row3.0,
                 row4.0, row5.0, row6.0, row7.0];
    let mut g = [row0.1, row1.1, row2.1, row3.1,
                 row4.1, row5.1, row6.1, row7.1];
    let mut b = [row0.2, row1.2, row2.2, row3.2,
                 row4.2, row5.2, row6.2, row7.2];

    // Step 2: Transpose 8x8 (SIMD)
    // After: r[row] = original column (x_tile + row) of source
    transpose_8x8_u8_neon(&mut r);
    transpose_8x8_u8_neon(&mut g);
    transpose_8x8_u8_neon(&mut b);

    // Step 3: Row-wise reversed store
    // For rotate90: dst(y, H-1-x)
    // Each transposed row goes to dst_y = x_tile + row
    // All start at dst_x = H - 1 - y_tile - 7 (rightmost position of the 8-pixel run)
    // We reverse each row with vrev64_u8, then store contiguously
    let dst_x_start = height - 1 - y_tile - 7;

    for row in 0..8 {
        let dst_y = x_tile + row;
        let dst_x_base = dst_x_start * 3;

        // Reverse the row (right-to-left store)
        let rr = vrev64_u8(r[row]);
        let gg = vrev64_u8(g[row]);
        let bb = vrev64_u8(b[row]);

        let out = uint8x8x3_t(rr, gg, bb);
        vst3_u8(dst.as_mut_ptr().add(dst_y * dst_stride + dst_x_base), out);
    }
}

#[cfg(target_arch = "aarch64")]
/// Grayscale 90° rotation using 8x8 tiles with NEON SIMD
///
/// Pipeline:
/// 1. Load 8x8 tile rows directly
/// 2. Transpose the 8x8 block
/// 3. Store each row reversed (for 90° CW rotation)
///
/// For rotate90: src(x,y) -> dst(y, height-1-x)
/// After transpose: each row becomes a column, then we reverse the column
unsafe fn rotate_90_gray_tiled(
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
            // Only process full 8x8 tiles with SIMD
            if y_tile + TILE_SIZE <= height && x_tile + TILE_SIZE <= width {
                rotate_90_gray_8x8_tile(src, dst, x_tile, y_tile, src_stride, dst_stride, height);
            } else {
                // Fallback for partial tiles at borders
                let y_max = (y_tile + TILE_SIZE).min(height);
                let x_max = (x_tile + TILE_SIZE).min(width);
                for y in y_tile..y_max {
                    for x in x_tile..x_max {
                        let src_idx = y * src_stride + x;
                        let dst_x = height - 1 - y;
                        let dst_y = x;
                        let dst_idx = dst_y * dst_stride + dst_x;

                        dst[dst_idx] = src[src_idx];
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
/// Process a single 8x8 grayscale tile for 90° CW rotation
///
/// Strategy:
/// 1. Load 8 rows directly (no deinterleaving needed)
/// 2. Transpose the 8x8 block (SIMD)
/// 3. Store each row reversed (for 90° CW rotation)
#[inline(always)]
unsafe fn rotate_90_gray_8x8_tile(
    src: &[u8],
    dst: &mut [u8],
    x_tile: usize,
    y_tile: usize,
    src_stride: usize,
    dst_stride: usize,
    height: usize,
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
    transpose_8x8_u8_neon(&mut rows);

    // Step 3: Store each row reversed (for 90° CW rotation)
    // For rotate90: dst(y, H-1-x)
    // After transpose, each row goes to a different y position, and we reverse horizontally
    let dst_x_start = height - 1 - y_tile - 7;

    for row in 0..8 {
        let dst_y = x_tile + row;
        let dst_x = dst_x_start;

        // Reverse the row using 64-bit reversal
        let reversed = vrev64_u8(rows[row]);
        vst1_u8(dst.as_mut_ptr().add(dst_y * dst_stride + dst_x), reversed);
    }
}

#[cfg(target_arch = "aarch64")]
/// Scalar fallback for non-RGB images
unsafe fn rotate_90_scalar(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    channels: usize,
    src_stride: usize,
    dst_stride: usize,
) {
    let tile_size = width.min(height).min(16);

    for y0 in (0..height).step_by(tile_size) {
        for x0 in (0..width).step_by(tile_size) {
            let y_max = (y0 + tile_size).min(height);
            let x_max = (x0 + tile_size).min(width);

            for y in y0..y_max {
                for x in x0..x_max {
                    let src_idx = y * src_stride + x * channels;
                    let dst_x = height - 1 - y;
                    let dst_y = x;
                    let dst_idx = dst_y * dst_stride + dst_x * channels;

                    std::ptr::copy_nonoverlapping(
                        src.as_ptr().add(src_idx),
                        dst.as_mut_ptr().add(dst_idx),
                        channels,
                    );
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
/// Rotate 270° clockwise using tiled NEON SIMD
///
/// For RGB images, uses the optimal pipeline:
/// deinterleave (vld3) → transpose per-channel → reinterleave (vst3)
///
/// For grayscale images, uses a simpler pipeline:
/// load rows → transpose → store as columns
pub unsafe fn rotate_270_cw_neon(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    channels: usize,
) {
    let src_stride = width * channels;
    let dst_stride = height * channels;

    // Use optimized SIMD path for RGB and grayscale
    if channels == 3 {
        rotate_270_rgb_tiled(src, dst, width, height, src_stride, dst_stride);
    } else if channels == 1 {
        rotate_270_gray_tiled(src, dst, width, height, src_stride, dst_stride);
    } else {
        // Fallback for other channel counts
        rotate_270_scalar(src, dst, width, height, channels, src_stride, dst_stride);
    }
}

#[cfg(target_arch = "aarch64")]
/// RGB 270° rotation using 8x8 tiles with deinterleave → transpose → reinterleave
unsafe fn rotate_270_rgb_tiled(
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
                rotate_270_rgb_8x8_tile(src, dst, x_tile, y_tile, src_stride, dst_stride, width);
            } else {
                // Fallback for borders
                for y in y_tile..y_max {
                    for x in x_tile..x_max {
                        let src_idx = y * src_stride + x * 3;
                        let dst_x = y;
                        let dst_y = width - 1 - x;
                        let dst_idx = dst_y * dst_stride + dst_x * 3;

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
/// Process a single 8x8 RGB tile for 270° CW rotation
#[inline(always)]
unsafe fn rotate_270_rgb_8x8_tile(
    src: &[u8],
    dst: &mut [u8],
    x_tile: usize,
    y_tile: usize,
    src_stride: usize,
    dst_stride: usize,
    width: usize,
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
    let mut r = [row0.0, row1.0, row2.0, row3.0,
                 row4.0, row5.0, row6.0, row7.0];
    let mut g = [row0.1, row1.1, row2.1, row3.1,
                 row4.1, row5.1, row6.1, row7.1];
    let mut b = [row0.2, row1.2, row2.2, row3.2,
                 row4.2, row5.2, row6.2, row7.2];

    // Step 3: Transpose each channel
    transpose_8x8_u8_neon(&mut r);
    transpose_8x8_u8_neon(&mut g);
    transpose_8x8_u8_neon(&mut b);

    // Step 4: Store as 8 columns (transposed rows become columns in dst)
    // For rotate270: dst_x = y, dst_y = W-1-x
    // After transpose, r[i]/g[i]/b[i] represents pixels from source column x_tile+i
    // These form a vertical column in dst at y = x_tile (constant for all i)
    // and at y = W-1-x_tile-i (reversed order on y-axis)
    for i in 0..8 {
        let dst_x = y_tile;  // x position in dst (same as source y)
        let dst_y = width - 1 - x_tile - i;  // y position in dst (reversed)

        // No reversal needed - pixels stay in order
        let out = uint8x8x3_t(r[i], g[i], b[i]);
        vst3_u8(dst.as_mut_ptr().add(dst_y * dst_stride + dst_x * 3), out);
    }
}

#[cfg(target_arch = "aarch64")]
/// Grayscale 270° rotation using 8x8 tiles with NEON SIMD
///
/// Pipeline:
/// 1. Load 8x8 tile rows directly
/// 2. Transpose the 8x8 block
/// 3. Store transposed rows as columns (for 270° CW rotation)
///
/// For rotate270: src(x,y) -> dst(y, width-1-x)
/// After transpose: rows become columns
unsafe fn rotate_270_gray_tiled(
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
                rotate_270_gray_8x8_tile(src, dst, x_tile, y_tile, src_stride, dst_stride, width);
            } else {
                // Fallback for borders
                for y in y_tile..y_max {
                    for x in x_tile..x_max {
                        let src_idx = y * src_stride + x;
                        let dst_x = y;
                        let dst_y = width - 1 - x;
                        let dst_idx = dst_y * dst_stride + dst_x;

                        dst[dst_idx] = src[src_idx];
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
/// Process a single 8x8 grayscale tile for 270° CW rotation
///
/// Strategy:
/// 1. Load 8 rows directly (no deinterleaving needed)
/// 2. Transpose the 8x8 block (SIMD)
/// 3. Store each transposed row as a column
#[inline(always)]
unsafe fn rotate_270_gray_8x8_tile(
    src: &[u8],
    dst: &mut [u8],
    x_tile: usize,
    y_tile: usize,
    src_stride: usize,
    dst_stride: usize,
    width: usize,
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
    transpose_8x8_u8_neon(&mut rows);

    // Step 3: Store each transposed row as a column
    // For rotate270: dst_x = y, dst_y = W-1-x
    // After transpose, rows[i] becomes column i in dst
    for i in 0..8 {
        let dst_x = y_tile;
        let dst_y = width - 1 - x_tile - i;

        vst1_u8(dst.as_mut_ptr().add(dst_y * dst_stride + dst_x), rows[i]);
    }
}

#[cfg(target_arch = "aarch64")]
/// Scalar fallback for non-RGB images (270°)
unsafe fn rotate_270_scalar(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    channels: usize,
    src_stride: usize,
    dst_stride: usize,
) {
    let tile_size = width.min(height).min(16);

    for y0 in (0..height).step_by(tile_size) {
        for x0 in (0..width).step_by(tile_size) {
            let y_max = (y0 + tile_size).min(height);
            let x_max = (x0 + tile_size).min(width);

            for y in y0..y_max {
                for x in x0..x_max {
                    let src_idx = y * src_stride + x * channels;
                    let dst_x = y;
                    let dst_y = width - 1 - x;
                    let dst_idx = dst_y * dst_stride + dst_x * channels;

                    std::ptr::copy_nonoverlapping(
                        src.as_ptr().add(src_idx),
                        dst.as_mut_ptr().add(dst_idx),
                        channels,
                    );
                }
            }
        }
    }
}

// ============================================================================
// Fallback implementations for non-AArch64
// ============================================================================

#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn rotate_90_cw_neon(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    channels: usize,
) {
    let src_stride = width * channels;
    let dst_stride = height * channels;

    // Use 16x16 tiles for better cache locality
    let tile_size = width.min(height).min(256);

    for y0 in (0..height).step_by(tile_size) {
        for x0 in (0..width).step_by(tile_size) {
            let y_max = (y0 + tile_size).min(height);
            let x_max = (x0 + tile_size).min(width);

            for y in y0..y_max {
                for x in x0..x_max {
                    let src_idx = y * src_stride + x * channels;
                    let dst_x = height - 1 - y;
                    let dst_y = x;
                    let dst_idx = dst_y * dst_stride + dst_x * channels;

                    std::ptr::copy_nonoverlapping(
                        src.as_ptr().add(src_idx),
                        dst.as_mut_ptr().add(dst_idx),
                        channels,
                    );
                }
            }
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn rotate_270_cw_neon(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    channels: usize,
) {
    let src_stride = width * channels;
    let dst_stride = height * channels;

    // Use 16x16 tiles for better cache locality
    let tile_size = width.min(height).min(256);

    for y0 in (0..height).step_by(tile_size) {
        for x0 in (0..width).step_by(tile_size) {
            let y_max = (y0 + tile_size).min(height);
            let x_max = (x0 + tile_size).min(width);

            for y in y0..y_max {
                for x in x0..x_max {
                    let src_idx = y * src_stride + x * channels;
                    let dst_x = y;
                    let dst_y = width - 1 - x;
                    let dst_idx = dst_y * dst_stride + dst_x * channels;

                    std::ptr::copy_nonoverlapping(
                        src.as_ptr().add(src_idx),
                        dst.as_mut_ptr().add(dst_idx),
                        channels,
                    );
                }
            }
        }
    }
}
