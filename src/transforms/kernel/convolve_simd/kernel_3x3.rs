// 3x3 kernel convolution implementations (Gaussian [1 2 1] / 4)
//
// Provides both 1D horizontal/vertical passes and separable implementation.
//
// OPTIMIZATION: Uses symmetric folding for vertical pass to reduce arithmetic:
//   [1 2 1] kernel -> (row[-1] + row[1]) + (row[0] << 1), then >> 2
//   This reduces 3 multiplies to 2 adds + 1 shift!

use crate::core::FusableImage;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ============================================================================
// Public entry points for 3x3 kernel
// ============================================================================

#[cfg(target_arch = "aarch64")]
pub(crate) unsafe fn convolve_1d_horizontal_neon_3(
    image: &mut FusableImage,
    kernel: &[i32],
    _scale: i32,
) {
    let width = image.width;
    let height = image.height;
    let channels = image.channels;
    let data = &mut image.data;

    if channels != 3 {
        // Fallback to scalar for non-RGB
        super::super::convolve::convolve_1d_horizontal(image, kernel, 4);
        return;
    }

    let mut output = vec![0u8; data.len()];
    let radius = 1;

    const TILE: usize = 16;

    for y in 0..height {
        let row_offset = y * width * channels;

        // Handle left edge
        for x in 0..width.min(radius) {
            for c in 0..3 {
                let mut sum: u32 = 0;
                for k in 0..3 {
                    let px =
                        (x as i32 + k as i32 - radius as i32).clamp(0, width as i32 - 1) as usize;
                    sum = sum.wrapping_add(data[row_offset + px * 3 + c] as u32 * [1u32, 2, 1][k]);
                }
                output[row_offset + x * 3 + c] = (sum >> 2) as u8;
            }
        }

        // SIMD middle - process 16 pixels at a time
        let simd_start = radius;
        let simd_end = width.saturating_sub(radius);
        let simd_chunks = if simd_end > simd_start {
            (simd_end - simd_start) / TILE
        } else {
            0
        };

        for chunk in 0..simd_chunks {
            let base_x = simd_start + chunk * TILE;
            let out_offset = row_offset + base_x * 3;

            // Load three 16-pixel RGB chunks: x-1, x, x+1
            let p0 = vld3q_u8(data.as_ptr().add(row_offset + (base_x - 1) * 3) as *const u8);
            let p1 = vld3q_u8(data.as_ptr().add(row_offset + base_x * 3) as *const u8);
            let p2 = vld3q_u8(data.as_ptr().add(row_offset + (base_x + 1) * 3) as *const u8);

            // Process low 8 pixels: [1 2 1] kernel
            let r_lo = blur3_u8_lo(p0.0, p1.0, p2.0);
            let g_lo = blur3_u8_lo(p0.1, p1.1, p2.1);
            let b_lo = blur3_u8_lo(p0.2, p1.2, p2.2);

            // Process high 8 pixels
            let r_hi = blur3_u8_hi(p0.0, p1.0, p2.0);
            let g_hi = blur3_u8_hi(p0.1, p1.1, p2.1);
            let b_hi = blur3_u8_hi(p0.2, p1.2, p2.2);

            // Store low 8
            let rgb_out_lo = uint8x8x3_t(r_lo, g_lo, b_lo);
            vst3_u8(output.as_mut_ptr().add(out_offset) as *mut u8, rgb_out_lo);
            // Store high 8
            let rgb_out_hi = uint8x8x3_t(r_hi, g_hi, b_hi);
            vst3_u8(
                output.as_mut_ptr().add(out_offset + 8 * 3) as *mut u8,
                rgb_out_hi,
            );
        }

        // Handle right edge
        for x in simd_end.max(radius)..width {
            for c in 0..3 {
                let mut sum: u32 = 0;
                for k in 0..3 {
                    let px =
                        (x as i32 + k as i32 - radius as i32).clamp(0, width as i32 - 1) as usize;
                    sum = sum.wrapping_add(data[row_offset + px * 3 + c] as u32 * [1u32, 2, 1][k]);
                }
                output[row_offset + x * 3 + c] = (sum >> 2) as u8;
            }
        }
    }

    data.copy_from_slice(&output);
}

#[cfg(target_arch = "aarch64")]
pub(crate) unsafe fn convolve_1d_vertical_neon_3(
    image: &mut FusableImage,
    _kernel: &[i32],
    _scale: i32,
) {
    let width = image.width;
    let height = image.height;
    let channels = image.channels;

    if channels != 3 {
        // Fallback to scalar for non-RGB
        super::super::convolve::convolve_1d_vertical(image, &[1, 2, 1][..], 4);
        return;
    }

    let data = &mut image.data;
    let mut output = vec![0u8; data.len()];
    let radius = 1;

    // Process 8 columns at a time - this makes vertical blur efficient
    const COLS_PER_ITER: usize = 8;

    for x_block in 0..((width + COLS_PER_ITER - 1) / COLS_PER_ITER) {
        let x_start = x_block * COLS_PER_ITER;
        let x_end = (x_start + COLS_PER_ITER).min(width);

        // Top edge - scalar
        for y in 0..height.min(radius) {
            for x in x_start..x_end {
                for c in 0..3 {
                    let mut sum: u32 = 0;
                    for k in 0..3 {
                        let py = (y as i32 + k as i32 - radius as i32).clamp(0, height as i32 - 1)
                            as usize;
                        sum = sum
                            .wrapping_add(data[(py * width + x) * 3 + c] as u32 * [1u32, 2, 1][k]);
                    }
                    output[(y * width + x) * 3 + c] = (sum >> 2) as u8;
                }
            }
        }

        // SIMD middle - process all middle rows at once for these 8 columns
        let use_simd = x_end - x_start == COLS_PER_ITER;
        let simd_start = radius;
        let simd_end = height.saturating_sub(radius);

        if use_simd {
            for y in simd_start..simd_end {
                // Load 3 rows of pixels for these 8 columns
                let row_minus1 = data.as_ptr().add(((y - 1) * width + x_start) * 3) as *const u8;
                let row0 = data.as_ptr().add((y * width + x_start) * 3) as *const u8;
                let row1 = data.as_ptr().add(((y + 1) * width + x_start) * 3) as *const u8;

                // Load 8 RGB pixels from each row
                let p_minus1 = vld3q_u8(row_minus1);
                let p0 = vld3q_u8(row0);
                let p1 = vld3q_u8(row1);

                // Apply [1 2 1] kernel vertically to all 8 pixels
                let r_blur = blur3_scalar_to_u8(
                    vget_low_u8(p_minus1.0),
                    vget_low_u8(p0.0),
                    vget_low_u8(p1.0),
                );
                let g_blur = blur3_scalar_to_u8(
                    vget_low_u8(p_minus1.1),
                    vget_low_u8(p0.1),
                    vget_low_u8(p1.1),
                );
                let b_blur = blur3_scalar_to_u8(
                    vget_low_u8(p_minus1.2),
                    vget_low_u8(p0.2),
                    vget_low_u8(p1.2),
                );

                // Store results
                let out_ptr = output.as_mut_ptr().add((y * width + x_start) * 3) as *mut u8;
                let rgb_out = uint8x8x3_t(r_blur, g_blur, b_blur);
                vst3_u8(out_ptr, rgb_out);
            }
        } else {
            // Scalar for partial blocks
            for y in simd_start..simd_end {
                for x in x_start..x_end {
                    for c in 0..3 {
                        let mut sum: u32 = 0;
                        for k in 0..3 {
                            let py = (y as i32 + k as i32 - radius as i32)
                                .clamp(0, height as i32 - 1)
                                as usize;
                            sum = sum.wrapping_add(
                                data[(py * width + x) * 3 + c] as u32 * [1u32, 2, 1][k],
                            );
                        }
                        output[(y * width + x) * 3 + c] = (sum >> 2) as u8;
                    }
                }
            }
        }

        // Bottom edge - scalar
        for y in simd_end.max(radius)..height {
            for x in x_start..x_end {
                for c in 0..3 {
                    let mut sum: u32 = 0;
                    for k in 0..3 {
                        let py = (y as i32 + k as i32 - radius as i32).clamp(0, height as i32 - 1)
                            as usize;
                        sum = sum
                            .wrapping_add(data[(py * width + x) * 3 + c] as u32 * [1u32, 2, 1][k]);
                    }
                    output[(y * width + x) * 3 + c] = (sum >> 2) as u8;
                }
            }
        }
    }

    data.copy_from_slice(&output);
}

#[cfg(target_arch = "aarch64")]
pub(crate) unsafe fn convolve_separable_neon_3(
    image: &mut FusableImage,
    _kernel: &[i32],
    _scale: i32,
) {
    let width = image.width;
    let height = image.height;
    let channels = image.channels;
    let data = &mut image.data;

    if channels == 1 {
        convolve_separable_gray_neon_3(image);
        return;
    }

    if channels != 3 {
        // Fallback to scalar for non-RGB
        super::super::convolve::convolve_1d_horizontal(image, &[1, 2, 1][..], 4);
        super::super::convolve::convolve_1d_vertical(image, &[1, 2, 1][..], 4);
        return;
    }

    let mut temp = vec![0u8; data.len()];
    let radius = 1;
    const TILE: usize = 16;
    let row_bytes = width * 3;

    // ============================================================================
    // HORIZONTAL PASS: u8 → u8 (process 16 RGB pixels per iteration)
    // ============================================================================

    for y in 0..height {
        let row_offset = y * row_bytes;

        // Handle left edge
        for x in 0..width.min(radius) {
            for c in 0..3 {
                let mut sum: u32 = 0;
                for k in 0..3 {
                    let px =
                        (x as i32 + k as i32 - radius as i32).clamp(0, width as i32 - 1) as usize;
                    sum = sum.wrapping_add(data[row_offset + px * 3 + c] as u32 * [1u32, 2, 1][k]);
                }
                temp[row_offset + x * 3 + c] = (sum >> 2) as u8;
            }
        }

        // SIMD middle - process 16 pixels at a time
        let simd_start = radius;
        let simd_end = width.saturating_sub(radius);
        let simd_chunks = if simd_end > simd_start {
            (simd_end - simd_start) / TILE
        } else {
            0
        };

        let mut simd_processed_end = simd_start;

        for chunk in 0..simd_chunks {
            let base_x = simd_start + chunk * TILE;
            let out_offset = row_offset + base_x * 3;

            // Load three 16-pixel RGB chunks: x-1, x, x+1
            let p0 = vld3q_u8(data.as_ptr().add(row_offset + (base_x - 1) * 3) as *const u8);
            let p1 = vld3q_u8(data.as_ptr().add(row_offset + base_x * 3) as *const u8);
            let p2 = vld3q_u8(data.as_ptr().add(row_offset + (base_x + 1) * 3) as *const u8);

            // Process low 8 pixels: [1 2 1] kernel
            let r_lo = blur3_u8_lo(p0.0, p1.0, p2.0);
            let g_lo = blur3_u8_lo(p0.1, p1.1, p2.1);
            let b_lo = blur3_u8_lo(p0.2, p1.2, p2.2);

            // Process high 8 pixels
            let r_hi = blur3_u8_hi(p0.0, p1.0, p2.0);
            let g_hi = blur3_u8_hi(p0.1, p1.1, p2.1);
            let b_hi = blur3_u8_hi(p0.2, p1.2, p2.2);

            // Store low 8 pixels
            let rgb_out_lo = uint8x8x3_t(r_lo, g_lo, b_lo);
            vst3_u8(temp.as_mut_ptr().add(out_offset) as *mut u8, rgb_out_lo);

            // Store high 8 pixels
            let rgb_out_hi = uint8x8x3_t(r_hi, g_hi, b_hi);
            vst3_u8(
                temp.as_mut_ptr().add(out_offset + 8 * 3) as *mut u8,
                rgb_out_hi,
            );

            simd_processed_end = base_x + TILE;
        }

        // Handle middle pixels that SIMD didn't process (scalar fallback)
        for x in simd_processed_end..simd_end {
            for c in 0..3 {
                let mut sum: u32 = 0;
                for k in 0..3 {
                    let px =
                        (x as i32 + k as i32 - radius as i32).clamp(0, width as i32 - 1) as usize;
                    sum = sum.wrapping_add(data[row_offset + px * 3 + c] as u32 * [1u32, 2, 1][k]);
                }
                temp[row_offset + x * 3 + c] = (sum >> 2) as u8;
            }
        }

        // Handle right edge
        for x in simd_end..width {
            for c in 0..3 {
                let mut sum: u32 = 0;
                for k in 0..3 {
                    let px =
                        (x as i32 + k as i32 - radius as i32).clamp(0, width as i32 - 1) as usize;
                    sum = sum.wrapping_add(data[row_offset + px * 3 + c] as u32 * [1u32, 2, 1][k]);
                }
                temp[row_offset + x * 3 + c] = (sum >> 2) as u8;
            }
        }
    }

    // ============================================================================
    // VERTICAL PASS: u8 → u8 with SYMMETRIC FOLDING and FULL ROW PROCESSING
    // OPTIMIZATION 1: (row[-1] + row[1]) + (row[0] << 1) >> 2 instead of multiply
    // OPTIMIZATION 2: Process FULL ROW at once for cache efficiency (not column blocks)
    // This is critical for large images - we read each row only once instead of N times!
    // ============================================================================
    let mut output = vec![0u8; data.len()];
    const SIMD_WIDTH: usize = 16; // Process 16 bytes at a time

    // Top edge - scalar (first row only for radius=1)
    for y in 0..height.min(radius) {
        for i in 0..row_bytes {
            let mut sum: u32 = 0;
            for k in 0..3 {
                let py = (y as i32 + k as i32 - radius as i32).clamp(0, height as i32 - 1) as usize;
                sum = sum.wrapping_add(temp[py * row_bytes + i] as u32 * [1u32, 2, 1][k]);
            }
            output[y * row_bytes + i] = (sum >> 2) as u8;
        }
    }

    // Middle rows - process full row with SIMD (16 bytes at a time)
    let simd_start = radius;
    let simd_end = height.saturating_sub(radius);

    for y in simd_start..simd_end {
        let row_minus1 = temp.as_ptr().add((y - 1) * row_bytes);
        let row0 = temp.as_ptr().add(y * row_bytes);
        let row1 = temp.as_ptr().add((y + 1) * row_bytes);
        let out_row = output.as_mut_ptr().add(y * row_bytes);

        // Process in chunks of SIMD_WIDTH bytes
        let mut byte_idx = 0;
        let simd_chunks = row_bytes / SIMD_WIDTH;

        for _ in 0..simd_chunks {
            // Load 16 u8 values from each of the 3 rows
            let v_minus1 = vld1q_u8(row_minus1.add(byte_idx));
            let v0 = vld1q_u8(row0.add(byte_idx));
            let v1 = vld1q_u8(row1.add(byte_idx));

            // CRITICAL: Widen to u16 BEFORE arithmetic to avoid overflow!
            // [1 2 1] kernel with symmetric folding: (row[-1] + row[1]) + (row[0] << 1), then >> 2
            let v_minus1_16 = vmovl_u8(vget_low_u8(v_minus1)); // 8 x u16
            let v0_16 = vmovl_u8(vget_low_u8(v0)); // 8 x u16
            let v1_16 = vmovl_u8(vget_low_u8(v1)); // 8 x u16

            let v_minus1_16_hi = vmovl_u8(vget_high_u8(v_minus1)); // 8 x u16
            let v0_16_hi = vmovl_u8(vget_high_u8(v0)); // 8 x u16
            let v1_16_hi = vmovl_u8(vget_high_u8(v1)); // 8 x u16

            // (row[-1] + row[1]) + (row[0] << 1)
            let sum_lo = vaddq_u16(v_minus1_16, v1_16);
            let center_lo = vshlq_n_u16(v0_16, 1);
            let total_lo = vaddq_u16(sum_lo, center_lo);

            let sum_hi = vaddq_u16(v_minus1_16_hi, v1_16_hi);
            let center_hi = vshlq_n_u16(v0_16_hi, 1);
            let total_hi = vaddq_u16(sum_hi, center_hi);

            // >> 2 and narrow to u8
            let result_lo = vshrn_n_u16(total_lo, 2);
            let result_hi = vshrn_n_u16(total_hi, 2);
            let result = vcombine_u8(result_lo, result_hi);

            vst1q_u8(out_row.add(byte_idx), result);
            byte_idx += SIMD_WIDTH;
        }

        // Handle remaining bytes (scalar)
        for i in byte_idx..row_bytes {
            let mut sum: u32 = 0;
            sum = sum.wrapping_add(temp[(y - 1) * row_bytes + i] as u32 * 1);
            sum = sum.wrapping_add(temp[y * row_bytes + i] as u32 * 2);
            sum = sum.wrapping_add(temp[(y + 1) * row_bytes + i] as u32 * 1);
            output[y * row_bytes + i] = (sum >> 2) as u8;
        }
    }

    // Bottom edge - scalar (last row only for radius=1)
    for y in simd_end.max(radius)..height {
        for i in 0..row_bytes {
            let mut sum: u32 = 0;
            for k in 0..3 {
                let py = (y as i32 + k as i32 - radius as i32).clamp(0, height as i32 - 1) as usize;
                sum = sum.wrapping_add(temp[py * row_bytes + i] as u32 * [1u32, 2, 1][k]);
            }
            output[y * row_bytes + i] = (sum >> 2) as u8;
        }
    }

    data.copy_from_slice(&output);
}

// ============================================================================
// Helper functions for 3-tap blur
// ============================================================================

/// Helper for 3-tap blur on low 8 pixels (kernel: [1 2 1] >> 2)
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn blur3_u8_lo(a: uint8x16_t, b: uint8x16_t, c: uint8x16_t) -> uint8x8_t {
    let mut sum = vmull_u8(vget_low_u8(a), vdup_n_u8(1));
    sum = vmlal_u8(sum, vget_low_u8(b), vdup_n_u8(2));
    sum = vmlal_u8(sum, vget_low_u8(c), vdup_n_u8(1));
    vshrn_n_u16(sum, 2)
}

/// Helper for 3-tap blur on high 8 pixels (kernel: [1 2 1] >> 2)
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn blur3_u8_hi(a: uint8x16_t, b: uint8x16_t, c: uint8x16_t) -> uint8x8_t {
    let mut sum = vmull_u8(vget_high_u8(a), vdup_n_u8(1));
    sum = vmlal_u8(sum, vget_high_u8(b), vdup_n_u8(2));
    sum = vmlal_u8(sum, vget_high_u8(c), vdup_n_u8(1));
    vshrn_n_u16(sum, 2)
}

/// Helper to apply [1 2 1] >> 2 kernel to three vectors (element-wise)
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn blur3_scalar_to_u8(a: uint8x8_t, b: uint8x8_t, c: uint8x8_t) -> uint8x8_t {
    let sum = vmull_u8(a, vdup_n_u8(1));
    let sum = vmlal_u8(sum, b, vdup_n_u8(2));
    let sum = vmlal_u8(sum, c, vdup_n_u8(1));
    vshrn_n_u16(sum, 2)
}

#[cfg(target_arch = "aarch64")]
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn blur3_gray_u8(p0: uint8x16_t, p1: uint8x16_t, p2: uint8x16_t) -> uint8x16_t {
    let ends_lo = vaddl_u8(vget_low_u8(p0), vget_low_u8(p2));
    let tot_lo = vmlal_u8(ends_lo, vget_low_u8(p1), vdup_n_u8(2));
    let res_lo = vrshrn_n_u16(tot_lo, 2);

    let ends_hi = vaddl_u8(vget_high_u8(p0), vget_high_u8(p2));
    let tot_hi = vmlal_u8(ends_hi, vget_high_u8(p1), vdup_n_u8(2));
    let res_hi = vrshrn_n_u16(tot_hi, 2);

    vcombine_u8(res_lo, res_hi)
}

#[cfg(target_arch = "aarch64")]
unsafe fn convolve_separable_gray_neon_3(image: &mut FusableImage) {
    let width = image.width;
    let height = image.height;
    let data = &mut image.data;
    let total_bytes = data.len();
    let mut temp = Vec::<u8>::with_capacity(total_bytes);
    unsafe { temp.set_len(total_bytes); }
    let radius = 1;
    const TILE: usize = 16;

    // HORIZONTAL PASS
    for y in 0..height {
        let row_offset = y * width;
        let in_ptr = data.as_ptr().add(row_offset);
        let out_ptr = temp.as_mut_ptr().add(row_offset);

        // Left edge (x = 0)
        let left_val = *in_ptr as u32;
        let mid_val = *in_ptr as u32;
        let right_val = if width > 1 { *in_ptr.add(1) as u32 } else { left_val };
        *out_ptr = ((left_val + mid_val * 2 + right_val + 2) >> 2) as u8;

        let simd_start = radius;
        let simd_end = width.saturating_sub(radius);
        let simd_chunks = if simd_end > simd_start { (simd_end - simd_start) / TILE } else { 0 };

        let mut x = simd_start;
        for _ in 0..simd_chunks {
            let p0 = vld1q_u8(in_ptr.add(x - 1));
            let p1 = vld1q_u8(in_ptr.add(x));
            let p2 = vld1q_u8(in_ptr.add(x + 1));

            let combined = blur3_gray_u8(p0, p1, p2);
            vst1q_u8(out_ptr.add(x), combined);
            x += TILE;
        }

        // Middle remainder
        for rem_x in x..simd_end {
            let p0 = *in_ptr.add(rem_x - 1) as u32;
            let p1 = *in_ptr.add(rem_x) as u32;
            let p2 = *in_ptr.add(rem_x + 1) as u32;
            *out_ptr.add(rem_x) = ((p0 + p1 * 2 + p2 + 2) >> 2) as u8;
        }

        // Right edge
        if width > 1 {
            let rx = width - 1;
            let p0 = *in_ptr.add(rx - 1) as u32;
            let p1 = *in_ptr.add(rx) as u32;
            let p2 = *in_ptr.add(rx) as u32;
            *out_ptr.add(rx) = ((p0 + p1 * 2 + p2 + 2) >> 2) as u8;
        }
    }

    // VERTICAL PASS
    let row_chunks = width / 16;
    for y in 0..height {
        let prev_y = y.saturating_sub(1);
        let next_y = (y + 1).min(height - 1);

        let prev_ptr = temp.as_ptr().add(prev_y * width);
        let curr_ptr = temp.as_ptr().add(y * width);
        let next_ptr = temp.as_ptr().add(next_y * width);
        let out_ptr = data.as_mut_ptr().add(y * width);

        for chunk in 0..row_chunks {
            let offset = chunk * 16;
            let r_prev = vld1q_u8(prev_ptr.add(offset));
            let r_curr = vld1q_u8(curr_ptr.add(offset));
            let r_next = vld1q_u8(next_ptr.add(offset));

            let res = blur3_gray_u8(r_prev, r_curr, r_next);
            vst1q_u8(out_ptr.add(offset), res);
        }

        for x in (row_chunks * 16)..width {
            let p0 = *prev_ptr.add(x) as u32;
            let p1 = *curr_ptr.add(x) as u32;
            let p2 = *next_ptr.add(x) as u32;
            *out_ptr.add(x) = ((p0 + p1 * 2 + p2 + 2) >> 2) as u8;
        }
    }
}
