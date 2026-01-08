// 5x5 kernel convolution implementations (Gaussian [1 4 6 4 1] / 16)
//
// NEON SIMD implementations for aarch64

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
use crate::core::FusableImage;

// ============================================================================
// Public entry points for 5x5 kernel (NEON implementations)
// ============================================================================

#[cfg(target_arch = "aarch64")]
pub unsafe fn convolve_1d_horizontal_neon_5(image: &mut FusableImage, kernel: &[i32], _scale: i32) {
    let width = image.width;
    let height = image.height;
    let channels = image.channels;
    let data = &mut image.data;

    if channels != 3 {
        // Fallback to scalar for non-RGB
        super::super::super::convolve::convolve_1d_horizontal(image, kernel, 16);
        return;
    }

    let mut output = vec![0u8; data.len()];
    let radius = 2;

    const TILE: usize = 16;

    for y in 0..height {
        let row_offset = y * width * channels;

        // Handle left edge
        for x in 0..width.min(radius) {
            for c in 0..3 {
                let mut sum: u32 = 0;
                for k in 0..5 {
                    let px = (x as i32 + k as i32 - radius as i32).clamp(0, width as i32 - 1) as usize;
                    sum = sum.wrapping_add(data[row_offset + px * 3 + c] as u32 * kernel[k] as u32);
                }
                output[row_offset + x * 3 + c] = (sum >> 4) as u8;
            }
        }

        // SIMD middle - process 16 pixels at a time
        let simd_start = radius;
        let simd_end = width.saturating_sub(radius);
        let simd_chunks = if simd_end > simd_start { (simd_end - simd_start) / TILE } else { 0 };

        // Track where SIMD processing ends (or would end)
        let mut simd_processed_end = simd_start;

        for chunk in 0..simd_chunks {
            let base_x = simd_start + chunk * TILE;
            let out_offset = row_offset + base_x * 3;

            // Load five 16-pixel RGB chunks: x-2, x-1, x, x+1, x+2
            let p_2 = vld3q_u8(data.as_ptr().add(row_offset + (base_x - 2) * 3) as *const u8);
            let p_1 = vld3q_u8(data.as_ptr().add(row_offset + (base_x - 1) * 3) as *const u8);
            let p0 = vld3q_u8(data.as_ptr().add(row_offset + base_x * 3) as *const u8);
            let p1 = vld3q_u8(data.as_ptr().add(row_offset + (base_x + 1) * 3) as *const u8);
            let p2 = vld3q_u8(data.as_ptr().add(row_offset + (base_x + 2) * 3) as *const u8);

            // Process low 8 pixels: [1 4 6 4 1] kernel
            let r_lo = blur5_u8_lo(p_2.0, p_1.0, p0.0, p1.0, p2.0);
            let g_lo = blur5_u8_lo(p_2.1, p_1.1, p0.1, p1.1, p2.1);
            let b_lo = blur5_u8_lo(p_2.2, p_1.2, p0.2, p1.2, p2.2);

            // Process high 8 pixels
            let r_hi = blur5_u8_hi(p_2.0, p_1.0, p0.0, p1.0, p2.0);
            let g_hi = blur5_u8_hi(p_2.1, p_1.1, p0.1, p1.1, p2.1);
            let b_hi = blur5_u8_hi(p_2.2, p_1.2, p0.2, p1.2, p2.2);

            // Store low 8
            let rgb_out_lo = uint8x8x3_t(r_lo, g_lo, b_lo);
            vst3_u8(output.as_mut_ptr().add(out_offset) as *mut u8, rgb_out_lo);
            // Store high 8
            let rgb_out_hi = uint8x8x3_t(r_hi, g_hi, b_hi);
            vst3_u8(output.as_mut_ptr().add(out_offset + 8 * 3) as *mut u8, rgb_out_hi);

            simd_processed_end = base_x + TILE;
        }

        // CRITICAL FIX: Handle middle pixels that SIMD didn't process (scalar fallback)
        // This is needed when simd_chunks = 0 or when pixels don't fit in full 16-pixel chunks
        for x in simd_processed_end..simd_end {
            for c in 0..3 {
                let mut sum: u32 = 0;
                for k in 0..5 {
                    let px = (x as i32 + k as i32 - radius as i32).clamp(0, width as i32 - 1) as usize;
                    sum = sum.wrapping_add(data[row_offset + px * 3 + c] as u32 * kernel[k] as u32);
                }
                output[row_offset + x * 3 + c] = (sum >> 4) as u8;
            }
        }

        // Handle right edge
        for x in simd_end..width {
            for c in 0..3 {
                let mut sum: u32 = 0;
                for k in 0..5 {
                    let px = (x as i32 + k as i32 - radius as i32).clamp(0, width as i32 - 1) as usize;
                    sum = sum.wrapping_add(data[row_offset + px * 3 + c] as u32 * kernel[k] as u32);
                }
                output[row_offset + x * 3 + c] = (sum >> 4) as u8;
            }
        }
    }

    data.copy_from_slice(&output);
}

#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn convolve_1d_horizontal_neon_5(image: &mut FusableImage, kernel: &[i32], _scale: i32) {
    // Fallback to scalar implementation for non-ARM platforms
    super::super::super::convolve::convolve_1d_horizontal(image, kernel, 16);
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn convolve_1d_vertical_neon_5(image: &mut FusableImage, kernel: &[i32], _scale: i32) {
    let width = image.width;
    let height = image.height;
    let channels = image.channels;

    if channels != 3 {
        // Fallback to scalar for non-RGB
        super::super::super::convolve::convolve_1d_vertical(image, kernel, 16);
        return;
    }

    let data = &mut image.data;
    let mut output = vec![0u8; data.len()];
    let radius = 2;

    // Process 8 columns at a time
    const COLS_PER_ITER: usize = 8;

    for x_block in 0..((width + COLS_PER_ITER - 1) / COLS_PER_ITER) {
        let x_start = x_block * COLS_PER_ITER;
        let x_end = (x_start + COLS_PER_ITER).min(width);

        // Top edge - scalar
        for y in 0..height.min(radius) {
            for x in x_start..x_end {
                for c in 0..3 {
                    let mut sum: u32 = 0;
                    for k in 0..5 {
                        let py = (y as i32 + k as i32 - radius as i32).clamp(0, height as i32 - 1) as usize;
                        sum = sum.wrapping_add(data[(py * width + x) * 3 + c] as u32 * kernel[k] as u32);
                    }
                    output[(y * width + x) * 3 + c] = (sum >> 4) as u8;
                }
            }
        }

        // SIMD middle - process all middle rows at once for these 8 columns
        // Only use SIMD if we have a full block of 8 columns
        let use_simd = x_end - x_start == COLS_PER_ITER;
        let simd_start = radius;
        let simd_end = height.saturating_sub(radius);

        if use_simd {
            for y in simd_start..simd_end {
                // Load 5 rows of pixels for these 8 columns
                let row_2 = data.as_ptr().add(((y - 2) * width + x_start) * 3) as *const u8;
                let row_1 = data.as_ptr().add(((y - 1) * width + x_start) * 3) as *const u8;
                let row0 = data.as_ptr().add((y * width + x_start) * 3) as *const u8;
                let row1 = data.as_ptr().add(((y + 1) * width + x_start) * 3) as *const u8;
                let row2 = data.as_ptr().add(((y + 2) * width + x_start) * 3) as *const u8;

                // Load 8 RGB pixels from each row
                let p_2 = vld3q_u8(row_2);
                let p_1 = vld3q_u8(row_1);
                let p0 = vld3q_u8(row0);
                let p1 = vld3q_u8(row1);
                let p2 = vld3q_u8(row2);

                // Apply [1 4 6 4 1] kernel vertically to all 8 pixels
                let r_blur = blur5_scalar_to_u8(vget_low_u8(p_2.0), vget_low_u8(p_1.0), vget_low_u8(p0.0), vget_low_u8(p1.0), vget_low_u8(p2.0));
                let g_blur = blur5_scalar_to_u8(vget_low_u8(p_2.1), vget_low_u8(p_1.1), vget_low_u8(p0.1), vget_low_u8(p1.1), vget_low_u8(p2.1));
                let b_blur = blur5_scalar_to_u8(vget_low_u8(p_2.2), vget_low_u8(p_1.2), vget_low_u8(p0.2), vget_low_u8(p1.2), vget_low_u8(p2.2));

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
                        for k in 0..5 {
                            let py = (y as i32 + k as i32 - radius as i32).clamp(0, height as i32 - 1) as usize;
                            sum = sum.wrapping_add(data[(py * width + x) * 3 + c] as u32 * kernel[k] as u32);
                        }
                        output[(y * width + x) * 3 + c] = (sum >> 4) as u8;
                    }
                }
            }
        }

        // Bottom edge - scalar
        for y in simd_end.max(radius)..height {
            for x in x_start..x_end {
                for c in 0..3 {
                    let mut sum: u32 = 0;
                    for k in 0..5 {
                        let py = (y as i32 + k as i32 - radius as i32).clamp(0, height as i32 - 1) as usize;
                        sum = sum.wrapping_add(data[(py * width + x) * 3 + c] as u32 * kernel[k] as u32);
                    }
                    output[(y * width + x) * 3 + c] = (sum >> 4) as u8;
                }
            }
        }
    }

    data.copy_from_slice(&output);
}

#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn convolve_1d_vertical_neon_5(image: &mut FusableImage, kernel: &[i32], _scale: i32) {
    // Fallback to scalar implementation for non-ARM platforms
    super::super::super::convolve::convolve_1d_vertical(image, kernel, 16);
}

/// Full separable 5x5 convolution with u8 intermediates and full row processing
/// Kernel: [1 4 6 4 1] / 16 (Pascal's triangle row 4)
#[cfg(target_arch = "aarch64")]
pub unsafe fn convolve_separable_neon_5(image: &mut FusableImage, _kernel: &[i32], _scale: i32) {
    let width = image.width;
    let height = image.height;
    let channels = image.channels;
    let data = &mut image.data;

    if channels != 3 {
        // Fallback to scalar for non-RGB
        super::super::super::convolve::convolve_1d_horizontal(image, &[1, 4, 6, 4, 1], 16);
        super::super::super::convolve::convolve_1d_vertical(image, &[1, 4, 6, 4, 1], 16);
        return;
    }

    let mut temp = vec![0u8; data.len()];
    let radius = 2;
    const TILE: usize = 16;
    let row_bytes = width * 3;

    // ============================================================================
    // HORIZONTAL PASS: u8 -> u8 (process 16 RGB pixels per iteration)
    // ============================================================================
    for y in 0..height {
        let row_offset = y * row_bytes;

        // Handle left edge
        for x in 0..width.min(radius) {
            for c in 0..3 {
                let mut sum: u32 = 0;
                for k in 0..5 {
                    let px = (x as i32 + k as i32 - radius as i32).clamp(0, width as i32 - 1) as usize;
                    sum = sum.wrapping_add(data[row_offset + px * 3 + c] as u32 * [1u32, 4, 6, 4, 1][k]);
                }
                temp[row_offset + x * 3 + c] = (sum >> 4) as u8;
            }
        }

        // SIMD middle - process 16 pixels at a time
        let simd_start = radius;
        let simd_end = width.saturating_sub(radius);
        let simd_chunks = if simd_end > simd_start { (simd_end - simd_start) / TILE } else { 0 };

        let mut simd_processed_end = simd_start;

        for chunk in 0..simd_chunks {
            let base_x = simd_start + chunk * TILE;
            let out_offset = row_offset + base_x * 3;

            // Load five 16-pixel RGB chunks: x-2, x-1, x, x+1, x+2
            let p_2 = vld3q_u8(data.as_ptr().add(row_offset + (base_x - 2) * 3) as *const u8);
            let p_1 = vld3q_u8(data.as_ptr().add(row_offset + (base_x - 1) * 3) as *const u8);
            let p0 = vld3q_u8(data.as_ptr().add(row_offset + base_x * 3) as *const u8);
            let p1 = vld3q_u8(data.as_ptr().add(row_offset + (base_x + 1) * 3) as *const u8);
            let p2 = vld3q_u8(data.as_ptr().add(row_offset + (base_x + 2) * 3) as *const u8);

            // Process low 8 pixels: [1 4 6 4 1] kernel
            let r_lo = blur5_u8_lo(p_2.0, p_1.0, p0.0, p1.0, p2.0);
            let g_lo = blur5_u8_lo(p_2.1, p_1.1, p0.1, p1.1, p2.1);
            let b_lo = blur5_u8_lo(p_2.2, p_1.2, p0.2, p1.2, p2.2);

            // Process high 8 pixels
            let r_hi = blur5_u8_hi(p_2.0, p_1.0, p0.0, p1.0, p2.0);
            let g_hi = blur5_u8_hi(p_2.1, p_1.1, p0.1, p1.1, p2.1);
            let b_hi = blur5_u8_hi(p_2.2, p_1.2, p0.2, p1.2, p2.2);

            // Store low 8 pixels
            let rgb_out_lo = uint8x8x3_t(r_lo, g_lo, b_lo);
            vst3_u8(temp.as_mut_ptr().add(out_offset) as *mut u8, rgb_out_lo);

            // Store high 8 pixels
            let rgb_out_hi = uint8x8x3_t(r_hi, g_hi, b_hi);
            vst3_u8(temp.as_mut_ptr().add(out_offset + 8 * 3) as *mut u8, rgb_out_hi);

            simd_processed_end = base_x + TILE;
        }

        // Handle middle pixels that SIMD didn't process (scalar fallback)
        for x in simd_processed_end..simd_end {
            for c in 0..3 {
                let mut sum: u32 = 0;
                for k in 0..5 {
                    let px = (x as i32 + k as i32 - radius as i32).clamp(0, width as i32 - 1) as usize;
                    sum = sum.wrapping_add(data[row_offset + px * 3 + c] as u32 * [1u32, 4, 6, 4, 1][k]);
                }
                temp[row_offset + x * 3 + c] = (sum >> 4) as u8;
            }
        }

        // Handle right edge
        for x in simd_end..width {
            for c in 0..3 {
                let mut sum: u32 = 0;
                for k in 0..5 {
                    let px = (x as i32 + k as i32 - radius as i32).clamp(0, width as i32 - 1) as usize;
                    sum = sum.wrapping_add(data[row_offset + px * 3 + c] as u32 * [1u32, 4, 6, 4, 1][k]);
                }
                temp[row_offset + x * 3 + c] = (sum >> 4) as u8;
            }
        }
    }

    // ============================================================================
    // VERTICAL PASS: u8 -> u8 with SYMMETRIC FOLDING and FULL ROW PROCESSING
    // OPTIMIZATION 1: (row[-2] + row[2]) + 4*(row[-1] + row[1]) + 6*row[0] >> 4
    //                This reduces 5 multiplies to 3 multiplies + 2 adds!
    // OPTIMIZATION 2: Process full rows at once instead of column blocks
    //                This is critical for large images - we read each row only once!
    // ============================================================================
    let mut output = vec![0u8; data.len()];
    const SIMD_WIDTH: usize = 16; // Process 16 bytes at a time

    // Top edge - scalar (first 2 rows)
    for y in 0..height.min(radius) {
        for i in 0..row_bytes {
            let mut sum: u32 = 0;
            for k in 0..5 {
                let py = (y as i32 + k as i32 - radius as i32).clamp(0, height as i32 - 1) as usize;
                sum = sum.wrapping_add(temp[py * row_bytes + i] as u32 * [1u32, 4, 6, 4, 1][k]);
            }
            output[y * row_bytes + i] = (sum >> 4) as u8;
        }
    }

    // Middle rows - process full row with SIMD (16 bytes at a time)
    let simd_start = radius;
    let simd_end = height.saturating_sub(radius);

    for y in simd_start..simd_end {
        let row_minus2 = temp.as_ptr().add((y - 2) * row_bytes);
        let row_minus1 = temp.as_ptr().add((y - 1) * row_bytes);
        let row0 = temp.as_ptr().add(y * row_bytes);
        let row1 = temp.as_ptr().add((y + 1) * row_bytes);
        let row2 = temp.as_ptr().add((y + 2) * row_bytes);
        let out_row = output.as_mut_ptr().add(y * row_bytes);

        // Process in chunks of SIMD_WIDTH bytes
        let mut byte_idx = 0;
        let simd_chunks = row_bytes / SIMD_WIDTH;

        for _ in 0..simd_chunks {
            // Load 16 u8 values from each of the 5 rows
            let v_minus2 = vld1q_u8(row_minus2.add(byte_idx));
            let v_minus1 = vld1q_u8(row_minus1.add(byte_idx));
            let v0 = vld1q_u8(row0.add(byte_idx));
            let v1 = vld1q_u8(row1.add(byte_idx));
            let v2 = vld1q_u8(row2.add(byte_idx));

            // CRITICAL: Widen to u16 BEFORE arithmetic to avoid overflow!
            // SYMMETRIC FOLDING for [1 4 6 4 1] kernel:
            //   (row[-2] + row[2]) + 4*(row[-1] + row[1]) + 6*row[0], then >> 4
            // This reduces 5 multiplies to 3 multiplies + 2 adds!

            // Low 8 pixels
            let sum_04_lo = vaddq_u16(vmovl_u8(vget_low_u8(v_minus2)), vmovl_u8(vget_low_u8(v2)));
            let sum_13_lo = vaddq_u16(vmovl_u8(vget_low_u8(v_minus1)), vmovl_u8(vget_low_u8(v1)));
            let center_lo = vmovl_u8(vget_low_u8(v0));

            // (row[-2] + row[2]) + 4*(row[-1] + row[1]) + 6*row[0]
            let mut total_lo = vshlq_n_u16(sum_13_lo, 2); // 4 * sum_13
            total_lo = vmlaq_u16(total_lo, sum_04_lo, vdupq_n_u16(1)); // + sum_04
            total_lo = vmlaq_u16(total_lo, center_lo, vdupq_n_u16(6)); // + 6*center

            // High 8 pixels
            let sum_04_hi = vaddq_u16(vmovl_u8(vget_high_u8(v_minus2)), vmovl_u8(vget_high_u8(v2)));
            let sum_13_hi = vaddq_u16(vmovl_u8(vget_high_u8(v_minus1)), vmovl_u8(vget_high_u8(v1)));
            let center_hi = vmovl_u8(vget_high_u8(v0));

            // (row[-2] + row[2]) + 4*(row[-1] + row[1]) + 6*row[0]
            let mut total_hi = vshlq_n_u16(sum_13_hi, 2); // 4 * sum_13
            total_hi = vmlaq_u16(total_hi, sum_04_hi, vdupq_n_u16(1)); // + sum_04
            total_hi = vmlaq_u16(total_hi, center_hi, vdupq_n_u16(6)); // + 6*center

            // Normalize: >> 4 and narrow to u8
            let result_lo = vshrn_n_u16(total_lo, 4);
            let result_hi = vshrn_n_u16(total_hi, 4);
            let result = vcombine_u8(result_lo, result_hi);

            vst1q_u8(out_row.add(byte_idx), result);
            byte_idx += SIMD_WIDTH;
        }

        // Handle remaining bytes (scalar)
        for i in byte_idx..row_bytes {
            let mut sum: u32 = 0;
            for k in 0..5 {
                let py = (y as i32 + k as i32 - radius as i32).clamp(0, height as i32 - 1) as usize;
                sum = sum.wrapping_add(temp[py * row_bytes + i] as u32 * [1u32, 4, 6, 4, 1][k]);
            }
            output[y * row_bytes + i] = (sum >> 4) as u8;
        }
    }

    // Bottom edge - scalar (last 2 rows)
    for y in simd_end.max(radius)..height {
        for i in 0..row_bytes {
            let mut sum: u32 = 0;
            for k in 0..5 {
                let py = (y as i32 + k as i32 - radius as i32).clamp(0, height as i32 - 1) as usize;
                sum = sum.wrapping_add(temp[py * row_bytes + i] as u32 * [1u32, 4, 6, 4, 1][k]);
            }
            output[y * row_bytes + i] = (sum >> 4) as u8;
        }
    }

    data.copy_from_slice(&output);
}

#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn convolve_separable_neon_5(image: &mut FusableImage, _kernel: &[i32], _scale: i32) {
    // Fallback to scalar implementation for non-ARM platforms
    super::super::super::convolve::convolve_1d_horizontal(image, &[1, 4, 6, 4, 1], 16);
    super::super::super::convolve::convolve_1d_vertical(image, &[1, 4, 6, 4, 1], 16);
}

// ============================================================================
// Helper functions for 5-tap blur
// ============================================================================

/// Helper for 5-tap blur on low 8 pixels (kernel: [1 4 6 4 1] >> 4)
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn blur5_u8_lo(a: uint8x16_t, b: uint8x16_t, c: uint8x16_t, d: uint8x16_t, e: uint8x16_t) -> uint8x8_t {
    let mut sum = vmull_u8(vget_low_u8(a), vdup_n_u8(1));
    sum = vmlal_u8(sum, vget_low_u8(b), vdup_n_u8(4));
    sum = vmlal_u8(sum, vget_low_u8(c), vdup_n_u8(6));
    sum = vmlal_u8(sum, vget_low_u8(d), vdup_n_u8(4));
    sum = vmlal_u8(sum, vget_low_u8(e), vdup_n_u8(1));
    vshrn_n_u16(sum, 4)
}

/// Helper for 5-tap blur on high 8 pixels (kernel: [1 4 6 4 1] >> 4)
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn blur5_u8_hi(a: uint8x16_t, b: uint8x16_t, c: uint8x16_t, d: uint8x16_t, e: uint8x16_t) -> uint8x8_t {
    let mut sum = vmull_u8(vget_high_u8(a), vdup_n_u8(1));
    sum = vmlal_u8(sum, vget_high_u8(b), vdup_n_u8(4));
    sum = vmlal_u8(sum, vget_high_u8(c), vdup_n_u8(6));
    sum = vmlal_u8(sum, vget_high_u8(d), vdup_n_u8(4));
    sum = vmlal_u8(sum, vget_high_u8(e), vdup_n_u8(1));
    vshrn_n_u16(sum, 4)
}

/// Helper to apply [1 4 6 4 1] >> 4 kernel to five vectors (element-wise)
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn blur5_scalar_to_u8(a: uint8x8_t, b: uint8x8_t, c: uint8x8_t, d: uint8x8_t, e: uint8x8_t) -> uint8x8_t {
    let sum = vmull_u8(a, vdup_n_u8(1));
    let sum = vmlal_u8(sum, b, vdup_n_u8(4));
    let sum = vmlal_u8(sum, c, vdup_n_u8(6));
    let sum = vmlal_u8(sum, d, vdup_n_u8(4));
    let sum = vmlal_u8(sum, e, vdup_n_u8(1));
    vshrn_n_u16(sum, 4)
}
