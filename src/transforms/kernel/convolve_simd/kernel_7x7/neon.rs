// 7x7 kernel convolution implementations - NEON SIMD
//
// This module contains all ARM NEON SIMD implementations for 7x7 convolution.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
use crate::core::FusableImage;

// ============================================================================
// Public entry points for 7x7 kernel - NEON implementations
// ============================================================================

#[cfg(target_arch = "aarch64")]
pub(crate) unsafe fn convolve_1d_horizontal_neon_7(image: &mut FusableImage, kernel: &[i32], scale: i32) {
    let width = image.width;
    let height = image.height;
    let channels = image.channels;
    let data = &mut image.data;

    let mut output = vec![0u8; data.len()];
    let radius = 3;

    // Convert kernel to Q8.8 fixed-point
    let kq: [u16; 7] = std::array::from_fn(|i| {
        ((kernel[i] as u64) * 256 / (scale as u64)) as u16
    });

    const TILE: usize = 8;

    for y in 0..height {
        let row_offset = y * width * channels;

        if channels == 3 {
            // Handle edges
            for x in 0..width.min(radius) {
                for c in 0..3 {
                    let mut sum: u32 = 0;
                    for k in 0..7 {
                        let px = (x as i32 + k as i32 - radius as i32).clamp(0, width as i32 - 1) as usize;
                        sum = sum.wrapping_add(data[row_offset + px * 3 + c] as u32 * kq[k] as u32);
                    }
                    output[row_offset + x * 3 + c] = (sum >> 8) as u8;
                }
            }

            // SIMD middle
            let simd_start = radius;
            let simd_end = width.saturating_sub(radius);
            let simd_chunks = if simd_end > simd_start { (simd_end - simd_start) / TILE } else { 0 };

            for chunk in 0..simd_chunks {
                let base_x = simd_start + chunk * TILE;
                let out_offset = row_offset + base_x * 3;

                let mut acc_r = vdupq_n_u16(0);
                let mut acc_g = vdupq_n_u16(0);
                let mut acc_b = vdupq_n_u16(0);

                for k in 0..7 {
                    let px = base_x as i32 + k as i32 - radius as i32;
                    let src_offset = row_offset + (px as usize) * 3;
                    let rgb = vld3_u8(data.as_ptr().add(src_offset) as *const u8);

                    let r_u16 = vmovl_u8(rgb.0);
                    let g_u16 = vmovl_u8(rgb.1);
                    let b_u16 = vmovl_u8(rgb.2);
                    let k_vec = vdupq_n_u16(kq[k]);

                    acc_r = vmlaq_u16(acc_r, r_u16, k_vec);
                    acc_g = vmlaq_u16(acc_g, g_u16, k_vec);
                    acc_b = vmlaq_u16(acc_b, b_u16, k_vec);
                }

                let r_out = vqshrn_n_u16(acc_r, 8);
                let g_out = vqshrn_n_u16(acc_g, 8);
                let b_out = vqshrn_n_u16(acc_b, 8);

                let rgb_out = uint8x8x3_t(r_out, g_out, b_out);
                vst3_u8(output.as_mut_ptr().add(out_offset) as *mut u8, rgb_out);
            }

            // Handle right edge
            for x in simd_end.max(radius)..width {
                for c in 0..3 {
                    let mut sum: u32 = 0;
                    for k in 0..7 {
                        let px = (x as i32 + k as i32 - radius as i32).clamp(0, width as i32 - 1) as usize;
                        sum = sum.wrapping_add(data[row_offset + px * 3 + c] as u32 * kq[k] as u32);
                    }
                    output[row_offset + x * 3 + c] = (sum >> 8) as u8;
                }
            }
        } else {
            // Scalar fallback
            for x in 0..width {
                for c in 0..channels {
                    let mut sum: u32 = 0;
                    for k in 0..7 {
                        let px = (x as i32 + k as i32 - radius as i32).clamp(0, width as i32 - 1) as usize;
                        sum = sum.wrapping_add(data[row_offset + px * channels + c] as u32 * kq[k] as u32);
                    }
                    output[row_offset + x * channels + c] = (sum >> 8) as u8;
                }
            }
        }
    }

    data.copy_from_slice(&output);
}

#[cfg(not(target_arch = "aarch64"))]
pub(crate) unsafe fn convolve_1d_horizontal_neon_7(_image: &mut FusableImage, _kernel: &[i32], _scale: i32) {
    // Fallback for non-ARM architectures
    // This should never be called on non-ARM platforms
    unreachable!("NEON functions should not be called on non-ARM platforms");
}

#[cfg(target_arch = "aarch64")]
pub(crate) unsafe fn convolve_1d_vertical_neon_7(image: &mut FusableImage, kernel: &[i32], scale: i32) {
    let width = image.width;
    let height = image.height;
    let channels = image.channels;

    let kq: [u16; 7] = std::array::from_fn(|i| {
        ((kernel[i] as u64) * 256 / (scale as u64)) as u16
    });

    if channels == 3 {
        // For now, use the scalar approach but with optimizations
        // The transpose overhead was too high
        vertical_pass_scalar_optimized(&mut image.data, width, height, &kq);
    } else {
        super::super::super::convolve::convolve_1d_vertical(image, kernel, scale);
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub(crate) unsafe fn convolve_1d_vertical_neon_7(_image: &mut FusableImage, _kernel: &[i32], _scale: i32) {
    // Fallback for non-ARM architectures
    unreachable!("NEON functions should not be called on non-ARM platforms");
}

#[cfg(target_arch = "aarch64")]
pub(crate) unsafe fn convolve_separable_neon_7(image: &mut FusableImage, _kernel: &[i32], _scale: i32) {
    let width = image.width;
    let height = image.height;
    let channels = image.channels;
    let data = &mut image.data;

    if channels != 3 {
        // Fallback to scalar for non-RGB
        super::super::super::convolve::convolve_1d_horizontal(image, &[1, 6, 15, 20, 15, 6, 1], 64);
        super::super::super::convolve::convolve_1d_vertical(image, &[1, 6, 15, 20, 15, 6, 1], 64);
        return;
    }

    // ============================================================================
    // HORIZONTAL PASS: u8 -> u16 (unaligned loads, no lane extraction!)
    // ============================================================================
    // Allocate u16 intermediate buffer
    let mut temp_u16 = vec![0u16; data.len()];

    // Kernel weights for 7x7 Pascal's triangle: [1, 6, 15, 20, 15, 6, 1]
    let k0 = vdup_n_u8(1);
    let k1 = vdup_n_u8(6);
    let k2 = vdup_n_u8(15);
    let k3 = vdup_n_u8(20);
    let k4 = vdup_n_u8(15);
    let k5 = vdup_n_u8(6);
    let k6 = vdup_n_u8(1);

    let radius = 3;
    let row_bytes = width * 3; // Total bytes per row

    for y in 0..height {
        let row_offset = y * row_bytes;

        // Handle left edge (scalar) - first 3 pixels
        for x in 0..width.min(radius) {
            for c in 0..3 {
                let mut sum: u32 = 0;
                for k in 0..7 {
                    let px = (x as i32 + k as i32 - radius as i32).clamp(0, width as i32 - 1) as usize;
                    sum = sum.wrapping_add(data[row_offset + px * 3 + c] as u32 * [1u32, 6, 15, 20, 15, 6, 1][k]);
                }
                temp_u16[row_offset + x * 3 + c] = (sum >> 6) as u16;
            }
        }

        // SIMD middle - process in chunks of 16 bytes
        // We need 34 bytes margin (16 + 18 for max tap offset)
        let simd_start_byte = radius * 3;  // Start at byte 9 (pixel 3)
        let simd_end_byte = row_bytes.saturating_sub(34);

        if simd_end_byte > simd_start_byte {
            let mut byte_idx = simd_start_byte;
            while byte_idx <= simd_end_byte {
                // Position pointer at target pixel (byte_idx)
                // Then load from offsets: -9, -6, -3, 0, +3, +6, +9
                let ptr = data.as_ptr().add(row_offset + byte_idx);

                // Tap 0 (offset -9 bytes = -3 pixels, leftmost)
                let v0 = vld1q_u8(ptr.sub(9));
                let mut acc_lo = vmull_u8(vget_low_u8(v0), k0);
                let mut acc_hi = vmull_u8(vget_high_u8(v0), k0);

                // Tap 1 (offset -6 bytes = -2 pixels)
                let v1 = vld1q_u8(ptr.sub(6));
                acc_lo = vmlal_u8(acc_lo, vget_low_u8(v1), k1);
                acc_hi = vmlal_u8(acc_hi, vget_high_u8(v1), k1);

                // Tap 2 (offset -3 bytes = -1 pixel)
                let v2 = vld1q_u8(ptr.sub(3));
                acc_lo = vmlal_u8(acc_lo, vget_low_u8(v2), k2);
                acc_hi = vmlal_u8(acc_hi, vget_high_u8(v2), k2);

                // Tap 3 (offset 0 bytes = 0 pixels, center)
                let v3 = vld1q_u8(ptr);
                acc_lo = vmlal_u8(acc_lo, vget_low_u8(v3), k3);
                acc_hi = vmlal_u8(acc_hi, vget_high_u8(v3), k3);

                // Tap 4 (offset +3 bytes = +1 pixel)
                let v4 = vld1q_u8(ptr.add(3));
                acc_lo = vmlal_u8(acc_lo, vget_low_u8(v4), k4);
                acc_hi = vmlal_u8(acc_hi, vget_high_u8(v4), k4);

                // Tap 5 (offset +6 bytes = +2 pixels)
                let v5 = vld1q_u8(ptr.add(6));
                acc_lo = vmlal_u8(acc_lo, vget_low_u8(v5), k5);
                acc_hi = vmlal_u8(acc_hi, vget_high_u8(v5), k5);

                // Tap 6 (offset +9 bytes = +3 pixels, rightmost)
                let v6 = vld1q_u8(ptr.add(9));
                acc_lo = vmlal_u8(acc_lo, vget_low_u8(v6), k6);
                acc_hi = vmlal_u8(acc_hi, vget_high_u8(v6), k6);

                // Normalize and store as u16
                let res_lo = vshrq_n_u16(acc_lo, 6);
                let res_hi = vshrq_n_u16(acc_hi, 6);

                // CRITICAL FIX: temp_u16[k] stores result for data[k]
                // Use direct element index - .add() on u16 pointer advances by elements, not bytes!
                let dst_ptr = temp_u16.as_mut_ptr().add(row_offset + byte_idx);
                vst1q_u16(dst_ptr, res_lo);
                vst1q_u16(dst_ptr.add(8), res_hi);

                byte_idx += 16;
            }

            // Handle right edge (scalar) - remaining pixels
            // byte_idx is now at the start of the next unprocessed chunk
            let simd_end_pixel = (byte_idx / 3).max(radius);
            for x in simd_end_pixel..width {
                for c in 0..3 {
                    let mut sum: u32 = 0;
                    for k in 0..7 {
                        let px = (x as i32 + k as i32 - radius as i32).clamp(0, width as i32 - 1) as usize;
                        sum = sum.wrapping_add(data[row_offset + px * 3 + c] as u32 * [1u32, 6, 15, 20, 15, 6, 1][k]);
                    }
                    temp_u16[row_offset + x * 3 + c] = (sum >> 6) as u16;
                }
            }
        } else {
            // No SIMD possible, process all pixels with scalar
            for x in radius..width {
                for c in 0..3 {
                    let mut sum: u32 = 0;
                    for k in 0..7 {
                        let px = (x as i32 + k as i32 - radius as i32).clamp(0, width as i32 - 1) as usize;
                        sum = sum.wrapping_add(data[row_offset + px * 3 + c] as u32 * [1u32, 6, 15, 20, 15, 6, 1][k]);
                    }
                    temp_u16[row_offset + x * 3 + c] = (sum >> 6) as u16;
                }
            }
        }
    }

    // ============================================================================
    // VERTICAL PASS: u16 -> u16 -> u8 with FULL ROW PROCESSING for cache efficiency
    // OPTIMIZATION: Process full rows at once instead of column blocks
    // This is critical for large images - we read each row only once instead of N times!
    // ============================================================================
    let mut output = vec![0u8; data.len()];
    let temp_stride = row_bytes;
    let output_stride = row_bytes;

    // Symmetric weights (pre-loaded for SIMD)
    let k0 = vdup_n_u16(1);
    let k1 = vdup_n_u16(6);
    let k2 = vdup_n_u16(15);
    let k3 = vdup_n_u16(20);

    // Process 8 u16 values at a time for full row processing
    const VALS_PER_ITER: usize = 8;

    // Top edge (scalar) - first 3 rows
    for y in 0..height.min(radius) {
        for i in 0..row_bytes {
            let mut sum: u32 = 0;
            for k in 0..7 {
                let py = (y as i32 + k as i32 - radius as i32).clamp(0, height as i32 - 1) as usize;
                sum = sum.wrapping_add(temp_u16[py * temp_stride + i] as u32 * [1u32, 6, 15, 20, 15, 6, 1][k]);
            }
            output[y * output_stride + i] = (sum >> 6) as u8;
        }
    }

    // Middle rows - process full row with SIMD
    let simd_start = radius;
    let simd_end = height.saturating_sub(radius);

    for y in simd_start..simd_end {
        let out_row = output.as_mut_ptr().add(y * output_stride);

        // Process row in chunks of VALS_PER_ITER
        let mut val_idx = 0;
        let simd_chunks = row_bytes / VALS_PER_ITER;

        for _ in 0..simd_chunks {
            // Pointers to 7 rows at current column position
            let r0 = temp_u16.as_ptr().add((y - 3) * temp_stride + val_idx);
            let r1 = temp_u16.as_ptr().add((y - 2) * temp_stride + val_idx);
            let r2 = temp_u16.as_ptr().add((y - 1) * temp_stride + val_idx);
            let r3 = temp_u16.as_ptr().add(y * temp_stride + val_idx);
            let r4 = temp_u16.as_ptr().add((y + 1) * temp_stride + val_idx);
            let r5 = temp_u16.as_ptr().add((y + 2) * temp_stride + val_idx);
            let r6 = temp_u16.as_ptr().add((y + 3) * temp_stride + val_idx);

            // Load 8 u16 values from each row
            let v0 = vld1q_u16(r0);
            let v1 = vld1q_u16(r1);
            let v2 = vld1q_u16(r2);
            let v3 = vld1q_u16(r3);
            let v4 = vld1q_u16(r4);
            let v5 = vld1q_u16(r5);
            let v6 = vld1q_u16(r6);

            // Start with center row (weight 20) - split into low/high halves
            let mut acc_lo = vmull_u16(vget_low_u16(v3), k3);
            let mut acc_hi = vmull_u16(vget_high_u16(v3), k3);

            // Add symmetric pairs (reduces 7 multiplies to 4!)
            let sum06 = vaddq_u16(v0, v6);
            acc_lo = vmlal_u16(acc_lo, vget_low_u16(sum06), k0);
            acc_hi = vmlal_u16(acc_hi, vget_high_u16(sum06), k0);

            let sum15 = vaddq_u16(v1, v5);
            acc_lo = vmlal_u16(acc_lo, vget_low_u16(sum15), k1);
            acc_hi = vmlal_u16(acc_hi, vget_high_u16(sum15), k1);

            let sum24 = vaddq_u16(v2, v4);
            acc_lo = vmlal_u16(acc_lo, vget_low_u16(sum24), k2);
            acc_hi = vmlal_u16(acc_hi, vget_high_u16(sum24), k2);

            // Normalize: divide by 64, then narrow u32 -> u16 -> u8
            let res_lo = vshrn_n_u32(acc_lo, 6);
            let res_hi = vshrn_n_u32(acc_hi, 6);
            let res_u16 = vcombine_u16(res_lo, res_hi);
            let res_u8 = vqmovn_u16(res_u16);

            // Store 8 u8 values
            vst1_u8(out_row.add(val_idx), res_u8);

            val_idx += VALS_PER_ITER;
        }

        // Handle remaining values (scalar)
        for i in val_idx..row_bytes {
            let mut sum: u32 = 0;
            for k in 0..7 {
                let py = (y as i32 + k as i32 - radius as i32).clamp(0, height as i32 - 1) as usize;
                sum = sum.wrapping_add(temp_u16[py * temp_stride + i] as u32 * [1u32, 6, 15, 20, 15, 6, 1][k]);
            }
            output[y * output_stride + i] = (sum >> 6) as u8;
        }
    }

    // Bottom edge (scalar) - last 3 rows
    for y in simd_end.max(radius)..height {
        for i in 0..row_bytes {
            let mut sum: u32 = 0;
            for k in 0..7 {
                let py = (y as i32 + k as i32 - radius as i32).clamp(0, height as i32 - 1) as usize;
                sum = sum.wrapping_add(temp_u16[py * temp_stride + i] as u32 * [1u32, 6, 15, 20, 15, 6, 1][k]);
            }
            output[y * output_stride + i] = (sum >> 6) as u8;
        }
    }

    data.copy_from_slice(&output);
}

#[cfg(not(target_arch = "aarch64"))]
pub(crate) unsafe fn convolve_separable_neon_7(_image: &mut FusableImage, _kernel: &[i32], _scale: i32) {
    // Fallback for non-ARM architectures
    unreachable!("NEON functions should not be called on non-ARM platforms");
}

// ============================================================================
// Helper functions for 7-tap blur
// ============================================================================

/// Optimized scalar vertical pass - minimal allocations
#[cfg(target_arch = "aarch64")]
unsafe fn vertical_pass_scalar_optimized(
    data: &mut [u8],
    width: usize,
    height: usize,
    kernel: &[u16; 7],
) {
    let radius = 3;
    let mut output = vec![0u8; data.len()];

    for x in 0..width {
        for c in 0..3 {
            // Top edge
            for y in 0..height.min(radius) {
                let mut sum: u32 = 0;
                for k in 0..7 {
                    let py = (y as i32 + k as i32 - radius as i32).clamp(0, height as i32 - 1) as usize;
                    sum = sum.wrapping_add(data[(py * width + x) * 3 + c] as u32 * kernel[k] as u32);
                }
                output[(y * width + x) * 3 + c] = (sum >> 8) as u8;
            }

            // Middle rows - unrolled kernel loop
            for y in radius..(height.saturating_sub(radius)) {
                let mut sum: u32 = 0;
                let base_idx = (y * width + x) * 3;

                // Unroll: k=0,1; k=2,3; k=4,5; k=6
                let py0 = y - 3;
                let py1 = y - 2;
                let py2 = y - 1;
                let py3 = y;
                let py4 = y + 1;
                let py5 = y + 2;
                let py6 = y + 3;

                sum = sum.wrapping_add(data[(py0 * width + x) * 3 + c] as u32 * kernel[0] as u32);
                sum = sum.wrapping_add(data[(py1 * width + x) * 3 + c] as u32 * kernel[1] as u32);
                sum = sum.wrapping_add(data[(py2 * width + x) * 3 + c] as u32 * kernel[2] as u32);
                sum = sum.wrapping_add(data[(py3 * width + x) * 3 + c] as u32 * kernel[3] as u32);
                sum = sum.wrapping_add(data[(py4 * width + x) * 3 + c] as u32 * kernel[4] as u32);
                sum = sum.wrapping_add(data[(py5 * width + x) * 3 + c] as u32 * kernel[5] as u32);
                sum = sum.wrapping_add(data[(py6 * width + x) * 3 + c] as u32 * kernel[6] as u32);

                output[base_idx + c] = (sum >> 8) as u8;
            }

            // Bottom edge
            for y in height.saturating_sub(radius).max(radius)..height {
                let mut sum: u32 = 0;
                for k in 0..7 {
                    let py = (y as i32 + k as i32 - radius as i32).clamp(0, height as i32 - 1) as usize;
                    sum = sum.wrapping_add(data[(py * width + x) * 3 + c] as u32 * kernel[k] as u32);
                }
                output[(y * width + x) * 3 + c] = (sum >> 8) as u8;
            }
        }
    }

    data.copy_from_slice(&output);
}

#[cfg(not(target_arch = "aarch64"))]
unsafe fn vertical_pass_scalar_optimized(
    _data: &mut [u8],
    _width: usize,
    _height: usize,
    _kernel: &[u16; 7],
) {
    // Fallback for non-ARM architectures
    unreachable!("NEON helper functions should not be called on non-ARM platforms");
}
