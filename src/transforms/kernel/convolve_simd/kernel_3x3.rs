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

    // ============================================================================
    // FUSED rolling separable [1 2 1] x [1 2 1]:
    //   horizontal -> 3-row planar buffer, vertical emitted one row behind,
    //   so `data` is read once and written once (halves memory traffic vs the
    //   two-pass version, which is the bottleneck for this memory-bound op).
    // ============================================================================
    let mut output = vec![0u8; data.len()];
    let plane = width; // bytes per channel plane per row
    let slot_bytes = 3 * plane;
    let row_bytes = width * 3;
    let mut h_buf = vec![0u8; 3 * slot_bytes];

    unsafe fn horizontal_planar(
        data: &[u8],
        sy: usize,
        width: usize,
        h_buf: &mut [u8],
        slot: usize,
        plane: usize,
        slot_bytes: usize,
    ) {
        let row_offset = sy * width * 3;
        let base = slot * slot_bytes;
        const TILE: usize = 16;

        // Left edge (x = 0)
        for c in 0..3 {
            let mut sum: u32 = 0;
            for k in 0..3 {
                let px = (k as i32 - 1).clamp(0, width as i32 - 1) as usize;
                sum += data[row_offset + px * 3 + c] as u32 * [1, 2, 1][k];
            }
            h_buf[base + c * plane] = (sum >> 2) as u8;
        }

        // SIMD middle
        let simd_start = 1usize;
        let simd_end = width.saturating_sub(1);
        let chunks = if simd_end > simd_start {
            (simd_end - simd_start) / TILE
        } else {
            0
        };
        for chunk in 0..chunks {
            let bx = simd_start + chunk * TILE;
            let p0 = vld3q_u8(data.as_ptr().add(row_offset + (bx - 1) * 3) as *const u8);
            let p1 = vld3q_u8(data.as_ptr().add(row_offset + bx * 3) as *const u8);
            let p2 = vld3q_u8(data.as_ptr().add(row_offset + (bx + 1) * 3) as *const u8);

            let r = vcombine_u8(blur3_u8_lo(p0.0, p1.0, p2.0), blur3_u8_hi(p0.0, p1.0, p2.0));
            let g = vcombine_u8(blur3_u8_lo(p0.1, p1.1, p2.1), blur3_u8_hi(p0.1, p1.1, p2.1));
            let b = vcombine_u8(blur3_u8_lo(p0.2, p1.2, p2.2), blur3_u8_hi(p0.2, p1.2, p2.2));

            vst1q_u8(h_buf.as_mut_ptr().add(base + bx), r);
            vst1q_u8(h_buf.as_mut_ptr().add(base + plane + bx), g);
            vst1q_u8(h_buf.as_mut_ptr().add(base + 2 * plane + bx), b);
        }

        // Remainder + right edge (scalar, clamped)
        for x in (simd_start + chunks * TILE)..width {
            for c in 0..3 {
                let mut sum: u32 = 0;
                for k in 0..3 {
                    let px = (x as i32 + k as i32 - 1).clamp(0, width as i32 - 1) as usize;
                    sum += data[row_offset + px * 3 + c] as u32 * [1, 2, 1][k];
                }
                h_buf[base + c * plane + x] = (sum >> 2) as u8;
            }
        }
    }

    unsafe fn vertical_emit(
        h_buf: &[u8],
        s0: usize,
        s1: usize,
        s2: usize,
        width: usize,
        output: &mut [u8],
        oy: usize,
        row_bytes: usize,
        plane: usize,
        slot_bytes: usize,
    ) {
        let out_row = oy * row_bytes;
        let b0 = s0 * slot_bytes;
        let b1 = s1 * slot_bytes;
        let b2 = s2 * slot_bytes;
        const TILE: usize = 16;

        // Left edge (x = 0)
        for c in 0..3 {
            let mut sum: u32 = 0;
            sum += h_buf[b0 + c * plane] as u32 * 1;
            sum += h_buf[b1 + c * plane] as u32 * 2;
            sum += h_buf[b2 + c * plane] as u32 * 1;
            output[out_row + c] = (sum >> 2) as u8;
        }

        // SIMD middle
        let simd_start = 1usize;
        let simd_end = width.saturating_sub(1);
        let chunks = if simd_end > simd_start {
            (simd_end - simd_start) / TILE
        } else {
            0
        };
        for chunk in 0..chunks {
            let bx = simd_start + chunk * TILE;
            let mut rgb = [vdupq_n_u8(0); 3];
            for c in 0..3 {
                let v0 = vld1q_u8(h_buf.as_ptr().add(b0 + c * plane + bx));
                let v1 = vld1q_u8(h_buf.as_ptr().add(b1 + c * plane + bx));
                let v2 = vld1q_u8(h_buf.as_ptr().add(b2 + c * plane + bx));
                rgb[c] = vcombine_u8(
                    blur3_scalar_to_u8(vget_low_u8(v0), vget_low_u8(v1), vget_low_u8(v2)),
                    blur3_scalar_to_u8(vget_high_u8(v0), vget_high_u8(v1), vget_high_u8(v2)),
                );
            }
            vst3q_u8(output.as_mut_ptr().add(out_row + bx * 3), uint8x16x3_t(rgb[0], rgb[1], rgb[2]));
        }

        // Remainder + right edge (scalar)
        for x in (simd_start + chunks * TILE)..width {
            for c in 0..3 {
                let mut sum: u32 = 0;
                sum += h_buf[b0 + c * plane + x] as u32 * 1;
                sum += h_buf[b1 + c * plane + x] as u32 * 2;
                sum += h_buf[b2 + c * plane + x] as u32 * 1;
                output[out_row + x * 3 + c] = (sum >> 2) as u8;
            }
        }
    }

    // Process virtual rows 0..=height; h[row] for the clamped source row, then
    // emit the vertical output for row y-1 once h[y-2..y] are available.
    for y in 0..=height {
        let sy = y.min(height - 1);
        let slot = y % 3;
        horizontal_planar(data, sy, width, &mut h_buf, slot, plane, slot_bytes);
        if y == 1 {
            // Top border row 0: clamped window rows 0,0,1 (h[0] in slot 0,
            // h[1] in slot 1).
            vertical_emit(
                &h_buf,
                0,
                0,
                1,
                width,
                &mut output,
                0,
                row_bytes,
                plane,
                slot_bytes,
            );
        }
        if y >= 2 {
            let oy = y - 1;
            vertical_emit(
                &h_buf,
                (y - 2) % 3,
                (y - 1) % 3,
                y % 3,
                width,
                &mut output,
                oy,
                row_bytes,
                plane,
                slot_bytes,
            );
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

#[inline(always)]
unsafe fn blur3_gray_u8(p0: uint8x16_t, p1: uint8x16_t, p2: uint8x16_t) -> uint8x16_t {
    let ends_lo = vaddl_u8(vget_low_u8(p0), vget_low_u8(p2));
    let tot_lo = vmlal_u8(ends_lo, vget_low_u8(p1), vdup_n_u8(2));
    let res_lo = vshrn_n_u16(tot_lo, 2);

    let ends_hi = vaddl_u8(vget_high_u8(p0), vget_high_u8(p2));
    let tot_hi = vmlal_u8(ends_hi, vget_high_u8(p1), vdup_n_u8(2));
    let res_hi = vshrn_n_u16(tot_hi, 2);

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
        *out_ptr = ((left_val + mid_val * 2 + right_val) >> 2) as u8;

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
            *out_ptr.add(rem_x) = ((p0 + p1 * 2 + p2) >> 2) as u8;
        }

        // Right edge
        if width > 1 {
            let rx = width - 1;
            let p0 = *in_ptr.add(rx - 1) as u32;
            let p1 = *in_ptr.add(rx) as u32;
            let p2 = *in_ptr.add(rx) as u32;
            *out_ptr.add(rx) = ((p0 + p1 * 2 + p2) >> 2) as u8;
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
            *out_ptr.add(x) = ((p0 + p1 * 2 + p2) >> 2) as u8;
        }
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
mod fused_tests {
    use super::*;
    use crate::core::FusableImage;

    fn scalar_two_pass(data: &[u8], width: usize, height: usize) -> Vec<u8> {
        let ch = 3usize;
        let mut h = vec![0u8; data.len()];
        // horizontal [1 2 1] >> 2 (truncate)
        for y in 0..height {
            for x in 0..width {
                for c in 0..ch {
                    let mut sum: u32 = 0;
                    for k in 0..3 {
                        let px = (x as i32 + k as i32 - 1).clamp(0, width as i32 - 1) as usize;
                        sum += data[(y * width + px) * ch + c] as u32 * [1, 2, 1][k];
                    }
                    h[(y * width + x) * ch + c] = (sum >> 2) as u8;
                }
            }
        }
        let mut out = vec![0u8; data.len()];
        for y in 0..height {
            for x in 0..width {
                for c in 0..ch {
                    let mut sum: u32 = 0;
                    for k in 0..3 {
                        let py = (y as i32 + k as i32 - 1).clamp(0, height as i32 - 1) as usize;
                        sum += h[(py * width + x) * ch + c] as u32 * [1, 2, 1][k];
                    }
                    out[(y * width + x) * ch + c] = (sum >> 2) as u8;
                }
            }
        }
        out
    }

    /// Scalar two-pass [1 2 1] / 4 reference for 1-channel images, truncating each pass
    /// (the library's canonical Gaussian convention; the NEON gray path must match).
    fn scalar_two_pass_gray(data: &[u8], width: usize, height: usize) -> Vec<u8> {
        let mut h = vec![0u8; data.len()];
        for y in 0..height {
            for x in 0..width {
                let mut sum: u32 = 0;
                for k in 0..3 {
                    let px = (x as i32 + k as i32 - 1).clamp(0, width as i32 - 1) as usize;
                    sum += data[y * width + px] as u32 * [1, 2, 1][k];
                }
                h[y * width + x] = (sum >> 2) as u8;
            }
        }
        let mut out = vec![0u8; data.len()];
        for y in 0..height {
            for x in 0..width {
                let mut sum: u32 = 0;
                for k in 0..3 {
                    let py = (y as i32 + k as i32 - 1).clamp(0, height as i32 - 1) as usize;
                    sum += h[py * width + x] as u32 * [1, 2, 1][k];
                }
                out[y * width + x] = (sum >> 2) as u8;
            }
        }
        out
    }

    #[test]
    fn test_fused_3x3_matches_scalar() {
        let (w, h) = (32usize, 32usize);
        let mut data: Vec<u8> = (0..w * h * 3)
            .map(|i| ((i as u64 * 2654435761) % 256) as u8)
            .collect();
        let expected = scalar_two_pass(&data, w, h);

        let mut img = FusableImage::new(&mut data, w, h, 3);
        unsafe {
            convolve_separable_neon_3(&mut img, &[1, 2, 1], 4);
        }

        let mut mismatches = 0usize;
        let mut max_diff = 0i32;
        for i in 0..data.len() {
            let diff = (data[i] as i32 - expected[i] as i32).abs();
            if diff > 0 {
                mismatches += 1;
                max_diff = max_diff.max(diff);
                if mismatches <= 8 {
                    let px = i / 3;
                    eprintln!(
                        "  idx={} (x={}, y={}, c={}): fused={} expected={}",
                        i,
                        px % w,
                        px / w,
                        i % 3,
                        data[i],
                        expected[i]
                    );
                }
            }
        }
        assert_eq!(
            mismatches,
            0,
            "fused 3x3 mismatch: {} mismatches, max_diff={}",
            mismatches,
            max_diff
        );
    }

    #[test]
    fn test_fused_3x3_gray_matches_scalar_truncation() {
        // Non-multiple of 16 so SIMD tiles, scalar remainder, and both edges are all exercised.
        let (w, h) = (33usize, 17usize);
        let mut data: Vec<u8> = (0..w * h)
            .map(|i| ((i as u64 * 40503) % 256) as u8)
            .collect();
        let expected = scalar_two_pass_gray(&data, w, h);

        let mut img = FusableImage::new(&mut data, w, h, 1);
        unsafe {
            convolve_separable_neon_3(&mut img, &[1, 2, 1], 4);
        }

        let mut mismatches = 0usize;
        let mut max_diff = 0i32;
        for i in 0..data.len() {
            let diff = (data[i] as i32 - expected[i] as i32).abs();
            if diff > 0 {
                mismatches += 1;
                max_diff = max_diff.max(diff);
                if mismatches <= 8 {
                    eprintln!(
                        "  idx={} (x={}, y={}): gray={} expected={}",
                        i,
                        i % w,
                        i / w,
                        data[i],
                        expected[i]
                    );
                }
            }
        }
        assert_eq!(
            mismatches,
            0,
            "fused 3x3 gray mismatch: {} mismatches, max_diff={}",
            mismatches,
            max_diff
        );
    }
}
