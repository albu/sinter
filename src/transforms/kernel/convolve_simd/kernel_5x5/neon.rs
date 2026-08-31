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

    if channels == 1 {
        convolve_separable_gray_neon_5(image);
        return;
    }

    if channels != 3 {
        // Fallback to scalar for non-RGB
        super::super::super::convolve::convolve_1d_horizontal(image, &[1, 4, 6, 4, 1], 16);
        super::super::super::convolve::convolve_1d_vertical(image, &[1, 4, 6, 4, 1], 16);
        return;
    }

    // ============================================================================
    // FUSED rolling separable [1 4 6 4 1] x [1 4 6 4 1], interleaved:
    //   For horizontal [1 4 6 4 1] on interleaved RGB, byte i's horizontal
    //   neighbours are bytes i-6 / i-3 / i / i+3 / i+6 — always the same
    //   channel — so both passes run on plain contiguous byte vectors
    //   (vld1q/vst1q, no vld3/vst3 de-interleaving).
    //   horizontal -> 5-row ring buffer, vertical emitted two rows behind and
    //   written straight back into `data`, so `data` is read once and written
    //   once with no second full-size buffer.
    // ============================================================================
    const TAPS: usize = 5;
    const RADIUS: usize = 2;
    const COEFS: [u8; TAPS] = [1, 4, 6, 4, 1];
    let rb = RADIUS * 3; // bytes to the first/last same-channel tap
    let row_bytes = width * 3;
    // Zeroed ring buffer: each slot is refilled every TAPS iterations, so any
    // read touches a value written at most TAPS-1 iterations earlier.
    let mut ring = vec![0u8; TAPS * row_bytes];

    unsafe fn horizontal(
        data: &[u8],
        sy: usize,
        row_bytes: usize,
        ring: &mut [u8],
        slot: usize,
    ) {
        let row_off = sy * row_bytes;
        let dst = slot * row_bytes;
        const RADIUS: usize = 2;
        const COEFS: [u32; 5] = [1, 4, 6, 4, 1];
        let rb = RADIUS * 3;

        // Left border: pixel-clamped taps. For very narrow rows
        // (row_bytes <= 2*rb) the border covers the whole row.
        for i in 0..rb.min(row_bytes) {
            let px = i / 3;
            let c = i % 3;
            let mut sum: u32 = 0;
            for k in 0..5 {
                let spx = (px as i32 + k as i32 - RADIUS as i32).clamp(0, row_bytes as i32) as usize;
                let spx = spx.min((row_bytes / 3) - 1);
                sum += data[row_off + spx * 3 + c] as u32 * COEFS[k];
            }
            ring[dst + i] = (sum >> 4) as u8;
        }

        // SIMD middle: output bytes [rb, row_bytes-rb), 16 at a time.
        let chunks = if row_bytes > 2 * rb {
            (row_bytes - 2 * rb) / 16
        } else {
            0
        };
        for k in 0..chunks {
            let bx = rb + k * 16;
            let mut taps: [uint8x16_t; 5] = [vdupq_n_u8(0); 5];
            for (t, tap) in taps.iter_mut().enumerate() {
                let off = (t * 3) as isize - rb as isize; // -6 / -3 / 0 / +3 / +6
                let ptr = (row_off as isize + bx as isize + off) as usize;
                *tap = vld1q_u8(data.as_ptr().add(ptr));
            }
            let r = blur5_bytes(&taps);
            vst1q_u8(ring.as_mut_ptr().add(dst + bx), r);
        }

        // Right border + remainder: pixel-clamped taps
        let right_start = (rb + chunks * 16).min(row_bytes);
        for x in right_start..row_bytes {
            let px = x / 3;
            let c = x % 3;
            let w = row_bytes / 3;
            let mut sum: u32 = 0;
            for k in 0..5 {
                let spx = (px as i32 + k as i32 - RADIUS as i32).clamp(0, w as i32 - 1) as usize;
                sum += data[row_off + spx * 3 + c] as u32 * COEFS[k];
            }
            ring[dst + x] = (sum >> 4) as u8;
        }
    }

    unsafe fn vertical(
        ring: &[u8],
        slots: &[usize; 5],
        row_bytes: usize,
        data: &mut [u8],
        oy: usize,
    ) {
        const COEFS: [u8; 5] = [1, 4, 6, 4, 1];
        let out_row = oy * row_bytes;
        let bases: [usize; 5] = [
            slots[0] * row_bytes,
            slots[1] * row_bytes,
            slots[2] * row_bytes,
            slots[3] * row_bytes,
            slots[4] * row_bytes,
        ];

        // x = 0..3 (first pixel)
        for i in 0..3 {
            let mut sum: u32 = 0;
            for k in 0..5 {
                sum += ring[bases[k] + i] as u32 * COEFS[k] as u32;
            }
            data[out_row + i] = (sum >> 4) as u8;
        }

        let chunks = if row_bytes > 6 { (row_bytes - 6) / 16 } else { 0 };
        for k in 0..chunks {
            let bx = 3 + k * 16;
            let v0 = vld1q_u8(ring.as_ptr().add(bases[0] + bx));
            let mut acc_lo = vmull_u8(vget_low_u8(v0), vdup_n_u8(COEFS[0]));
            let mut acc_hi = vmull_u8(vget_high_u8(v0), vdup_n_u8(COEFS[0]));
            for t in 1..5 {
                let v = vld1q_u8(ring.as_ptr().add(bases[t] + bx));
                acc_lo = vmlal_u8(acc_lo, vget_low_u8(v), vdup_n_u8(COEFS[t]));
                acc_hi = vmlal_u8(acc_hi, vget_high_u8(v), vdup_n_u8(COEFS[t]));
            }
            let r = vcombine_u8(vshrn_n_u16(acc_lo, 4), vshrn_n_u16(acc_hi, 4));
            vst1q_u8(data.as_mut_ptr().add(out_row + bx), r);
        }

        for x in (3 + chunks * 16)..row_bytes {
            let mut sum: u32 = 0;
            for k in 0..5 {
                sum += ring[bases[k] + x] as u32 * COEFS[k] as u32;
            }
            data[out_row + x] = (sum >> 4) as u8;
        }
    }

    // Iterate RADIUS rows past the end: H keeps writing the clamped bottom row
    // into consecutive slots so the final vertical window can read duplicated
    // rows from distinct slots. Emit at y covers output row y - RADIUS;
    // overwrite safety holds because the oldest slot read was written exactly
    // TAPS-1 iterations earlier.
    let clamp_row = |r: i64| r.clamp(0, height as i64 - 1) as usize;
    for y in 0..=(height - 1 + RADIUS) {
        let sy = y.min(height - 1);
        horizontal(data, sy, row_bytes, &mut ring, y % TAPS);
        if y >= RADIUS {
            let oy = y - RADIUS;
            let slots: [usize; 5] = [
                clamp_row(oy as i64 - 2) % TAPS,
                clamp_row(oy as i64 - 1) % TAPS,
                clamp_row(oy as i64) % TAPS,
                clamp_row(oy as i64 + 1) % TAPS,
                clamp_row(oy as i64 + 2) % TAPS,
            ];
            vertical(&ring, &slots, row_bytes, data, oy);
        }
    }
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

/// [1 4 6 4 1] >> 4 (truncating) on 16 interleaved bytes; horizontal taps
/// sit at +-6/+-3/0 bytes, so one lane vector holds 16 single-channel taps.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn blur5_bytes(t: &[uint8x16_t; 5]) -> uint8x16_t {
    let c1 = vdup_n_u8(1);
    let c4 = vdup_n_u8(4);
    let c6 = vdup_n_u8(6);
    let mut lo = vmull_u8(vget_low_u8(t[0]), c1);
    lo = vmlal_u8(lo, vget_low_u8(t[1]), c4);
    lo = vmlal_u8(lo, vget_low_u8(t[2]), c6);
    lo = vmlal_u8(lo, vget_low_u8(t[3]), c4);
    lo = vmlal_u8(lo, vget_low_u8(t[4]), c1);
    let mut hi = vmull_u8(vget_high_u8(t[0]), c1);
    hi = vmlal_u8(hi, vget_high_u8(t[1]), c4);
    hi = vmlal_u8(hi, vget_high_u8(t[2]), c6);
    hi = vmlal_u8(hi, vget_high_u8(t[3]), c4);
    hi = vmlal_u8(hi, vget_high_u8(t[4]), c1);
    vcombine_u8(vshrn_n_u16(lo, 4), vshrn_n_u16(hi, 4))
}

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

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn blur5_gray_u8(p_m2: uint8x16_t, p_m1: uint8x16_t, p_0: uint8x16_t, p_p1: uint8x16_t, p_p2: uint8x16_t) -> uint8x16_t {
    let outer_lo = vaddl_u8(vget_low_u8(p_m2), vget_low_u8(p_p2));
    let inner_lo = vaddl_u8(vget_low_u8(p_m1), vget_low_u8(p_p1));
    let mut sum_lo = vmlal_u8(outer_lo, vget_low_u8(p_0), vdup_n_u8(6));
    sum_lo = vaddq_u16(sum_lo, vshlq_n_u16(inner_lo, 2));
    let res_lo = vshrn_n_u16(sum_lo, 4);

    let outer_hi = vaddl_u8(vget_high_u8(p_m2), vget_high_u8(p_p2));
    let inner_hi = vaddl_u8(vget_high_u8(p_m1), vget_high_u8(p_p1));
    let mut sum_hi = vmlal_u8(outer_hi, vget_high_u8(p_0), vdup_n_u8(6));
    sum_hi = vaddq_u16(sum_hi, vshlq_n_u16(inner_hi, 2));
    let res_hi = vshrn_n_u16(sum_hi, 4);

    vcombine_u8(res_lo, res_hi)
}

#[cfg(target_arch = "aarch64")]
unsafe fn convolve_separable_gray_neon_5(image: &mut FusableImage) {
    let width = image.width;
    let height = image.height;
    let data = &mut image.data;
    let total_bytes = data.len();
    let mut temp = Vec::<u8>::with_capacity(total_bytes);
    unsafe { temp.set_len(total_bytes); }
    let radius = 2;
    const TILE: usize = 16;

    // HORIZONTAL PASS
    for y in 0..height {
        let row_offset = y * width;
        let in_ptr = data.as_ptr().add(row_offset);
        let out_ptr = temp.as_mut_ptr().add(row_offset);

        // Left edge (x = 0, 1)
        for x in 0..width.min(radius) {
            let mut sum: u32 = 0;
            for k in 0..5 {
                let px = (x as i32 + k as i32 - radius as i32).clamp(0, width as i32 - 1) as usize;
                sum = sum.wrapping_add(*in_ptr.add(px) as u32 * [1u32, 4, 6, 4, 1][k]);
            }
            *out_ptr.add(x) = (sum >> 4) as u8;
        }

        let simd_start = radius;
        let simd_end = width.saturating_sub(radius);
        let simd_chunks = if simd_end > simd_start { (simd_end - simd_start) / TILE } else { 0 };

        let mut x = simd_start;
        for _ in 0..simd_chunks {
            let p_2 = vld1q_u8(in_ptr.add(x - 2));
            let p_1 = vld1q_u8(in_ptr.add(x - 1));
            let p0  = vld1q_u8(in_ptr.add(x));
            let p1  = vld1q_u8(in_ptr.add(x + 1));
            let p2  = vld1q_u8(in_ptr.add(x + 2));

            let combined = blur5_gray_u8(p_2, p_1, p0, p1, p2);
            vst1q_u8(out_ptr.add(x), combined);
            x += TILE;
        }

        // Remainder
        for rem_x in x..simd_end {
            let mut sum: u32 = 0;
            for k in 0..5 {
                let px = (rem_x as i32 + k as i32 - radius as i32).clamp(0, width as i32 - 1) as usize;
                sum = sum.wrapping_add(*in_ptr.add(px) as u32 * [1u32, 4, 6, 4, 1][k]);
            }
            *out_ptr.add(rem_x) = (sum >> 4) as u8;
        }

        // Right edge
        for rx in simd_end.max(radius)..width {
            let mut sum: u32 = 0;
            for k in 0..5 {
                let px = (rx as i32 + k as i32 - radius as i32).clamp(0, width as i32 - 1) as usize;
                sum = sum.wrapping_add(*in_ptr.add(px) as u32 * [1u32, 4, 6, 4, 1][k]);
            }
            *out_ptr.add(rx) = (sum >> 4) as u8;
        }
    }

    // VERTICAL PASS
    let row_chunks = width / 16;
    for y in 0..height {
        let y_m2 = (y as i32 - 2).clamp(0, height as i32 - 1) as usize;
        let y_m1 = (y as i32 - 1).clamp(0, height as i32 - 1) as usize;
        let y_p1 = (y as i32 + 1).clamp(0, height as i32 - 1) as usize;
        let y_p2 = (y as i32 + 2).clamp(0, height as i32 - 1) as usize;

        let ptr_m2 = temp.as_ptr().add(y_m2 * width);
        let ptr_m1 = temp.as_ptr().add(y_m1 * width);
        let ptr_0  = temp.as_ptr().add(y * width);
        let ptr_p1 = temp.as_ptr().add(y_p1 * width);
        let ptr_p2 = temp.as_ptr().add(y_p2 * width);
        let out_ptr = data.as_mut_ptr().add(y * width);

        for chunk in 0..row_chunks {
            let offset = chunk * 16;
            let r_m2 = vld1q_u8(ptr_m2.add(offset));
            let r_m1 = vld1q_u8(ptr_m1.add(offset));
            let r_0  = vld1q_u8(ptr_0.add(offset));
            let r_p1 = vld1q_u8(ptr_p1.add(offset));
            let r_p2 = vld1q_u8(ptr_p2.add(offset));

            let combined = blur5_gray_u8(r_m2, r_m1, r_0, r_p1, r_p2);
            vst1q_u8(out_ptr.add(offset), combined);
        }

        for x in (row_chunks * 16)..width {
            let sum = *ptr_m2.add(x) as u32
                + *ptr_m1.add(x) as u32 * 4
                + *ptr_0.add(x) as u32 * 6
                + *ptr_p1.add(x) as u32 * 4
                + *ptr_p2.add(x) as u32;
            *out_ptr.add(x) = (sum >> 4) as u8;
        }
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;
    use crate::core::FusableImage;

    /// Scalar two-pass separable reference for 3-channel images, truncating
    /// each pass (the library's canonical Gaussian convention).
    fn scalar_two_pass_rgb(data: &[u8], width: usize, height: usize) -> Vec<u8> {
        let coefs = [1u32, 4, 6, 4, 1];
        let ch = 3usize;
        let radius = 2usize;
        let mut h = vec![0u8; data.len()];
        for y in 0..height {
            for x in 0..width {
                for c in 0..ch {
                    let mut sum: u32 = 0;
                    for (kk, kv) in coefs.iter().enumerate() {
                        let px = (x as i32 + kk as i32 - radius as i32).clamp(0, width as i32 - 1)
                            as usize;
                        sum += data[(y * width + px) * ch + c] as u32 * kv;
                    }
                    h[(y * width + x) * ch + c] = (sum >> 4) as u8;
                }
            }
        }
        let mut out = vec![0u8; data.len()];
        for y in 0..height {
            for x in 0..width {
                for c in 0..ch {
                    let mut sum: u32 = 0;
                    for (kk, kv) in coefs.iter().enumerate() {
                        let py = (y as i32 + kk as i32 - radius as i32).clamp(0, height as i32 - 1)
                            as usize;
                        sum += h[(py * width + x) * ch + c] as u32 * kv;
                    }
                    out[(y * width + x) * ch + c] = (sum >> 4) as u8;
                }
            }
        }
        out
    }

    #[test]
    fn test_fused_interleaved_5x5_rgb_edge_sizes() {
        // Border-heavy and tiny sizes: exercises left/right border clamping,
        // the SIMD remainder, and the rolling emit order for small heights.
        for &(w, h) in &[(1usize, 1usize), (2, 2), (3, 3), (5, 4), (7, 5), (16, 3), (33, 17), (32, 32)] {
            let data: Vec<u8> = (0..w * h * 3)
                .map(|i| ((i as u64 * 2654435761) % 256) as u8)
                .collect();
            let expected = scalar_two_pass_rgb(&data, w, h);
            let mut d = data.clone();
            let mut img = FusableImage::new(&mut d, w, h, 3);
            unsafe {
                convolve_separable_neon_5(&mut img, &[1, 4, 6, 4, 1], 16);
            }
            let mismatches = d
                .iter()
                .zip(expected.iter())
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(mismatches, 0, "{w}x{h}: {mismatches} wrong bytes");
        }
    }


    /// Scalar two-pass separable reference for 1-channel images, truncating each pass
    /// (the library's canonical Gaussian convention; the NEON gray path must match).
    fn scalar_two_pass_gray(
        data: &[u8],
        width: usize,
        height: usize,
        kernel: &[u32],
        scale: u32,
    ) -> Vec<u8> {
        let r = kernel.len() / 2;
        let mut h = vec![0u8; data.len()];
        for y in 0..height {
            for x in 0..width {
                let mut sum: u32 = 0;
                for (kk, kv) in kernel.iter().enumerate() {
                    let px = (x as i32 + kk as i32 - r as i32).clamp(0, width as i32 - 1) as usize;
                    sum += data[y * width + px] as u32 * kv;
                }
                h[y * width + x] = (sum / scale) as u8;
            }
        }
        let mut out = vec![0u8; data.len()];
        for y in 0..height {
            for x in 0..width {
                let mut sum: u32 = 0;
                for (kk, kv) in kernel.iter().enumerate() {
                    let py = (y as i32 + kk as i32 - r as i32).clamp(0, height as i32 - 1) as usize;
                    sum += h[py * width + x] as u32 * kv;
                }
                out[y * width + x] = (sum / scale) as u8;
            }
        }
        out
    }

    #[test]
    fn test_gray_5x5_matches_scalar_truncation() {
        // Non-multiple of 16 so SIMD tiles, scalar remainder, and both edges are all exercised.
        let (w, h) = (33usize, 17usize);
        let mut data: Vec<u8> = (0..w * h)
            .map(|i| ((i as u64 * 40503) % 256) as u8)
            .collect();
        let expected = scalar_two_pass_gray(&data, w, h, &[1, 4, 6, 4, 1], 16);

        let mut img = FusableImage::new(&mut data, w, h, 1);
        unsafe {
            convolve_separable_neon_5(&mut img, &[1, 4, 6, 4, 1], 16);
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
            "gray 5x5 mismatch: {} mismatches, max_diff={}",
            mismatches,
            max_diff
        );
    }
}
