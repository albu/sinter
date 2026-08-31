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

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;
    use crate::core::FusableImage;

    /// Scalar two-pass separable reference for 3-channel images, truncating
    /// each pass (the library's canonical Gaussian convention).
    fn scalar_two_pass_rgb(data: &[u8], width: usize, height: usize) -> Vec<u8> {
        let coefs = [2u32, 7, 14, 18, 14, 7, 2];
        let ch = 3usize;
        let radius = 3usize;
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
                    h[(y * width + x) * ch + c] = (sum >> 6) as u8;
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
                    out[(y * width + x) * ch + c] = (sum >> 6) as u8;
                }
            }
        }
        out
    }

    #[test]
    fn test_fused_interleaved_7x7_rgb_edge_sizes() {
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
                convolve_separable_neon_7(&mut img, &[2, 7, 14, 18, 14, 7, 2], 64);
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
    fn test_gray_7x7_matches_scalar_truncation() {
        // Non-multiple of 16 so SIMD tiles, scalar remainder, and both edges are all exercised.
        let (w, h) = (33usize, 17usize);
        let mut data: Vec<u8> = (0..w * h)
            .map(|i| ((i as u64 * 40503) % 256) as u8)
            .collect();
        let expected = scalar_two_pass_gray(&data, w, h, &[2, 7, 14, 18, 14, 7, 2], 64);

        let mut img = FusableImage::new(&mut data, w, h, 1);
        unsafe {
            convolve_separable_neon_7(&mut img, &[2, 7, 14, 18, 14, 7, 2], 64);
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
            "gray 7x7 mismatch: {} mismatches, max_diff={}",
            mismatches,
            max_diff
        );
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

    if channels == 1 {
        convolve_separable_gray_neon_7(image);
        return;
    }

    if channels != 3 {
        // Fallback to scalar for non-RGB
        super::super::super::convolve::convolve_1d_horizontal(image, &[2, 7, 14, 18, 14, 7, 2], 64);
        super::super::super::convolve::convolve_1d_vertical(image, &[2, 7, 14, 18, 14, 7, 2], 64);
        return;
    }

    // ============================================================================
    // FUSED rolling separable [2 7 14 18 14 7 2] x [2 7 14 18 14 7 2], interleaved:
    //   For horizontal 7-tap on interleaved RGB, byte i's horizontal neighbours
    //   are bytes i-9/-6/-3/0/+3/+6/+9 — always the same channel — so both
    //   passes run on plain contiguous byte vectors (vld1q/vst1q) with u8
    //   accumulates widened per tap. This replaces the previous u16
    //   intermediate (2x memory traffic) with a 7-row ring buffer; the
    //   vertical pass emits three rows behind and writes straight back into
    //   `data`, so `data` is read once and written once.
    // ============================================================================
    const TAPS: usize = 7;
    const RADIUS: usize = 3;
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
        const RADIUS: usize = 3;
        const COEFS: [u32; 7] = [2, 7, 14, 18, 14, 7, 2];
        let rb = RADIUS * 3;

        // Left border: pixel-clamped taps. For very narrow rows
        // (row_bytes <= 2*rb) the border covers the whole row.
        for i in 0..rb.min(row_bytes) {
            let px = i / 3;
            let c = i % 3;
            let mut sum: u32 = 0;
            for k in 0..7 {
                let spx = (px as i32 + k as i32 - RADIUS as i32).clamp(0, row_bytes as i32) as usize;
                let spx = spx.min((row_bytes / 3) - 1);
                sum += data[row_off + spx * 3 + c] as u32 * COEFS[k];
            }
            ring[dst + i] = (sum >> 6) as u8;
        }

        // SIMD middle: output bytes [rb, row_bytes-rb), 16 at a time.
        let chunks = if row_bytes > 2 * rb {
            (row_bytes - 2 * rb) / 16
        } else {
            0
        };
        for k in 0..chunks {
            let bx = rb + k * 16;
            let mut taps: [uint8x16_t; 7] = [vdupq_n_u8(0); 7];
            for (t, tap) in taps.iter_mut().enumerate() {
                let off = (t * 3) as isize - rb as isize; // -9 .. 0 .. +9
                let ptr = (row_off as isize + bx as isize + off) as usize;
                *tap = vld1q_u8(data.as_ptr().add(ptr));
            }
            let r = blur7_bytes(&taps);
            vst1q_u8(ring.as_mut_ptr().add(dst + bx), r);
        }

        // Right border + remainder: pixel-clamped taps
        let right_start = (rb + chunks * 16).min(row_bytes);
        for x in right_start..row_bytes {
            let px = x / 3;
            let c = x % 3;
            let w = row_bytes / 3;
            let mut sum: u32 = 0;
            for k in 0..7 {
                let spx = (px as i32 + k as i32 - RADIUS as i32).clamp(0, w as i32 - 1) as usize;
                sum += data[row_off + spx * 3 + c] as u32 * COEFS[k];
            }
            ring[dst + x] = (sum >> 6) as u8;
        }
    }

    unsafe fn vertical(
        ring: &[u8],
        slots: &[usize; 7],
        row_bytes: usize,
        data: &mut [u8],
        oy: usize,
    ) {
        const COEFS: [u8; 7] = [2, 7, 14, 18, 14, 7, 2];
        let out_row = oy * row_bytes;
        let mut bases = [0usize; 7];
        for (b, s) in bases.iter_mut().zip(slots.iter()) {
            *b = s * row_bytes;
        }

        // x = 0..3 (first pixel)
        for i in 0..3 {
            let mut sum: u32 = 0;
            for k in 0..7 {
                sum += ring[bases[k] + i] as u32 * COEFS[k] as u32;
            }
            data[out_row + i] = (sum >> 6) as u8;
        }

        let chunks = if row_bytes > 6 { (row_bytes - 6) / 16 } else { 0 };
        for k in 0..chunks {
            let bx = 3 + k * 16;
            let v0 = vld1q_u8(ring.as_ptr().add(bases[0] + bx));
            let mut acc_lo = vmull_u8(vget_low_u8(v0), vdup_n_u8(COEFS[0]));
            let mut acc_hi = vmull_u8(vget_high_u8(v0), vdup_n_u8(COEFS[0]));
            for t in 1..7 {
                let v = vld1q_u8(ring.as_ptr().add(bases[t] + bx));
                acc_lo = vmlal_u8(acc_lo, vget_low_u8(v), vdup_n_u8(COEFS[t]));
                acc_hi = vmlal_u8(acc_hi, vget_high_u8(v), vdup_n_u8(COEFS[t]));
            }
            let r = vcombine_u8(vshrn_n_u16(acc_lo, 6), vshrn_n_u16(acc_hi, 6));
            vst1q_u8(data.as_mut_ptr().add(out_row + bx), r);
        }

        for x in (3 + chunks * 16)..row_bytes {
            let mut sum: u32 = 0;
            for k in 0..7 {
                sum += ring[bases[k] + x] as u32 * COEFS[k] as u32;
            }
            data[out_row + x] = (sum >> 6) as u8;
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
            let mut slots = [0usize; 7];
            for (k, s) in slots.iter_mut().enumerate() {
                *s = clamp_row(oy as i64 + k as i64 - RADIUS as i64) % TAPS;
            }
            vertical(&ring, &slots, row_bytes, data, oy);
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub(crate) unsafe fn convolve_separable_neon_7(_image: &mut FusableImage, _kernel: &[i32], _scale: i32) {
    // Fallback for non-ARM architectures
    unreachable!("NEON functions should not be called on non-ARM platforms");
}

// ============================================================================
// Helper functions for 7-tap blur
// ============================================================================

/// [2 7 14 18 14 7 2] >> 6 (truncating) on 16 interleaved bytes; horizontal
/// taps sit at +-9/+-6/+-3/0 bytes, so one lane vector holds 16 single-channel
/// taps. Max sum 64*255 = 16320 fits u16.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn blur7_bytes(t: &[uint8x16_t; 7]) -> uint8x16_t {
    let c2 = vdup_n_u8(2);
    let c7 = vdup_n_u8(7);
    let c14 = vdup_n_u8(14);
    let c18 = vdup_n_u8(18);
    let mut lo = vmull_u8(vget_low_u8(t[0]), c2);
    lo = vmlal_u8(lo, vget_low_u8(t[1]), c7);
    lo = vmlal_u8(lo, vget_low_u8(t[2]), c14);
    lo = vmlal_u8(lo, vget_low_u8(t[3]), c18);
    lo = vmlal_u8(lo, vget_low_u8(t[4]), c14);
    lo = vmlal_u8(lo, vget_low_u8(t[5]), c7);
    lo = vmlal_u8(lo, vget_low_u8(t[6]), c2);
    let mut hi = vmull_u8(vget_high_u8(t[0]), c2);
    hi = vmlal_u8(hi, vget_high_u8(t[1]), c7);
    hi = vmlal_u8(hi, vget_high_u8(t[2]), c14);
    hi = vmlal_u8(hi, vget_high_u8(t[3]), c18);
    hi = vmlal_u8(hi, vget_high_u8(t[4]), c14);
    hi = vmlal_u8(hi, vget_high_u8(t[5]), c7);
    hi = vmlal_u8(hi, vget_high_u8(t[6]), c2);
    vcombine_u8(vshrn_n_u16(lo, 6), vshrn_n_u16(hi, 6))
}

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

#[cfg(target_arch = "aarch64")]
unsafe fn convolve_separable_gray_neon_7(image: &mut FusableImage) {
    let width = image.width;
    let height = image.height;
    let data = &mut image.data;
    let mut temp = vec![0u8; data.len()];
    let radius = 3;
    const TILE: usize = 16;

    let k0 = vdup_n_u8(2);
    let k1 = vdup_n_u8(7);
    let k2 = vdup_n_u8(14);
    let k3 = vdup_n_u8(18);
    let k4 = vdup_n_u8(14);
    let k5 = vdup_n_u8(7);
    let k6 = vdup_n_u8(2);

    // HORIZONTAL PASS
    for y in 0..height {
        let row_offset = y * width;
        let in_ptr = data.as_ptr().add(row_offset);
        let out_ptr = temp.as_mut_ptr().add(row_offset);

        // Left edge (x = 0, 1, 2)
        for x in 0..width.min(radius) {
            let mut sum: u32 = 0;
            for k in 0..7 {
                let px = (x as i32 + k as i32 - radius as i32).clamp(0, width as i32 - 1) as usize;
                sum = sum.wrapping_add(*in_ptr.add(px) as u32 * [2u32, 7, 14, 18, 14, 7, 2][k]);
            }
            *out_ptr.add(x) = (sum >> 6) as u8;
        }

        let simd_start = radius;
        let simd_end = width.saturating_sub(radius);
        let simd_chunks = if simd_end > simd_start { (simd_end - simd_start) / TILE } else { 0 };

        let mut x = simd_start;
        for _ in 0..simd_chunks {
            let p_3 = vld1q_u8(in_ptr.add(x - 3));
            let p_2 = vld1q_u8(in_ptr.add(x - 2));
            let p_1 = vld1q_u8(in_ptr.add(x - 1));
            let p0  = vld1q_u8(in_ptr.add(x));
            let p1  = vld1q_u8(in_ptr.add(x + 1));
            let p2  = vld1q_u8(in_ptr.add(x + 2));
            let p3  = vld1q_u8(in_ptr.add(x + 3));

            // Low 8
            let mut sum_lo = vmull_u8(vget_low_u8(p_3), k0);
            sum_lo = vmlal_u8(sum_lo, vget_low_u8(p_2), k1);
            sum_lo = vmlal_u8(sum_lo, vget_low_u8(p_1), k2);
            sum_lo = vmlal_u8(sum_lo, vget_low_u8(p0), k3);
            sum_lo = vmlal_u8(sum_lo, vget_low_u8(p1), k4);
            sum_lo = vmlal_u8(sum_lo, vget_low_u8(p2), k5);
            sum_lo = vmlal_u8(sum_lo, vget_low_u8(p3), k6);
            let lo = vshrn_n_u16(sum_lo, 6);

            // High 8
            let mut sum_hi = vmull_u8(vget_high_u8(p_3), k0);
            sum_hi = vmlal_u8(sum_hi, vget_high_u8(p_2), k1);
            sum_hi = vmlal_u8(sum_hi, vget_high_u8(p_1), k2);
            sum_hi = vmlal_u8(sum_hi, vget_high_u8(p0), k3);
            sum_hi = vmlal_u8(sum_hi, vget_high_u8(p1), k4);
            sum_hi = vmlal_u8(sum_hi, vget_high_u8(p2), k5);
            sum_hi = vmlal_u8(sum_hi, vget_high_u8(p3), k6);
            let hi = vshrn_n_u16(sum_hi, 6);

            vst1q_u8(out_ptr.add(x), vcombine_u8(lo, hi));
            x += TILE;
        }

        // Remainder
        for rem_x in x..simd_end {
            let mut sum: u32 = 0;
            for k in 0..7 {
                let px = (rem_x as i32 + k as i32 - radius as i32).clamp(0, width as i32 - 1) as usize;
                sum = sum.wrapping_add(*in_ptr.add(px) as u32 * [2u32, 7, 14, 18, 14, 7, 2][k]);
            }
            *out_ptr.add(rem_x) = (sum >> 6) as u8;
        }

        // Right edge
        for rx in simd_end.max(radius)..width {
            let mut sum: u32 = 0;
            for k in 0..7 {
                let px = (rx as i32 + k as i32 - radius as i32).clamp(0, width as i32 - 1) as usize;
                sum = sum.wrapping_add(*in_ptr.add(px) as u32 * [2u32, 7, 14, 18, 14, 7, 2][k]);
            }
            *out_ptr.add(rx) = (sum >> 6) as u8;
        }
    }

    // VERTICAL PASS
    let row_chunks = width / 16;
    for y in 0..height {
        let y_m3 = (y as i32 - 3).clamp(0, height as i32 - 1) as usize;
        let y_m2 = (y as i32 - 2).clamp(0, height as i32 - 1) as usize;
        let y_m1 = (y as i32 - 1).clamp(0, height as i32 - 1) as usize;
        let y_p1 = (y as i32 + 1).clamp(0, height as i32 - 1) as usize;
        let y_p2 = (y as i32 + 2).clamp(0, height as i32 - 1) as usize;
        let y_p3 = (y as i32 + 3).clamp(0, height as i32 - 1) as usize;

        let ptr_m3 = temp.as_ptr().add(y_m3 * width);
        let ptr_m2 = temp.as_ptr().add(y_m2 * width);
        let ptr_m1 = temp.as_ptr().add(y_m1 * width);
        let ptr_0  = temp.as_ptr().add(y * width);
        let ptr_p1 = temp.as_ptr().add(y_p1 * width);
        let ptr_p2 = temp.as_ptr().add(y_p2 * width);
        let ptr_p3 = temp.as_ptr().add(y_p3 * width);
        let out_ptr = data.as_mut_ptr().add(y * width);

        for chunk in 0..row_chunks {
            let offset = chunk * 16;
            let r_m3 = vld1q_u8(ptr_m3.add(offset));
            let r_m2 = vld1q_u8(ptr_m2.add(offset));
            let r_m1 = vld1q_u8(ptr_m1.add(offset));
            let r_0  = vld1q_u8(ptr_0.add(offset));
            let r_p1 = vld1q_u8(ptr_p1.add(offset));
            let r_p2 = vld1q_u8(ptr_p2.add(offset));
            let r_p3 = vld1q_u8(ptr_p3.add(offset));

            // Low 8
            let mut sum_lo = vmull_u8(vget_low_u8(r_m3), k0);
            sum_lo = vmlal_u8(sum_lo, vget_low_u8(r_m2), k1);
            sum_lo = vmlal_u8(sum_lo, vget_low_u8(r_m1), k2);
            sum_lo = vmlal_u8(sum_lo, vget_low_u8(r_0), k3);
            sum_lo = vmlal_u8(sum_lo, vget_low_u8(r_p1), k4);
            sum_lo = vmlal_u8(sum_lo, vget_low_u8(r_p2), k5);
            sum_lo = vmlal_u8(sum_lo, vget_low_u8(r_p3), k6);
            let lo = vshrn_n_u16(sum_lo, 6);

            // High 8
            let mut sum_hi = vmull_u8(vget_high_u8(r_m3), k0);
            sum_hi = vmlal_u8(sum_hi, vget_high_u8(r_m2), k1);
            sum_hi = vmlal_u8(sum_hi, vget_high_u8(r_m1), k2);
            sum_hi = vmlal_u8(sum_hi, vget_high_u8(r_0), k3);
            sum_hi = vmlal_u8(sum_hi, vget_high_u8(r_p1), k4);
            sum_hi = vmlal_u8(sum_hi, vget_high_u8(r_p2), k5);
            sum_hi = vmlal_u8(sum_hi, vget_high_u8(r_p3), k6);
            let hi = vshrn_n_u16(sum_hi, 6);

            vst1q_u8(out_ptr.add(offset), vcombine_u8(lo, hi));
        }

        for x in (row_chunks * 16)..width {
            let sum = *ptr_m3.add(x) as u32 * 2
                + *ptr_m2.add(x) as u32 * 7
                + *ptr_m1.add(x) as u32 * 14
                + *ptr_0.add(x) as u32 * 18
                + *ptr_p1.add(x) as u32 * 14
                + *ptr_p2.add(x) as u32 * 7
                + *ptr_p3.add(x) as u32 * 2;
            *out_ptr.add(x) = (sum >> 6) as u8;
        }
    }
}
