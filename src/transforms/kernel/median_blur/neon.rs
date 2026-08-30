// ARM NEON SIMD implementation of 3x3 median filter using a 19-op sorting network.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use super::sorting_network::median9;

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn vcas(a: &mut uint8x16_t, b: &mut uint8x16_t) {
    let min = vminq_u8(*a, *b);
    let max = vmaxq_u8(*a, *b);
    *a = min;
    *b = max;
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn vmedian9(
    mut p0: uint8x16_t,
    mut p1: uint8x16_t,
    mut p2: uint8x16_t,
    mut p3: uint8x16_t,
    mut p4: uint8x16_t,
    mut p5: uint8x16_t,
    mut p6: uint8x16_t,
    mut p7: uint8x16_t,
    mut p8: uint8x16_t,
) -> uint8x16_t {
    vcas(&mut p1, &mut p2); vcas(&mut p4, &mut p5); vcas(&mut p7, &mut p8);
    vcas(&mut p0, &mut p1); vcas(&mut p3, &mut p4); vcas(&mut p6, &mut p7);
    vcas(&mut p1, &mut p2); vcas(&mut p4, &mut p5); vcas(&mut p7, &mut p8);
    vcas(&mut p0, &mut p3); vcas(&mut p5, &mut p8); vcas(&mut p4, &mut p7);
    vcas(&mut p3, &mut p6); vcas(&mut p1, &mut p4); vcas(&mut p2, &mut p5);
    vcas(&mut p4, &mut p7); vcas(&mut p4, &mut p2); vcas(&mut p6, &mut p4);
    vcas(&mut p4, &mut p2);
    p4
}

/// Apply 3x3 median filter using ARM NEON vectorization
#[cfg(target_arch = "aarch64")]
pub unsafe fn apply_median_blur_3x3_neon(
    data: &mut [u8],
    width: usize,
    height: usize,
    channels: usize,
) {
    if width < 3 || height < 3 {
        super::sorting_network::apply_median_blur_3x3_scalar(data, width, height, channels);
        return;
    }

    let mut output = vec![0u8; data.len()];
    let stride = width * channels;

    // 1. Process top and bottom border rows with scalar
    for y in [0, height - 1] {
        let y_prev = y.saturating_sub(1);
        let y_next = (y + 1).min(height - 1);
        let row_curr = y * stride;
        let row_prev = y_prev * stride;
        let row_next = y_next * stride;

        for x in 0..width {
            let x_prev = x.saturating_sub(1);
            let x_next = (x + 1).min(width - 1);
            for c in 0..channels {
                let p = [
                    data[row_prev + x_prev * channels + c],
                    data[row_prev + x * channels + c],
                    data[row_prev + x_next * channels + c],
                    data[row_curr + x_prev * channels + c],
                    data[row_curr + x * channels + c],
                    data[row_curr + x_next * channels + c],
                    data[row_next + x_prev * channels + c],
                    data[row_next + x * channels + c],
                    data[row_next + x_next * channels + c],
                ];
                output[row_curr + x * channels + c] = median9(p);
            }
        }
    }

    // 2. Process interior rows
    let step_bytes = if channels == 3 { 3 } else { 1 };
    let row_len_bytes = width * channels;

    for y in 1..(height - 1) {
        let prev_ptr = data.as_ptr().add((y - 1) * stride);
        let curr_ptr = data.as_ptr().add(y * stride);
        let next_ptr = data.as_ptr().add((y + 1) * stride);
        let out_ptr = output.as_mut_ptr().add(y * stride);

        // Process left border pixel (x=0)
        let row_curr = y * stride;
        let row_prev = (y - 1) * stride;
        let row_next = (y + 1) * stride;
        for c in 0..channels {
            let p = [
                data[row_prev + c],
                data[row_prev + c],
                data[row_prev + channels + c],
                data[row_curr + c],
                data[row_curr + c],
                data[row_curr + channels + c],
                data[row_next + c],
                data[row_next + c],
                data[row_next + channels + c],
            ];
            output[row_curr + c] = median9(p);
        }

        // SIMD interior
        let mut byte_idx = step_bytes;
        // Ensure byte_idx + 16 + step_bytes <= row_len_bytes
        let simd_end = row_len_bytes.saturating_sub(16 + step_bytes);

        while byte_idx <= simd_end {
            let p0 = vld1q_u8(prev_ptr.add(byte_idx - step_bytes));
            let p1 = vld1q_u8(prev_ptr.add(byte_idx));
            let p2 = vld1q_u8(prev_ptr.add(byte_idx + step_bytes));

            let p3 = vld1q_u8(curr_ptr.add(byte_idx - step_bytes));
            let p4 = vld1q_u8(curr_ptr.add(byte_idx));
            let p5 = vld1q_u8(curr_ptr.add(byte_idx + step_bytes));

            let p6 = vld1q_u8(next_ptr.add(byte_idx - step_bytes));
            let p7 = vld1q_u8(next_ptr.add(byte_idx));
            let p8 = vld1q_u8(next_ptr.add(byte_idx + step_bytes));

            let res = vmedian9(p0, p1, p2, p3, p4, p5, p6, p7, p8);
            vst1q_u8(out_ptr.add(byte_idx), res);

            byte_idx += 16;
        }

        // Remainder pixels in the row
        let x_start = byte_idx / channels;
        for x in x_start..width {
            let x_prev = x.saturating_sub(1);
            let x_next = (x + 1).min(width - 1);
            for c in 0..channels {
                let p = [
                    data[row_prev + x_prev * channels + c],
                    data[row_prev + x * channels + c],
                    data[row_prev + x_next * channels + c],
                    data[row_curr + x_prev * channels + c],
                    data[row_curr + x * channels + c],
                    data[row_curr + x_next * channels + c],
                    data[row_next + x_prev * channels + c],
                    data[row_next + x * channels + c],
                    data[row_next + x_next * channels + c],
                ];
                output[row_curr + x * channels + c] = median9(p);
            }
        }
    }

    data.copy_from_slice(&output);
}
