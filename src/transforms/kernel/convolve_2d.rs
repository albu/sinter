// Fast 2D 3x3 convolution and specialized kernel operations
//
// Provides specialized, highly-optimized SIMD and branchless scalar implementations
// for Sharpen, Emboss, EdgeDetection (Laplacian), and generic 3x3 convolutions.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use crate::core::FusableImage;

// ============================================================================
// Specialized Sharpen
// ============================================================================

pub fn apply_sharpen(image: &mut FusableImage, strength: f32) {
    let width = image.width;
    let height = image.height;
    let channels = image.channels;

    if width < 3 || height < 3 {
        apply_sharpen_scalar(&mut image.data, width, height, channels, strength);
        return;
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        apply_sharpen_neon(&mut image.data, width, height, channels, strength);
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        apply_sharpen_scalar(&mut image.data, width, height, channels, strength);
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn apply_sharpen_neon(
    data: &mut [u8],
    width: usize,
    height: usize,
    channels: usize,
    strength: f32,
) {
    let mut output = vec![0u8; data.len()];
    let stride = width * channels;
    let s = strength;
    let center_weight = 1.0 + 4.0 * s;
    let neighbor_weight = s;

    // Top and bottom border rows (scalar)
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
                let c_val = data[row_curr + x * channels + c] as f32;
                let n_val = data[row_prev + x * channels + c] as f32;
                let s_val = data[row_next + x * channels + c] as f32;
                let w_val = data[row_curr + x_prev * channels + c] as f32;
                let e_val = data[row_curr + x_next * channels + c] as f32;

                let res = center_weight * c_val - neighbor_weight * (n_val + s_val + w_val + e_val);
                output[row_curr + x * channels + c] = res.clamp(0.0, 255.0) as u8;
            }
        }
    }

    let step_bytes = if channels == 3 { 3 } else { 1 };
    let row_len_bytes = width * channels;

    let w_center_i16 = (center_weight * 256.0).round() as i16;
    let w_neighbor_i16 = (neighbor_weight * 256.0).round() as i16;

    let v_center_w = vdupq_n_s16(w_center_i16);
    let v_neigh_w = vdupq_n_s16(w_neighbor_i16);

    for y in 1..(height - 1) {
        let prev_ptr = data.as_ptr().add((y - 1) * stride);
        let curr_ptr = data.as_ptr().add(y * stride);
        let next_ptr = data.as_ptr().add((y + 1) * stride);
        let out_ptr = output.as_mut_ptr().add(y * stride);

        // Left border pixel (x = 0)
        let row_curr = y * stride;
        let row_prev = (y - 1) * stride;
        let row_next = (y + 1) * stride;
        for c in 0..channels {
            let c_val = data[row_curr + c] as f32;
            let n_val = data[row_prev + c] as f32;
            let s_val = data[row_next + c] as f32;
            let w_val = data[row_curr + c] as f32;
            let e_val = data[row_curr + channels + c] as f32;

            let res = center_weight * c_val - neighbor_weight * (n_val + s_val + w_val + e_val);
            output[row_curr + c] = res.clamp(0.0, 255.0) as u8;
        }

        // SIMD interior
        let mut byte_idx = step_bytes;
        let simd_end = row_len_bytes.saturating_sub(16 + step_bytes);

        while byte_idx <= simd_end {
            let c_vec = vld1q_u8(curr_ptr.add(byte_idx));
            let n_vec = vld1q_u8(prev_ptr.add(byte_idx));
            let s_vec = vld1q_u8(next_ptr.add(byte_idx));
            let w_vec = vld1q_u8(curr_ptr.add(byte_idx - step_bytes));
            let e_vec = vld1q_u8(curr_ptr.add(byte_idx + step_bytes));

            // Low 8 bytes
            let c_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(c_vec)));
            let n_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(n_vec)));
            let s_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(s_vec)));
            let w_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(w_vec)));
            let e_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(e_vec)));

            let neigh_sum_lo = vaddq_s16(vaddq_s16(n_lo, s_lo), vaddq_s16(w_lo, e_lo));
            let prod_c_lo_l = vmull_s16(vget_low_s16(c_lo), vget_low_s16(v_center_w));
            let prod_c_lo_h = vmull_s16(vget_high_s16(c_lo), vget_high_s16(v_center_w));
            let prod_n_lo_l = vmull_s16(vget_low_s16(neigh_sum_lo), vget_low_s16(v_neigh_w));
            let prod_n_lo_h = vmull_s16(vget_high_s16(neigh_sum_lo), vget_high_s16(v_neigh_w));

            let diff_lo_l = vsubq_s32(prod_c_lo_l, prod_n_lo_l);
            let diff_lo_h = vsubq_s32(prod_c_lo_h, prod_n_lo_h);

            let res_lo_l = vshrn_n_s32(diff_lo_l, 8);
            let res_lo_h = vshrn_n_s32(diff_lo_h, 8);
            let res_lo_16 = vcombine_s16(res_lo_l, res_lo_h);
            let res_lo_u8 = vqmovun_s16(res_lo_16);

            // High 8 bytes
            let c_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(c_vec)));
            let n_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(n_vec)));
            let s_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(s_vec)));
            let w_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(w_vec)));
            let e_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(e_vec)));

            let neigh_sum_hi = vaddq_s16(vaddq_s16(n_hi, s_hi), vaddq_s16(w_hi, e_hi));
            let prod_c_hi_l = vmull_s16(vget_low_s16(c_hi), vget_low_s16(v_center_w));
            let prod_c_hi_h = vmull_s16(vget_high_s16(c_hi), vget_high_s16(v_center_w));
            let prod_n_hi_l = vmull_s16(vget_low_s16(neigh_sum_hi), vget_low_s16(v_neigh_w));
            let prod_n_hi_h = vmull_s16(vget_high_s16(neigh_sum_hi), vget_high_s16(v_neigh_w));

            let diff_hi_l = vsubq_s32(prod_c_hi_l, prod_n_hi_l);
            let diff_hi_h = vsubq_s32(prod_c_hi_h, prod_n_hi_h);

            let res_hi_l = vshrn_n_s32(diff_hi_l, 8);
            let res_hi_h = vshrn_n_s32(diff_hi_h, 8);
            let res_hi_16 = vcombine_s16(res_hi_l, res_hi_h);
            let res_hi_u8 = vqmovun_s16(res_hi_16);

            let result = vcombine_u8(res_lo_u8, res_hi_u8);
            vst1q_u8(out_ptr.add(byte_idx), result);

            byte_idx += 16;
        }

        // Remainder pixels
        let x_start = byte_idx / channels;
        for x in x_start..width {
            let x_prev = x.saturating_sub(1);
            let x_next = (x + 1).min(width - 1);

            for c in 0..channels {
                let c_val = data[row_curr + x * channels + c] as f32;
                let n_val = data[row_prev + x * channels + c] as f32;
                let s_val = data[row_next + x * channels + c] as f32;
                let w_val = data[row_curr + x_prev * channels + c] as f32;
                let e_val = data[row_curr + x_next * channels + c] as f32;

                let res = center_weight * c_val - neighbor_weight * (n_val + s_val + w_val + e_val);
                output[row_curr + x * channels + c] = res.clamp(0.0, 255.0) as u8;
            }
        }
    }

    data.copy_from_slice(&output);
}

fn apply_sharpen_scalar(
    data: &mut [u8],
    width: usize,
    height: usize,
    channels: usize,
    strength: f32,
) {
    let mut output = vec![0u8; data.len()];
    let stride = width * channels;
    let s = strength;
    let center_weight = 1.0 + 4.0 * s;
    let neighbor_weight = s;

    for y in 0..height {
        let y_prev = y.saturating_sub(1);
        let y_next = (y + 1).min(height - 1);
        let row_curr = y * stride;
        let row_prev = y_prev * stride;
        let row_next = y_next * stride;

        for x in 0..width {
            let x_prev = x.saturating_sub(1);
            let x_next = (x + 1).min(width - 1);

            for c in 0..channels {
                let c_val = data[row_curr + x * channels + c] as f32;
                let n_val = data[row_prev + x * channels + c] as f32;
                let s_val = data[row_next + x * channels + c] as f32;
                let w_val = data[row_curr + x_prev * channels + c] as f32;
                let e_val = data[row_curr + x_next * channels + c] as f32;

                let res = center_weight * c_val - neighbor_weight * (n_val + s_val + w_val + e_val);
                output[row_curr + x * channels + c] = res.clamp(0.0, 255.0) as u8;
            }
        }
    }

    data.copy_from_slice(&output);
}

// ============================================================================
// Specialized Laplacian Edge Detection
// Kernel: [0, 1, 0; 1, -4, 1; 0, 1, 0]
// Output = clamp((N + S + E + W) - 4*center, 0, 255)
// ============================================================================

pub fn apply_laplacian(image: &mut FusableImage) {
    let width = image.width;
    let height = image.height;
    let channels = image.channels;

    if width < 3 || height < 3 {
        apply_laplacian_scalar(&mut image.data, width, height, channels);
        return;
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        apply_laplacian_neon(&mut image.data, width, height, channels);
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        apply_laplacian_scalar(&mut image.data, width, height, channels);
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn apply_laplacian_neon(data: &mut [u8], width: usize, height: usize, channels: usize) {
    let mut output = vec![0u8; data.len()];
    let stride = width * channels;

    // Top and bottom border rows
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
                let c_val = data[row_curr + x * channels + c] as i32;
                let n_val = data[row_prev + x * channels + c] as i32;
                let s_val = data[row_next + x * channels + c] as i32;
                let w_val = data[row_curr + x_prev * channels + c] as i32;
                let e_val = data[row_curr + x_next * channels + c] as i32;

                let res = (n_val + s_val + w_val + e_val) - 4 * c_val;
                output[row_curr + x * channels + c] = res.clamp(0, 255) as u8;
            }
        }
    }

    let step_bytes = if channels == 3 { 3 } else { 1 };
    let row_len_bytes = width * channels;

    for y in 1..(height - 1) {
        let prev_ptr = data.as_ptr().add((y - 1) * stride);
        let curr_ptr = data.as_ptr().add(y * stride);
        let next_ptr = data.as_ptr().add((y + 1) * stride);
        let out_ptr = output.as_mut_ptr().add(y * stride);

        // Left border pixel
        let row_curr = y * stride;
        let row_prev = (y - 1) * stride;
        let row_next = (y + 1) * stride;
        for c in 0..channels {
            let c_val = data[row_curr + c] as i32;
            let n_val = data[row_prev + c] as i32;
            let s_val = data[row_next + c] as i32;
            let w_val = data[row_curr + c] as i32;
            let e_val = data[row_curr + channels + c] as i32;

            let res = (n_val + s_val + w_val + e_val) - 4 * c_val;
            output[row_curr + c] = res.clamp(0, 255) as u8;
        }

        // SIMD interior
        let mut byte_idx = step_bytes;
        let simd_end = row_len_bytes.saturating_sub(16 + step_bytes);

        while byte_idx <= simd_end {
            let c_vec = vld1q_u8(curr_ptr.add(byte_idx));
            let n_vec = vld1q_u8(prev_ptr.add(byte_idx));
            let s_vec = vld1q_u8(next_ptr.add(byte_idx));
            let w_vec = vld1q_u8(curr_ptr.add(byte_idx - step_bytes));
            let e_vec = vld1q_u8(curr_ptr.add(byte_idx + step_bytes));

            // Low 8 bytes
            let c_lo = vmovl_u8(vget_low_u8(c_vec));
            let n_lo = vmovl_u8(vget_low_u8(n_vec));
            let s_lo = vmovl_u8(vget_low_u8(s_vec));
            let w_lo = vmovl_u8(vget_low_u8(w_vec));
            let e_lo = vmovl_u8(vget_low_u8(e_vec));

            let sum_lo = vaddq_u16(vaddq_u16(n_lo, s_lo), vaddq_u16(w_lo, e_lo));
            let c4_lo = vshlq_n_u16(c_lo, 2);
            let diff_lo = vsubq_s16(vreinterpretq_s16_u16(sum_lo), vreinterpretq_s16_u16(c4_lo));
            let res_lo = vqmovun_s16(diff_lo);

            // High 8 bytes
            let c_hi = vmovl_u8(vget_high_u8(c_vec));
            let n_hi = vmovl_u8(vget_high_u8(n_vec));
            let s_hi = vmovl_u8(vget_high_u8(s_vec));
            let w_hi = vmovl_u8(vget_high_u8(w_vec));
            let e_hi = vmovl_u8(vget_high_u8(e_vec));

            let sum_hi = vaddq_u16(vaddq_u16(n_hi, s_hi), vaddq_u16(w_hi, e_hi));
            let c4_hi = vshlq_n_u16(c_hi, 2);
            let diff_hi = vsubq_s16(vreinterpretq_s16_u16(sum_hi), vreinterpretq_s16_u16(c4_hi));
            let res_hi = vqmovun_s16(diff_hi);

            let result = vcombine_u8(res_lo, res_hi);
            vst1q_u8(out_ptr.add(byte_idx), result);

            byte_idx += 16;
        }

        // Remainder pixels
        let x_start = byte_idx / channels;
        for x in x_start..width {
            let x_prev = x.saturating_sub(1);
            let x_next = (x + 1).min(width - 1);

            for c in 0..channels {
                let c_val = data[row_curr + x * channels + c] as i32;
                let n_val = data[row_prev + x * channels + c] as i32;
                let s_val = data[row_next + x * channels + c] as i32;
                let w_val = data[row_curr + x_prev * channels + c] as i32;
                let e_val = data[row_curr + x_next * channels + c] as i32;

                let res = (n_val + s_val + w_val + e_val) - 4 * c_val;
                output[row_curr + x * channels + c] = res.clamp(0, 255) as u8;
            }
        }
    }

    data.copy_from_slice(&output);
}

fn apply_laplacian_scalar(data: &mut [u8], width: usize, height: usize, channels: usize) {
    let mut output = vec![0u8; data.len()];
    let stride = width * channels;

    for y in 0..height {
        let y_prev = y.saturating_sub(1);
        let y_next = (y + 1).min(height - 1);
        let row_curr = y * stride;
        let row_prev = y_prev * stride;
        let row_next = y_next * stride;

        for x in 0..width {
            let x_prev = x.saturating_sub(1);
            let x_next = (x + 1).min(width - 1);

            for c in 0..channels {
                let c_val = data[row_curr + x * channels + c] as i32;
                let n_val = data[row_prev + x * channels + c] as i32;
                let s_val = data[row_next + x * channels + c] as i32;
                let w_val = data[row_curr + x_prev * channels + c] as i32;
                let e_val = data[row_curr + x_next * channels + c] as i32;

                let res = (n_val + s_val + w_val + e_val) - 4 * c_val;
                output[row_curr + x * channels + c] = res.clamp(0, 255) as u8;
            }
        }
    }

    data.copy_from_slice(&output);
}

pub fn apply_emboss(image: &mut FusableImage, kernel: &[i32; 9]) {
    let width = image.width;
    let height = image.height;
    let channels = image.channels;

    if width < 3 || height < 3 {
        convolve_3x3_fast(image, kernel, 256, 0);
        return;
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        apply_emboss_neon(&mut image.data, width, height, channels, kernel);
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        convolve_3x3_fast(image, kernel, 256, 0);
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn apply_emboss_neon(
    data: &mut [u8],
    width: usize,
    height: usize,
    channels: usize,
    kernel: &[i32; 9],
) {
    let mut output = Vec::with_capacity(data.len());
    unsafe { output.set_len(data.len()); }
    let stride = width * channels;

    // Top and bottom border rows (scalar)
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
                    data[row_prev + x_prev * channels + c] as i32,
                    data[row_prev + x * channels + c] as i32,
                    data[row_prev + x_next * channels + c] as i32,
                    data[row_curr + x_prev * channels + c] as i32,
                    data[row_curr + x * channels + c] as i32,
                    data[row_curr + x_next * channels + c] as i32,
                    data[row_next + x_prev * channels + c] as i32,
                    data[row_next + x * channels + c] as i32,
                    data[row_next + x_next * channels + c] as i32,
                ];

                let mut sum = 0i32;
                for i in 0..9 {
                    sum += p[i] * kernel[i];
                }
                output[row_curr + x * channels + c] = (sum / 256).clamp(0, 255) as u8;
            }
        }
    }

    let step_bytes = if channels == 3 { 3 } else { 1 };
    let row_len_bytes = width * channels;

    let vk0 = vdupq_n_s16(kernel[0] as i16);
    let vk1 = vdupq_n_s16(kernel[1] as i16);
    let vk2 = vdupq_n_s16(kernel[2] as i16);
    let vk3 = vdupq_n_s16(kernel[3] as i16);
    let vk4 = vdupq_n_s16(kernel[4] as i16);
    let vk5 = vdupq_n_s16(kernel[5] as i16);
    let vk6 = vdupq_n_s16(kernel[6] as i16);
    let vk7 = vdupq_n_s16(kernel[7] as i16);
    let vk8 = vdupq_n_s16(kernel[8] as i16);

    for y in 1..(height - 1) {
        let prev_ptr = data.as_ptr().add((y - 1) * stride);
        let curr_ptr = data.as_ptr().add(y * stride);
        let next_ptr = data.as_ptr().add((y + 1) * stride);
        let out_ptr = output.as_mut_ptr().add(y * stride);

        // Left border pixel
        let row_curr = y * stride;
        let row_prev = (y - 1) * stride;
        let row_next = (y + 1) * stride;
        for c in 0..channels {
            let p = [
                data[row_prev + c] as i32,
                data[row_prev + c] as i32,
                data[row_prev + channels + c] as i32,
                data[row_curr + c] as i32,
                data[row_curr + c] as i32,
                data[row_curr + channels + c] as i32,
                data[row_next + c] as i32,
                data[row_next + c] as i32,
                data[row_next + channels + c] as i32,
            ];
            let mut sum = 0i32;
            for i in 0..9 {
                sum += p[i] * kernel[i];
            }
            output[row_curr + c] = (sum / 256).clamp(0, 255) as u8;
        }

        // SIMD interior
        let mut byte_idx = step_bytes;
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

            // Low 8 bytes
            let v_p0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(p0)));
            let v_p1 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(p1)));
            let v_p2 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(p2)));
            let v_p3 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(p3)));
            let v_p4 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(p4)));
            let v_p5 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(p5)));
            let v_p6 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(p6)));
            let v_p7 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(p7)));
            let v_p8 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(p8)));

            let mut sum_lo_l = vmull_s16(vget_low_s16(v_p0), vget_low_s16(vk0));
            sum_lo_l = vmlal_s16(sum_lo_l, vget_low_s16(v_p1), vget_low_s16(vk1));
            sum_lo_l = vmlal_s16(sum_lo_l, vget_low_s16(v_p2), vget_low_s16(vk2));
            sum_lo_l = vmlal_s16(sum_lo_l, vget_low_s16(v_p3), vget_low_s16(vk3));
            sum_lo_l = vmlal_s16(sum_lo_l, vget_low_s16(v_p4), vget_low_s16(vk4));
            sum_lo_l = vmlal_s16(sum_lo_l, vget_low_s16(v_p5), vget_low_s16(vk5));
            sum_lo_l = vmlal_s16(sum_lo_l, vget_low_s16(v_p6), vget_low_s16(vk6));
            sum_lo_l = vmlal_s16(sum_lo_l, vget_low_s16(v_p7), vget_low_s16(vk7));
            sum_lo_l = vmlal_s16(sum_lo_l, vget_low_s16(v_p8), vget_low_s16(vk8));

            let mut sum_lo_h = vmull_s16(vget_high_s16(v_p0), vget_high_s16(vk0));
            sum_lo_h = vmlal_s16(sum_lo_h, vget_high_s16(v_p1), vget_high_s16(vk1));
            sum_lo_h = vmlal_s16(sum_lo_h, vget_high_s16(v_p2), vget_high_s16(vk2));
            sum_lo_h = vmlal_s16(sum_lo_h, vget_high_s16(v_p3), vget_high_s16(vk3));
            sum_lo_h = vmlal_s16(sum_lo_h, vget_high_s16(v_p4), vget_high_s16(vk4));
            sum_lo_h = vmlal_s16(sum_lo_h, vget_high_s16(v_p5), vget_high_s16(vk5));
            sum_lo_h = vmlal_s16(sum_lo_h, vget_high_s16(v_p6), vget_high_s16(vk6));
            sum_lo_h = vmlal_s16(sum_lo_h, vget_high_s16(v_p7), vget_high_s16(vk7));
            sum_lo_h = vmlal_s16(sum_lo_h, vget_high_s16(v_p8), vget_high_s16(vk8));

            let res_lo_l = vshrn_n_s32(sum_lo_l, 8);
            let res_lo_h = vshrn_n_s32(sum_lo_h, 8);
            let res_lo_16 = vcombine_s16(res_lo_l, res_lo_h);
            let res_lo_u8 = vqmovun_s16(res_lo_16);

            // High 8 bytes
            let v_p0_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(p0)));
            let v_p1_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(p1)));
            let v_p2_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(p2)));
            let v_p3_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(p3)));
            let v_p4_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(p4)));
            let v_p5_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(p5)));
            let v_p6_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(p6)));
            let v_p7_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(p7)));
            let v_p8_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(p8)));

            let mut sum_hi_l = vmull_s16(vget_low_s16(v_p0_hi), vget_low_s16(vk0));
            sum_hi_l = vmlal_s16(sum_hi_l, vget_low_s16(v_p1_hi), vget_low_s16(vk1));
            sum_hi_l = vmlal_s16(sum_hi_l, vget_low_s16(v_p2_hi), vget_low_s16(vk2));
            sum_hi_l = vmlal_s16(sum_hi_l, vget_low_s16(v_p3_hi), vget_low_s16(vk3));
            sum_hi_l = vmlal_s16(sum_hi_l, vget_low_s16(v_p4_hi), vget_low_s16(vk4));
            sum_hi_l = vmlal_s16(sum_hi_l, vget_low_s16(v_p5_hi), vget_low_s16(vk5));
            sum_hi_l = vmlal_s16(sum_hi_l, vget_low_s16(v_p6_hi), vget_low_s16(vk6));
            sum_hi_l = vmlal_s16(sum_hi_l, vget_low_s16(v_p7_hi), vget_low_s16(vk7));
            sum_hi_l = vmlal_s16(sum_hi_l, vget_low_s16(v_p8_hi), vget_low_s16(vk8));

            let mut sum_hi_h = vmull_s16(vget_high_s16(v_p0_hi), vget_high_s16(vk0));
            sum_hi_h = vmlal_s16(sum_hi_h, vget_high_s16(v_p1_hi), vget_high_s16(vk1));
            sum_hi_h = vmlal_s16(sum_hi_h, vget_high_s16(v_p2_hi), vget_high_s16(vk2));
            sum_hi_h = vmlal_s16(sum_hi_h, vget_high_s16(v_p3_hi), vget_high_s16(vk3));
            sum_hi_h = vmlal_s16(sum_hi_h, vget_high_s16(v_p4_hi), vget_high_s16(vk4));
            sum_hi_h = vmlal_s16(sum_hi_h, vget_high_s16(v_p5_hi), vget_high_s16(vk5));
            sum_hi_h = vmlal_s16(sum_hi_h, vget_high_s16(v_p6_hi), vget_high_s16(vk6));
            sum_hi_h = vmlal_s16(sum_hi_h, vget_high_s16(v_p7_hi), vget_high_s16(vk7));
            sum_hi_h = vmlal_s16(sum_hi_h, vget_high_s16(v_p8_hi), vget_high_s16(vk8));

            let res_hi_l = vshrn_n_s32(sum_hi_l, 8);
            let res_hi_h = vshrn_n_s32(sum_hi_h, 8);
            let res_hi_16 = vcombine_s16(res_hi_l, res_hi_h);
            let res_hi_u8 = vqmovun_s16(res_hi_16);

            let result = vcombine_u8(res_lo_u8, res_hi_u8);
            vst1q_u8(out_ptr.add(byte_idx), result);

            byte_idx += 16;
        }

        // Remainder pixels
        let x_start = byte_idx / channels;
        for x in x_start..width {
            let x_prev = x.saturating_sub(1);
            let x_next = (x + 1).min(width - 1);

            for c in 0..channels {
                let p = [
                    data[row_prev + x_prev * channels + c] as i32,
                    data[row_prev + x * channels + c] as i32,
                    data[row_prev + x_next * channels + c] as i32,
                    data[row_curr + x_prev * channels + c] as i32,
                    data[row_curr + x * channels + c] as i32,
                    data[row_curr + x_next * channels + c] as i32,
                    data[row_next + x_prev * channels + c] as i32,
                    data[row_next + x * channels + c] as i32,
                    data[row_next + x_next * channels + c] as i32,
                ];

                let mut sum = 0i32;
                for i in 0..9 {
                    sum += p[i] * kernel[i];
                }
                output[row_curr + x * channels + c] = (sum / 256).clamp(0, 255) as u8;
            }
        }
    }

    data.copy_from_slice(&output);
}

// ============================================================================
// Generic fast 3x3 convolution with interior/border split
// ============================================================================

pub fn convolve_3x3_fast(
    image: &mut FusableImage,
    kernel: &[i32; 9],
    scale: i32,
    offset: i32,
) {
    let width = image.width;
    let height = image.height;
    let channels = image.channels;
    let data = &mut image.data;

    let mut output = vec![0u8; data.len()];
    let stride = width * channels;

    // Top and bottom border rows
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
                    data[row_prev + x_prev * channels + c] as i32,
                    data[row_prev + x * channels + c] as i32,
                    data[row_prev + x_next * channels + c] as i32,
                    data[row_curr + x_prev * channels + c] as i32,
                    data[row_curr + x * channels + c] as i32,
                    data[row_curr + x_next * channels + c] as i32,
                    data[row_next + x_prev * channels + c] as i32,
                    data[row_next + x * channels + c] as i32,
                    data[row_next + x_next * channels + c] as i32,
                ];

                let mut sum = 0i32;
                for i in 0..9 {
                    sum += p[i] * kernel[i];
                }
                let val = (sum / scale).saturating_add(offset);
                output[row_curr + x * channels + c] = val.clamp(0, 255) as u8;
            }
        }
    }

    // Interior rows (1..height-1) with zero bounds check on y
    for y in 1..(height - 1) {
        let row_prev = (y - 1) * stride;
        let row_curr = y * stride;
        let row_next = (y + 1) * stride;

        // Left border pixel (x = 0)
        for c in 0..channels {
            let p = [
                data[row_prev + c] as i32,
                data[row_prev + c] as i32,
                data[row_prev + channels + c] as i32,
                data[row_curr + c] as i32,
                data[row_curr + c] as i32,
                data[row_curr + channels + c] as i32,
                data[row_next + c] as i32,
                data[row_next + c] as i32,
                data[row_next + channels + c] as i32,
            ];
            let mut sum = 0i32;
            for i in 0..9 {
                sum += p[i] * kernel[i];
            }
            let val = (sum / scale).saturating_add(offset);
            output[row_curr + c] = val.clamp(0, 255) as u8;
        }

        // Interior pixels (1..width-1) with ZERO bounds checks on x and y
        for x in 1..(width - 1) {
            let p_prev = row_prev + (x - 1) * channels;
            let p_curr = row_curr + (x - 1) * channels;
            let p_next = row_next + (x - 1) * channels;

            for c in 0..channels {
                let p = [
                    data[p_prev + c] as i32,
                    data[p_prev + channels + c] as i32,
                    data[p_prev + 2 * channels + c] as i32,
                    data[p_curr + c] as i32,
                    data[p_curr + channels + c] as i32,
                    data[p_curr + 2 * channels + c] as i32,
                    data[p_next + c] as i32,
                    data[p_next + channels + c] as i32,
                    data[p_next + 2 * channels + c] as i32,
                ];

                let mut sum = 0i32;
                for i in 0..9 {
                    sum += p[i] * kernel[i];
                }
                let val = (sum / scale).saturating_add(offset);
                output[row_curr + x * channels + c] = val.clamp(0, 255) as u8;
            }
        }

        // Right border pixel (x = width - 1)
        let x = width - 1;
        for c in 0..channels {
            let p = [
                data[row_prev + (x - 1) * channels + c] as i32,
                data[row_prev + x * channels + c] as i32,
                data[row_prev + x * channels + c] as i32,
                data[row_curr + (x - 1) * channels + c] as i32,
                data[row_curr + x * channels + c] as i32,
                data[row_curr + x * channels + c] as i32,
                data[row_next + (x - 1) * channels + c] as i32,
                data[row_next + x * channels + c] as i32,
                data[row_next + x * channels + c] as i32,
            ];
            let mut sum = 0i32;
            for i in 0..9 {
                sum += p[i] * kernel[i];
            }
            let val = (sum / scale).saturating_add(offset);
            output[row_curr + x * channels + c] = val.clamp(0, 255) as u8;
        }
    }

    data.copy_from_slice(&output);
}
