// ARM64 NEON SIMD implementation for multiplicative noise

use crate::core::image::FusableImage;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// ARM64 NEON: True SIMD - load 8 pixels, convert to f32, multiply by single noise, clamp, store
#[cfg(target_arch = "aarch64")]
pub fn apply_noise_vectorized_simd(image: &mut FusableImage<'_>, noise_factors: &[f32]) {
    let data = &mut image.data;
    let len = data.len();
    let vector_count = len / 8; // Process 8 pixels at a time
    let remainder = len % 8;

    unsafe {
        // Process 8 pixels at a time with true SIMD
        for i in 0..vector_count {
            let idx = i * 8;

            // Load 8 pixels as u8
            let pixels_u8 = vld1_u8(data.as_ptr().add(idx));

            // Widen u8 -> u16 (8 -> 16-bit)
            let pixels_u16 = vmovl_u8(pixels_u8);

            // Convert low and high halves to f32
            let low_u32 = vmovl_u16(vget_low_u16(pixels_u16));
            let high_u32 = vmovl_u16(vget_high_u16(pixels_u16));
            let pixels_f0 = vcvtq_f32_u32(low_u32);
            let pixels_f1 = vcvtq_f32_u32(high_u32);

            // Broadcast single noise factor to all 8 pixels
            let noise_scalar = *noise_factors.get_unchecked(i);
            let noise = vdupq_n_f32(noise_scalar);

            // SIMD multiply both halves by same noise
            let multiplied0 = vmulq_f32(pixels_f0, noise);
            let multiplied1 = vmulq_f32(pixels_f1, noise);

            // Clamp to [0, 255] using vmin/vmax
            let clamped0 = vmaxq_f32(vdupq_n_f32(0.0), vminq_f32(vdupq_n_f32(255.0), multiplied0));
            let clamped1 = vmaxq_f32(vdupq_n_f32(0.0), vminq_f32(vdupq_n_f32(255.0), multiplied1));

            let result_u32_0 = vcvtq_u32_f32(clamped0);
            let result_u32_1 = vcvtq_u32_f32(clamped1);

            // Narrow u32 -> u16 with saturation
            let _result_u16 = vcombine_u16(vqmovn_u32(result_u32_0), vqmovn_u32(result_u32_1));

            // Narrow u16 -> u8 with saturation
            let result_u8 = vqmovn_u16(_result_u16);

            // Store result
            vst1_u8(data.as_mut_ptr().add(idx), result_u8);
        }

        // Handle remaining pixels (0-7) scalar
        let base_idx = vector_count * 8;
        if remainder > 0 {
            let noise_scalar = noise_factors.get(vector_count).copied().unwrap_or(1.0);
            for j in 0..remainder {
                let v = data[base_idx + j] as f32 * noise_scalar;
                data[base_idx + j] = v.clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// Scalar fallback for vectorized approach (8 pixels per noise value)
#[cfg(not(target_arch = "aarch64"))]
pub fn apply_noise_vectorized_simd(image: &mut FusableImage<'_>, noise_factors: &[f32]) {
    let data = &mut image.data;
    let len = data.len();
    let vector_count = (len + 7) / 8;

    for i in 0..vector_count {
        let idx = i * 8;
        let remaining = len - idx;
        let count = remaining.min(8);
        let noise = noise_factors[i];

        for j in 0..count {
            let v = data[idx + j] as f32 * noise;
            data[idx + j] = v.clamp(0.0, 255.0) as u8;
        }
    }
}
