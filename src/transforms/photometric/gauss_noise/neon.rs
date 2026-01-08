// NEON SIMD implementation for Gaussian Noise
//
// This module contains the ARM NEON-optimized implementation of the Gaussian Noise transform.
// For non-ARM platforms, a scalar fallback is provided.

const LUT_SIZE: usize = 1024;
const LUT_MASK: u32 = (LUT_SIZE - 1) as u32;

/// Apply Gaussian Noise using NEON SIMD instructions
///
/// This is the ARM-optimized implementation that processes 8 pixels at a time.
#[cfg(target_arch = "aarch64")]
pub fn apply_gauss_noise_neon(
    data: &mut [u8],
    lut: &[i16; LUT_SIZE],
    strength: i16,
    mean_offset: i16,
) {
    use std::arch::aarch64::*;

    let mut rng_state: u64 = 0xDEADBEEF;

    let len = data.len();
    let chunks = len / 8;

    // Process 8 pixels at a time using NEON
    for i in 0..chunks {
        let base = i * 8;

        // Generate 8 random indices using xorshift*
        rng_state ^= rng_state >> 12;
        rng_state ^= rng_state << 25;
        rng_state ^= rng_state >> 27;
        let idx_base = (rng_state as u32) & LUT_MASK;

        // LUT lookup for 8 noise values
        // Since NEON table lookup is complex, use hybrid approach
        let mut noise_i16 = [0i16; 8];
        for j in 0..8 {
            let idx = (idx_base.wrapping_add(j as u32).wrapping_mul(0x9E3779B9)) & LUT_MASK;
            noise_i16[j] = lut[idx as usize];
        }

        unsafe {
            // Load 8 pixels
            let pixels = vld1_u8(data.as_ptr().add(base));

            // Convert to i16 (widening)
            let pixels_wide_unsigned = vmovl_u8(pixels);
            let mut pixels_wide = vreinterpretq_s16_u16(pixels_wide_unsigned);

            // Add mean offset
            let mean_vec = vdupq_n_s16(mean_offset);
            pixels_wide = vqaddq_s16(pixels_wide, mean_vec);

            // Load noise values into NEON register
            let noise = vld1q_s16(noise_i16.as_ptr());

            // Apply fixed-point scaling: (noise * strength) >> 7
            let scaled = vqrdmulhq_s16(noise, vdupq_n_s16(strength));

            // Add to pixels
            let result = vqaddq_s16(pixels_wide, scaled);

            // Narrow back to u8 with saturation
            let result_u8 = vqmovun_s16(result);

            // Store result
            vst1_u8(data.as_mut_ptr().add(base), result_u8);
        }
    }

    // Handle remaining pixels (scalar)
    if chunks * 8 < len {
        for px in data.iter_mut().skip(chunks * 8) {
            rng_state ^= rng_state >> 12;
            rng_state ^= rng_state << 25;
            rng_state ^= rng_state >> 27;
            let idx = (rng_state as u32) & LUT_MASK;

            // LUT lookup (unit Gaussian in Q8.7 format)
            let noise = lut[idx as usize];

            // Apply scaling: (noise * strength) >> 7
            // Use i32 to prevent overflow
            let scaled = ((noise as i32 * strength as i32) >> 7) as i16;

            // Add to pixel with mean offset
            let px_i16 = *px as i16 + mean_offset;
            let result = px_i16 + scaled;

            *px = result.clamp(0, 255) as u8;
        }
    }
}

/// Fallback implementation for non-ARM platforms
///
/// This scalar implementation is used on platforms without NEON support.
#[cfg(not(target_arch = "aarch64"))]
pub fn apply_gauss_noise_neon(
    data: &mut [u8],
    lut: &[i16; LUT_SIZE],
    strength: i16,
    mean_offset: i16,
) {
    let mut rng_state: u64 = 0xDEADBEEF;

    for px in data.iter_mut() {
        // Generate random index using xorshift*
        rng_state ^= rng_state >> 12;
        rng_state ^= rng_state << 25;
        rng_state ^= rng_state >> 27;
        let idx = (rng_state as u32) & LUT_MASK;

        // LUT lookup (unit Gaussian in Q8.7 format)
        let noise = lut[idx as usize];

        // Apply scaling: (noise * strength) >> 7
        // Use i32 to prevent overflow
        let scaled = ((noise as i32 * strength as i32) >> 7) as i16;

        // Add to pixel with mean offset
        let px_i16 = *px as i16 + mean_offset;
        let result = px_i16 + scaled;

        *px = result.clamp(0, 255) as u8;
    }
}
