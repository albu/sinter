// NEON SIMD implementation for RGB Shift
//
// Shifts each color channel by a specified amount using NEON.

/// Apply RGB shift using NEON SIMD
///
/// Processes 8 pixels at a time (24 bytes = 8 RGB pixels).
/// Uses deinterleaved loads (vld3) and saturating addition.
///
/// # Safety
/// Caller must ensure that:
/// - `data.len()` is a multiple of 3 (valid RGB data)
/// - All pointer arithmetic within the function stays within bounds
#[cfg(target_arch = "aarch64")]
pub unsafe fn rgb_shift_neon(data: &mut [u8], r_shift: i16, g_shift: i16, b_shift: i16) {
    use std::arch::aarch64::*;

    let len = data.len();
    let mut i = 0;

    let r_shift_s16 = vdupq_n_s16(r_shift);
    let g_shift_s16 = vdupq_n_s16(g_shift);
    let b_shift_s16 = vdupq_n_s16(b_shift);

    // Process 8 pixels (24 bytes) at a time
    while i + 24 <= len {
        let rgb = vld3_u8(data.as_ptr().add(i));

        // Widen u8 -> u16 -> s16 and add signed shift
        let r_u16 = vmovl_u8(rgb.0);
        let g_u16 = vmovl_u8(rgb.1);
        let b_u16 = vmovl_u8(rgb.2);

        let r_s16 = vaddq_s16(vreinterpretq_s16_u16(r_u16), r_shift_s16);
        let g_s16 = vaddq_s16(vreinterpretq_s16_u16(g_u16), g_shift_s16);
        let b_s16 = vaddq_s16(vreinterpretq_s16_u16(b_u16), b_shift_s16);

        // Saturating narrow s16 -> u8 (clamps [0, 255])
        let r_out = vqmovun_s16(r_s16);
        let g_out = vqmovun_s16(g_s16);
        let b_out = vqmovun_s16(b_s16);

        let result = uint8x8x3_t(r_out, g_out, b_out);
        vst3_u8(data.as_mut_ptr().add(i), result);

        i += 24;
    }

    // Handle remaining pixels (must be multiple of 3 for RGB)
    while i + 3 <= len {
        // Use scalar for remaining pixels
        data[i] = (data[i] as i16 + r_shift).clamp(0, 255) as u8;
        data[i + 1] = (data[i + 1] as i16 + g_shift).clamp(0, 255) as u8;
        data[i + 2] = (data[i + 2] as i16 + b_shift).clamp(0, 255) as u8;
        i += 3;
    }
}

/// Fallback implementation for non-ARM64 platforms
#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn rgb_shift_neon(data: &mut [u8], r_shift: i16, g_shift: i16, b_shift: i16) {
    let len = data.len();
    let mut i = 0;

    while i + 3 <= len {
        data[i] = (data[i] as i16 + r_shift).clamp(0, 255) as u8;
        data[i + 1] = (data[i + 1] as i16 + g_shift).clamp(0, 255) as u8;
        data[i + 2] = (data[i + 2] as i16 + b_shift).clamp(0, 255) as u8;
        i += 3;
    }
}
