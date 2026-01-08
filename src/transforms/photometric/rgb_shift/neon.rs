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
pub unsafe fn rgb_shift_neon(data: &mut [u8], r_shift: i8, g_shift: i8, b_shift: i8) {
    use std::arch::aarch64::*;

    let pixels_per_iter = 8; // Process 8 pixels at a time
    let len = data.len();
    let mut i = 0;

    // Broadcast shift values to all lanes
    let r_shift_vec = vdup_n_u8(r_shift as u8);
    let g_shift_vec = vdup_n_u8(g_shift as u8);
    let b_shift_vec = vdup_n_u8(b_shift as u8);

    // Process 8 pixels (24 bytes) at a time
    while i + 24 <= len {
        // Load 8 RGB pixels, deinterleave into R, G, B vectors
        // SAFETY: Loop condition ensures i + 24 <= len, so we have 24 valid bytes
        let rgb = vld3_u8(data.as_ptr().add(i));

        // Apply shifts with saturating addition
        // vqadd_u8 saturates at 0 and 255
        let r_out = vqadd_u8(rgb.0, r_shift_vec);
        let g_out = vqadd_u8(rgb.1, g_shift_vec);
        let b_out = vqadd_u8(rgb.2, b_shift_vec);

        // Store back interleaved - need to construct uint8x8x3_t
        // SAFETY: i is valid for 24 bytes as ensured by loop condition
        let result = uint8x8x3_t(r_out, g_out, b_out);
        vst3_u8(data.as_mut_ptr().add(i), result);

        i += 24;
    }

    // Handle remaining pixels (must be multiple of 3 for RGB)
    while i + 3 <= len {
        // Use scalar for remaining pixels
        data[i] = (data[i] as i16 + r_shift as i16).clamp(0, 255) as u8;
        data[i + 1] = (data[i + 1] as i16 + g_shift as i16).clamp(0, 255) as u8;
        data[i + 2] = (data[i + 2] as i16 + b_shift as i16).clamp(0, 255) as u8;
        i += 3;
    }
}

/// Fallback implementation for non-ARM64 platforms
#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn rgb_shift_neon(data: &mut [u8], r_shift: i8, g_shift: i8, b_shift: i8) {
    let len = data.len();
    let mut i = 0;

    while i + 3 <= len {
        data[i] = (data[i] as i16 + r_shift as i16).clamp(0, 255) as u8;
        data[i + 1] = (data[i + 1] as i16 + g_shift as i16).clamp(0, 255) as u8;
        data[i + 2] = (data[i + 2] as i16 + b_shift as i16).clamp(0, 255) as u8;
        i += 3;
    }
}
