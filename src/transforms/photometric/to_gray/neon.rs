// NEON SIMD implementation for RGB to Grayscale
//
// Uses vld3_u8 for RGB deinterleave and fixed-point weights.

/// Convert RGB to grayscale using NEON SIMD
///
/// Processes 8 pixels at a time (24 bytes = 8 RGB pixels).
/// Uses deinterleaved loads (vld3) for efficient RGB separation.
#[cfg(target_arch = "aarch64")]
pub unsafe fn to_gray_neon(src: &[u8], dst: &mut [u8], pixel_count: usize) {
    use std::arch::aarch64::*;

    let pixels_per_iter = 8; // Process 8 pixels at a time
    let mut i = 0;

    // Fixed-point weights (Q8): 0.299≈77, 0.587≈150, 0.114≈29
    let r_weight = vdupq_n_u16(77);
    let g_weight = vdupq_n_u16(150);
    let b_weight = vdupq_n_u16(29);
    let round_const = vdupq_n_u16(128);

    // Process 8 pixels (24 bytes) at a time
    while i + pixels_per_iter <= pixel_count {
        // Load 8 RGB pixels, deinterleave into R, G, B vectors
        // vld3_u8 loads interleaved data and returns 3 separate registers
        let rgb = vld3_u8(src.as_ptr().add(i * 3));

        // Extract individual channels as uint8x8_t
        let r8 = rgb.0;
        let g8 = rgb.1;
        let b8 = rgb.2;

        // Widen to u16 for multiplication
        let r16 = vmovl_u8(r8);
        let g16 = vmovl_u8(g8);
        let b16 = vmovl_u8(b8);

        // Fixed-point: gray = (77*r + 150*g + 29*b + 128) >> 8
        let mut gray = vmlaq_u16(round_const, r_weight, r16);
        gray = vmlaq_u16(gray, g_weight, g16);
        gray = vmlaq_u16(gray, b_weight, b16);

        // Shift right by 8
        let gray_shifted = vshrq_n_u16(gray, 8);

        // Narrow back to u8
        let gray8 = vmovn_u16(gray_shifted);

        // Store 8 grayscale values
        vst1_u8(dst.as_mut_ptr().add(i), gray8);

        i += pixels_per_iter;
    }

    // Handle remaining pixels
    for j in i..pixel_count {
        let r = src[j * 3] as u32;
        let g = src[j * 3 + 1] as u32;
        let b = src[j * 3 + 2] as u32;
        dst[j] = ((77 * r + 150 * g + 29 * b + 128) >> 8) as u8;
    }
}

/// Fallback for non-ARM64 platforms
#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn to_gray_neon(src: &[u8], dst: &mut [u8], pixel_count: usize) {
    for i in 0..pixel_count {
        let r = src[i * 3] as u32;
        let g = src[i * 3 + 1] as u32;
        let b = src[i * 3 + 2] as u32;
        dst[i] = ((77 * r + 150 * g + 29 * b + 128) >> 8) as u8;
    }
}
