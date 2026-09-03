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

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn lut_lookup16(
    v: std::arch::aarch64::uint8x16_t,
    t0: std::arch::aarch64::uint8x16x4_t,
    t1: std::arch::aarch64::uint8x16x4_t,
    t2: std::arch::aarch64::uint8x16x4_t,
    t3: std::arch::aarch64::uint8x16x4_t,
) -> std::arch::aarch64::uint8x16_t {
    use std::arch::aarch64::*;
    let res0 = vqtbl4q_u8(t0, v);
    let res1 = vqtbl4q_u8(t1, vsubq_u8(v, vdupq_n_u8(64)));
    let res2 = vqtbl4q_u8(t2, vsubq_u8(v, vdupq_n_u8(128)));
    let res3 = vqtbl4q_u8(t3, vsubq_u8(v, vdupq_n_u8(192)));
    let mask1 = vcgeq_u8(v, vdupq_n_u8(64));
    let mask2 = vcgeq_u8(v, vdupq_n_u8(128));
    let mask3 = vcgeq_u8(v, vdupq_n_u8(192));
    let mut result = vbslq_u8(mask1, res1, res0);
    result = vbslq_u8(mask2, res2, result);
    vbslq_u8(mask3, res3, result)
}

/// Convert RGB to grayscale while applying a 3-channel LUT in registers.
/// Single memory pass: loads RGB, applies LUT in registers, computes grayscale, stores 1 channel.
#[cfg(target_arch = "aarch64")]
pub unsafe fn lut_to_gray_neon(
    src: &[u8],
    dst: &mut [u8],
    pixel_count: usize,
    luts: &[[u8; 256]; 3],
) {
    use std::arch::aarch64::*;

    if luts[0] == luts[1] && luts[1] == luts[2] {
        let t0 = vld1q_u8_x4(luts[0].as_ptr());
        let t1 = vld1q_u8_x4(luts[0].as_ptr().add(64));
        let t2 = vld1q_u8_x4(luts[0].as_ptr().add(128));
        let t3 = vld1q_u8_x4(luts[0].as_ptr().add(192));

        let r_weight = vdupq_n_u16(77);
        let g_weight = vdupq_n_u16(150);
        let b_weight = vdupq_n_u16(29);
        let round_const = vdupq_n_u16(128);

        let chunks = pixel_count / 16;
        let src_ptr = src.as_ptr();
        let dst_ptr = dst.as_mut_ptr();

        for i in 0..chunks {
            let off = i * 48;
            let rgb = vld3q_u8(src_ptr.add(off));
            let r = lut_lookup16(rgb.0, t0, t1, t2, t3);
            let g = lut_lookup16(rgb.1, t0, t1, t2, t3);
            let b = lut_lookup16(rgb.2, t0, t1, t2, t3);

            // Low 8
            let r_low = vmovl_u8(vget_low_u8(r));
            let g_low = vmovl_u8(vget_low_u8(g));
            let b_low = vmovl_u8(vget_low_u8(b));
            let mut gray_low = vmlaq_u16(round_const, r_weight, r_low);
            gray_low = vmlaq_u16(gray_low, g_weight, g_low);
            gray_low = vmlaq_u16(gray_low, b_weight, b_low);
            let gray8_low = vmovn_u16(vshrq_n_u16(gray_low, 8));

            // High 8
            let r_high = vmovl_u8(vget_high_u8(r));
            let g_high = vmovl_u8(vget_high_u8(g));
            let b_high = vmovl_u8(vget_high_u8(b));
            let mut gray_high = vmlaq_u16(round_const, r_weight, r_high);
            gray_high = vmlaq_u16(gray_high, g_weight, g_high);
            gray_high = vmlaq_u16(gray_high, b_weight, b_high);
            let gray8_high = vmovn_u16(vshrq_n_u16(gray_high, 8));

            let gray16 = vcombine_u8(gray8_low, gray8_high);
            vst1q_u8(dst_ptr.add(i * 16), gray16);
        }

        for j in (chunks * 16)..pixel_count {
            let r = luts[0][*src_ptr.add(j * 3) as usize] as u32;
            let g = luts[0][*src_ptr.add(j * 3 + 1) as usize] as u32;
            let b = luts[0][*src_ptr.add(j * 3 + 2) as usize] as u32;
            *dst_ptr.add(j) = ((77 * r + 150 * g + 29 * b + 128) >> 8) as u8;
        }
        return;
    }


    let r0 = vld1q_u8_x4(luts[0].as_ptr());
    let r1 = vld1q_u8_x4(luts[0].as_ptr().add(64));
    let r2 = vld1q_u8_x4(luts[0].as_ptr().add(128));
    let r3 = vld1q_u8_x4(luts[0].as_ptr().add(192));
    let g0 = vld1q_u8_x4(luts[1].as_ptr());
    let g1 = vld1q_u8_x4(luts[1].as_ptr().add(64));
    let g2 = vld1q_u8_x4(luts[1].as_ptr().add(128));
    let g3 = vld1q_u8_x4(luts[1].as_ptr().add(192));
    let b0 = vld1q_u8_x4(luts[2].as_ptr());
    let b1 = vld1q_u8_x4(luts[2].as_ptr().add(64));
    let b2 = vld1q_u8_x4(luts[2].as_ptr().add(128));
    let b3 = vld1q_u8_x4(luts[2].as_ptr().add(192));

    let r_weight = vdupq_n_u16(77);
    let g_weight = vdupq_n_u16(150);
    let b_weight = vdupq_n_u16(29);
    let round_const = vdupq_n_u16(128);

    let chunks = pixel_count / 16;
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 48;
        let rgb = vld3q_u8(src_ptr.add(off));
        let r = lut_lookup16(rgb.0, r0, r1, r2, r3);
        let g = lut_lookup16(rgb.1, g0, g1, g2, g3);
        let b = lut_lookup16(rgb.2, b0, b1, b2, b3);

        // Low 8
        let r_low = vmovl_u8(vget_low_u8(r));
        let g_low = vmovl_u8(vget_low_u8(g));
        let b_low = vmovl_u8(vget_low_u8(b));
        let mut gray_low = vmlaq_u16(round_const, r_weight, r_low);
        gray_low = vmlaq_u16(gray_low, g_weight, g_low);
        gray_low = vmlaq_u16(gray_low, b_weight, b_low);
        let gray8_low = vmovn_u16(vshrq_n_u16(gray_low, 8));

        // High 8
        let r_high = vmovl_u8(vget_high_u8(r));
        let g_high = vmovl_u8(vget_high_u8(g));
        let b_high = vmovl_u8(vget_high_u8(b));
        let mut gray_high = vmlaq_u16(round_const, r_weight, r_high);
        gray_high = vmlaq_u16(gray_high, g_weight, g_high);
        gray_high = vmlaq_u16(gray_high, b_weight, b_high);
        let gray8_high = vmovn_u16(vshrq_n_u16(gray_high, 8));

        let gray16 = vcombine_u8(gray8_low, gray8_high);
        vst1q_u8(dst_ptr.add(i * 16), gray16);
    }

    // Tail
    for j in (chunks * 16)..pixel_count {
        let r = luts[0][*src_ptr.add(j * 3) as usize] as u32;
        let g = luts[1][*src_ptr.add(j * 3 + 1) as usize] as u32;
        let b = luts[2][*src_ptr.add(j * 3 + 2) as usize] as u32;
        *dst_ptr.add(j) = ((77 * r + 150 * g + 29 * b + 128) >> 8) as u8;
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

#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn lut_to_gray_neon(
    src: &[u8],
    dst: &mut [u8],
    pixel_count: usize,
    luts: &[[u8; 256]; 3],
) {
    for i in 0..pixel_count {
        let r = luts[0][src[i * 3] as usize] as u32;
        let g = luts[1][src[i * 3 + 1] as usize] as u32;
        let b = luts[2][src[i * 3 + 2] as usize] as u32;
        dst[i] = ((77 * r + 150 * g + 29 * b + 128) >> 8) as u8;
    }
}

