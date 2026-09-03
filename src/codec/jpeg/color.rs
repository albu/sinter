/// Fast fixed-point YCbCr to RGB conversion with NEON SIMD acceleration

#[inline(always)]
pub fn clamp_u8(val: i32) -> u8 {
    if val <= 0 {
        0
    } else if val >= 255 {
        255
    } else {
        val as u8
    }
}

/// Convert a single Y, Cb, Cr triplet to RGB
#[inline(always)]
pub fn ycbcr_to_rgb(y: u8, cb: u8, cr: u8) -> (u8, u8, u8) {
    let y_val = y as i32;
    let cb_shift = (cb as i32) - 128;
    let cr_shift = (cr as i32) - 128;

    let r = y_val + ((91881 * cr_shift + 32768) >> 16);
    let g = y_val - ((22554 * cb_shift + 46802 * cr_shift + 32768) >> 16);
    let b = y_val + ((116130 * cb_shift + 32768) >> 16);

    (clamp_u8(r), clamp_u8(g), clamp_u8(b))
}

/// Convert an array of Y, Cb, Cr pixels to interleaved RGB [R, G, B, R, G, B, ...]
pub fn ycbcr_to_rgb_slice(y_plane: &[u8], cb_plane: &[u8], cr_plane: &[u8], rgb_out: &mut [u8]) {
    assert_eq!(y_plane.len(), cb_plane.len());
    assert_eq!(y_plane.len(), cr_plane.len());
    assert_eq!(y_plane.len() * 3, rgb_out.len());

    let count = y_plane.len();
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        while i + 8 <= count {
            let y_raw = vld1_u8(y_plane.as_ptr().add(i));
            let cb_raw = vld1_u8(cb_plane.as_ptr().add(i));
            let cr_raw = vld1_u8(cr_plane.as_ptr().add(i));

            let y_16 = vreinterpretq_s16_u16(vmovl_u8(y_raw));
            let cb_16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(cb_raw)), vdupq_n_s16(128));
            let cr_16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(cr_raw)), vdupq_n_s16(128));

            // Fixed point arithmetic on 8 pixels:
            // R = Y + 1.402 * Cr
            // B = Y + 1.772 * Cb
            // G = Y - 0.344136 * Cb - 0.714136 * Cr
            let r_low = vaddq_s32(vshll_n_s16(vget_low_s16(y_16), 16), vmulq_n_s32(vmovl_s16(vget_low_s16(cr_16)), 91881));
            let r_high = vaddq_s32(vshll_n_s16(vget_high_s16(y_16), 16), vmulq_n_s32(vmovl_s16(vget_high_s16(cr_16)), 91881));

            let b_low = vaddq_s32(vshll_n_s16(vget_low_s16(y_16), 16), vmulq_n_s32(vmovl_s16(vget_low_s16(cb_16)), 116130));
            let b_high = vaddq_s32(vshll_n_s16(vget_high_s16(y_16), 16), vmulq_n_s32(vmovl_s16(vget_high_s16(cb_16)), 116130));

            let g_low = vsubq_s32(
                vshll_n_s16(vget_low_s16(y_16), 16),
                vaddq_s32(
                    vmulq_n_s32(vmovl_s16(vget_low_s16(cb_16)), 22554),
                    vmulq_n_s32(vmovl_s16(vget_low_s16(cr_16)), 46802),
                ),
            );
            let g_high = vsubq_s32(
                vshll_n_s16(vget_high_s16(y_16), 16),
                vaddq_s32(
                    vmulq_n_s32(vmovl_s16(vget_high_s16(cb_16)), 22554),
                    vmulq_n_s32(vmovl_s16(vget_high_s16(cr_16)), 46802),
                ),
            );

            let r_s16 = vcombine_s16(vrshrn_n_s32(r_low, 16), vrshrn_n_s32(r_high, 16));
            let g_s16 = vcombine_s16(vrshrn_n_s32(g_low, 16), vrshrn_n_s32(g_high, 16));
            let b_s16 = vcombine_s16(vrshrn_n_s32(b_low, 16), vrshrn_n_s32(b_high, 16));

            let r_u8 = vqmovun_s16(r_s16);
            let g_u8 = vqmovun_s16(g_s16);
            let b_u8 = vqmovun_s16(b_s16);

            let rgb_vec = uint8x8x3_t(r_u8, g_u8, b_u8);
            vst3_u8(rgb_out.as_mut_ptr().add(i * 3), rgb_vec);

            i += 8;
        }
    }

    // Scalar fallback for remaining pixels
    while i < count {
        let (r, g, b) = ycbcr_to_rgb(y_plane[i], cb_plane[i], cr_plane[i]);
        rgb_out[i * 3] = r;
        rgb_out[i * 3 + 1] = g;
        rgb_out[i * 3 + 2] = b;
        i += 1;
    }
}

/// Convert exactly 16 pixels of Y, Cb, Cr to interleaved RGB using NEON vector registers
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn ycbcr_to_rgb_16(y_ptr: *const u8, cb_ptr: *const u8, cr_ptr: *const u8, rgb_ptr: *mut u8) {
    use std::arch::aarch64::*;

    // Process first 8 pixels
    let y_raw0 = vld1_u8(y_ptr);
    let cb_raw0 = vld1_u8(cb_ptr);
    let cr_raw0 = vld1_u8(cr_ptr);

    let y_16_0 = vreinterpretq_s16_u16(vmovl_u8(y_raw0));
    let cb_16_0 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(cb_raw0)), vdupq_n_s16(128));
    let cr_16_0 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(cr_raw0)), vdupq_n_s16(128));

    let r_low0 = vaddq_s32(vshll_n_s16(vget_low_s16(y_16_0), 16), vmulq_n_s32(vmovl_s16(vget_low_s16(cr_16_0)), 91881));
    let r_high0 = vaddq_s32(vshll_n_s16(vget_high_s16(y_16_0), 16), vmulq_n_s32(vmovl_s16(vget_high_s16(cr_16_0)), 91881));
    let b_low0 = vaddq_s32(vshll_n_s16(vget_low_s16(y_16_0), 16), vmulq_n_s32(vmovl_s16(vget_low_s16(cb_16_0)), 116130));
    let b_high0 = vaddq_s32(vshll_n_s16(vget_high_s16(y_16_0), 16), vmulq_n_s32(vmovl_s16(vget_high_s16(cb_16_0)), 116130));

    let g_low0 = vsubq_s32(
        vshll_n_s16(vget_low_s16(y_16_0), 16),
        vaddq_s32(
            vmulq_n_s32(vmovl_s16(vget_low_s16(cb_16_0)), 22554),
            vmulq_n_s32(vmovl_s16(vget_low_s16(cr_16_0)), 46802),
        ),
    );
    let g_high0 = vsubq_s32(
        vshll_n_s16(vget_high_s16(y_16_0), 16),
        vaddq_s32(
            vmulq_n_s32(vmovl_s16(vget_high_s16(cb_16_0)), 22554),
            vmulq_n_s32(vmovl_s16(vget_high_s16(cr_16_0)), 46802),
        ),
    );

    let r_s16_0 = vcombine_s16(vrshrn_n_s32(r_low0, 16), vrshrn_n_s32(r_high0, 16));
    let g_s16_0 = vcombine_s16(vrshrn_n_s32(g_low0, 16), vrshrn_n_s32(g_high0, 16));
    let b_s16_0 = vcombine_s16(vrshrn_n_s32(b_low0, 16), vrshrn_n_s32(b_high0, 16));

    let rgb_vec0 = uint8x8x3_t(vqmovun_s16(r_s16_0), vqmovun_s16(g_s16_0), vqmovun_s16(b_s16_0));
    vst3_u8(rgb_ptr, rgb_vec0);

    // Process second 8 pixels
    let y_raw1 = vld1_u8(y_ptr.add(8));
    let cb_raw1 = vld1_u8(cb_ptr.add(8));
    let cr_raw1 = vld1_u8(cr_ptr.add(8));

    let y_16_1 = vreinterpretq_s16_u16(vmovl_u8(y_raw1));
    let cb_16_1 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(cb_raw1)), vdupq_n_s16(128));
    let cr_16_1 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(cr_raw1)), vdupq_n_s16(128));

    let r_low1 = vaddq_s32(vshll_n_s16(vget_low_s16(y_16_1), 16), vmulq_n_s32(vmovl_s16(vget_low_s16(cr_16_1)), 91881));
    let r_high1 = vaddq_s32(vshll_n_s16(vget_high_s16(y_16_1), 16), vmulq_n_s32(vmovl_s16(vget_high_s16(cr_16_1)), 91881));
    let b_low1 = vaddq_s32(vshll_n_s16(vget_low_s16(y_16_1), 16), vmulq_n_s32(vmovl_s16(vget_low_s16(cb_16_1)), 116130));
    let b_high1 = vaddq_s32(vshll_n_s16(vget_high_s16(y_16_1), 16), vmulq_n_s32(vmovl_s16(vget_high_s16(cb_16_1)), 116130));

    let g_low1 = vsubq_s32(
        vshll_n_s16(vget_low_s16(y_16_1), 16),
        vaddq_s32(
            vmulq_n_s32(vmovl_s16(vget_low_s16(cb_16_1)), 22554),
            vmulq_n_s32(vmovl_s16(vget_low_s16(cr_16_1)), 46802),
        ),
    );
    let g_high1 = vsubq_s32(
        vshll_n_s16(vget_high_s16(y_16_1), 16),
        vaddq_s32(
            vmulq_n_s32(vmovl_s16(vget_high_s16(cb_16_1)), 22554),
            vmulq_n_s32(vmovl_s16(vget_high_s16(cr_16_1)), 46802),
        ),
    );

    let r_s16_1 = vcombine_s16(vrshrn_n_s32(r_low1, 16), vrshrn_n_s32(r_high1, 16));
    let g_s16_1 = vcombine_s16(vrshrn_n_s32(g_low1, 16), vrshrn_n_s32(g_high1, 16));
    let b_s16_1 = vcombine_s16(vrshrn_n_s32(b_low1, 16), vrshrn_n_s32(b_high1, 16));

    let rgb_vec1 = uint8x8x3_t(vqmovun_s16(r_s16_1), vqmovun_s16(g_s16_1), vqmovun_s16(b_s16_1));
    vst3_u8(rgb_ptr.add(24), rgb_vec1);
}
