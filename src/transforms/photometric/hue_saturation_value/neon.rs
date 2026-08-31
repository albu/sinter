use crate::core::FusableImage;
use crate::transforms::photometric::hue_saturation_value::HueSaturationValue;
use std::arch::aarch64::*;

/// Apply saturation/value scaling only (hue unchanged) — exact hexcone model.
///
/// For hue_shift == 0 the full HSV round-trip simplifies algebraically to
///
///   RGB' = vs*ss*RGB + vs*(1-ss)*V*[1,1,1],  V = max(R,G,B)
///
/// (derived from C = V*S/255 = V - min, m = V - C; verified bit-exact against
/// the brute-force HSV round-trip). This is the SAME model as the hue-shift
/// SIMD path — previously the hue==0 branch used a luma-weighted matrix
/// (0.299/0.587/0.114) that disagreed with it by up to ~50 for sat_scale 1.3.
///
/// Q8.8 fixed-point, 16 pixels per iteration.
#[cfg(target_arch = "aarch64")]
pub(super) unsafe fn apply_satval_neon(image: &mut FusableImage, sat_scale: f32, val_scale: f32) {
    let data = &mut image.data;
    let len = data.len();
    let px = len / 3;
    let chunks = px / 16;

    let m = sat_scale * val_scale;
    let k = val_scale * (1.0 - sat_scale);
    let m_i = (m * 256.0).round() as i16;
    let k_i = (k * 256.0).round() as i16;
    let vm = vdupq_n_s16(m_i);
    let vk = vdupq_n_s16(k_i);
    let round = vdupq_n_s32(128);

    for i in 0..chunks {
        let off = i * 48;
        let rgb = vld3q_u8(data.as_ptr().add(off));
        let vmax = vmaxq_u8(vmaxq_u8(rgb.0, rgb.1), rgb.2);
        let rr = vcombine_u8(
            blend8(
                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(rgb.0))),
                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(vmax))),
                vm, vk, round,
            ),
            blend8(
                vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(rgb.0))),
                vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(vmax))),
                vm, vk, round,
            ),
        );
        let gg = vcombine_u8(
            blend8(
                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(rgb.1))),
                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(vmax))),
                vm, vk, round,
            ),
            blend8(
                vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(rgb.1))),
                vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(vmax))),
                vm, vk, round,
            ),
        );
        let bb = vcombine_u8(
            blend8(
                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(rgb.2))),
                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(vmax))),
                vm, vk, round,
            ),
            blend8(
                vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(rgb.2))),
                vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(vmax))),
                vm, vk, round,
            ),
        );
        vst3q_u8(data.as_mut_ptr().add(off), uint8x16x3_t(rr, gg, bb));
    }

    // Tail: leftover complete pixels (len is a multiple of 3 for RGB).
    let tail_start = chunks * 48;
    let mut i = tail_start;
    while i + 3 <= len {
        let v = data[i].max(data[i + 1]).max(data[i + 2]) as f32;
        for c in 0..3 {
            let val = m * data[i + c] as f32 + k * v;
            data[i + c] = val.clamp(0.0, 255.0).round() as u8;
        }
        i += 3;
    }
}

/// Exact sat/val scaling for the clip regime (ss > 1 or vs > 1).
///
/// When S*ss or V*vs would exceed 255 the simple formula
/// `RGB' = vs*ss*RGB + vs*(1-ss)*V` is wrong (it ignores the saturation clip:
/// C' = min(vs*ss*C, V')). This computes the exact hexcone result from the
/// sorted channels:
///
///   M = max, L = min, U = mid, C = M - L
///   V' = min(M*vs, 255)
///   C' = min(vs*ss*C, V')
///   X' = C' * (U-L)/C
///   m' = V' - C'
///   out = m' + (C' on max channel, X' on mid channel, 0 on min channel)
///
/// Float per 4 pixels; bit-consistent with the hue-shift SIMD path (verified
/// against a brute-force HSV round-trip to max 1).
#[cfg(target_arch = "aarch64")]
pub(super) unsafe fn apply_satval_neon_exact(image: &mut FusableImage, sat_scale: f32, val_scale: f32) {
    let data = &mut image.data;
    let len = data.len();
    let px = len / 3;
    let chunks = px / 16;

    let vs = vdupq_n_f32(val_scale);
    let vss = vdupq_n_f32(sat_scale);
    let v255 = vdupq_n_f32(255.0);
    let zero = vdupq_n_f32(0.0);

    let mut i = 0;
    while i + 16 <= px {
        let rgb = vld3q_u8(data.as_ptr().add(i * 3));
        let (r0, r1, r2, r3) = expand_u8_to_f32(rgb.0);
        let (g0, g1, g2, g3) = expand_u8_to_f32(rgb.1);
        let (b0, b1, b2, b3) = expand_u8_to_f32(rgb.2);
        let (ro0, go0, bo0) = satval_batch(r0, g0, b0, vs, vss, v255, zero);
        let (ro1, go1, bo1) = satval_batch(r1, g1, b1, vs, vss, v255, zero);
        let (ro2, go2, bo2) = satval_batch(r2, g2, b2, vs, vss, v255, zero);
        let (ro3, go3, bo3) = satval_batch(r3, g3, b3, vs, vss, v255, zero);
        let r_out = pack_f32_to_u8(ro0, ro1, ro2, ro3);
        let g_out = pack_f32_to_u8(go0, go1, go2, go3);
        let b_out = pack_f32_to_u8(bo0, bo1, bo2, bo3);
        vst3q_u8(data.as_mut_ptr().add(i * 3), uint8x16x3_t(r_out, g_out, b_out));
        i += 16;
    }

    // Tail: exact scalar (i is a pixel index; each iteration consumes 3 bytes).
    while i < px {
        let off = i * 3;
        let (r, g, b) = (data[off] as f32, data[off + 1] as f32, data[off + 2] as f32);
        let mm = r.max(g).max(b);
        let ll = r.min(g).min(b);
        let uu = r + g + b - mm - ll;
        let cc = mm - ll;
        let vp = (mm * val_scale).clamp(0.0, 255.0);
        let cp = if mm > 0.0 {
            (vp * cc * sat_scale / mm).min(vp)
        } else {
            0.0
        };
        let xp = if cc > 0.0 { cp * (uu - ll) / cc } else { 0.0 };
        let mp = vp - cp;
        let (rm, gm, bm) = (r == mm, g == mm, b == mm);
        let (rl, gl, bl) = (r == ll, g == ll, b == ll);
        data[off] = (mp + if rm { cp } else if rl { 0.0 } else { xp }).clamp(0.0, 255.0).round() as u8;
        data[off + 1] = (mp + if gm { cp } else if gl { 0.0 } else { xp }).clamp(0.0, 255.0).round() as u8;
        data[off + 2] = (mp + if bm { cp } else if bl { 0.0 } else { xp }).clamp(0.0, 255.0).round() as u8;
        i += 1;
    }
}

/// One 4-pixel batch of the exact sat/val transform (see apply_satval_neon_exact).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn satval_batch(
    r: float32x4_t,
    g: float32x4_t,
    b: float32x4_t,
    vs: float32x4_t,
    vss: float32x4_t,
    v255: float32x4_t,
    zero: float32x4_t,
) -> (float32x4_t, float32x4_t, float32x4_t) {
    let m = vmaxq_f32(vmaxq_f32(r, g), b);
    let l = vminq_f32(vminq_f32(r, g), b);
    let c = vsubq_f32(m, l);
    let u = vsubq_f32(vaddq_f32(vaddq_f32(r, g), b), vaddq_f32(m, l));
    let vp = vminq_f32(vmulq_f32(m, vs), v255);
    // C' = V' * S' / 255 = min(V' * C * ss / M, V')  (V' clip handled by vp).
    let cp = vminq_f32(vmulq_f32(vmulq_f32(vmulq_f32(vp, c), vss), vrecip(m)), vp);
    let xp = vmulq_f32(vmulq_f32(cp, vsubq_f32(u, l)), vrecip(c));
    let mp = vsubq_f32(vp, cp);

    let max_r = vceqq_f32(r, m);
    let max_g = vceqq_f32(g, m);
    let max_b = vceqq_f32(b, m);
    let min_r = vceqq_f32(r, l);
    let min_g = vceqq_f32(g, l);
    let min_b = vceqq_f32(b, l);

    let ro = vaddq_f32(mp, vbslq_f32(max_r, cp, vbslq_f32(min_r, zero, xp)));
    let go = vaddq_f32(mp, vbslq_f32(max_g, cp, vbslq_f32(min_g, zero, xp)));
    let bo = vaddq_f32(mp, vbslq_f32(max_b, cp, vbslq_f32(min_b, zero, xp)));
    (ro, go, bo)
}

/// Per-channel Q8.8 blend: (m*ch + k*V + 128) >> 8, saturated to u8.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn blend8(
    ch: int16x8_t,
    vmax: int16x8_t,
    vm: int16x8_t,
    vk: int16x8_t,
    round: int32x4_t,
) -> uint8x8_t {
    let lo = vshrn_n_s32(
        vaddq_s32(
            vmlal_s16(vmull_s16(vget_low_s16(ch), vget_low_s16(vm)), vget_low_s16(vmax), vget_low_s16(vk)),
            round,
        ),
        8,
    );
    let hi = vshrn_n_s32(
        vaddq_s32(
            vmlal_s16(vmull_s16(vget_high_s16(ch), vget_high_s16(vm)), vget_high_s16(vmax), vget_high_s16(vk)),
            round,
        ),
        8,
    );
    vqmovun_s16(vcombine_s16(lo, hi))
}

/// Execute using NEON SIMD implementation
///
/// This processes 16 pixels at a time using floating point arithmetic for
/// Hue calculation and saturation/value scaling.
///
/// It follows the logic of the fast scalar implementation:
/// 1. RGB -> HSV (Hue as degrees, S/V as 0-255)
/// 2. Apply shift/scale
/// 3. HSV -> RGB
pub(super) fn execute_fast_simd(hsv: &HueSaturationValue, image: &mut FusableImage) {
    // Only RGB images are supported for hue rotation.
    // Grayscale falls back to scalar implementation.
    if image.channels != 3 {
        super::fast_impl::execute_fast_grayscale(hsv, image);
        return;
    }

    unsafe {
        let mut ptr = image.data.as_mut_ptr();
        let len = image.data.len();
        let pixel_count = len / 3;
        let mut i = 0;

        let h_shift = vdupq_n_f32(hsv.hue_shift);
        let s_scale = vdupq_n_f32(hsv.sat_scale);
        let v_scale = vdupq_n_f32(hsv.val_scale);

        // Process 16 pixels at a time
        while i + 16 <= pixel_count {
            // Load 3 channels, interleaved (R0 G0 B0, R1 G1 B1...)
            // vld3q_u8 deinterleaves them into 3 vectors of 16 elements
            let rgb = vld3q_u8(ptr);

            // Expand u8x16 to f32x4 (4 batches)
            let (r0, r1, r2, r3) = expand_u8_to_f32(rgb.0);
            let (g0, g1, g2, g3) = expand_u8_to_f32(rgb.1);
            let (b0, b1, b2, b3) = expand_u8_to_f32(rgb.2);

            let (ro0, go0, bo0) = process_batch(r0, g0, b0, h_shift, s_scale, v_scale);
            let (ro1, go1, bo1) = process_batch(r1, g1, b1, h_shift, s_scale, v_scale);
            let (ro2, go2, bo2) = process_batch(r2, g2, b2, h_shift, s_scale, v_scale);
            let (ro3, go3, bo3) = process_batch(r3, g3, b3, h_shift, s_scale, v_scale);

            // Pack f32x4 back to u8x16
            let r_out = pack_f32_to_u8(ro0, ro1, ro2, ro3);
            let g_out = pack_f32_to_u8(go0, go1, go2, go3);
            let b_out = pack_f32_to_u8(bo0, bo1, bo2, bo3);

            vst3q_u8(ptr, uint8x16x3_t(r_out, g_out, b_out));

            ptr = ptr.add(16 * 3);
            i += 16;
        }

        // Handle remaining pixels using scalar implementation
        if i < pixel_count {
            let executed_len = i * 3;
            let remainder_slice = &mut image.data[executed_len..];
            // We need to create a temporary FusableImage wrapper for the slice
            // Note: The dimensions don't strictly matter for pixel-wise ops,
            // but we set width=remainder, height=1 to be safe.
            let mut temp_img = FusableImage::new(remainder_slice, pixel_count - i, 1, 3);
            super::fast_impl::execute_fast(hsv, &mut temp_img);
        }
    }
}

/// Expand uint8x16_t to four float32x4_t vectors
#[inline(always)]
unsafe fn expand_u8_to_f32(v: uint8x16_t) -> (float32x4_t, float32x4_t, float32x4_t, float32x4_t) {
    // u8x16 -> u16x8
    let low_u16 = vmovl_u8(vget_low_u8(v));
    let high_u16 = vmovl_u8(vget_high_u8(v));

    // u16x8 -> u32x4
    let ll_u32 = vmovl_u16(vget_low_u16(low_u16));
    let lh_u32 = vmovl_u16(vget_high_u16(low_u16));
    let hl_u32 = vmovl_u16(vget_low_u16(high_u16));
    let hh_u32 = vmovl_u16(vget_high_u16(high_u16));

    // u32x4 -> f32x4
    (
        vcvtq_f32_u32(ll_u32),
        vcvtq_f32_u32(lh_u32),
        vcvtq_f32_u32(hl_u32),
        vcvtq_f32_u32(hh_u32),
    )
}

/// Pack four float32x4_t vectors into one uint8x16_t
#[inline(always)]
unsafe fn pack_f32_to_u8(
    v0: float32x4_t,
    v1: float32x4_t,
    v2: float32x4_t,
    v3: float32x4_t,
) -> uint8x16_t {
    // f32 -> u32 (Round to nearest)
    let u0 = vcvtaq_u32_f32(v0);
    let u1 = vcvtaq_u32_f32(v1);
    let u2 = vcvtaq_u32_f32(v2);
    let u3 = vcvtaq_u32_f32(v3);

    // u32 -> u16 (Saturating narrow)
    let u16_0 = vcombine_u16(vqmovn_u32(u0), vqmovn_u32(u1));
    let u16_1 = vcombine_u16(vqmovn_u32(u2), vqmovn_u32(u3));

    // u16 -> u8 (Saturating narrow)
    vcombine_u8(vqmovn_u16(u16_0), vqmovn_u16(u16_1))
}

#[inline(always)]
unsafe fn vrecip(v: float32x4_t) -> float32x4_t {
    let est = vrecpeq_f32(v);
    let step = vrecpsq_f32(v, est);
    let est2 = vmulq_f32(est, step);
    let step2 = vrecpsq_f32(v, est2);
    vmulq_f32(est2, step2)
}

/// Process a batch of 4 pixels (in float32)
#[inline(always)]
unsafe fn process_batch(
    r: float32x4_t,
    g: float32x4_t,
    b: float32x4_t,
    h_shift: float32x4_t,
    s_scale: float32x4_t,
    v_scale: float32x4_t,
) -> (float32x4_t, float32x4_t, float32x4_t) {
    const INV_255: f32 = 1.0 / 255.0;
    const INV_60: f32 = 1.0 / 60.0;

    let zeroes = vdupq_n_f32(0.0);
    let ones = vdupq_n_f32(1.0);
    let two_five_five = vdupq_n_f32(255.0);
    let sixty = vdupq_n_f32(60.0);
    let three_sixty = vdupq_n_f32(360.0);

    // --- RGB to HSV ---
    let max = vmaxq_f32(vmaxq_f32(r, g), b);
    let min = vminq_f32(vminq_f32(r, g), b);
    let delta = vsubq_f32(max, min);

    // Value (0..255)
    let v_old = max;

    // Saturation (0..255)
    // S = (Delta * 255) / Max
    let max_gt_0 = vcgtq_f32(max, zeroes);
    let inv_max = vrecip(max);
    let s_val = vmulq_f32(vmulq_f32(delta, two_five_five), inv_max);
    let s_old = vbslq_f32(max_gt_0, s_val, zeroes);

    // Hue (0..360)
    let delta_is_zero = vceqq_f32(delta, zeroes);
    let inv_delta = vrecip(delta);

    // Terms for Hue calculation
    let max_is_r = vceqq_f32(max, r);
    let max_is_g = vceqq_f32(max, g);

    // Compute all terms
    let term_r = vmulq_f32(vsubq_f32(g, b), inv_delta);
    let term_g = vaddq_f32(vmulq_f32(vsubq_f32(b, r), inv_delta), vdupq_n_f32(2.0));
    let term_b = vaddq_f32(vmulq_f32(vsubq_f32(r, g), inv_delta), vdupq_n_f32(4.0));

    // Select based on Max channel
    let h_raw = vbslq_f32(max_is_r, term_r, vbslq_f32(max_is_g, term_g, term_b));
    let h_scaled = vmulq_f32(h_raw, sixty);

    // Fix negative hue (e.g. R=Max, G < B -> term < 0)
    let h_lt_0 = vcltq_f32(h_scaled, zeroes);
    let h_pos = vbslq_f32(h_lt_0, vaddq_f32(h_scaled, three_sixty), h_scaled);

    let h_old = vbslq_f32(delta_is_zero, zeroes, h_pos);

    // --- Adjust ---

    // Hue Shift
    let h_shifted = vaddq_f32(h_old, h_shift);
    // Modulo 360
    let h_ge_360 = vcgeq_f32(h_shifted, three_sixty);
    let h_mod = vbslq_f32(h_ge_360, vsubq_f32(h_shifted, three_sixty), h_shifted);
    let h_lt_0_final = vcltq_f32(h_mod, zeroes);
    let h_final = vbslq_f32(h_lt_0_final, vaddq_f32(h_mod, three_sixty), h_mod);

    // Saturation Scale
    let s_scaled = vmulq_f32(s_old, s_scale);
    let s_final = vminq_f32(vmaxq_f32(s_scaled, zeroes), two_five_five);

    // Value Scale
    let v_scaled = vmulq_f32(v_old, v_scale);
    let v_final = vminq_f32(vmaxq_f32(v_scaled, zeroes), two_five_five);

    // --- HSV to RGB ---

    // C = V * S / 255
    let c = vmulq_n_f32(vmulq_f32(v_final, s_final), INV_255);

    // m = V - C
    let m = vsubq_f32(v_final, c);

    // X = C * (1 - abs((H / 60) % 2 - 1))
    let hp = vmulq_n_f32(h_final, INV_60);

    // hp % 2
    let hp_half = vmulq_n_f32(hp, 0.5);
    let hp_half_floor = vcvtq_f32_s32(vcvtq_s32_f32(hp_half)); // floor(hp/2)
    let hp_mod_2 = vsubq_f32(hp, vmulq_n_f32(hp_half_floor, 2.0));

    let x_term = vabsq_f32(vsubq_f32(hp_mod_2, ones));
    let x = vmulq_f32(c, vsubq_f32(ones, x_term));

    // Sector reconstruction
    // 0 <= hp < 1: C, X, 0
    // 1 <= hp < 2: X, C, 0
    // 2 <= hp < 3: 0, C, X
    // 3 <= hp < 4: 0, X, C
    // 4 <= hp < 5: X, 0, C
    // 5 <= hp < 6: C, 0, X

    let lt_1 = vcltq_f32(hp, ones);
    let lt_2 = vcltq_f32(hp, vdupq_n_f32(2.0));
    let lt_3 = vcltq_f32(hp, vdupq_n_f32(3.0));
    let lt_4 = vcltq_f32(hp, vdupq_n_f32(4.0));
    let lt_5 = vcltq_f32(hp, vdupq_n_f32(5.0));

    // R
    let r_temp = vbslq_f32(
        lt_1,
        c,
        vbslq_f32(
            lt_2,
            x,
            vbslq_f32(lt_3, zeroes, vbslq_f32(lt_4, zeroes, vbslq_f32(lt_5, x, c))),
        ),
    );

    // G
    let g_temp = vbslq_f32(
        lt_1,
        x,
        vbslq_f32(
            lt_2,
            c,
            vbslq_f32(lt_3, c, vbslq_f32(lt_4, x, vbslq_f32(lt_5, zeroes, zeroes))),
        ),
    );

    // B
    let b_temp = vbslq_f32(
        lt_1,
        zeroes,
        vbslq_f32(
            lt_2,
            zeroes,
            vbslq_f32(lt_3, x, vbslq_f32(lt_4, c, vbslq_f32(lt_5, c, x))),
        ),
    );

    (
        vaddq_f32(r_temp, m),
        vaddq_f32(g_temp, m),
        vaddq_f32(b_temp, m),
    )
}
