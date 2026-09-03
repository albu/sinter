// NEON SIMD optimization for resize
//
// Supports both nearest-neighbor and bilinear interpolation.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use super::ResizeInterpolation;

// Minimum dimensions to use SIMD (avoid overhead for small images)
const SIMD_MIN_WIDTH: usize = 16;
const SIMD_MIN_HEIGHT: usize = 16;

/// Apply resize using NEON SIMD optimization
///
/// # Safety
/// - `src` and `dst` must be valid for reads/writes of their respective sizes
/// - Coordinates must be within bounds
pub unsafe fn resize_neon(
    src: &[u8],
    dst: &mut [u8],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    channels: usize,
    interpolation: ResizeInterpolation,
) {
    match interpolation {
        ResizeInterpolation::Nearest => {
            resize_nearest(
                src, dst, src_width, src_height, dst_width, dst_height, channels,
            );
        }
        ResizeInterpolation::Bilinear
        | ResizeInterpolation::Bicubic
        | ResizeInterpolation::Lanczos4 => {
            // Bicubic and Lanczos4 fall back to bilinear for now
            resize_bilinear(
                src, dst, src_width, src_height, dst_width, dst_height, channels,
            );
        }
    }
}

/// Fused resize + LUT application
pub unsafe fn resize_with_lut_neon(
    src: &[u8],
    dst: &mut [u8],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    channels: usize,
    interpolation: ResizeInterpolation,
    luts: &[[u8; 256]; 3],
) {
    if interpolation == ResizeInterpolation::Nearest
        && channels == 3
        && src_width == dst_width * 2
        && src_height == dst_height * 2
    {
        resize_nearest_down2_rgb_lut_neon(src, dst, src_width, src_height, dst_width, dst_height, luts);
        return;
    }

    if (interpolation == ResizeInterpolation::Bilinear || interpolation == ResizeInterpolation::Bicubic)
        && channels == 3
        && src_width == dst_width * 2
        && src_height == dst_height * 2
    {
        resize_bilinear_down2_rgb_lut_neon(src, dst, src_width, src_height, dst_width, dst_height, luts);
        return;
    }

    // Default: resize first, then apply LUT in place
    resize_neon(src, dst, src_width, src_height, dst_width, dst_height, channels, interpolation);
    let mut view = crate::core::FusableImage::new(dst, dst_width, dst_height, channels);
    if luts[0] == luts[1] && luts[1] == luts[2] {
        crate::transforms::runtime::lut::LutExecutor::apply(&mut view, &luts[0]);
    } else if channels == 3 {
        crate::transforms::runtime::lut::LutExecutor::apply_rgb_luts(&mut view, luts);
    }
}



// ============================================================================
// Nearest-Neighbor Interpolation
// ============================================================================

fn resize_nearest(
    src: &[u8],
    dst: &mut [u8],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    channels: usize,
) {
    unsafe {
        #[cfg(target_arch = "aarch64")]
        if dst_width >= SIMD_MIN_WIDTH && dst_height >= SIMD_MIN_HEIGHT {
            if channels == 1 {
                if src_width == dst_width * 2 && src_height == dst_height * 2 {
                    resize_nearest_down2_gray_neon(src, dst, src_width, src_height, dst_width, dst_height);
                    return;
                }
                resize_nearest_gray_neon(src, dst, src_width, src_height, dst_width, dst_height);
                return;
            } else if channels == 3 {
                if src_width == dst_width * 2 && src_height == dst_height * 2 {
                    resize_nearest_down2_rgb_neon(src, dst, src_width, src_height, dst_width, dst_height);
                    return;
                }
                resize_nearest_rgb_neon(src, dst, src_width, src_height, dst_width, dst_height);
                return;
            }
        }

        resize_nearest_scalar(
            src, dst, src_width, src_height, dst_width, dst_height, channels,
        );
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn resize_nearest_down2_gray_neon(
    src: &[u8],
    dst: &mut [u8],
    src_width: usize,
    _src_height: usize,
    dst_width: usize,
    dst_height: usize,
) {
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let src_stride = src_width;
    let dst_stride = dst_width;

    for y in 0..dst_height {
        let s_row = src_ptr.add(y * 2 * src_stride);
        let d_row = dst_ptr.add(y * dst_stride);
        let mut x = 0;
        while x + 16 <= dst_width {
            let pair = vld2q_u8(s_row.add(x * 2));
            vst1q_u8(d_row.add(x), pair.0);
            x += 16;
        }
        while x < dst_width {
            *d_row.add(x) = *s_row.add(x * 2);
            x += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn resize_nearest_down2_rgb_neon(
    src: &[u8],
    dst: &mut [u8],
    src_width: usize,
    _src_height: usize,
    dst_width: usize,
    dst_height: usize,
) {
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let src_stride = src_width * 3;
    let dst_stride = dst_width * 3;

    for y in 0..dst_height {
        let s_row = src_ptr.add(y * 2 * src_stride);
        let d_row = dst_ptr.add(y * dst_stride);
        let mut x = 0;
        while x + 4 <= dst_width {
            let in_x = x * 6;
            let d = d_row.add(x * 3);
            let s = s_row.add(in_x);

            *d = *s;
            *d.add(1) = *s.add(1);
            *d.add(2) = *s.add(2);

            *d.add(3) = *s.add(6);
            *d.add(4) = *s.add(7);
            *d.add(5) = *s.add(8);

            *d.add(6) = *s.add(12);
            *d.add(7) = *s.add(13);
            *d.add(8) = *s.add(14);

            *d.add(9) = *s.add(18);
            *d.add(10) = *s.add(19);
            *d.add(11) = *s.add(20);

            x += 4;
        }
        while x < dst_width {
            let in_x = x * 6;
            let d = d_row.add(x * 3);
            let s = s_row.add(in_x);
            *d = *s;
            *d.add(1) = *s.add(1);
            *d.add(2) = *s.add(2);
            x += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn resize_nearest_down2_rgb_lut_neon(
    src: &[u8],
    dst: &mut [u8],
    src_width: usize,
    _src_height: usize,
    dst_width: usize,
    dst_height: usize,
    luts: &[[u8; 256]; 3],
) {
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let src_stride = src_width * 3;
    let dst_stride = dst_width * 3;
    let lut_r = luts[0].as_ptr();
    let lut_g = luts[1].as_ptr();
    let lut_b = luts[2].as_ptr();

    for y in 0..dst_height {
        let s_row = src_ptr.add(y * 2 * src_stride);
        let d_row = dst_ptr.add(y * dst_stride);
        let mut x = 0;
        while x + 4 <= dst_width {
            let in_x = x * 6;
            let d = d_row.add(x * 3);
            let s = s_row.add(in_x);

            *d = *lut_r.add(*s as usize);
            *d.add(1) = *lut_g.add(*s.add(1) as usize);
            *d.add(2) = *lut_b.add(*s.add(2) as usize);

            *d.add(3) = *lut_r.add(*s.add(6) as usize);
            *d.add(4) = *lut_g.add(*s.add(7) as usize);
            *d.add(5) = *lut_b.add(*s.add(8) as usize);

            *d.add(6) = *lut_r.add(*s.add(12) as usize);
            *d.add(7) = *lut_g.add(*s.add(13) as usize);
            *d.add(8) = *lut_b.add(*s.add(14) as usize);

            *d.add(9) = *lut_r.add(*s.add(18) as usize);
            *d.add(10) = *lut_g.add(*s.add(19) as usize);
            *d.add(11) = *lut_b.add(*s.add(20) as usize);

            x += 4;
        }
        while x < dst_width {
            let in_x = x * 6;
            let d = d_row.add(x * 3);
            let s = s_row.add(in_x);
            *d = *lut_r.add(*s as usize);
            *d.add(1) = *lut_g.add(*s.add(1) as usize);
            *d.add(2) = *lut_b.add(*s.add(2) as usize);
            x += 1;
        }
    }
}


#[cfg(target_arch = "aarch64")]
unsafe fn resize_nearest_rgb_neon(
    src: &[u8],
    dst: &mut [u8],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) {
    let src_stride = src_width * 3;
    let dst_stride = dst_width * 3;

    let mut x_src_offsets = Vec::<usize>::with_capacity(dst_width);
    let x_scale = (src_width as f32) / (dst_width as f32);

    for i in 0..dst_width {
        let x_src = ((i as f32) * x_scale).floor() as usize;
        x_src_offsets.push(x_src.min(src_width - 1) * 3);
    }

    let y_scale = (src_height as f32) / (dst_height as f32);
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    for y_new in 0..dst_height {
        let y_src = ((y_new as f32) * y_scale).floor() as usize;
        let y_src = y_src.min(src_height - 1);

        let src_row = src_ptr.add(y_src * src_stride);
        let dst_row = dst_ptr.add(y_new * dst_stride);

        let mut x = 0;
        while x + 4 <= dst_width {
            let x0 = *x_src_offsets.get_unchecked(x);
            let x1 = *x_src_offsets.get_unchecked(x + 1);
            let x2 = *x_src_offsets.get_unchecked(x + 2);
            let x3 = *x_src_offsets.get_unchecked(x + 3);

            let d = dst_row.add(x * 3);
            *d = *src_row.add(x0);
            *d.add(1) = *src_row.add(x0 + 1);
            *d.add(2) = *src_row.add(x0 + 2);

            *d.add(3) = *src_row.add(x1);
            *d.add(4) = *src_row.add(x1 + 1);
            *d.add(5) = *src_row.add(x1 + 2);

            *d.add(6) = *src_row.add(x2);
            *d.add(7) = *src_row.add(x2 + 1);
            *d.add(8) = *src_row.add(x2 + 2);

            *d.add(9) = *src_row.add(x3);
            *d.add(10) = *src_row.add(x3 + 1);
            *d.add(11) = *src_row.add(x3 + 2);

            x += 4;
        }

        while x < dst_width {
            let x0 = *x_src_offsets.get_unchecked(x);
            let d = dst_row.add(x * 3);
            *d = *src_row.add(x0);
            *d.add(1) = *src_row.add(x0 + 1);
            *d.add(2) = *src_row.add(x0 + 2);
            x += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn resize_nearest_gray_neon(
    src: &[u8],
    dst: &mut [u8],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) {
    let mut x_src_coords = Vec::<usize>::with_capacity(dst_width);
    let x_scale = (src_width as f32) / (dst_width as f32);

    for i in 0..dst_width {
        let x_src = ((i as f32) * x_scale).floor() as usize;
        x_src_coords.push(x_src.min(src_width - 1));
    }

    let y_scale = (src_height as f32) / (dst_height as f32);
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    for y_new in 0..dst_height {
        let y_src = ((y_new as f32) * y_scale).floor() as usize;
        let y_src = y_src.min(src_height - 1);

        let src_row = src_ptr.add(y_src * src_width);
        let dst_row = dst_ptr.add(y_new * dst_width);

        let mut x = 0;
        while x + 8 <= dst_width {
            let x0 = *x_src_coords.get_unchecked(x);
            let x1 = *x_src_coords.get_unchecked(x + 1);
            let x2 = *x_src_coords.get_unchecked(x + 2);
            let x3 = *x_src_coords.get_unchecked(x + 3);
            let x4 = *x_src_coords.get_unchecked(x + 4);
            let x5 = *x_src_coords.get_unchecked(x + 5);
            let x6 = *x_src_coords.get_unchecked(x + 6);
            let x7 = *x_src_coords.get_unchecked(x + 7);

            let d = dst_row.add(x);
            *d = *src_row.add(x0);
            *d.add(1) = *src_row.add(x1);
            *d.add(2) = *src_row.add(x2);
            *d.add(3) = *src_row.add(x3);
            *d.add(4) = *src_row.add(x4);
            *d.add(5) = *src_row.add(x5);
            *d.add(6) = *src_row.add(x6);
            *d.add(7) = *src_row.add(x7);

            x += 8;
        }

        while x < dst_width {
            let x0 = *x_src_coords.get_unchecked(x);
            *dst_row.add(x) = *src_row.add(x0);
            x += 1;
        }
    }
}

fn resize_nearest_scalar(
    src: &[u8],
    dst: &mut [u8],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    channels: usize,
) {
    let x_scale = (src_width as f32) / (dst_width as f32);
    let y_scale = (src_height as f32) / (dst_height as f32);
    let src_stride = src_width * channels;
    let dst_stride = dst_width * channels;

    for y_new in 0..dst_height {
        let y_src = ((y_new as f32) * y_scale).floor() as usize;
        let y_src = y_src.min(src_height - 1);

        for x_new in 0..dst_width {
            let x_src = ((x_new as f32) * x_scale).floor() as usize;
            let x_src = x_src.min(src_width - 1);

            let src_idx = y_src * src_stride + x_src * channels;
            let dst_idx = y_new * dst_stride + x_new * channels;

            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr().add(src_idx),
                    dst.as_mut_ptr().add(dst_idx),
                    channels,
                );
            }
        }
    }
}

// ============================================================================
// Bilinear Interpolation
// ============================================================================

fn resize_bilinear(
    src: &[u8],
    dst: &mut [u8],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    channels: usize,
) {
    unsafe {
        // Use SIMD for beneficial image sizes with RGB or grayscale
        #[cfg(target_arch = "aarch64")]
        if dst_width >= SIMD_MIN_WIDTH && dst_height >= SIMD_MIN_HEIGHT {
            if channels == 3 {
                if src_width == dst_width * 2 && src_height == dst_height * 2 {
                    resize_bilinear_down2_rgb_neon(src, dst, src_width, src_height, dst_width, dst_height);
                    return;
                }
                resize_bilinear_rgb_neon(src, dst, src_width, src_height, dst_width, dst_height);
                return;
            } else if channels == 1 {
                if src_width == dst_width * 2 && src_height == dst_height * 2 {
                    resize_bilinear_down2_gray_neon(src, dst, src_width, src_height, dst_width, dst_height);
                    return;
                }
                resize_bilinear_gray_neon(src, dst, src_width, src_height, dst_width, dst_height);
                return;
            }
        }

        // Scalar fallback for small images or unsupported channel counts
        resize_bilinear_scalar(
            src, dst, src_width, src_height, dst_width, dst_height, channels,
        );
    }
}

/// Scalar bilinear resize (always available as fallback)
fn resize_bilinear_scalar(
    src: &[u8],
    dst: &mut [u8],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    channels: usize,
) {
    let x_scale = (src_width as f32) / (dst_width as f32);
    let y_scale = (src_height as f32) / (dst_height as f32);
    let src_stride = src_width * channels;
    let dst_stride = dst_width * channels;

    for y_new in 0..dst_height {
        let y_src = (y_new as f32 + 0.5) * y_scale - 0.5;
        let y0_f = y_src.floor();
        let y0 = if y_src < 0.0 { 0 } else { (y0_f as i32).min(src_height as i32 - 1) };
        let y1 = (y0 + 1).min(src_height as i32 - 1);
        let dy = if y_src < 0.0 || y_src >= (src_height - 1) as f32 { 0.0 } else { y_src - y0_f };

        for x_new in 0..dst_width {
            let x_src = (x_new as f32 + 0.5) * x_scale - 0.5;
            let x0_f = x_src.floor();
            let x0 = if x_src < 0.0 { 0 } else { (x0_f as i32).min(src_width as i32 - 1) };
            let x1 = (x0 + 1).min(src_width as i32 - 1);
            let dx = if x_src < 0.0 || x_src >= (src_width - 1) as f32 { 0.0 } else { x_src - x0_f };

            for ch in 0..channels {
                let val = bilinear_sample(
                    src, x0, y0, x1, y1, dx, dy, src_width, src_height, src_stride, ch,
                );
                let dst_idx = y_new * dst_stride + x_new * channels + ch;
                dst[dst_idx] = val;
            }
        }
    }
}

// ============================================================================
// NEON SIMD Bilinear Implementations
// ============================================================================

#[cfg(target_arch = "aarch64")]
unsafe fn resize_bilinear_down2_rgb_neon(
    src: &[u8],
    dst: &mut [u8],
    src_width: usize,
    _src_height: usize,
    dst_width: usize,
    dst_height: usize,
) {
    let src_stride = src_width * 3;
    let dst_stride = dst_width * 3;
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    for y_new in 0..dst_height {
        let row0 = src_ptr.add(2 * y_new * src_stride);
        let row1 = src_ptr.add((2 * y_new + 1) * src_stride);
        let dst_row = dst_ptr.add(y_new * dst_stride);

        let mut x_new = 0;
        let mut in_x = 0;

        // Process 16 output RGB pixels per iteration (from 32 input RGB pixels = 96 bytes)
        while x_new + 16 <= dst_width && in_x + 96 <= src_stride {
            let r0_a = vld3q_u8(row0.add(in_x));
            let r0_b = vld3q_u8(row0.add(in_x + 48));
            let r1_a = vld3q_u8(row1.add(in_x));
            let r1_b = vld3q_u8(row1.add(in_x + 48));

            // Red channel
            let r0_even_r = vuzp1q_u8(r0_a.0, r0_b.0);
            let r0_odd_r = vuzp2q_u8(r0_a.0, r0_b.0);
            let r1_even_r = vuzp1q_u8(r1_a.0, r1_b.0);
            let r1_odd_r = vuzp2q_u8(r1_a.0, r1_b.0);
            let top_r = vrhaddq_u8(r0_even_r, r0_odd_r);
            let bot_r = vrhaddq_u8(r1_even_r, r1_odd_r);
            let final_r = vrhaddq_u8(top_r, bot_r);

            // Green channel
            let r0_even_g = vuzp1q_u8(r0_a.1, r0_b.1);
            let r0_odd_g = vuzp2q_u8(r0_a.1, r0_b.1);
            let r1_even_g = vuzp1q_u8(r1_a.1, r1_b.1);
            let r1_odd_g = vuzp2q_u8(r1_a.1, r1_b.1);
            let top_g = vrhaddq_u8(r0_even_g, r0_odd_g);
            let bot_g = vrhaddq_u8(r1_even_g, r1_odd_g);
            let final_g = vrhaddq_u8(top_g, bot_g);

            // Blue channel
            let r0_even_b = vuzp1q_u8(r0_a.2, r0_b.2);
            let r0_odd_b = vuzp2q_u8(r0_a.2, r0_b.2);
            let r1_even_b = vuzp1q_u8(r1_a.2, r1_b.2);
            let r1_odd_b = vuzp2q_u8(r1_a.2, r1_b.2);
            let top_b = vrhaddq_u8(r0_even_b, r0_odd_b);
            let bot_b = vrhaddq_u8(r1_even_b, r1_odd_b);
            let final_b = vrhaddq_u8(top_b, bot_b);

            vst3q_u8(
                dst_row.add(x_new * 3),
                uint8x16x3_t(final_r, final_g, final_b),
            );

            x_new += 16;
            in_x += 96;
        }

        while x_new < dst_width {
            let p00 = row0.add(in_x);
            let p10 = if in_x + 3 < src_stride { row0.add(in_x + 3) } else { p00 };
            let p01 = row1.add(in_x);
            let p11 = if in_x + 3 < src_stride { row1.add(in_x + 3) } else { p01 };

            for c in 0..3 {
                let top = (*p00.add(c) as u32 + *p10.add(c) as u32 + 1) >> 1;
                let bot = (*p01.add(c) as u32 + *p11.add(c) as u32 + 1) >> 1;
                *dst_row.add(x_new * 3 + c) = ((top + bot + 1) >> 1) as u8;
            }

            x_new += 1;
            in_x += 6;
        }
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

#[cfg(target_arch = "aarch64")]
unsafe fn resize_bilinear_down2_rgb_lut_neon(
    src: &[u8],
    dst: &mut [u8],
    src_width: usize,
    _src_height: usize,
    dst_width: usize,
    dst_height: usize,
    luts: &[[u8; 256]; 3],
) {
    let src_stride = src_width * 3;
    let dst_stride = dst_width * 3;
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    let is_1c = luts[0] == luts[1] && luts[1] == luts[2];

    let (r0, r1, r2, r3) = (
        vld1q_u8_x4(luts[0].as_ptr()),
        vld1q_u8_x4(luts[0].as_ptr().add(64)),
        vld1q_u8_x4(luts[0].as_ptr().add(128)),
        vld1q_u8_x4(luts[0].as_ptr().add(192)),
    );

    let (g0, g1, g2, g3) = if is_1c {
        (r0, r1, r2, r3)
    } else {
        (
            vld1q_u8_x4(luts[1].as_ptr()),
            vld1q_u8_x4(luts[1].as_ptr().add(64)),
            vld1q_u8_x4(luts[1].as_ptr().add(128)),
            vld1q_u8_x4(luts[1].as_ptr().add(192)),
        )
    };

    let (b0, b1, b2, b3) = if is_1c {
        (r0, r1, r2, r3)
    } else {
        (
            vld1q_u8_x4(luts[2].as_ptr()),
            vld1q_u8_x4(luts[2].as_ptr().add(64)),
            vld1q_u8_x4(luts[2].as_ptr().add(128)),
            vld1q_u8_x4(luts[2].as_ptr().add(192)),
        )
    };

    for y_new in 0..dst_height {
        let row0 = src_ptr.add(2 * y_new * src_stride);
        let row1 = src_ptr.add((2 * y_new + 1) * src_stride);
        let dst_row = dst_ptr.add(y_new * dst_stride);

        let mut x_new = 0;
        let mut in_x = 0;

        while x_new + 16 <= dst_width && in_x + 96 <= src_stride {
            let r0_a = vld3q_u8(row0.add(in_x));
            let r0_b = vld3q_u8(row0.add(in_x + 48));
            let r1_a = vld3q_u8(row1.add(in_x));
            let r1_b = vld3q_u8(row1.add(in_x + 48));

            // Red channel
            let r0_even_r = vuzp1q_u8(r0_a.0, r0_b.0);
            let r0_odd_r = vuzp2q_u8(r0_a.0, r0_b.0);
            let r1_even_r = vuzp1q_u8(r1_a.0, r1_b.0);
            let r1_odd_r = vuzp2q_u8(r1_a.0, r1_b.0);
            let top_r = vrhaddq_u8(r0_even_r, r0_odd_r);
            let bot_r = vrhaddq_u8(r1_even_r, r1_odd_r);
            let final_r = vrhaddq_u8(top_r, bot_r);
            let lut_r = lut_lookup16(final_r, r0, r1, r2, r3);

            // Green channel
            let r0_even_g = vuzp1q_u8(r0_a.1, r0_b.1);
            let r0_odd_g = vuzp2q_u8(r0_a.1, r0_b.1);
            let r1_even_g = vuzp1q_u8(r1_a.1, r1_b.1);
            let r1_odd_g = vuzp2q_u8(r1_a.1, r1_b.1);
            let top_g = vrhaddq_u8(r0_even_g, r0_odd_g);
            let bot_g = vrhaddq_u8(r1_even_g, r1_odd_g);
            let final_g = vrhaddq_u8(top_g, bot_g);
            let lut_g = lut_lookup16(final_g, g0, g1, g2, g3);

            // Blue channel
            let r0_even_b = vuzp1q_u8(r0_a.2, r0_b.2);
            let r0_odd_b = vuzp2q_u8(r0_a.2, r0_b.2);
            let r1_even_b = vuzp1q_u8(r1_a.2, r1_b.2);
            let r1_odd_b = vuzp2q_u8(r1_a.2, r1_b.2);
            let top_b = vrhaddq_u8(r0_even_b, r0_odd_b);
            let bot_b = vrhaddq_u8(r1_even_b, r1_odd_b);
            let final_b = vrhaddq_u8(top_b, bot_b);
            let lut_b = lut_lookup16(final_b, b0, b1, b2, b3);

            vst3q_u8(
                dst_row.add(x_new * 3),
                uint8x16x3_t(lut_r, lut_g, lut_b),
            );

            x_new += 16;
            in_x += 96;
        }

        while x_new < dst_width {
            let p00 = row0.add(in_x);
            let p10 = if in_x + 3 < src_stride { row0.add(in_x + 3) } else { p00 };
            let p01 = row1.add(in_x);
            let p11 = if in_x + 3 < src_stride { row1.add(in_x + 3) } else { p01 };

            for c in 0..3 {
                let top = (*p00.add(c) as u32 + *p10.add(c) as u32 + 1) >> 1;
                let bot = (*p01.add(c) as u32 + *p11.add(c) as u32 + 1) >> 1;
                let raw_val = ((top + bot + 1) >> 1) as u8;
                *dst_row.add(x_new * 3 + c) = luts[c][raw_val as usize];
            }

            x_new += 1;
            in_x += 6;
        }
    }
}

/// Fast 2x downsample for Grayscale using single-cycle `vrhadd` NEON operations

#[cfg(target_arch = "aarch64")]
unsafe fn resize_bilinear_down2_gray_neon(
    src: &[u8],
    dst: &mut [u8],
    src_width: usize,
    _src_height: usize,
    dst_width: usize,
    dst_height: usize,
) {
    let src_stride = src_width;
    let dst_stride = dst_width;
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    for y_new in 0..dst_height {
        let row0 = src_ptr.add(2 * y_new * src_stride);
        let row1 = src_ptr.add((2 * y_new + 1) * src_stride);
        let dst_row = dst_ptr.add(y_new * dst_stride);

        let mut x_new = 0;
        let mut in_x = 0;

        while x_new + 32 <= dst_width && in_x + 64 <= src_stride {
            let r0_0 = vld1q_u8(row0.add(in_x));
            let r0_1 = vld1q_u8(row0.add(in_x + 16));
            let r0_2 = vld1q_u8(row0.add(in_x + 32));
            let r0_3 = vld1q_u8(row0.add(in_x + 48));

            let r1_0 = vld1q_u8(row1.add(in_x));
            let r1_1 = vld1q_u8(row1.add(in_x + 16));
            let r1_2 = vld1q_u8(row1.add(in_x + 32));
            let r1_3 = vld1q_u8(row1.add(in_x + 48));

            let r0_even0 = vuzp1q_u8(r0_0, r0_1);
            let r0_odd0 = vuzp2q_u8(r0_0, r0_1);
            let r0_even1 = vuzp1q_u8(r0_2, r0_3);
            let r0_odd1 = vuzp2q_u8(r0_2, r0_3);

            let r1_even0 = vuzp1q_u8(r1_0, r1_1);
            let r1_odd0 = vuzp2q_u8(r1_0, r1_1);
            let r1_even1 = vuzp1q_u8(r1_2, r1_3);
            let r1_odd1 = vuzp2q_u8(r1_2, r1_3);

            let top0 = vrhaddq_u8(r0_even0, r0_odd0);
            let top1 = vrhaddq_u8(r0_even1, r0_odd1);
            let bot0 = vrhaddq_u8(r1_even0, r1_odd0);
            let bot1 = vrhaddq_u8(r1_even1, r1_odd1);

            let res0 = vrhaddq_u8(top0, bot0);
            let res1 = vrhaddq_u8(top1, bot1);

            vst1q_u8(dst_row.add(x_new), res0);
            vst1q_u8(dst_row.add(x_new + 16), res1);

            x_new += 32;
            in_x += 64;
        }

        while x_new + 16 <= dst_width && in_x + 32 <= src_stride {
            let r0_0 = vld1q_u8(row0.add(in_x));
            let r0_1 = vld1q_u8(row0.add(in_x + 16));
            let r1_0 = vld1q_u8(row1.add(in_x));
            let r1_1 = vld1q_u8(row1.add(in_x + 16));

            let r0_even = vuzp1q_u8(r0_0, r0_1);
            let r0_odd = vuzp2q_u8(r0_0, r0_1);
            let r1_even = vuzp1q_u8(r1_0, r1_1);
            let r1_odd = vuzp2q_u8(r1_0, r1_1);

            let top = vrhaddq_u8(r0_even, r0_odd);
            let bot = vrhaddq_u8(r1_even, r1_odd);
            let res = vrhaddq_u8(top, bot);

            vst1q_u8(dst_row.add(x_new), res);

            x_new += 16;
            in_x += 32;
        }

        while x_new < dst_width {
            let top = (*row0.add(in_x) as u32 + *row0.add(in_x + 1) as u32 + 1) >> 1;
            let bot = (*row1.add(in_x) as u32 + *row1.add(in_x + 1) as u32 + 1) >> 1;
            *dst_row.add(x_new) = ((top + bot + 1) >> 1) as u8;

            x_new += 1;
            in_x += 2;
        }
    }
}

/// RGB bilinear resize using NEON SIMD with Fixed-Point Arithmetic (Q11)
/// Optimized "Per-Pixel" Vectorization to avoid de-interleaving overhead
#[cfg(target_arch = "aarch64")]
unsafe fn resize_bilinear_rgb_neon(
    src: &[u8],
    dst: &mut [u8],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) {
    let src_stride = src_width * 3;
    let dst_stride = dst_width * 3;
    let x_scale = (src_width as f32) / (dst_width as f32);
    let y_scale = (src_height as f32) / (dst_height as f32);

    const SCALE: u16 = 2048;

    let mut x0_offsets = Vec::with_capacity(dst_width);
    let mut dx_weights = Vec::with_capacity(dst_width);

    for i in 0..dst_width {
        let x_src = (i as f32 + 0.5) * x_scale - 0.5;
        let x0_f = x_src.floor();
        let x0 = if x_src < 0.0 { 0 } else { (x0_f as usize).min(src_width - 1) };

        x0_offsets.push(x0 * 3);

        let dx_f = if x_src < 0.0 || x_src >= (src_width - 1) as f32 { 0.0 } else { x_src - x0_f };
        let w = (dx_f * (SCALE as f32)).round() as u16;
        dx_weights.push(w);
    }

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let max_safe_off = src_stride.saturating_sub(8);

    for y_new in 0..dst_height {
        let y_src = (y_new as f32 + 0.5) * y_scale - 0.5;
        let y0_f = y_src.floor();
        let y0 = if y_src < 0.0 { 0 } else { (y0_f as usize).min(src_height - 1) };
        let y1 = (y0 + 1).min(src_height - 1);

        let dy_f = if y_src < 0.0 || y_src >= (src_height - 1) as f32 { 0.0 } else { y_src - y0_f };
        let dy = (dy_f * (SCALE as f32)).round() as u16;
        let idy = SCALE - dy;

        let row0_base = y0 * src_stride;
        let row1_base = y1 * src_stride;
        let dst_row_base = y_new * dst_stride;

        for x_new in 0..dst_width {
            let off = *x0_offsets.get_unchecked(x_new);
            let dx = *dx_weights.get_unchecked(x_new);
            let idx = SCALE - dx;

            let out_idx = dst_row_base + x_new * 3;
            let dst_p = dst_ptr.add(out_idx);

            if off <= max_safe_off {
                let ptr0 = src_ptr.add(row0_base + off);
                let ptr1 = src_ptr.add(row1_base + off);

                let v0_u8 = vld1_u8(ptr0);
                let v1_u8 = vld1_u8(ptr1);

                let v0_u16 = vmovl_u8(v0_u8);
                let v1_u16 = vmovl_u8(v1_u8);

                let v0_p0 = vget_low_u16(v0_u16);
                let v1_p0 = vget_low_u16(v1_u16);

                let v0_p1 = vget_low_u16(vextq_u16(v0_u16, v0_u16, 3));
                let v1_p1 = vget_low_u16(vextq_u16(v1_u16, v1_u16, 3));

                let top = vmlal_n_u16(vmull_n_u16(v0_p0, idx), v0_p1, dx);
                let bot = vmlal_n_u16(vmull_n_u16(v1_p0, idx), v1_p1, dx);

                let top_16 = vrshrn_n_u32(top, 11);
                let bot_16 = vrshrn_n_u32(bot, 11);

                let res = vmlal_n_u16(vmull_n_u16(top_16, idy), bot_16, dy);
                let res_16 = vrshrn_n_u32(res, 11);

                *dst_p = vget_lane_u16::<0>(res_16) as u8;
                *dst_p.add(1) = vget_lane_u16::<1>(res_16) as u8;
                *dst_p.add(2) = vget_lane_u16::<2>(res_16) as u8;
            } else {
                let ptr0 = src_ptr.add(row0_base + off);
                let ptr1 = src_ptr.add(row1_base + off);

                let x1_off = if off + 3 < src_stride { 3 } else { 0 };

                let idx_u32 = idx as u32;
                let dx_u32 = dx as u32;
                let idy_u32 = idy as u32;
                let dy_u32 = dy as u32;

                for ch in 0..3 {
                    let p00 = *ptr0.add(ch) as u32;
                    let p10 = *ptr0.add(x1_off + ch) as u32;
                    let p01 = *ptr1.add(ch) as u32;
                    let p11 = *ptr1.add(x1_off + ch) as u32;

                    let top = (p00 * idx_u32 + p10 * dx_u32 + 1024) >> 11;
                    let bot = (p01 * idx_u32 + p11 * dx_u32 + 1024) >> 11;
                    let res = (top * idy_u32 + bot * dy_u32 + 1024) >> 11;

                    *dst_p.add(ch) = res.min(255) as u8;
                }
            }
        }
    }
}

/// Grayscale bilinear resize using row cache & NEON SIMD vertical blend
#[cfg(target_arch = "aarch64")]
unsafe fn resize_bilinear_gray_neon(
    src: &[u8],
    dst: &mut [u8],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) {
    let src_stride = src_width;
    let dst_stride = dst_width;
    let x_scale = (src_width as f32) / (dst_width as f32);
    let y_scale = (src_height as f32) / (dst_height as f32);

    const SCALE: i32 = 2048;

    let mut x_table = Vec::<(u32, u32, i32, i32)>::with_capacity(dst_width);
    for i in 0..dst_width {
        let x_src = (i as f32 + 0.5) * x_scale - 0.5;
        let x0_f = x_src.floor();
        let x0 = if x_src < 0.0 { 0 } else { (x0_f as usize).min(src_width - 1) } as u32;
        let x1 = (x0 + 1).min(src_width as u32 - 1);
        let dx_f = if x_src < 0.0 || x_src >= (src_width - 1) as f32 { 0.0 } else { x_src - x0_f };
        let dx = (dx_f * (SCALE as f32)).round() as i32;
        let idx = SCALE - dx;
        x_table.push((x0, x1, dx, idx));
    }

    let mut buf0 = Vec::<i16>::with_capacity(dst_width);
    let mut buf1 = Vec::<i16>::with_capacity(dst_width);
    unsafe {
        buf0.set_len(dst_width);
        buf1.set_len(dst_width);
    }

    let mut prev_y0 = usize::MAX;
    let mut prev_y1 = usize::MAX;

    let h_interpolate = |src_row: *const u8, out_buf: *mut i16| {
        let mut x = 0;
        while x + 4 <= dst_width {
            let (x0_0, x1_0, dx_0, idx_0) = *x_table.get_unchecked(x);
            let (x0_1, x1_1, dx_1, idx_1) = *x_table.get_unchecked(x + 1);
            let (x0_2, x1_2, dx_2, idx_2) = *x_table.get_unchecked(x + 2);
            let (x0_3, x1_3, dx_3, idx_3) = *x_table.get_unchecked(x + 3);

            let v0 = (*src_row.add(x0_0 as usize) as i32 * idx_0 + *src_row.add(x1_0 as usize) as i32 * dx_0 + 1024) >> 11;
            let v1 = (*src_row.add(x0_1 as usize) as i32 * idx_1 + *src_row.add(x1_1 as usize) as i32 * dx_1 + 1024) >> 11;
            let v2 = (*src_row.add(x0_2 as usize) as i32 * idx_2 + *src_row.add(x1_2 as usize) as i32 * dx_2 + 1024) >> 11;
            let v3 = (*src_row.add(x0_3 as usize) as i32 * idx_3 + *src_row.add(x1_3 as usize) as i32 * dx_3 + 1024) >> 11;

            *out_buf.add(x) = v0 as i16;
            *out_buf.add(x + 1) = v1 as i16;
            *out_buf.add(x + 2) = v2 as i16;
            *out_buf.add(x + 3) = v3 as i16;
            x += 4;
        }
        while x < dst_width {
            let (x0, x1, dx, idx) = *x_table.get_unchecked(x);
            let val = (*src_row.add(x0 as usize) as i32 * idx + *src_row.add(x1 as usize) as i32 * dx + 1024) >> 11;
            *out_buf.add(x) = val as i16;
            x += 1;
        }
    };

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    for y_new in 0..dst_height {
        let y_src = (y_new as f32 + 0.5) * y_scale - 0.5;
        let y0_f = y_src.floor();
        let y0 = if y_src < 0.0 { 0 } else { (y0_f as usize).min(src_height - 1) };
        let y1 = (y0 + 1).min(src_height - 1);

        let dy_f = if y_src < 0.0 || y_src >= (src_height - 1) as f32 { 0.0 } else { y_src - y0_f };
        let dy = (dy_f * (SCALE as f32)).round() as i16;
        let idy = (SCALE as i16) - dy;

        if y0 == prev_y0 {
        } else if y0 == prev_y1 {
            std::mem::swap(&mut buf0, &mut buf1);
            prev_y0 = y0;
            prev_y1 = usize::MAX;
        } else {
            h_interpolate(src_ptr.add(y0 * src_stride), buf0.as_mut_ptr());
            prev_y0 = y0;
        }

        if y1 == y0 {
        } else if y1 == prev_y1 {
        } else {
            h_interpolate(src_ptr.add(y1 * src_stride), buf1.as_mut_ptr());
            prev_y1 = y1;
        }

        let dst_row = dst_ptr.add(y_new * dst_stride);
        let b0_ptr = buf0.as_ptr();
        let b1_ptr = if y1 == y0 { buf0.as_ptr() } else { buf1.as_ptr() };

        let mut i = 0;
        while i + 16 <= dst_width {
            let b0_lo = vld1q_s16(b0_ptr.add(i));
            let b0_hi = vld1q_s16(b0_ptr.add(i + 8));
            let b1_lo = vld1q_s16(b1_ptr.add(i));
            let b1_hi = vld1q_s16(b1_ptr.add(i + 8));

            let acc0 = vmlal_n_s16(vmull_n_s16(vget_low_s16(b0_lo), idy), vget_low_s16(b1_lo), dy);
            let acc1 = vmlal_n_s16(vmull_n_s16(vget_high_s16(b0_lo), idy), vget_high_s16(b1_lo), dy);
            let acc2 = vmlal_n_s16(vmull_n_s16(vget_low_s16(b0_hi), idy), vget_low_s16(b1_hi), dy);
            let acc3 = vmlal_n_s16(vmull_n_s16(vget_high_s16(b0_hi), idy), vget_high_s16(b1_hi), dy);

            let r0 = vrshrn_n_s32(acc0, 11);
            let r1 = vrshrn_n_s32(acc1, 11);
            let r2 = vrshrn_n_s32(acc2, 11);
            let r3 = vrshrn_n_s32(acc3, 11);

            let u0 = vqmovun_s16(vcombine_s16(r0, r1));
            let u1 = vqmovun_s16(vcombine_s16(r2, r3));
            let res = vcombine_u8(u0, u1);

            vst1q_u8(dst_row.add(i), res);
            i += 16;
        }

        while i < dst_width {
            let v0 = *b0_ptr.add(i) as i32;
            let v1 = *b1_ptr.add(i) as i32;
            let val = (v0 * (idy as i32) + v1 * (dy as i32) + 1024) >> 11;
            *dst_row.add(i) = (val as u8).min(255);
            i += 1;
        }
    }
}

/// Bilinear interpolation sample
#[inline]
fn bilinear_sample(
    data: &[u8],
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    dx: f32,
    dy: f32,
    width: usize,
    height: usize,
    stride: usize,
    channel: usize,
) -> u8 {
    // channels = stride / width
    let channels = stride / width;

    let sample = |x: i32, y: i32| -> f32 {
        if x < 0 || x >= width as i32 || y < 0 || y >= height as i32 {
            0.0
        } else {
            let idx = y as usize * stride + x as usize * channels + channel;
            if idx < data.len() {
                data[idx] as f32
            } else {
                0.0
            }
        }
    };

    let v00 = sample(x0, y0);
    let v10 = sample(x1, y0);
    let v01 = sample(x0, y1);
    let v11 = sample(x1, y1);

    let top = v00 * (1.0 - dx) + v10 * dx;
    let bottom = v01 * (1.0 - dx) + v11 * dx;
    let result = top * (1.0 - dy) + bottom * dy;

    result.clamp(0.0, 255.0) as u8
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::FusableImage;

    #[test]
    fn test_resize_nearest_upscale_rgb() {
        let mut data = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let img = FusableImage::new(&mut data[..], 2, 2, 3);
        let mut dst = vec![0u8; 4 * 4 * 3];

        unsafe {
            resize_neon(
                &img.data,
                &mut dst,
                2,
                2,
                4,
                4,
                3,
                ResizeInterpolation::Nearest,
            );
        }

        assert_eq!(dst[0], 1); // top-left R
        assert_eq!(dst[1], 2); // top-left G
        assert_eq!(dst[2], 3); // top-left B
    }

    #[test]
    fn test_resize_bilinear_upscale_rgb() {
        // Test bilinear with a simple 2x2 pattern
        let mut data = vec![
            0u8, 0, 0, // (0,0): black
            255, 0, 0, // (1,0): red
            0, 255, 0, // (0,1): green
            255, 255, 0, // (1,1): yellow
        ];
        let img = FusableImage::new(&mut data[..], 2, 2, 3);
        let mut dst = vec![0u8; 4 * 4 * 3];

        unsafe {
            resize_neon(
                &img.data,
                &mut dst,
                2,
                2,
                4,
                4,
                3,
                ResizeInterpolation::Bilinear,
            );
        }

        // Top-left corner (0,0) should be close to black
        assert_eq!(dst[0], 0); // R
        assert_eq!(dst[1], 0); // G
        assert_eq!(dst[2], 0); // B

        // Bilinear should produce intermediate values, not just the original values
        // Check that we have some variation in the output
        let mut has_red = false;
        let mut has_green = false;
        let mut has_intermediate = false;

        for i in 0..dst.len() / 3 {
            let r = dst[i * 3];
            let g = dst[i * 3 + 1];
            if r > 200 {
                has_red = true;
            }
            if g > 200 {
                has_green = true;
            }
            // Check for intermediate values (50-150 range indicates interpolation)
            if r > 50 && r < 150 {
                has_intermediate = true;
            }
            if g > 50 && g < 150 {
                has_intermediate = true;
            }
        }

        // Bilinear should have red and green from the source pixels
        assert!(has_red, "Should have red values from source");
        assert!(has_green, "Should have green values from source");
        assert!(
            has_intermediate,
            "Bilinear should produce intermediate values"
        );
    }

    #[test]
    fn test_resize_bilinear_vs_nearest() {
        let mut data = vec![0u8, 0, 0, 255, 0, 0, 0, 255, 0, 255, 255, 255];
        let img = FusableImage::new(&mut data[..], 2, 2, 3);

        let mut dst_nearest = vec![0u8; 3 * 3 * 3];
        let mut dst_bilinear = vec![0u8; 3 * 3 * 3];

        unsafe {
            resize_neon(
                &img.data,
                &mut dst_nearest,
                2,
                2,
                3,
                3,
                3,
                ResizeInterpolation::Nearest,
            );

            resize_neon(
                &img.data,
                &mut dst_bilinear,
                2,
                2,
                3,
                3,
                3,
                ResizeInterpolation::Bilinear,
            );
        }

        // Nearest and bilinear should produce different results
        // (unless all sampled pixels have the same value)
        let results_differ = dst_nearest != dst_bilinear;
        assert!(
            results_differ,
            "Nearest and bilinear should produce different results"
        );
    }

    #[test]
    fn test_resize_nearest_upscale_grayscale() {
        // Test grayscale (1-channel) resize with NEON
        let mut data = vec![10u8, 20, 30, 40];
        let img = FusableImage::new(&mut data[..], 2, 2, 1);
        let mut dst = vec![0u8; 4 * 4 * 1];

        unsafe {
            resize_neon(
                &img.data,
                &mut dst,
                2,
                2,
                4,
                4,
                1,
                ResizeInterpolation::Nearest,
            );
        }

        assert_eq!(dst[0], 10); // (0,0) -> 10
        assert_eq!(dst[1], 10); // (1,0) -> 10
        assert_eq!(dst[2], 20); // (2,0) -> 20
        assert_eq!(dst[3], 20); // (3,0) -> 20
        assert_eq!(dst[8], 30); // (0,2) -> 30
        assert_eq!(dst[15], 40); // (3,3) -> 40
    }

    #[test]
    fn test_resize_interpolation_enum() {
        // Test i32 conversions
        assert_eq!(ResizeInterpolation::Nearest.to_i32(), 0);
        assert_eq!(ResizeInterpolation::Bilinear.to_i32(), 1);
        assert_eq!(ResizeInterpolation::Bicubic.to_i32(), 2);
        assert_eq!(ResizeInterpolation::Lanczos4.to_i32(), 3);

        assert_eq!(
            ResizeInterpolation::from_i32(0),
            Some(ResizeInterpolation::Nearest)
        );
        assert_eq!(
            ResizeInterpolation::from_i32(1),
            Some(ResizeInterpolation::Bilinear)
        );
        assert_eq!(
            ResizeInterpolation::from_i32(2),
            Some(ResizeInterpolation::Bicubic)
        );
        assert_eq!(
            ResizeInterpolation::from_i32(3),
            Some(ResizeInterpolation::Lanczos4)
        );
        assert_eq!(ResizeInterpolation::from_i32(99), None);

        // Test string conversions
        assert_eq!(ResizeInterpolation::Nearest.to_str(), "nearest");
        assert_eq!(ResizeInterpolation::Bilinear.to_str(), "bilinear");
        assert_eq!(ResizeInterpolation::Bicubic.to_str(), "bicubic");
        assert_eq!(ResizeInterpolation::Lanczos4.to_str(), "lanczos4");

        assert_eq!(
            ResizeInterpolation::from_str("nearest"),
            Some(ResizeInterpolation::Nearest)
        );
        assert_eq!(
            ResizeInterpolation::from_str("BILINEAR"),
            Some(ResizeInterpolation::Bilinear)
        ); // case insensitive
        assert_eq!(
            ResizeInterpolation::from_str("BiCubic"),
            Some(ResizeInterpolation::Bicubic)
        );
        assert_eq!(
            ResizeInterpolation::from_str("lanczos"),
            Some(ResizeInterpolation::Lanczos4)
        );
        assert_eq!(ResizeInterpolation::from_str("invalid"), None);
    }
}
