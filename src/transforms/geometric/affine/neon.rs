// ARM NEON SIMD implementation for affine transforms
//
// Uses fixed-point Q16.16 coordinate stepping, NEON SIMD 4-way fused multiply-accumulate,
// and zero-overhead memory buffers.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use crate::core::{BarrierImage, FusableImage};
use crate::transforms::geometric::affine::{Affine, AffineBorderMode, AffineInterpolation};

/// Execute using optimized NEON implementation
#[cfg(target_arch = "aarch64")]
pub(super) fn execute_neon(affine: &Affine, image: &FusableImage) -> BarrierImage {
    let (out_width, out_height) = affine.output_size.unwrap_or((image.width, image.height));
    let channels = image.channels;
    let in_width = image.width;
    let in_height = image.height;
    let data = &image.data;

    let len = out_width * out_height * channels;
    let mut transformed_data = Vec::<u8>::with_capacity(len);
    unsafe { transformed_data.set_len(len); }

    // Build inverse transformation matrix
    let [a, b, c, d, e, f] = affine.build_inverse_matrix(in_width, in_height);

    match affine.interpolation {
        AffineInterpolation::Nearest => {
            let dx_fp = (a * 65536.0).round() as i64;
            let dy_fp = (d * 65536.0).round() as i64;

            if channels == 3 {
                let in_stride = in_width * 3;
                let in_ptr = data.as_ptr();
                let out_ptr = transformed_data.as_mut_ptr();
                let iw = in_width as i32;
                let ih = in_height as i32;

                for y_out in 0..out_height {
                    let mut x_fp = ((b * y_out as f32 + c) * 65536.0).round() as i64;
                    let mut y_fp = ((e * y_out as f32 + f) * 65536.0).round() as i64;
                    let row_out_idx = y_out * out_width * 3;

                    for x_out in 0..out_width {
                        let xi = ((x_fp + 32768) >> 16) as i32;
                        let yi = ((y_fp + 32768) >> 16) as i32;
                        let out_idx = row_out_idx + x_out * 3;

                        if (xi as u32) < in_width as u32 && (yi as u32) < in_height as u32 {
                            unsafe {
                                let src_p = in_ptr.add(yi as usize * in_stride + xi as usize * 3);
                                let dst_p = out_ptr.add(out_idx);
                                *dst_p = *src_p;
                                *dst_p.add(1) = *src_p.add(1);
                                *dst_p.add(2) = *src_p.add(2);
                            }
                        } else {
                            let (r, g, b_val) = sample_rgb(data, iw, ih, xi, yi, affine.border_mode);
                            unsafe {
                                let dst_p = out_ptr.add(out_idx);
                                *dst_p = r as u8;
                                *dst_p.add(1) = g as u8;
                                *dst_p.add(2) = b_val as u8;
                            }
                        }

                        x_fp += dx_fp;
                        y_fp += dy_fp;
                    }
                }
            } else if channels == 1 {
                let in_ptr = data.as_ptr();
                let out_ptr = transformed_data.as_mut_ptr();
                let iw = in_width as i32;
                let ih = in_height as i32;

                for y_out in 0..out_height {
                    let mut x_fp = ((b * y_out as f32 + c) * 65536.0).round() as i64;
                    let mut y_fp = ((e * y_out as f32 + f) * 65536.0).round() as i64;
                    let row_out_idx = y_out * out_width;

                    for x_out in 0..out_width {
                        let xi = ((x_fp + 32768) >> 16) as i32;
                        let yi = ((y_fp + 32768) >> 16) as i32;
                        let out_idx = row_out_idx + x_out;

                        if (xi as u32) < in_width as u32 && (yi as u32) < in_height as u32 {
                            unsafe {
                                *out_ptr.add(out_idx) = *in_ptr.add(yi as usize * in_width + xi as usize);
                            }
                        } else {
                            unsafe {
                                *out_ptr.add(out_idx) = sample_single(data, iw, ih, 1, xi, yi, 0, affine.border_mode) as u8;
                            }
                        }

                        x_fp += dx_fp;
                        y_fp += dy_fp;
                    }
                }
            } else {
                for y_out in 0..out_height {
                    let mut x_fp = ((b * y_out as f32 + c) * 65536.0).round() as i64;
                    let mut y_fp = ((e * y_out as f32 + f) * 65536.0).round() as i64;
                    let row_out_idx = y_out * out_width * channels;

                    for x_out in 0..out_width {
                        let xi = ((x_fp + 32768) >> 16) as i32;
                        let yi = ((y_fp + 32768) >> 16) as i32;
                        let out_idx = row_out_idx + x_out * channels;

                        if xi >= 0 && xi < in_width as i32 && yi >= 0 && yi < in_height as i32 {
                            let in_idx = (yi as usize * in_width + xi as usize) * channels;
                            for ch in 0..channels {
                                transformed_data[out_idx + ch] = data[in_idx + ch];
                            }
                        } else {
                            let (bx, by) = map_border(xi, yi, in_width as i32, in_height as i32, affine.border_mode);
                            if let (Some(sx), Some(sy)) = (bx, by) {
                                let in_idx = (sy * in_width + sx) * channels;
                                for ch in 0..channels {
                                    transformed_data[out_idx + ch] = data[in_idx + ch];
                                }
                            } else if let AffineBorderMode::Constant { value } = affine.border_mode {
                                for ch in 0..channels {
                                    transformed_data[out_idx + ch] = value;
                                }
                            }
                        }

                        x_fp += dx_fp;
                        y_fp += dy_fp;
                    }
                }
            }
        }

        AffineInterpolation::Bilinear => {
            let dx_fp = (a * 65536.0).round() as i64;
            let dy_fp = (d * 65536.0).round() as i64;

            if channels == 3 {
                let in_stride = in_width * 3;
                let in_ptr = data.as_ptr();
                let out_ptr = transformed_data.as_mut_ptr();
                let max_x = in_width.saturating_sub(1) as u32;
                let max_y = in_height.saturating_sub(1) as u32;
                let iw = in_width as i32;
                let ih = in_height as i32;

                let tbl_mask = unsafe { vld1_u8([0, 1, 2, 255, 3, 4, 5, 255].as_ptr()) };

                for y_out in 0..out_height {
                    let mut x_fp = ((b * y_out as f32 + c) * 65536.0).round() as i64;
                    let mut y_fp = ((e * y_out as f32 + f) * 65536.0).round() as i64;
                    let row_out_idx = y_out * out_width * 3;

                    for x_out in 0..out_width {
                        let x0 = (x_fp >> 16) as i32;
                        let y0 = (y_fp >> 16) as i32;
                        let fx = ((x_fp >> 8) & 0xFF) as u32;
                        let fy = ((y_fp >> 8) & 0xFF) as u32;

                        let w00 = (256 - fx) * (256 - fy);
                        let w10 = fx * (256 - fy);
                        let w01 = (256 - fx) * fy;
                        let w11 = fx * fy;

                        let out_idx = row_out_idx + x_out * 3;

                        if (x0 as u32) < max_x && (y0 as u32) < max_y {
                            unsafe {
                                let top_ptr = in_ptr.add(y0 as usize * in_stride + x0 as usize * 3);
                                let bot_ptr = top_ptr.add(in_stride);

                                let raw_top = vld1_u8(top_ptr);
                                let raw_bot = vld1_u8(bot_ptr);

                                let top_perm = vtbl1_u8(raw_top, tbl_mask);
                                let bot_perm = vtbl1_u8(raw_bot, tbl_mask);

                                let top_u16 = vmovl_u8(top_perm);
                                let bot_u16 = vmovl_u8(bot_perm);

                                let top0 = vmovl_u16(vget_low_u16(top_u16));
                                let top1 = vmovl_u16(vget_high_u16(top_u16));
                                let bot0 = vmovl_u16(vget_low_u16(bot_u16));
                                let bot1 = vmovl_u16(vget_high_u16(bot_u16));

                                let mut acc = vmulq_n_u32(top0, w00);
                                acc = vmlaq_n_u32(acc, top1, w10);
                                acc = vmlaq_n_u32(acc, bot0, w01);
                                acc = vmlaq_n_u32(acc, bot1, w11);

                                let res_u16 = vrshrn_n_u32(acc, 16);
                                let dst_p = out_ptr.add(out_idx);
                                *dst_p = vget_lane_u16::<0>(res_u16) as u8;
                                *dst_p.add(1) = vget_lane_u16::<1>(res_u16) as u8;
                                *dst_p.add(2) = vget_lane_u16::<2>(res_u16) as u8;
                            }
                        } else {
                            let (r00, g00, b00) = sample_rgb(data, iw, ih, x0, y0, affine.border_mode);
                            let (r10, g10, b10) = sample_rgb(data, iw, ih, x0 + 1, y0, affine.border_mode);
                            let (r01, g01, b01) = sample_rgb(data, iw, ih, x0, y0 + 1, affine.border_mode);
                            let (r11, g11, b11) = sample_rgb(data, iw, ih, x0 + 1, y0 + 1, affine.border_mode);

                            let r = (r00 * w00 + r10 * w10 + r01 * w01 + r11 * w11 + 32768) >> 16;
                            let g = (g00 * w00 + g10 * w10 + g01 * w01 + g11 * w11 + 32768) >> 16;
                            let b = (b00 * w00 + b10 * w10 + b01 * w01 + b11 * w11 + 32768) >> 16;

                            unsafe {
                                let dst_p = out_ptr.add(out_idx);
                                *dst_p = r.min(255) as u8;
                                *dst_p.add(1) = g.min(255) as u8;
                                *dst_p.add(2) = b.min(255) as u8;
                            }
                        }

                        x_fp += dx_fp;
                        y_fp += dy_fp;
                    }
                }
            } else if channels == 1 {
                let in_ptr = data.as_ptr();
                let out_ptr = transformed_data.as_mut_ptr();
                let max_x = in_width.saturating_sub(1) as u32;
                let max_y = in_height.saturating_sub(1) as u32;

                if dy_fp == 0 && dx_fp >= 0 {
                    // Fast path: with a pure scale/translate affine the source row
                    // is constant for the whole output row (dy_fp == 0) and source
                    // x is monotonic (dx_fp >= 0). Process 8 output pixels per
                    // iteration: one 16-byte window load per row, a vtbl gather
                    // per corner, and u32 fixed-point bilinear blend. Bit-exact
                    // with gray_bilinear_pixel (same arithmetic).
                    for y_out in 0..out_height {
                        let mut x_fp = ((b * y_out as f32 + c) * 65536.0).round() as i64;
                        let y_fp = ((e * y_out as f32 + f) * 65536.0).round() as i64;
                        let row_out_idx = y_out * out_width;
                        let y0 = (y_fp >> 16) as i32;

                        if (y0 as u32) >= max_y {
                            // Top/bottom border row: scalar (exactly like the
                            // general path so borders stay bit-identical).
                            for x_out in 0..out_width {
                                let val = gray_bilinear_pixel(
                                    data, in_width, in_height, x_fp, y_fp,
                                    max_x, max_y, affine.border_mode,
                                );
                                unsafe { *out_ptr.add(row_out_idx + x_out) = val; }
                                x_fp += dx_fp;
                            }
                        } else {
                            let top_row = unsafe { in_ptr.add(y0 as usize * in_width) };
                            let bot_row = unsafe { top_row.add(in_width) };
                            let fy = ((y_fp >> 8) & 0xFF) as u32;
                            let mut x_out = 0usize;

                            while x_out < out_width {
                                let x0 = (x_fp >> 16) as i32;
                                if (x0 as u32) >= max_x {
                                    // Left/right border pixel: scalar.
                                    let val = gray_bilinear_pixel(
                                        data, in_width, in_height, x_fp, y_fp,
                                        max_x, max_y, affine.border_mode,
                                    );
                                    unsafe { *out_ptr.add(row_out_idx + x_out) = val; }
                                    x_fp += dx_fp;
                                    x_out += 1;
                                    continue;
                                }

                                // Try an 8-pixel vector block.
                                if x_out + 8 <= out_width {
                                    let mut min_x0 = x0;
                                    let mut max_x0 = x0;
                                    let mut block_ok = true;
                                    for i in 1..8 {
                                        let x0i = ((x_fp + (i as i64) * dx_fp) >> 16) as i32;
                                        if (x0i as u32) >= max_x {
                                            block_ok = false;
                                            break;
                                        }
                                        min_x0 = min_x0.min(x0i);
                                        max_x0 = max_x0.max(x0i);
                                    }
                                    if block_ok
                                        && (max_x0 - min_x0) <= 14
                                        && (min_x0 + 15) < in_width as i32
                                    {
                                        unsafe {
                                            let mut x0s = [0i32; 8];
                                            let mut fxs = [0u8; 8];
                                            for i in 0..8 {
                                                let x_fp_i = x_fp + (i as i64) * dx_fp;
                                                x0s[i] = (x_fp_i >> 16) as i32;
                                                fxs[i] = ((x_fp_i >> 8) & 0xFF) as u8;
                                            }
                                            let off = x0s[0] as usize;
                                            let top16 = vld1q_u8(top_row.add(off));
                                            let bot16 = vld1q_u8(bot_row.add(off));

                                            let mut idx_bytes = [0u8; 8];
                                            for i in 0..8 {
                                                idx_bytes[i] = (x0s[i] - x0s[0]) as u8;
                                            }
                                            let idx_v = vld1_u8(idx_bytes.as_ptr());
                                            let idx1_v = vadd_u8(idx_v, vdup_n_u8(1));
                                            let p00 = vqtbl1_u8(top16, idx_v);
                                            let p10 = vqtbl1_u8(top16, idx1_v);
                                            let p01 = vqtbl1_u8(bot16, idx_v);
                                            let p11 = vqtbl1_u8(bot16, idx1_v);

                                            let fx_v = vmovl_u8(vld1_u8(fxs.as_ptr()));
                                            // Factored bilinear (bit-exact with the scalar
                                            // formula by integer distributivity):
                                            //   top = p00*(256-fx) + p10*fx   (u32)
                                            //   bot = p01*(256-fx) + p11*fx   (u32)
                                            //   val = (top*(256-fy) + bot*fy + 32768) >> 16
                                            let ax_v = vsubq_u16(vdupq_n_u16(256), fx_v);
                                            let p00_u16 = vmovl_u8(p00);
                                            let p10_u16 = vmovl_u8(p10);
                                            let p01_u16 = vmovl_u8(p01);
                                            let p11_u16 = vmovl_u8(p11);

                                            let top_lo = vmlal_u16(
                                                vmull_u16(vget_low_u16(p00_u16), vget_low_u16(ax_v)),
                                                vget_low_u16(p10_u16),
                                                vget_low_u16(fx_v),
                                            );
                                            let top_hi = vmlal_u16(
                                                vmull_u16(vget_high_u16(p00_u16), vget_high_u16(ax_v)),
                                                vget_high_u16(p10_u16),
                                                vget_high_u16(fx_v),
                                            );
                                            let bot_lo = vmlal_u16(
                                                vmull_u16(vget_low_u16(p01_u16), vget_low_u16(ax_v)),
                                                vget_low_u16(p11_u16),
                                                vget_low_u16(fx_v),
                                            );
                                            let bot_hi = vmlal_u16(
                                                vmull_u16(vget_high_u16(p01_u16), vget_high_u16(ax_v)),
                                                vget_high_u16(p11_u16),
                                                vget_high_u16(fx_v),
                                            );

                                            let fy_w = vdupq_n_u32(fy);
                                            let fyc_w = vdupq_n_u32(256 - fy);
                                            let acc_lo = vaddq_u32(
                                                vmlaq_u32(vmulq_u32(top_lo, fyc_w), bot_lo, fy_w),
                                                vdupq_n_u32(32768),
                                            );
                                            let acc_hi = vaddq_u32(
                                                vmlaq_u32(vmulq_u32(top_hi, fyc_w), bot_hi, fy_w),
                                                vdupq_n_u32(32768),
                                            );

                                            let r_lo = vqmovn_u32(vshrq_n_u32(acc_lo, 16));
                                            let r_hi = vqmovn_u32(vshrq_n_u32(acc_hi, 16));
                                            let r_u8 = vqmovn_u16(vcombine_u16(r_lo, r_hi));
                                            vst1_u8(out_ptr.add(row_out_idx + x_out), r_u8);
                                        }
                                        x_fp += 8 * dx_fp;
                                        x_out += 8;
                                        continue;
                                    }
                                }

                                // Non-conforming pixel (drift > 14 or tail): scalar.
                                let val = gray_bilinear_pixel(
                                    data, in_width, in_height, x_fp, y_fp,
                                    max_x, max_y, affine.border_mode,
                                );
                                unsafe { *out_ptr.add(row_out_idx + x_out) = val; }
                                x_fp += dx_fp;
                                x_out += 1;
                            }
                        }
                    }
                } else {
                    // General path (rotation/shear, or x-mirror affines): walk
                    // each output row in 8-pixel blocks. Per lane the Q16
                    // coordinates are stepped in SIMD; the block's 2x2 corners
                    // live in a small source window that vqtbl4 gathers from
                    // four 16-byte row loads. Blocks failing the window guard
                    // (border lanes, span too large) fall back to
                    // gray_bilinear_pixel, keeping the path bit-exact with the
                    // scalar reference everywhere.
                    for y_out in 0..out_height {
                        let mut x_fp = ((b * y_out as f32 + c) * 65536.0).round() as i64;
                        let mut y_fp = ((e * y_out as f32 + f) * 65536.0).round() as i64;
                        let row_out_idx = y_out * out_width;

                        let mut x_out = 0usize;
                        while x_out < out_width {
                            if x_out + 8 <= out_width
                                && unsafe {
                                    affine_gray_block8(
                                        data,
                                        out_ptr.add(row_out_idx + x_out),
                                        in_width,
                                        in_height,
                                        x_fp,
                                        y_fp,
                                        dx_fp,
                                        dy_fp,
                                    )
                                }
                            {
                                x_fp += 8 * dx_fp;
                                y_fp += 8 * dy_fp;
                                x_out += 8;
                                continue;
                            }

                            let val = gray_bilinear_pixel(
                                data, in_width, in_height, x_fp, y_fp,
                                max_x, max_y, affine.border_mode,
                            );
                            unsafe { *out_ptr.add(row_out_idx + x_out) = val; }
                            x_fp += dx_fp;
                            y_fp += dy_fp;
                            x_out += 1;
                        }
                    }
                }
            } else {
                for y_out in 0..out_height {
                    let mut x_fp = ((b * y_out as f32 + c) * 65536.0).round() as i64;
                    let mut y_fp = ((e * y_out as f32 + f) * 65536.0).round() as i64;
                    let row_out_idx = y_out * out_width * channels;

                    for x_out in 0..out_width {
                        let x0 = (x_fp >> 16) as i32;
                        let y0 = (y_fp >> 16) as i32;
                        let fx = ((x_fp >> 8) & 0xFF) as u32;
                        let fy = ((y_fp >> 8) & 0xFF) as u32;

                        let w00 = (256 - fx) * (256 - fy);
                        let w10 = fx * (256 - fy);
                        let w01 = (256 - fx) * fy;
                        let w11 = fx * fy;

                        let out_idx = row_out_idx + x_out * channels;

                        for ch in 0..channels {
                            let v00 = sample_single(data, in_width as i32, in_height as i32, channels, x0, y0, ch, affine.border_mode);
                            let v10 = sample_single(data, in_width as i32, in_height as i32, channels, x0 + 1, y0, ch, affine.border_mode);
                            let v01 = sample_single(data, in_width as i32, in_height as i32, channels, x0, y0 + 1, ch, affine.border_mode);
                            let v11 = sample_single(data, in_width as i32, in_height as i32, channels, x0 + 1, y0 + 1, ch, affine.border_mode);

                            let val = (v00 * w00 + v10 * w10 + v01 * w01 + v11 * w11 + 32768) >> 16;
                            transformed_data[out_idx + ch] = val.min(255) as u8;
                        }

                        x_fp += dx_fp;
                        y_fp += dy_fp;
                    }
                }
            }
        }
    }

    BarrierImage::from_vec(transformed_data, out_width, out_height, channels)
}

#[inline(always)]
fn sample_rgb(data: &[u8], width: i32, height: i32, x: i32, y: i32, mode: AffineBorderMode) -> (u32, u32, u32) {
    if x >= 0 && x < width && y >= 0 && y < height {
        let idx = (y as usize * width as usize + x as usize) * 3;
        (data[idx] as u32, data[idx + 1] as u32, data[idx + 2] as u32)
    } else {
        match mode {
            AffineBorderMode::Constant { value } => (value as u32, value as u32, value as u32),
            AffineBorderMode::Replicate => {
                let cx = x.clamp(0, width - 1) as usize;
                let cy = y.clamp(0, height - 1) as usize;
                let idx = (cy * width as usize + cx) * 3;
                (data[idx] as u32, data[idx + 1] as u32, data[idx + 2] as u32)
            }
            AffineBorderMode::Reflect => {
                let cx = reflect_coord(x, width);
                let cy = reflect_coord(y, height);
                let idx = (cy * width as usize + cx) * 3;
                (data[idx] as u32, data[idx + 1] as u32, data[idx + 2] as u32)
            }
            AffineBorderMode::Wrap => {
                let cx = x.rem_euclid(width) as usize;
                let cy = y.rem_euclid(height) as usize;
                let idx = (cy * width as usize + cx) * 3;
                (data[idx] as u32, data[idx + 1] as u32, data[idx + 2] as u32)
            }
        }
    }
}

#[inline(always)]
fn sample_single(data: &[u8], width: i32, height: i32, channels: usize, x: i32, y: i32, ch: usize, mode: AffineBorderMode) -> u32 {
    if x >= 0 && x < width && y >= 0 && y < height {
        data[(y as usize * width as usize + x as usize) * channels + ch] as u32
    } else {
        match mode {
            AffineBorderMode::Constant { value } => value as u32,
            AffineBorderMode::Replicate => {
                let cx = x.clamp(0, width - 1) as usize;
                let cy = y.clamp(0, height - 1) as usize;
                data[(cy * width as usize + cx) * channels + ch] as u32
            }
            AffineBorderMode::Reflect => {
                let cx = reflect_coord(x, width);
                let cy = reflect_coord(y, height);
                data[(cy * width as usize + cx) * channels + ch] as u32
            }
            AffineBorderMode::Wrap => {
                let cx = x.rem_euclid(width) as usize;
                let cy = y.rem_euclid(height) as usize;
                data[(cy * width as usize + cx) * channels + ch] as u32
            }
        }
    }
}

/// Scalar grayscale bilinear sample at fixed-point (x_fp, y_fp).
///
/// This is the bit-exact reference for the grayscale bilinear path: the NEON
/// fast path reproduces the same arithmetic (`(sum + 32768) >> 16`, truncating
/// x0/y0, `fx = (x_fp >> 8) & 0xFF`), so results match byte-for-byte.
#[inline(always)]
fn gray_bilinear_pixel(
    data: &[u8],
    in_width: usize,
    in_height: usize,
    x_fp: i64,
    y_fp: i64,
    max_x: u32,
    max_y: u32,
    mode: AffineBorderMode,
) -> u8 {
    let x0 = (x_fp >> 16) as i32;
    let y0 = (y_fp >> 16) as i32;
    let fx = ((x_fp >> 8) & 0xFF) as u32;
    let fy = ((y_fp >> 8) & 0xFF) as u32;

    let w00 = (256 - fx) * (256 - fy);
    let w10 = fx * (256 - fy);
    let w01 = (256 - fx) * fy;
    let w11 = fx * fy;

    if (x0 as u32) < max_x && (y0 as u32) < max_y {
        let top = y0 as usize * in_width + x0 as usize;
        let p00 = data[top] as u32;
        let p10 = data[top + 1] as u32;
        let p01 = data[top + in_width] as u32;
        let p11 = data[top + in_width + 1] as u32;
        ((p00 * w00 + p10 * w10 + p01 * w01 + p11 * w11 + 32768) >> 16) as u8
    } else {
        let iw = in_width as i32;
        let ih = in_height as i32;
        let p00 = sample_single(data, iw, ih, 1, x0, y0, 0, mode);
        let p10 = sample_single(data, iw, ih, 1, x0 + 1, y0, 0, mode);
        let p01 = sample_single(data, iw, ih, 1, x0, y0 + 1, 0, mode);
        let p11 = sample_single(data, iw, ih, 1, x0 + 1, y0 + 1, 0, mode);
        ((p00 * w00 + p10 * w10 + p01 * w01 + p11 * w11 + 32768) >> 16).min(255) as u8
    }
}

/// Try to compute an 8-pixel bilinear block for the general affine path.
///
/// Writes 8 output pixels and returns true when every lane is interior and the
/// block's source corners fit a 4x16-byte window table:
///   - all corners strictly inside (x0 < w-1, y0 < h-1) so no border sampling,
///   - x span <= 13 (so a corner's +1 column stays inside a table row),
///   - y span <= 2 (so a corner's +1 row stays inside the 4-row table),
///   - a 16-byte load at (min_x0, min_y0..min_y0+3) stays in bounds.
/// Returns false writing nothing otherwise; the caller falls back to
/// gray_bilinear_pixel, so the combined path stays bit-exact with the scalar
/// reference (identical corners, weights, `(sum + 32768) >> 16` rounding).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn affine_gray_block8(
    data: &[u8],
    out_ptr: *mut u8,
    in_width: usize,
    in_height: usize,
    x_fp: i64,
    y_fp: i64,
    dx_fp: i64,
    dy_fp: i64,
) -> bool {
    // Q16 coordinates fit i32 for any image < ~32k pixels wide (coordinate
    // magnitude is bounded by ~65536 * (dim + translate)).
    let lane_off: [i32; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
    let off_lo = vld1q_s32(lane_off.as_ptr());
    let off_hi = vld1q_s32(lane_off.as_ptr().add(4));

    let x_lo = vaddq_s32(vdupq_n_s32(x_fp as i32), vmulq_n_s32(off_lo, dx_fp as i32));
    let x_hi = vaddq_s32(vdupq_n_s32(x_fp as i32), vmulq_n_s32(off_hi, dx_fp as i32));
    let y_lo = vaddq_s32(vdupq_n_s32(y_fp as i32), vmulq_n_s32(off_lo, dy_fp as i32));
    let y_hi = vaddq_s32(vdupq_n_s32(y_fp as i32), vmulq_n_s32(off_hi, dy_fp as i32));

    let x0_lo = vshrq_n_s32(x_lo, 16);
    let x0_hi = vshrq_n_s32(x_hi, 16);
    let y0_lo = vshrq_n_s32(y_lo, 16);
    let y0_hi = vshrq_n_s32(y_hi, 16);

    // x0/y0 are monotonic in the lane index (constant integer step), so the
    // window bounds are just the block endpoints -- no horizontal reductions.
    let x0_first = (x_fp >> 16) as i32;
    let x0_last = ((x_fp + 7 * dx_fp) >> 16) as i32;
    let y0_first = (y_fp >> 16) as i32;
    let y0_last = ((y_fp + 7 * dy_fp) >> 16) as i32;
    let (min_x0, max_x0) = if dx_fp >= 0 { (x0_first, x0_last) } else { (x0_last, x0_first) };
    let (min_y0, max_y0) = if dy_fp >= 0 { (y0_first, y0_last) } else { (y0_last, y0_first) };

    if min_x0 < 0
        || min_y0 < 0
        || max_x0 >= in_width as i32 - 1
        || max_y0 >= in_height as i32 - 1
        || min_x0 + 16 > in_width as i32
        || max_x0 - min_x0 > 13
        || max_y0 - min_y0 > 2
        || min_y0 + 4 > in_height as i32
    {
        return false;
    }

    // Per-lane window table index: (y0 - min_y0)*16 + (x0 - min_x0) <= 2*16+13.
    let xr16 = vcombine_u16(
        vreinterpret_u16_s16(vqmovn_s32(vsubq_s32(x0_lo, vdupq_n_s32(min_x0)))),
        vreinterpret_u16_s16(vqmovn_s32(vsubq_s32(x0_hi, vdupq_n_s32(min_x0)))),
    );
    let yr16 = vcombine_u16(
        vreinterpret_u16_s16(vqmovn_s32(vsubq_s32(y0_lo, vdupq_n_s32(min_y0)))),
        vreinterpret_u16_s16(vqmovn_s32(vsubq_s32(y0_hi, vdupq_n_s32(min_y0)))),
    );
    let idx16 = vmlaq_u16(xr16, yr16, vdupq_n_u16(16));
    let idx8 = vqmovn_u16(idx16);
    let idx8_p1 = vadd_u8(idx8, vdup_n_u8(1));
    let idx8_p16 = vadd_u8(idx8, vdup_n_u8(16));
    let idx8_p17 = vadd_u8(idx8, vdup_n_u8(17));

    let base = data.as_ptr().add(min_y0 as usize * in_width + min_x0 as usize);
    let t0 = vld1q_u8(base);
    let t1 = vld1q_u8(base.add(in_width));
    let t2 = vld1q_u8(base.add(2 * in_width));
    let t3 = vld1q_u8(base.add(3 * in_width));

    let tbl = uint8x16x4_t(t0, t1, t2, t3);
    let p00 = vqtbl4_u8(tbl, idx8);
    let p10 = vqtbl4_u8(tbl, idx8_p1);
    let p01 = vqtbl4_u8(tbl, idx8_p16);
    let p11 = vqtbl4_u8(tbl, idx8_p17);

    // Fractional parts: (fp >> 8) & 0xFF per lane.
    let fx16 = vcombine_u16(
        vreinterpret_u16_s16(vqmovn_s32(vandq_s32(vshrq_n_s32(x_lo, 8), vdupq_n_s32(0xFF)))),
        vreinterpret_u16_s16(vqmovn_s32(vandq_s32(vshrq_n_s32(x_hi, 8), vdupq_n_s32(0xFF)))),
    );
    let fy16 = vcombine_u16(
        vreinterpret_u16_s16(vqmovn_s32(vandq_s32(vshrq_n_s32(y_lo, 8), vdupq_n_s32(0xFF)))),
        vreinterpret_u16_s16(vqmovn_s32(vandq_s32(vshrq_n_s32(y_hi, 8), vdupq_n_s32(0xFF)))),
    );

    // Factored two-lerp, bit-exact with the scalar formula by integer
    // distributivity: (p00*w00 + p10*w10 + p01*w01 + p11*w11 + 32768) >> 16.
    let ax16 = vsubq_u16(vdupq_n_u16(256), fx16);
    let p00_16 = vmovl_u8(p00);
    let p10_16 = vmovl_u8(p10);
    let p01_16 = vmovl_u8(p01);
    let p11_16 = vmovl_u8(p11);

    let top_lo = vmlal_u16(
        vmull_u16(vget_low_u16(p00_16), vget_low_u16(ax16)),
        vget_low_u16(p10_16),
        vget_low_u16(fx16),
    );
    let top_hi = vmlal_u16(
        vmull_u16(vget_high_u16(p00_16), vget_high_u16(ax16)),
        vget_high_u16(p10_16),
        vget_high_u16(fx16),
    );
    let bot_lo = vmlal_u16(
        vmull_u16(vget_low_u16(p01_16), vget_low_u16(ax16)),
        vget_low_u16(p11_16),
        vget_low_u16(fx16),
    );
    let bot_hi = vmlal_u16(
        vmull_u16(vget_high_u16(p01_16), vget_high_u16(ax16)),
        vget_high_u16(p11_16),
        vget_high_u16(fx16),
    );

    let fy32_lo = vmovl_u16(vget_low_u16(fy16));
    let fy32_hi = vmovl_u16(vget_high_u16(fy16));
    let fyc_lo = vsubq_u32(vdupq_n_u32(256), fy32_lo);
    let fyc_hi = vsubq_u32(vdupq_n_u32(256), fy32_hi);
    let acc_lo = vaddq_u32(
        vmlaq_u32(vmulq_u32(top_lo, fyc_lo), bot_lo, fy32_lo),
        vdupq_n_u32(32768),
    );
    let acc_hi = vaddq_u32(
        vmlaq_u32(vmulq_u32(top_hi, fyc_hi), bot_hi, fy32_hi),
        vdupq_n_u32(32768),
    );

    let res = vcombine_u16(
        vqmovn_u32(vshrq_n_u32(acc_lo, 16)),
        vqmovn_u32(vshrq_n_u32(acc_hi, 16)),
    );
    vst1_u8(out_ptr, vqmovn_u16(res));
    true
}

#[inline(always)]
fn map_border(x: i32, y: i32, width: i32, height: i32, mode: AffineBorderMode) -> (Option<usize>, Option<usize>) {
    match mode {
        AffineBorderMode::Constant { .. } => (None, None),
        AffineBorderMode::Replicate => {
            let cx = x.clamp(0, width - 1) as usize;
            let cy = y.clamp(0, height - 1) as usize;
            (Some(cx), Some(cy))
        }
        AffineBorderMode::Reflect => {
            let cx = reflect_coord(x, width);
            let cy = reflect_coord(y, height);
            (Some(cx), Some(cy))
        }
        AffineBorderMode::Wrap => {
            let cx = x.rem_euclid(width) as usize;
            let cy = y.rem_euclid(height) as usize;
            (Some(cx), Some(cy))
        }
    }
}

#[inline(always)]
fn reflect_coord(mut p: i32, len: i32) -> usize {
    if len <= 1 {
        return 0;
    }
    if p < 0 {
        p = -p - 1;
    }
    if p >= len {
        let double_len = 2 * len;
        p = p % double_len;
        if p >= len {
            p = double_len - 1 - p;
        }
    }
    p.clamp(0, len - 1) as usize
}
