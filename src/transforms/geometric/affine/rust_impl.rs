// High-performance Rust implementation for affine transforms
//
// Uses fixed-point Q16.16 coordinate stepping, bundled RGB sampling,
// and branchless interior interpolation.
//
// Note: Reference implementation used for test verification and fallback.
#![allow(dead_code)]

use crate::core::{BarrierImage, FusableImage};
use crate::transforms::geometric::affine::{Affine, AffineBorderMode, AffineInterpolation};

/// Execute using optimized Rust implementation
pub(super) fn execute_rust(affine: &Affine, image: &FusableImage) -> BarrierImage {
    let (out_width, out_height) = affine.output_size.unwrap_or((image.width, image.height));
    let channels = image.channels;
    let in_width = image.width;
    let in_height = image.height;
    let data = &image.data;

    let mut transformed_data = vec![0u8; out_width * out_height * channels];

    // Build inverse transformation matrix
    let [a, b, c, d, e, f] = affine.build_inverse_matrix(in_width, in_height);

    match affine.interpolation {
        AffineInterpolation::Nearest => {
            let dx_fp = (a * 65536.0).round() as i64;
            let dy_fp = (d * 65536.0).round() as i64;

            for y_out in 0..out_height {
                let mut x_fp = ((b * y_out as f32 + c) * 65536.0).round() as i64;
                let mut y_fp = ((e * y_out as f32 + f) * 65536.0).round() as i64;

                let row_out_idx = y_out * out_width * channels;

                for x_out in 0..out_width {
                    // Round to nearest: add 0.5 (32768 in Q16.16) and shift right 16
                    let xi = ((x_fp + 32768) >> 16) as i32;
                    let yi = ((y_fp + 32768) >> 16) as i32;

                    let out_idx = row_out_idx + x_out * channels;

                    if xi >= 0 && xi < in_width as i32 && yi >= 0 && yi < in_height as i32 {
                        let in_idx = (yi as usize * in_width + xi as usize) * channels;
                        for ch in 0..channels {
                            transformed_data[out_idx + ch] = data[in_idx + ch];
                        }
                    } else {
                        // Border handling
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

        AffineInterpolation::Bilinear => {
            let dx_fp = (a * 65536.0).round() as i64;
            let dy_fp = (d * 65536.0).round() as i64;

            if channels == 3 {
                let in_stride = in_width * 3;
                let in_ptr = data.as_ptr();
                let out_ptr = transformed_data.as_mut_ptr();
                let max_x = in_width.saturating_sub(1) as u32;
                let max_y = in_height.saturating_sub(1) as u32;

                for y_out in 0..out_height {
                    let mut x_fp = ((b * y_out as f32 + c) * 65536.0).round() as i64;
                    let mut y_fp = ((e * y_out as f32 + f) * 65536.0).round() as i64;
                    let row_out_idx = y_out * out_width * 3;

                    for x_out in 0..out_width {
                        let x0 = (x_fp >> 16) as i32;
                        let y0 = (y_fp >> 16) as i32;
                        let fx = ((x_fp >> 8) & 0xFF) as u32; // 0..255
                        let fy = ((y_fp >> 8) & 0xFF) as u32; // 0..255

                        let w00 = (256 - fx) * (256 - fy);
                        let w10 = fx * (256 - fy);
                        let w01 = (256 - fx) * fy;
                        let w11 = fx * fy;

                        let out_idx = row_out_idx + x_out * 3;

                        if (x0 as u32) < max_x && (y0 as u32) < max_y {
                            unsafe {
                                let top_ptr = in_ptr.add(y0 as usize * in_stride + x0 as usize * 3);
                                let bot_ptr = top_ptr.add(in_stride);

                                let r00 = *top_ptr as u32;
                                let g00 = *top_ptr.add(1) as u32;
                                let b00 = *top_ptr.add(2) as u32;

                                let r10 = *top_ptr.add(3) as u32;
                                let g10 = *top_ptr.add(4) as u32;
                                let b10 = *top_ptr.add(5) as u32;

                                let r01 = *bot_ptr as u32;
                                let g01 = *bot_ptr.add(1) as u32;
                                let b01 = *bot_ptr.add(2) as u32;

                                let r11 = *bot_ptr.add(3) as u32;
                                let g11 = *bot_ptr.add(4) as u32;
                                let b11 = *bot_ptr.add(5) as u32;

                                let r = (r00 * w00 + r10 * w10 + r01 * w01 + r11 * w11 + 32768) >> 16;
                                let g = (g00 * w00 + g10 * w10 + g01 * w01 + g11 * w11 + 32768) >> 16;
                                let b = (b00 * w00 + b10 * w10 + b01 * w01 + b11 * w11 + 32768) >> 16;

                                *out_ptr.add(out_idx) = r as u8;
                                *out_ptr.add(out_idx + 1) = g as u8;
                                *out_ptr.add(out_idx + 2) = b as u8;
                            }
                        } else {
                            let (r00, g00, b00) = sample_rgb(data, in_width as i32, in_height as i32, x0, y0, affine.border_mode);
                            let (r10, g10, b10) = sample_rgb(data, in_width as i32, in_height as i32, x0 + 1, y0, affine.border_mode);
                            let (r01, g01, b01) = sample_rgb(data, in_width as i32, in_height as i32, x0, y0 + 1, affine.border_mode);
                            let (r11, g11, b11) = sample_rgb(data, in_width as i32, in_height as i32, x0 + 1, y0 + 1, affine.border_mode);

                            let r = (r00 * w00 + r10 * w10 + r01 * w01 + r11 * w11 + 32768) >> 16;
                            let g = (g00 * w00 + g10 * w10 + g01 * w01 + g11 * w11 + 32768) >> 16;
                            let b = (b00 * w00 + b10 * w10 + b01 * w01 + b11 * w11 + 32768) >> 16;

                            unsafe {
                                *out_ptr.add(out_idx) = r.min(255) as u8;
                                *out_ptr.add(out_idx + 1) = g.min(255) as u8;
                                *out_ptr.add(out_idx + 2) = b.min(255) as u8;
                            }
                        }

                        x_fp += dx_fp;
                        y_fp += dy_fp;
                    }
                }
            } else {
                // General channels
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
