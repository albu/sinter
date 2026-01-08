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
        // Use SIMD for beneficial image sizes with RGB or grayscale
        #[cfg(target_arch = "aarch64")]
        if dst_width >= SIMD_MIN_WIDTH && dst_height >= SIMD_MIN_HEIGHT {
            if channels == 3 {
                resize_nearest_rgb_neon(src, dst, src_width, src_height, dst_width, dst_height);
                return;
            } else if channels == 1 {
                resize_nearest_gray_neon(src, dst, src_width, src_height, dst_width, dst_height);
                return;
            }
        }

        resize_nearest_scalar(
            src, dst, src_width, src_height, dst_width, dst_height, channels,
        );
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

    // Pre-compute source X coordinates for all destination X positions
    let mut x_src_coords = vec![0usize; dst_width];
    let x_scale = (src_width as f32) / (dst_width as f32);

    for i in 0..dst_width {
        let x_src = ((i as f32) * x_scale).round() as usize;
        x_src_coords[i] = x_src.min(src_width - 1);
    }

    let y_scale = (src_height as f32) / (dst_height as f32);

    for y_new in 0..dst_height {
        let y_src = ((y_new as f32) * y_scale).round() as usize;
        let y_src = y_src.min(src_height - 1);

        let src_row_base = y_src * src_stride;
        let dst_row_base = y_new * dst_stride;

        let mut x_new = 0;

        while x_new + 8 <= dst_width {
            let src_ptr = src.as_ptr().add(src_row_base);

            let mut r_vals = [0u8; 8];
            let mut g_vals = [0u8; 8];
            let mut b_vals = [0u8; 8];

            for i in 0..8 {
                let x_src = x_src_coords[x_new + i];
                let src_idx = x_src * 3;
                r_vals[i] = *src_ptr.add(src_idx);
                g_vals[i] = *src_ptr.add(src_idx + 1);
                b_vals[i] = *src_ptr.add(src_idx + 2);
            }

            let r = vld1_u8(r_vals.as_ptr());
            let g = vld1_u8(g_vals.as_ptr());
            let b = vld1_u8(b_vals.as_ptr());

            let dst_ptr = dst.as_mut_ptr().add(dst_row_base + x_new * 3);
            let out = uint8x8x3_t(r, g, b);
            vst3_u8(dst_ptr, out);

            x_new += 8;
        }

        while x_new < dst_width {
            let x_src = x_src_coords[x_new];
            let src_idx = src_row_base + x_src * 3;
            let dst_idx = dst_row_base + x_new * 3;

            std::ptr::copy_nonoverlapping(
                src.as_ptr().add(src_idx),
                dst.as_mut_ptr().add(dst_idx),
                3,
            );

            x_new += 1;
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
    // Pre-compute source X coordinates for all destination X positions
    let mut x_src_coords = vec![0usize; dst_width];
    let x_scale = (src_width as f32) / (dst_width as f32);

    for i in 0..dst_width {
        let x_src = ((i as f32) * x_scale).round() as usize;
        x_src_coords[i] = x_src.min(src_width - 1);
    }

    let y_scale = (src_height as f32) / (dst_height as f32);

    for y_new in 0..dst_height {
        let y_src = ((y_new as f32) * y_scale).round() as usize;
        let y_src = y_src.min(src_height - 1);

        let src_row_base = y_src * src_width;
        let dst_row_base = y_new * dst_width;

        let mut x_new = 0;

        // Process 16 pixels at a time using NEON
        while x_new + 16 <= dst_width {
            let mut vals = [0u8; 16];

            for i in 0..16 {
                let x_src = x_src_coords[x_new + i];
                vals[i] = *src.as_ptr().add(src_row_base + x_src);
            }

            let pixels = vld1q_u8(vals.as_ptr());
            vst1q_u8(dst.as_mut_ptr().add(dst_row_base + x_new), pixels);

            x_new += 16;
        }

        // Process remaining pixels
        while x_new < dst_width {
            let x_src = x_src_coords[x_new];
            dst[dst_row_base + x_new] = src[src_row_base + x_src];
            x_new += 1;
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
        let y_src = ((y_new as f32) * y_scale).round() as usize;
        let y_src = y_src.min(src_height - 1);

        for x_new in 0..dst_width {
            let x_src = ((x_new as f32) * x_scale).round() as usize;
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
                resize_bilinear_rgb_neon(src, dst, src_width, src_height, dst_width, dst_height);
                return;
            } else if channels == 1 {
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
        let y_src = (y_new as f32) * y_scale;
        let y0 = y_src.floor() as i32;
        let y1 = y0 + 1;
        let dy = y_src - y_src.floor();

        for x_new in 0..dst_width {
            let x_src = (x_new as f32) * x_scale;
            let x0 = x_src.floor() as i32;
            let x1 = x0 + 1;
            let dx = x_src - x_src.floor();

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

/// RGB bilinear resize using NEON SIMD
#[cfg(target_arch = "aarch64")]
/// RGB bilinear resize using NEON SIMD with Fixed-Point Arithmetic (Q11)
#[cfg(target_arch = "aarch64")]
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

    // Fixed point constants
    const SHIFT: i32 = 11;
    const SCALE: i32 = 1 << SHIFT; // 2048
    const ROUND: i32 = 1 << (SHIFT * 2 - 1);

    // Pre-compute X coordinates and integer weights
    let mut x0_offsets = vec![0usize; dst_width];
    let mut dx_weights = vec![0i16; dst_width];

    for i in 0..dst_width {
        let x_src = (i as f32) * x_scale;
        let x0_f = x_src.floor();
        let x0 = x0_f as usize;
        let x0_clamped = x0.min(src_width - 1);

        x0_offsets[i] = x0_clamped * 3;

        // Weight: range 0..2048
        let w = (x_src - x0_f) * (SCALE as f32);
        dx_weights[i] = w as i16;
    }

    // Process each row
    for y_new in 0..dst_height {
        let y_src = (y_new as f32) * y_scale;
        let y0_f = y_src.floor();
        let y0 = y0_f as usize;
        let y0_clamped = y0.min(src_height - 1);
        let y1_clamped = (y0 + 1).min(src_height - 1);

        let dy_f = y_src - y0_f;
        let dy = (dy_f * (SCALE as f32)) as i32;
        let idy = SCALE - dy;

        let row0_base = y0_clamped * src_stride;
        let row1_base = y1_clamped * src_stride;
        let dst_row_base = y_new * dst_stride;

        let mut x_new = 0;

        // Process pixel by pixel using SIMD for the RGBCalculation
        // We do this because pixel data is interleaved (RGB)
        // Loading 8 bytes gives us R0 G0 B0 R1 G1 B1 X X
        while x_new < dst_width {
            let off = x0_offsets[x_new];

            // Check if we can safely load 8 bytes (need 6 bytes valid: P0 and P1)
            // P1 is implicitly at off+3
            if off + 8 <= src_stride {
                let dx = dx_weights[x_new] as i16;
                let idx = (SCALE as i16) - dx;

                let ptr0 = src.as_ptr().add(row0_base + off);
                let ptr1 = src.as_ptr().add(row1_base + off);

                // Load 8 bytes: R0 G0 B0 R1 G1 B1 X X
                let v0_u8 = vld1_u8(ptr0);
                let v1_u8 = vld1_u8(ptr1);

                // Expand to u16 (0..255)
                let v0_u16 = vmovl_u8(v0_u8); // u16x8
                let v1_u16 = vmovl_u8(v1_u8);

                // Prepare P0 vector (R0 G0 B0 X) - low 4 lanes
                let v0_p0 = vget_low_u16(v0_u16);
                let v1_p0 = vget_low_u16(v1_u16);

                // Prepare P1 vector (R1 G1 B1 X) - offset by 3 lanes
                // We use vext to shift elements: extract starting from index 3
                let v0_p1_all = vextq_u16(v0_u16, v0_u16, 3);
                let v1_p1_all = vextq_u16(v1_u16, v1_u16, 3);
                let v0_p1 = vget_low_u16(v0_p1_all);
                let v1_p1 = vget_low_u16(v1_p1_all);

                // --- X Interpolation ---
                // res = p0 * idx + p1 * dx
                // Use i32 accumulator: vmull -> vmlal

                // Top row
                // Treat as s16 to match weights (they are positive anyway)
                let top = vmull_n_s16(vreinterpret_s16_u16(v0_p0), idx);
                let top = vmlal_n_s16(top, vreinterpret_s16_u16(v0_p1), dx);

                // Bot row
                let bot = vmull_n_s16(vreinterpret_s16_u16(v1_p0), idx);
                let bot = vmlal_n_s16(bot, vreinterpret_s16_u16(v1_p1), dx);

                // --- Y Interpolation ---
                // res = top * idy + bot * dy
                let res = vmulq_n_s32(top, idy);
                let res = vmlaq_n_s32(res, bot, dy);

                // Round and Shift (>> 22)
                let rounded = vaddq_s32(res, vdupq_n_s32(ROUND));
                let shifted = vshrq_n_s32(rounded, 22);

                // Pack to u8
                // vqmovun_s32 -> u16x4 (saturated)
                let res_u16 = vqmovun_s32(shifted);

                // Scalar store 3 bytes (R, G, B)
                // Extract 32-bit lane 0: [R, G, B, X]?
                // packed u16: R G B X
                // We need u8.
                let packed_val = vget_lane_u64(vreinterpret_u64_u16(res_u16), 0);

                let r = vget_lane_u16(res_u16, 0) as u8;
                let g = vget_lane_u16(res_u16, 1) as u8;
                let b = vget_lane_u16(res_u16, 2) as u8;

                let dst_ptr = dst.as_mut_ptr().add(dst_row_base + x_new * 3);
                *dst_ptr = r;
                *dst_ptr.add(1) = g;
                *dst_ptr.add(2) = b;
            } else {
                // Scalar Fallback for edges
                let dx = dx_weights[x_new] as i32;
                let idx = SCALE - dx;

                let ptr0 = src.as_ptr().add(row0_base + off);
                let ptr1 = src.as_ptr().add(row1_base + off);

                // Check bounds for p1
                let x1_off = if off + 3 < src_stride { 3 } else { 0 };

                for ch in 0..3 {
                    let p00 = *ptr0.add(ch) as i32;
                    let p10 = *ptr0.add(x1_off + ch) as i32;
                    let p01 = *ptr1.add(ch) as i32;
                    let p11 = *ptr1.add(x1_off + ch) as i32;

                    let top = p00 * idx + p10 * dx;
                    let bot = p01 * idx + p11 * dx;
                    let res = top * idy + bot * dy;

                    dst[dst_row_base + x_new * 3 + ch] = ((res + ROUND) >> 22).clamp(0, 255) as u8;
                }
            }
            x_new += 1;
        }
    }
}

/// Grayscale bilinear resize using NEON SIMD
#[cfg(target_arch = "aarch64")]
unsafe fn resize_bilinear_gray_neon(
    src: &[u8],
    dst: &mut [u8],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) {
    let x_scale = (src_width as f32) / (dst_width as f32);
    let y_scale = (src_height as f32) / (dst_height as f32);

    // Pre-compute X coordinates and weights
    let mut x0_coords = vec![0i32; dst_width];
    let mut x1_coords = vec![0i32; dst_width];
    let mut dx_weights = vec![0f32; dst_width];

    for i in 0..dst_width {
        let x_src = (i as f32) * x_scale;
        let x0_f = x_src.floor();
        x0_coords[i] = x0_f as i32;
        x1_coords[i] = x0_f as i32 + 1;
        dx_weights[i] = x_src - x0_f;
    }

    // Pre-compute Y coordinates and weights
    let mut y0_coords = vec![0i32; dst_height];
    let mut y1_coords = vec![0i32; dst_height];
    let mut dy_weights = vec![0f32; dst_height];

    for i in 0..dst_height {
        let y_src = (i as f32) * y_scale;
        let y0_f = y_src.floor();
        y0_coords[i] = y0_f as i32;
        y1_coords[i] = y0_f as i32 + 1;
        dy_weights[i] = y_src - y0_f;
    }

    for y_new in 0..dst_height {
        let y0 = y0_coords[y_new];
        let y1 = y1_coords[y_new];
        let dy = dy_weights[y_new];

        let y0_clamped = y0.clamp(0, src_height as i32 - 1) as usize;
        let y1_clamped = y1.clamp(0, src_height as i32 - 1) as usize;

        let row0_base = y0_clamped * src_width;
        let row1_base = y1_clamped * src_width;
        let dst_row_base = y_new * dst_width;

        let mut x_new = 0;

        // Process 4 pixels at a time using SIMD
        while x_new + 4 <= dst_width {
            let dx0 = dx_weights[x_new];
            let dx1 = dx_weights[x_new + 1];
            let dx2 = dx_weights[x_new + 2];
            let dx3 = dx_weights[x_new + 3];

            let idx0 = 1.0 - dx0;
            let idx1 = 1.0 - dx1;
            let idx2 = 1.0 - dx2;
            let idx3 = 1.0 - dx3;

            let idy = 1.0 - dy;
            let dy_vec = vdupq_n_f32(dy);
            let idy_vec = vdupq_n_f32(idy);

            // Load 4 pairs of corner values
            let mut v00_vals = [0f32; 4];
            let mut v10_vals = [0f32; 4];
            let mut v01_vals = [0f32; 4];
            let mut v11_vals = [0f32; 4];

            for i in 0..4 {
                let x0 = x0_coords[x_new + i].clamp(0, src_width as i32 - 1) as usize;
                let x1 = x1_coords[x_new + i].clamp(0, src_width as i32 - 1) as usize;

                v00_vals[i] = src[row0_base + x0] as f32;
                v10_vals[i] = src[row0_base + x1] as f32;
                v01_vals[i] = src[row1_base + x0] as f32;
                v11_vals[i] = src[row1_base + x1] as f32;
            }

            // Build SIMD vectors
            let v00 = vld1q_f32(v00_vals.as_ptr());
            let v10 = vld1q_f32(v10_vals.as_ptr());
            let v01 = vld1q_f32(v01_vals.as_ptr());
            let v11 = vld1q_f32(v11_vals.as_ptr());

            let dx = vld1q_f32([dx0, dx1, dx2, dx3].as_ptr());
            let idx = vld1q_f32([idx0, idx1, idx2, idx3].as_ptr());

            // Bilinear interpolation using SIMD
            let top = vfmaq_f32(vmulq_f32(v00, idx), v10, dx);
            let bottom = vfmaq_f32(vmulq_f32(v01, idx), v11, dx);
            let result = vfmaq_f32(vmulq_f32(top, idy_vec), bottom, dy_vec);

            // Clamp and convert to u8
            let clamped = vmaxq_f32(vdupq_n_f32(0.0), vminq_f32(vdupq_n_f32(255.0), result));

            // Store directly
            let dst_ptr = dst.as_mut_ptr().add(dst_row_base + x_new);
            *dst_ptr.add(0) = vgetq_lane_f32(clamped, 0).clamp(0.0, 255.0) as u8;
            *dst_ptr.add(1) = vgetq_lane_f32(clamped, 1).clamp(0.0, 255.0) as u8;
            *dst_ptr.add(2) = vgetq_lane_f32(clamped, 2).clamp(0.0, 255.0) as u8;
            *dst_ptr.add(3) = vgetq_lane_f32(clamped, 3).clamp(0.0, 255.0) as u8;

            x_new += 4;
        }

        // Handle remaining pixels
        while x_new < dst_width {
            let x0 = x0_coords[x_new].clamp(0, src_width as i32 - 1) as usize;
            let x1 = x1_coords[x_new].clamp(0, src_width as i32 - 1) as usize;
            let dx = dx_weights[x_new];

            let v00 = src[row0_base + x0] as f32;
            let v10 = src[row0_base + x1] as f32;
            let v01 = src[row1_base + x0] as f32;
            let v11 = src[row1_base + x1] as f32;

            let top = v00 * (1.0 - dx) + v10 * dx;
            let bottom = v01 * (1.0 - dx) + v11 * dx;
            let result = top * (1.0 - dy) + bottom * dy;

            dst[dst_row_base + x_new] = result.clamp(0.0, 255.0) as u8;
            x_new += 1;
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

        assert_eq!(dst[0], 10); // top-left (0,0)
        assert_eq!(dst[1], 20); // (1,0) -> index 1
        assert_eq!(dst[4], 30); // (0,1) -> index 4
        assert_eq!(dst[15], 40); // bottom-right (3,3)
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
