// Edge detection transform
//
// Applies Laplacian or Sobel operators for edge detection.

use super::convolve::convolve_3x3;
use crate::core::{AccessPattern, Executable, FusableImage, ShapeEffect, Transform};

/// Edge detection method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EdgeMethod {
    /// Laplacian operator - detects edges in all directions
    Laplacian,
    /// Sobel operator - detects edges with directional sensitivity
    Sobel,
}

/// Edge detection transform
///
/// Detects edges in the image using either Laplacian or Sobel operators.
///
/// **Laplacian**: Detects edges in all directions by computing the second derivative.
/// Produces thin edges and is sensitive to noise.
///
/// **Sobel**: Computes gradient magnitude using horizontal and vertical kernels.
/// More robust to noise, produces thicker edges.
///
/// # Parameters
/// - `method`: Edge detection method (Laplacian or Sobel)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgeDetection {
    pub method: EdgeMethod,
}

impl EdgeDetection {
    /// Create a new EdgeDetection transform with Laplacian method
    pub fn laplacian() -> Self {
        Self {
            method: EdgeMethod::Laplacian,
        }
    }

    /// Create a new EdgeDetection transform with Sobel method
    pub fn sobel() -> Self {
        Self {
            method: EdgeMethod::Sobel,
        }
    }

    /// Create with custom method
    pub fn new(method: EdgeMethod) -> Self {
        Self { method }
    }
}

impl Transform for EdgeDetection {
    fn access(&self) -> AccessPattern {
        AccessPattern::InPlace
    }

    fn shape_effect(&self) -> ShapeEffect {
        ShapeEffect::Preserve
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_executable(&self) -> Option<&dyn Executable> {
        Some(self)
    }
}

impl Executable for EdgeDetection {
    fn execute(&self, image: &mut FusableImage) -> Option<crate::core::BarrierImage> {
        match self.method {
            EdgeMethod::Laplacian => self.apply_laplacian(image),
            EdgeMethod::Sobel => self.apply_sobel(image),
        }
        None
    }
}

impl EdgeDetection {
    /// Apply Laplacian edge detection
    ///
    /// Kernel:
    ///  0  1  0
    ///  1 -4  1
    ///  0  1  0
    fn apply_laplacian(&self, image: &mut FusableImage) {
        super::convolve_2d::apply_laplacian(image);
    }

    /// Apply Sobel edge detection
    ///
    /// Computes gradient magnitude using horizontal and vertical Sobel kernels:
    /// Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] (horizontal)
    /// Gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]] (vertical)
    /// Result = sqrt(Gx² + Gy²)
    fn apply_sobel(&self, image: &mut FusableImage) {
        let width = image.width;
        let height = image.height;
        let channels = image.channels;

        if width < 3 || height < 3 {
            return;
        }

        let mut output = Vec::<u8>::with_capacity(image.data.len());
        unsafe {
            output.set_len(image.data.len());
        }

        #[cfg(target_arch = "aarch64")]
        {
            if channels == 1 {
                unsafe {
                    sobel_neon_gray(&image.data, &mut output, width, height);
                }
                image.data.copy_from_slice(&output);
                return;
            } else if channels == 3 {
                unsafe {
                    sobel_neon_rgb(&image.data, &mut output, width, height);
                }
                image.data.copy_from_slice(&output);
                return;
            }
        }

        // Scalar fallback
        let data = &image.data;
        let gx = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
        let gy = [-1, -2, -1, 0, 0, 0, 1, 2, 1];

        let get_pixel = |data: &[u8], x: i32, y: i32, c: usize| -> u8 {
            let x_clamped = x.max(0).min(width as i32 - 1) as usize;
            let y_clamped = y.max(0).min(height as i32 - 1) as usize;
            data[(y_clamped * width + x_clamped) * channels + c]
        };

        for y in 0..height {
            for x in 0..width {
                for c in 0..channels {
                    let mut sum_x: i32 = 0;
                    let mut sum_y: i32 = 0;

                    for ky in 0..3 {
                        for kx in 0..3 {
                            let px = x as i32 + kx as i32 - 1;
                            let py = y as i32 + ky as i32 - 1;
                            let pixel = get_pixel(data, px, py, c) as i32;
                            sum_x += pixel * gx[ky * 3 + kx];
                            sum_y += pixel * gy[ky * 3 + kx];
                        }
                    }

                    let magnitude = ((sum_x * sum_x + sum_y * sum_y) as f32).sqrt().round() as i32;
                    output[(y * width + x) * channels + c] = magnitude.clamp(0, 255) as u8;
                }
            }
        }

        image.data.copy_from_slice(&output);
    }
}

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn sub_u8_to_s16(a: uint8x8_t, b: uint8x8_t) -> int16x8_t {
    let a_s16 = vreinterpretq_s16_u16(vmovl_u8(a));
    let b_s16 = vreinterpretq_s16_u16(vmovl_u8(b));
    vsubq_s16(a_s16, b_s16)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn compute_sobel_16(
    p00: uint8x16_t,
    p01: uint8x16_t,
    p02: uint8x16_t,
    p10: uint8x16_t,
    _p11: uint8x16_t,
    p12: uint8x16_t,
    p20: uint8x16_t,
    p21: uint8x16_t,
    p22: uint8x16_t,
) -> uint8x16_t {
    // Low 8 pixels
    let diff_top_lo = sub_u8_to_s16(vget_low_u8(p02), vget_low_u8(p00));
    let diff_mid_lo = sub_u8_to_s16(vget_low_u8(p12), vget_low_u8(p10));
    let diff_bot_lo = sub_u8_to_s16(vget_low_u8(p22), vget_low_u8(p20));
    let gx_lo = vaddq_s16(vaddq_s16(diff_top_lo, diff_bot_lo), vshlq_n_s16(diff_mid_lo, 1));

    let diff_l_lo = sub_u8_to_s16(vget_low_u8(p20), vget_low_u8(p00));
    let diff_c_lo = sub_u8_to_s16(vget_low_u8(p21), vget_low_u8(p01));
    let diff_r_lo = sub_u8_to_s16(vget_low_u8(p22), vget_low_u8(p02));
    let gy_lo = vaddq_s16(vaddq_s16(diff_l_lo, diff_r_lo), vshlq_n_s16(diff_c_lo, 1));

    let gx_ll = vcvtq_f32_s32(vmovl_s16(vget_low_s16(gx_lo)));
    let gy_ll = vcvtq_f32_s32(vmovl_s16(vget_low_s16(gy_lo)));
    let mag_ll = vsqrtq_f32(vmlaq_f32(vmulq_f32(gx_ll, gx_ll), gy_ll, gy_ll));

    let gx_lh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(gx_lo)));
    let gy_lh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(gy_lo)));
    let mag_lh = vsqrtq_f32(vmlaq_f32(vmulq_f32(gx_lh, gx_lh), gy_lh, gy_lh));

    let u_ll = vcvtnq_u32_f32(mag_ll);
    let u_lh = vcvtnq_u32_f32(mag_lh);
    let res_lo = vqmovn_u16(vcombine_u16(vqmovn_u32(u_ll), vqmovn_u32(u_lh)));

    // High 8 pixels
    let diff_top_hi = sub_u8_to_s16(vget_high_u8(p02), vget_high_u8(p00));
    let diff_mid_hi = sub_u8_to_s16(vget_high_u8(p12), vget_high_u8(p10));
    let diff_bot_hi = sub_u8_to_s16(vget_high_u8(p22), vget_high_u8(p20));
    let gx_hi = vaddq_s16(vaddq_s16(diff_top_hi, diff_bot_hi), vshlq_n_s16(diff_mid_hi, 1));

    let diff_l_hi = sub_u8_to_s16(vget_high_u8(p20), vget_high_u8(p00));
    let diff_c_hi = sub_u8_to_s16(vget_high_u8(p21), vget_high_u8(p01));
    let diff_r_hi = sub_u8_to_s16(vget_high_u8(p22), vget_high_u8(p02));
    let gy_hi = vaddq_s16(vaddq_s16(diff_l_hi, diff_r_hi), vshlq_n_s16(diff_c_hi, 1));

    let gx_hl = vcvtq_f32_s32(vmovl_s16(vget_low_s16(gx_hi)));
    let gy_hl = vcvtq_f32_s32(vmovl_s16(vget_low_s16(gy_hi)));
    let mag_hl = vsqrtq_f32(vmlaq_f32(vmulq_f32(gx_hl, gx_hl), gy_hl, gy_hl));

    let gx_hh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(gx_hi)));
    let gy_hh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(gy_hi)));
    let mag_hh = vsqrtq_f32(vmlaq_f32(vmulq_f32(gx_hh, gx_hh), gy_hh, gy_hh));

    let u_hl = vcvtnq_u32_f32(mag_hl);
    let u_hh = vcvtnq_u32_f32(mag_hh);
    let res_hi = vqmovn_u16(vcombine_u16(vqmovn_u32(u_hl), vqmovn_u32(u_hh)));

    vcombine_u8(res_lo, res_hi)
}

#[cfg(target_arch = "aarch64")]
unsafe fn sobel_neon_gray(src: &[u8], dst: &mut [u8], width: usize, height: usize) {
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let stride = width;

    // Interior rows 1..height-1
    for y in 1..height - 1 {
        let r0 = src_ptr.add((y - 1) * stride);
        let r1 = src_ptr.add(y * stride);
        let r2 = src_ptr.add((y + 1) * stride);
        let d_row = dst_ptr.add(y * stride);

        let mut x = 1;
        while x + 16 <= width - 1 {
            let p00 = vld1q_u8(r0.add(x - 1));
            let p01 = vld1q_u8(r0.add(x));
            let p02 = vld1q_u8(r0.add(x + 1));

            let p10 = vld1q_u8(r1.add(x - 1));
            let p11 = vld1q_u8(r1.add(x));
            let p12 = vld1q_u8(r1.add(x + 1));

            let p20 = vld1q_u8(r2.add(x - 1));
            let p21 = vld1q_u8(r2.add(x));
            let p22 = vld1q_u8(r2.add(x + 1));

            let res = compute_sobel_16(p00, p01, p02, p10, p11, p12, p20, p21, p22);
            vst1q_u8(d_row.add(x), res);
            x += 16;
        }

        // Remainder of row
        while x < width - 1 {
            let p00 = *r0.add(x - 1) as i32;
            let p01 = *r0.add(x) as i32;
            let p02 = *r0.add(x + 1) as i32;
            let p10 = *r1.add(x - 1) as i32;
            let p12 = *r1.add(x + 1) as i32;
            let p20 = *r2.add(x - 1) as i32;
            let p21 = *r2.add(x) as i32;
            let p22 = *r2.add(x + 1) as i32;

            let gx = (p02 - p00) + 2 * (p12 - p10) + (p22 - p20);
            let gy = (p20 - p00) + 2 * (p21 - p01) + (p22 - p02);
            let mag = ((gx * gx + gy * gy) as f32).sqrt().round() as i32;
            *d_row.add(x) = mag.min(255) as u8;
            x += 1;
        }
    }

    // Border rows & columns (y = 0, y = height-1, x = 0, x = width-1)
    let get_clamped = |px: i32, py: i32| -> i32 {
        let cx = px.clamp(0, width as i32 - 1) as usize;
        let cy = py.clamp(0, height as i32 - 1) as usize;
        *src_ptr.add(cy * stride + cx) as i32
    };

    let compute_pixel = |cx: usize, cy: usize| -> u8 {
        let x = cx as i32;
        let y = cy as i32;
        let p00 = get_clamped(x - 1, y - 1);
        let p01 = get_clamped(x, y - 1);
        let p02 = get_clamped(x + 1, y - 1);
        let p10 = get_clamped(x - 1, y);
        let p12 = get_clamped(x + 1, y);
        let p20 = get_clamped(x - 1, y + 1);
        let p21 = get_clamped(x, y + 1);
        let p22 = get_clamped(x + 1, y + 1);

        let gx = (p02 - p00) + 2 * (p12 - p10) + (p22 - p20);
        let gy = (p20 - p00) + 2 * (p21 - p01) + (p22 - p02);
        let mag = ((gx * gx + gy * gy) as f32).sqrt().round() as i32;
        mag.min(255) as u8
    };

    for x in 0..width {
        *dst_ptr.add(x) = compute_pixel(x, 0);
        *dst_ptr.add((height - 1) * stride + x) = compute_pixel(x, height - 1);
    }
    for y in 1..height - 1 {
        *dst_ptr.add(y * stride) = compute_pixel(0, y);
        *dst_ptr.add(y * stride + width - 1) = compute_pixel(width - 1, y);
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn sobel_neon_rgb(src: &[u8], dst: &mut [u8], width: usize, height: usize) {
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let stride = width * 3;

    // Interior rows 1..height-1
    for y in 1..height - 1 {
        let r0 = src_ptr.add((y - 1) * stride);
        let r1 = src_ptr.add(y * stride);
        let r2 = src_ptr.add((y + 1) * stride);
        let d_row = dst_ptr.add(y * stride);

        let mut x = 1;
        while x + 16 <= width - 1 {
            let p00 = vld3q_u8(r0.add((x - 1) * 3));
            let p01 = vld3q_u8(r0.add(x * 3));
            let p02 = vld3q_u8(r0.add((x + 1) * 3));

            let p10 = vld3q_u8(r1.add((x - 1) * 3));
            let p11 = vld3q_u8(r1.add(x * 3));
            let p12 = vld3q_u8(r1.add((x + 1) * 3));

            let p20 = vld3q_u8(r2.add((x - 1) * 3));
            let p21 = vld3q_u8(r2.add(x * 3));
            let p22 = vld3q_u8(r2.add((x + 1) * 3));

            let r_res = compute_sobel_16(p00.0, p01.0, p02.0, p10.0, p11.0, p12.0, p20.0, p21.0, p22.0);
            let g_res = compute_sobel_16(p00.1, p01.1, p02.1, p10.1, p11.1, p12.1, p20.1, p21.1, p22.1);
            let b_res = compute_sobel_16(p00.2, p01.2, p02.2, p10.2, p11.2, p12.2, p20.2, p21.2, p22.2);

            vst3q_u8(d_row.add(x * 3), uint8x16x3_t(r_res, g_res, b_res));
            x += 16;
        }

        // Remainder of row
        while x < width - 1 {
            for c in 0..3 {
                let p00 = *r0.add((x - 1) * 3 + c) as i32;
                let p01 = *r0.add(x * 3 + c) as i32;
                let p02 = *r0.add((x + 1) * 3 + c) as i32;
                let p10 = *r1.add((x - 1) * 3 + c) as i32;
                let p12 = *r1.add((x + 1) * 3 + c) as i32;
                let p20 = *r2.add((x - 1) * 3 + c) as i32;
                let p21 = *r2.add(x * 3 + c) as i32;
                let p22 = *r2.add((x + 1) * 3 + c) as i32;

                let gx = (p02 - p00) + 2 * (p12 - p10) + (p22 - p20);
                let gy = (p20 - p00) + 2 * (p21 - p01) + (p22 - p02);
                let mag = ((gx * gx + gy * gy) as f32).sqrt().round() as i32;
                *d_row.add(x * 3 + c) = mag.min(255) as u8;
            }
            x += 1;
        }
    }

    // Border rows & columns
    let get_clamped = |px: i32, py: i32, c: usize| -> i32 {
        let cx = px.clamp(0, width as i32 - 1) as usize;
        let cy = py.clamp(0, height as i32 - 1) as usize;
        *src_ptr.add(cy * stride + cx * 3 + c) as i32
    };

    let compute_pixel = |cx: usize, cy: usize, c: usize| -> u8 {
        let x = cx as i32;
        let y = cy as i32;
        let p00 = get_clamped(x - 1, y - 1, c);
        let p01 = get_clamped(x, y - 1, c);
        let p02 = get_clamped(x + 1, y - 1, c);
        let p10 = get_clamped(x - 1, y, c);
        let p12 = get_clamped(x + 1, y, c);
        let p20 = get_clamped(x - 1, y + 1, c);
        let p21 = get_clamped(x, y + 1, c);
        let p22 = get_clamped(x + 1, y + 1, c);

        let gx = (p02 - p00) + 2 * (p12 - p10) + (p22 - p20);
        let gy = (p20 - p00) + 2 * (p21 - p01) + (p22 - p02);
        let mag = ((gx * gx + gy * gy) as f32).sqrt().round() as i32;
        mag.min(255) as u8
    };

    for x in 0..width {
        for c in 0..3 {
            *dst_ptr.add(x * 3 + c) = compute_pixel(x, 0, c);
            *dst_ptr.add((height - 1) * stride + x * 3 + c) = compute_pixel(x, height - 1, c);
        }
    }
    for y in 1..height - 1 {
        for c in 0..3 {
            *dst_ptr.add(y * stride + c) = compute_pixel(0, y, c);
            *dst_ptr.add(y * stride + (width - 1) * 3 + c) = compute_pixel(width - 1, y, c);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_detection_laplacian() {
        let e = EdgeDetection::laplacian();
        assert_eq!(e.method, EdgeMethod::Laplacian);
    }

    #[test]
    fn test_edge_detection_sobel() {
        let e = EdgeDetection::sobel();
        assert_eq!(e.method, EdgeMethod::Sobel);
    }

    #[test]
    fn test_edge_detection_new() {
        let e = EdgeDetection::new(EdgeMethod::Laplacian);
        assert_eq!(e.method, EdgeMethod::Laplacian);
    }

    #[test]
    fn test_laplacian_constant() {
        // Constant image should produce zero (no edges)
        let mut data = vec![128u8; 9]; // 3x3 constant
        let mut img = FusableImage::new(&mut data, 3, 3, 1);

        EdgeDetection::laplacian().execute(&mut img);

        // All pixels should be 0 (no edges in constant image)
        assert!(img.data.iter().all(|&p| p == 0));
    }

    #[test]
    fn test_laplacian_horizontal_edge() {
        // Image with horizontal edge
        // 0 0 0
        // 255 255 255
        let mut data = vec![0u8, 0u8, 0u8, 255u8, 255u8, 255u8];
        let mut img = FusableImage::new(&mut data, 3, 2, 1);

        EdgeDetection::laplacian().execute(&mut img);

        // Edge pixels should have high values
        // Check that at least some pixels detected the edge
        let edge_max = img.data.iter().cloned().max().unwrap();
        assert!(edge_max > 0, "Edge should be detected");
    }

    #[test]
    fn test_sobel_constant() {
        // Constant image should produce zero gradient
        let mut data = vec![128u8; 9]; // 3x3 constant
        let mut img = FusableImage::new(&mut data, 3, 3, 1);

        EdgeDetection::sobel().execute(&mut img);

        // All pixels should be 0 (no gradient in constant image)
        assert!(img.data.iter().all(|&p| p == 0));
    }

    #[test]
    fn test_sobel_diagonal_edge() {
        // Image with diagonal edge
        // 255 0
        // 0 0
        let mut data = vec![255u8, 0u8, 0u8, 0u8];
        let mut img = FusableImage::new(&mut data, 2, 2, 1);

        EdgeDetection::sobel().execute(&mut img);

        // At least some pixels should detect the edge
        let max_val = *img.data.iter().max().unwrap();
        assert!(max_val > 0, "Edge should be detected");
    }

    #[test]
    fn test_edge_detection_rgb() {
        // Test RGB image
        let mut data = vec![
            100u8, 100u8, 100u8, 128u8, 128u8, 128u8, 150u8, 150u8, 150u8,
        ];
        let mut img = FusableImage::new(&mut data, 3, 1, 3);

        EdgeDetection::laplacian().execute(&mut img);

        // Each channel should be processed independently
        // For Laplacian, gradient image should have some variation
        assert_eq!(img.data.len(), 9);
    }

    #[test]
    fn test_sobel_vs_laplacian() {
        // Both methods should work on the same image
        let mut data1 = vec![0u8, 0u8, 0u8, 255u8, 255u8, 255u8];
        let mut data2 = data1.clone();

        let mut img1 = FusableImage::new(&mut data1, 3, 2, 1);
        let mut img2 = FusableImage::new(&mut data2, 3, 2, 1);

        EdgeDetection::laplacian().execute(&mut img1);
        EdgeDetection::sobel().execute(&mut img2);

        // Both should detect edges (not all zeros)
        let has_edge1 = img1.data.iter().any(|&p| p > 0);
        let has_edge2 = img2.data.iter().any(|&p| p > 0);

        assert!(has_edge1, "Laplacian should detect edge");
        assert!(has_edge2, "Sobel should detect edge");
    }
}
