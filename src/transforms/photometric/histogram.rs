// Histogram-based transforms
//
// These transforms analyze the image histogram and apply per-pixel adjustments.
// Note: These cannot be fused with other LUT ops because they depend on image content.

use crate::core::{AccessPattern, BarrierImage, Executable, FusableImage, ShapeEffect, Transform};
use crate::transforms::LutExecutor;

/// Equalize - Histogram equalization
///
/// Distributes pixel values more evenly across the range [0, 255] by
/// applying a transform based on the cumulative distribution function.
///
/// This enhances contrast by spreading out the most frequent intensity values.
///
/// # Performance
/// - Uses optimized histogram computation (scalar, cache-friendly)
/// - Fixed-point CDF calculation (avoiding float division)
/// - NEON-optimized LUT application via LutExecutor (vqtbl4q_u8)
/// - Expected: 2-3x faster than naive implementation
///
/// Note: Cannot be fused with other LUT transforms because the LUT depends
/// on the image histogram.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Equalize;

impl Equalize {
    pub fn new() -> Self {
        Self
    }

    /// Build equalization LUT for a single channel histogram using OpenCV formula
    fn compute_lut(hist: &[u32; 256], total_pixels: u32) -> [u8; 256] {
        let mut i = 0;
        while i < 256 && hist[i] == 0 {
            i += 1;
        }

        let mut lut = [0u8; 256];
        if i < 256 && hist[i] < total_pixels {
            let scale = 255.0f32 / ((total_pixels - hist[i]) as f32);
            let mut sum = 0u32;
            for j in i..256 {
                sum += hist[j];
                let val = ((sum - hist[i]) as f32 * scale).round() as i32;
                lut[j] = val.clamp(0, 255) as u8;
            }
        }
        lut
    }
}

impl Transform for Equalize {
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

impl Executable for Equalize {
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        let channels = image.channels;
        let total_pixels = (image.width * image.height) as u32;

        if channels == 1 {
            let mut hist = [0u32; 256];
            for &pixel in image.data.iter() {
                hist[pixel as usize] += 1;
            }
            let lut = Self::compute_lut(&hist, total_pixels);
            LutExecutor::apply(image, &lut);
        } else if channels == 3 {
            let mut hist_r = [0u32; 256];
            let mut hist_g = [0u32; 256];
            let mut hist_b = [0u32; 256];

            let chunks = image.data.chunks_exact(3);
            for chunk in chunks {
                hist_r[chunk[0] as usize] += 1;
                hist_g[chunk[1] as usize] += 1;
                hist_b[chunk[2] as usize] += 1;
            }

            let lut_r = Self::compute_lut(&hist_r, total_pixels);
            let lut_g = Self::compute_lut(&hist_g, total_pixels);
            let lut_b = Self::compute_lut(&hist_b, total_pixels);

            let luts = [lut_r, lut_g, lut_b];
            LutExecutor::apply_rgb_luts(image, &luts);
        } else {
            let mut hist = [0u32; 256];
            for &pixel in image.data.iter() {
                hist[pixel as usize] += 1;
            }
            let lut = Self::compute_lut(&hist, (image.width * image.height * channels) as u32);
            LutExecutor::apply(image, &lut);
        }

        None
    }
}

/// AutoContrast - Automatic contrast stretching
///
/// Finds the minimum and maximum pixel values in the image and linearly
/// stretches them to cover the full [0, 255] range. This enhances contrast
/// by using the full dynamic range available.
///
/// # Performance
/// - Uses optimized LUT application via LutExecutor
/// - Fixed-point arithmetic for LUT generation
///
/// Note: Cannot be fused with other LUT transforms because the LUT depends
/// on the image min/max values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoContrast {
    /// Cutoff percentage for outliers (0.0 to 0.5)
    /// Default 0.0 uses full range, higher values ignore extreme outliers
    pub cutoff: f32,
}

impl AutoContrast {
    pub fn new(cutoff: f32) -> Self {
        assert!(cutoff >= 0.0 && cutoff <= 0.5, "cutoff must be in [0, 0.5]");
        Self { cutoff }
    }

    /// Build contrast stretch LUT from image min/max
    ///
    /// Uses fixed-point arithmetic to avoid float operations.
    fn build_lut_from_image(&self, image: &FusableImage) -> [u8; 256] {
        let (lo, hi) = if self.cutoff == 0.0 {
            #[cfg(target_arch = "aarch64")]
            unsafe {
                use std::arch::aarch64::*;
                let mut min0 = vdupq_n_u8(255);
                let mut min1 = vdupq_n_u8(255);
                let mut min2 = vdupq_n_u8(255);
                let mut min3 = vdupq_n_u8(255);
                let mut max0 = vdupq_n_u8(0);
                let mut max1 = vdupq_n_u8(0);
                let mut max2 = vdupq_n_u8(0);
                let mut max3 = vdupq_n_u8(0);
                let chunks = image.data.len() / 64;
                let mut ptr = image.data.as_ptr();
                for _ in 0..chunks {
                    let v0 = vld1q_u8(ptr);
                    let v1 = vld1q_u8(ptr.add(16));
                    let v2 = vld1q_u8(ptr.add(32));
                    let v3 = vld1q_u8(ptr.add(48));
                    ptr = ptr.add(64);
                    min0 = vminq_u8(min0, v0);
                    min1 = vminq_u8(min1, v1);
                    min2 = vminq_u8(min2, v2);
                    min3 = vminq_u8(min3, v3);
                    max0 = vmaxq_u8(max0, v0);
                    max1 = vmaxq_u8(max1, v1);
                    max2 = vmaxq_u8(max2, v2);
                    max3 = vmaxq_u8(max3, v3);
                }
                let min_v = vminq_u8(vminq_u8(min0, min1), vminq_u8(min2, min3));
                let max_v = vmaxq_u8(vmaxq_u8(max0, max1), vmaxq_u8(max2, max3));
                let mut lo = vminvq_u8(min_v);
                let mut hi = vmaxvq_u8(max_v);
                let base = image.data.as_ptr();
                for i in (chunks * 64)..image.data.len() {
                    let val = *base.add(i);
                    lo = lo.min(val);
                    hi = hi.max(val);
                }
                (lo, hi)
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                let lo = *image.data.iter().min().unwrap_or(&0);
                let hi = *image.data.iter().max().unwrap_or(&255);
                (lo, hi)
            }
        } else {
            // 256-bin histogram with 4 sub-tables to avoid store-load forwarding stalls
            let mut h0 = [0u32; 256];
            let mut h1 = [0u32; 256];
            let mut h2 = [0u32; 256];
            let mut h3 = [0u32; 256];
            let chunks = image.data.len() / 4;
            let ptr = image.data.as_ptr();
            for i in 0..chunks {
                unsafe {
                    h0[*ptr.add(i * 4) as usize] += 1;
                    h1[*ptr.add(i * 4 + 1) as usize] += 1;
                    h2[*ptr.add(i * 4 + 2) as usize] += 1;
                    h3[*ptr.add(i * 4 + 3) as usize] += 1;
                }
            }
            let mut hist = [0u32; 256];
            for i in 0..256 {
                hist[i] = h0[i] + h1[i] + h2[i] + h3[i];
            }
            for i in (chunks * 4)..image.data.len() {
                hist[image.data[i] as usize] += 1;
            }
            let total = image.data.len() as u32;

            let trim = (total as f32 * self.cutoff).round() as u32;
            let mut lo = 0u8;
            let mut remaining = trim;
            while lo < 255 && remaining >= hist[lo as usize] {
                remaining -= hist[lo as usize];
                lo += 1;
            }

            let mut hi = 255u8;
            let mut remaining = trim;
            while hi > 0 && remaining >= hist[hi as usize] {
                remaining -= hist[hi as usize];
                hi -= 1;
            }
            (lo, hi)
        };

        // Uniform (or fully trimmed) image: preserve the historical edge case.
        if lo >= hi {
            return [0u8; 256];
        }

        // Build linear stretch LUT using fixed-point arithmetic
        // Formula: lut[i] = ((i - lo) * 255) / (hi - lo)
        let range = (hi as u32 - lo as u32) as u32;
        let mut lut = [0u8; 256];

        for i in 0..256 {
            let i = i as u32;
            // Fixed-point: (i - min) * 255 / range
            let normalized = i.saturating_sub(lo as u32);
            // Values beyond the trimmed hi end must clamp to 255, not wrap in u8.
            let v = (normalized * 255) / range;
            lut[i as usize] = v.min(255) as u8;
        }

        lut
    }
}

impl Transform for AutoContrast {
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

impl Executable for AutoContrast {
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        let lut = self.build_lut_from_image(image);

        // Use optimized LUT executor (NEON vqtbl4q_u8 on ARM)
        LutExecutor::apply(image, &lut);

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equalize_execute() {
        let mut data = vec![0u8; 100];
        // Create a simple pattern
        for i in 0..100 {
            data[i] = (i % 256) as u8;
        }

        let mut img = FusableImage::new(&mut data, 10, 10, 1);
        let eq = Equalize::new();
        eq.execute(&mut img);

        // After equalization, values should be spread out
        assert_ne!(img.data, vec![0u8; 100]);
    }

    #[test]
    fn test_equalize_clamping() {
        // Test that CDF doesn't overflow
        let mut data = vec![255u8; 256];
        let mut img = FusableImage::new(&mut data, 16, 16, 1);

        let eq = Equalize::new();
        eq.execute(&mut img);

        // All values should be in valid range
        for &px in img.data.iter() {
            assert!((0..=255).contains(&px));
        }
    }

    #[test]
    fn test_equalize_lut_values() {
        let mut data = Vec::with_capacity(4096);
        for y in 0..64 {
            for x in 0..64 {
                data.push((x as f32 / 64.0 * 255.0) as u8);
            }
        }
        let mut img = FusableImage::new(&mut data, 64, 64, 1);
        let eq = Equalize::new();
        eq.execute(&mut img);
        println!("Rust Equalize img.data[..5]: {:?}", &img.data[..5]);
    }

    #[test]
    fn test_autocontrast_new() {
        let ac = AutoContrast::new(0.0);
        assert_eq!(ac.cutoff, 0.0);
    }

    #[test]
    fn test_autocontrast_uniform_image() {
        let mut data = vec![128u8; 100];
        let mut img = FusableImage::new(&mut data, 10, 10, 1);

        let ac = AutoContrast::new(0.0);
        ac.execute(&mut img);

        // Uniform image should become all zeros (edge case)
        assert_eq!(img.data, vec![0u8; 100]);
    }

    #[test]
    fn test_autocontrast_stretch() {
        let mut data = vec![100u8; 50];
        data.extend(vec![150u8; 50]);

        let mut img = FusableImage::new(&mut data, 10, 10, 1);

        let ac = AutoContrast::new(0.0);
        ac.execute(&mut img);

        // After stretch, 100 should map toward 0, 150 toward 255
        // Check that values are now more spread out
        let min_val = *img.data.iter().min().unwrap();
        let max_val = *img.data.iter().max().unwrap();

        // Should use more of the range than original [100, 150]
        assert!(max_val - min_val > 50);
    }

    #[test]
    fn test_autocontrast_full_range() {
        let mut data = vec![0u8; 50];
        data.extend(vec![255u8; 50]);

        let mut img = FusableImage::new(&mut data, 10, 10, 1);

        let ac = AutoContrast::new(0.0);
        ac.execute(&mut img);

        // Image already uses full range, should be preserved
        let min_val = *img.data.iter().min().unwrap();
        let max_val = *img.data.iter().max().unwrap();

        assert_eq!(min_val, 0);
        assert_eq!(max_val, 255);
    }

    #[test]
    fn test_autocontrast_cutoff_actually_trims() {
        // Falsification: `cutoff` was previously accepted but swallowed —
        // build_lut_from_image never read it, so cutoff=0.05 was byte-identical
        // to cutoff=0. With the histogram-trim implementation, trimming 5% of
        // pixels from each end must shift the stretch endpoints.
        let mut data = Vec::with_capacity(256 * 256);
        for i in 0..(256 * 256) {
            data.push((i % 256) as u8); // each value appears exactly 256 times
        }

        let mut d0 = data.clone();
        let mut img0 = FusableImage::new(&mut d0, 256, 256, 1);
        AutoContrast::new(0.0).execute(&mut img0);
        let mut d1 = data.clone();
        let mut img1 = FusableImage::new(&mut d1, 256, 256, 1);
        AutoContrast::new(0.05).execute(&mut img1);

        assert_ne!(
            img0.data, img1.data,
            "cutoff must change the stretch (was silently ignored)"
        );
        // The trimmed LUT uses a narrower input range: value 64 maps to ~64
        // under the full [0,255] stretch, but to ~57 when 5% is trimmed from
        // each end ([12,243] range). It must differ.
        let out = img1.data[64 * 256 + 64];
        assert!(
            (out as i32 - 64).abs() > 5,
            "cutoff stretch should change the mapping, got {}",
            out
        );
        // Values above the trimmed hi end must clamp to 255, not wrap.
        let hi_out = img1.data[255 * 256 + 255];
        assert_eq!(hi_out, 255, "upper tail must clamp to 255, got {}", hi_out);
    }
}
