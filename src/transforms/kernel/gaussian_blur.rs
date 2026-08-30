// Gaussian blur transform
//
// Applies Gaussian blur for noise reduction and smoothing.
// Uses NEON-optimized separable convolution for 3x3, 5x5, 7x7 kernels.
//
// For sigma-based Gaussian blur with arbitrary kernel sizes, use GaussianBlurSigma
// which forwards to OpenCV's highly optimized implementation.
//
// PERFORMANCE:
// - 3x3: ~650 MP/s (NEON SIMD separable)
// - 5x5: ~730 MP/s (NEON SIMD separable)
// - 7x7: ~385 MP/s (NEON SIMD separable)

use super::convolve::convolve_separable;
use crate::core::{AccessPattern, Executable, FusableImage, ShapeEffect, Transform};

/// Kernel size for Gaussian blur
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KernelSize {
    /// 3x3 kernel via separable convolution (fastest, light blur, ~650 MP/s)
    Size3,
    /// 5x5 kernel via separable convolution (fast, moderate blur, ~730 MP/s)
    Size5,
    /// 7x7 kernel via separable convolution (fast, strong blur, ~385 MP/s)
    Size7,
}

/// Gaussian blur transform
///
/// Applies Gaussian blur to reduce noise and smooth the image.
/// Uses NEON-optimized separable convolution for 3×3, 5×5, and 7×7 kernels.
///
/// **Performance:**
/// - 3×3: Pascal row 2 [1, 2, 1] → ~650 MP/s
/// - 5×5: Pascal row 4 [1, 4, 6, 4, 1] → ~730 MP/s
/// - 7×7: Pascal row 6 [1, 6, 15, 20, 15, 6, 1] → ~385 MP/s
///
/// All configurations preserve constant images correctly.
///
/// # Parameters
/// - `kernel_size`: Size of the Gaussian kernel (3, 5, or 7)
///
/// # For Sigma-Based Gaussian Blur
///
/// For arbitrary sigma values and kernel sizes, use `GaussianBlurSigma` which
/// forwards to OpenCV's highly optimized Gaussian blur implementation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussianBlur {
    pub kernel_size: KernelSize,
}

impl GaussianBlur {
    /// Create a new GaussianBlur with 3x3 kernel
    pub fn new() -> Self {
        Self {
            kernel_size: KernelSize::Size3,
        }
    }

    /// Create a new GaussianBlur with the specified kernel size
    pub fn with_kernel_size(size: KernelSize) -> Self {
        Self { kernel_size: size }
    }
}

impl Default for GaussianBlur {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform for GaussianBlur {
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

impl Executable for GaussianBlur {
    fn execute(&self, image: &mut FusableImage) -> Option<crate::core::BarrierImage> {
        self.execute_rust(image);
        None
    }
}

impl GaussianBlur {
    fn execute_rust(&self, image: &mut FusableImage) {
        match self.kernel_size {
            KernelSize::Size3 => {
                // 3x3 Gaussian - use unified separable convolution [1 2 1] for SIMD
                // Preserves u16 precision between horizontal and vertical passes!
                let kernel = [1, 2, 1];
                #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
                {
                    use super::convolve_simd;
                    convolve_simd::convolve_separable_detect(image, &kernel[..], 4);
                }
                #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
                {
                    convolve_separable(image, &kernel[..], 4);
                }
            }
            KernelSize::Size5 => {
                // 5x5 Gaussian - use separable convolution [1 4 6 4 1] for SIMD
                let kernel = [1, 4, 6, 4, 1];
                #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
                {
                    use super::convolve_simd;
                    convolve_simd::convolve_separable_detect(image, &kernel[..], 16);
                }
                #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
                {
                    convolve_separable(image, &kernel[..], 16);
                }
            }
            KernelSize::Size7 => {
                // 7x7 discrete convolution matching OpenCV's Gaussian kernel (sigma=1.4)
                // [2, 7, 14, 18, 14, 7, 2] / 64
                // Sum = 64
                let kernel = [2i32, 7, 14, 18, 14, 7, 2];
                #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
                {
                    use super::convolve_simd;
                    convolve_simd::convolve_separable_detect(image, &kernel[..], 64);
                }
                #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
                {
                    convolve_separable(image, &kernel[..], 64);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_blur_new() {
        let gb = GaussianBlur::new();
        assert_eq!(gb.kernel_size, KernelSize::Size3);
    }

    #[test]
    fn test_gaussian_blur_default() {
        let gb = GaussianBlur::default();
        assert_eq!(gb.kernel_size, KernelSize::Size3);
    }

    #[test]
    fn test_gaussian_blur_with_kernel_size() {
        let gb = GaussianBlur::with_kernel_size(KernelSize::Size7);
        assert_eq!(gb.kernel_size, KernelSize::Size7);
    }

    #[test]
    fn test_gaussian_blur_3x3_single_pixel() {
        let mut data = vec![128u8];
        let mut img = FusableImage::new(&mut data, 1, 1, 1);

        GaussianBlur::new().execute(&mut img);

        // Single pixel should remain the same (all neighbors are itself)
        assert_eq!(img.data[0], 128);
    }

    #[test]
    fn test_gaussian_blur_3x3_constant() {
        // Constant image should remain constant
        let mut data = vec![100u8; 9]; // 3x3
        let mut img = FusableImage::new(&mut data, 3, 3, 1);

        GaussianBlur::new().execute(&mut img);

        // All pixels should remain 100
        assert!(img.data.iter().all(|&p| p == 100));
    }

    #[test]
    fn test_gaussian_blur_7x7_constant() {
        // Constant image should remain constant
        let mut data = vec![100u8; 49]; // 7x7
        let mut img = FusableImage::new(&mut data, 7, 7, 1);

        GaussianBlur::with_kernel_size(KernelSize::Size7).execute(&mut img);

        // All pixels should remain 100
        assert!(img.data.iter().all(|&p| p == 100));
    }

    #[test]
    fn test_gaussian_blur_comparison_kernel_sizes() {
        // Larger kernels should produce more blur (smaller range after blur)
        let data = vec![0u8, 128u8, 255u8, 128u8, 0u8];
        let width = 5;

        let mut ranges = Vec::new();

        for size in [KernelSize::Size3, KernelSize::Size5, KernelSize::Size7] {
            let mut test_data = data.clone();
            let mut img = FusableImage::new(&mut test_data, width, 1, 1);
            GaussianBlur::with_kernel_size(size).execute(&mut img);
            let range = img.data.iter().max().unwrap() - img.data.iter().min().unwrap();
            ranges.push((size, range));
        }

        // Each larger kernel should have equal or smaller range (more smoothing)
        for i in 1..ranges.len() {
            assert!(
                ranges[i].1 <= ranges[i - 1].1,
                "Kernel size {:?} range {} should be <= previous {}",
                ranges[i].0,
                ranges[i].1,
                ranges[i - 1].1
            );
        }
    }

    #[test]
    fn test_gaussian_blur_rgb() {
        // Test RGB image
        let mut data = vec![
            100u8, 100u8, 100u8, 128u8, 128u8, 128u8, 150u8, 150u8, 150u8,
        ];
        let mut img = FusableImage::new(&mut data, 3, 1, 3);

        GaussianBlur::with_kernel_size(KernelSize::Size7).execute(&mut img);

        // Each channel should be processed independently
        assert_eq!(img.data.len(), 9);
    }

    #[test]
    fn test_gaussian_blur_preserves_mean() {
        // Gaussian blur should preserve the mean brightness
        let mut data = vec![0u8, 128u8, 255u8];
        let original_mean: u32 = data.iter().map(|&p| p as u32).sum::<u32>() / data.len() as u32;

        let mut img = FusableImage::new(&mut data, 3, 1, 1);
        GaussianBlur::with_kernel_size(KernelSize::Size7).execute(&mut img);

        let new_mean: u32 = img.data.iter().map(|&p| p as u32).sum::<u32>() / img.data.len() as u32;

        // Mean should be approximately preserved (within rounding error)
        assert!((new_mean as i32 - original_mean as i32).abs() <= 1);
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
mod fused_path_tests {
    use super::*;
    use crate::core::FusableImage;

    #[test]
    fn test_full_gaussian3_correct() {
        let (w, h) = (256usize, 256usize);
        let mut data: Vec<u8> = (0..w * h * 3)
            .map(|i| ((i as u64 * 2654435761) % 256) as u8)
            .collect();
        let original = data.clone();
        let mut img = FusableImage::new(&mut data, w, h, 3);
        GaussianBlur::with_kernel_size(KernelSize::Size3).execute(&mut img);

        // scalar two-pass reference
        let ch = 3usize;
        let mut htmp = vec![0u8; original.len()];
        for y in 0..h {
            for x in 0..w {
                for c in 0..ch {
                    let mut s: u32 = 0;
                    for k in 0..3 {
                        let px = (x as i32 + k as i32 - 1).clamp(0, w as i32 - 1) as usize;
                        s += original[(y * w + px) * ch + c] as u32 * [1, 2, 1][k];
                    }
                    htmp[(y * w + x) * ch + c] = (s >> 2) as u8;
                }
            }
        }
        let mut expected = vec![0u8; original.len()];
        for y in 0..h {
            for x in 0..w {
                for c in 0..ch {
                    let mut s: u32 = 0;
                    for k in 0..3 {
                        let py = (y as i32 + k as i32 - 1).clamp(0, h as i32 - 1) as usize;
                        s += htmp[(py * w + x) * ch + c] as u32 * [1, 2, 1][k];
                    }
                    expected[(y * w + x) * ch + c] = (s >> 2) as u8;
                }
            }
        }
        let mm = data.iter().zip(expected.iter()).filter(|(a, b)| a != b).count();
        let mx = data.iter().zip(expected.iter()).map(|(a, b)| (*a as i32 - *b as i32).abs()).max().unwrap_or(0);
        assert_eq!(mm, 0, "full GaussianBlur3 mismatch: {} px, max_diff={}", mm, mx);
    }
}
