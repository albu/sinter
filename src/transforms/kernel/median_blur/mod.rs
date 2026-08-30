// Median Blur transform
//
// Non-linear smoothing filter that replaces each pixel with the median of its neighborhood.
// Excellent for removing salt-and-pepper noise while preserving edges.

mod fast;
mod huang;
mod histogram;
#[cfg(target_arch = "aarch64")]
mod neon;
pub mod sorting_network;

use crate::core::{AccessPattern, BarrierImage, Executable, FusableImage, ShapeEffect, Transform};

pub use fast::{clipped_mean_3x3_scalar, median_edge_fast};

/// Kernel size for median blur
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MedianKernelSize {
    Kernel3,
    Kernel5,
}

/// Median blur uses high-performance branchless sorting networks (SIMD on ARM/x86)
/// or sliding-window histogram.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MedianMode {
    Fast,
}

impl Default for MedianMode {
    fn default() -> Self {
        MedianMode::Fast
    }
}

/// Median Blur transform
///
/// Non-linear smoothing filter that replaces each pixel with the median
/// value of pixels in its neighborhood. Unlike Gaussian blur, median blur
/// preserves edges while removing noise.
///
/// # Implementation
///
/// - **3x3 kernel**: Uses an exact 19-comparator branchless sorting network
///   with ARM NEON SIMD vectorization (processing 16 pixels per instruction).
/// - **5x5 kernel**: Uses multi-pass or sliding column-histogram algorithm.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MedianBlur {
    pub kernel_size: MedianKernelSize,
    pub mode: MedianMode,
}

impl MedianBlur {
    /// Create a new MedianBlur transform with specified mode
    pub fn new(kernel_size: MedianKernelSize) -> Self {
        Self {
            kernel_size,
            mode: MedianMode::default(),
        }
    }

    /// Create a new MedianBlur transform with specified mode
    pub fn with_mode(kernel_size: MedianKernelSize, mode: MedianMode) -> Self {
        Self { kernel_size, mode }
    }

    /// Create a 3x3 median blur
    pub fn kernel3() -> Self {
        Self {
            kernel_size: MedianKernelSize::Kernel3,
            mode: MedianMode::default(),
        }
    }

    /// Create a 5x5 median blur
    pub fn kernel5() -> Self {
        Self {
            kernel_size: MedianKernelSize::Kernel5,
            mode: MedianMode::Fast,
        }
    }

    /// Native Rust execution path
    fn execute_native(&self, image: &mut FusableImage) {
        match self.kernel_size {
            MedianKernelSize::Kernel3 => {
                #[cfg(target_arch = "aarch64")]
                unsafe {
                    neon::apply_median_blur_3x3_neon(
                        &mut image.data,
                        image.width,
                        image.height,
                        image.channels,
                    );
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    sorting_network::apply_median_blur_3x3_scalar(
                        &mut image.data,
                        image.width,
                        image.height,
                        image.channels,
                    );
                }
            }
            MedianKernelSize::Kernel5 => {
                #[cfg(target_arch = "aarch64")]
                unsafe {
                    neon::apply_median_blur_5x5_neon(
                        &mut image.data,
                        image.width,
                        image.height,
                        image.channels,
                    );
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    histogram::apply_median_blur_5x5(image);
                }
            }
        }
    }

    fn execute_3x3(&self, image: &mut FusableImage) {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon::apply_median_blur_3x3_neon(
                &mut image.data,
                image.width,
                image.height,
                image.channels,
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            sorting_network::apply_median_blur_3x3_scalar(
                &mut image.data,
                image.width,
                image.height,
                image.channels,
            );
        }
    }
}

impl Transform for MedianBlur {
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

impl Executable for MedianBlur {
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        self.execute_native(image);
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_median_blur_new() {
        let m = MedianBlur::new(MedianKernelSize::Kernel3);
        assert_eq!(m.kernel_size, MedianKernelSize::Kernel3);
        assert_eq!(m.mode, MedianMode::Fast);
    }

    #[test]
    fn test_median_blur_with_mode() {
        let m = MedianBlur::with_mode(MedianKernelSize::Kernel3, MedianMode::Fast);
        assert_eq!(m.kernel_size, MedianKernelSize::Kernel3);
        assert_eq!(m.mode, MedianMode::Fast);
    }

    #[test]
    fn test_median_blur_kernel3() {
        let m = MedianBlur::kernel3();
        assert_eq!(m.kernel_size, MedianKernelSize::Kernel3);
        assert_eq!(m.mode, MedianMode::Fast);
    }

    #[test]
    fn test_median_blur_kernel5() {
        let m = MedianBlur::kernel5();
        assert_eq!(m.kernel_size, MedianKernelSize::Kernel5);
        assert_eq!(m.mode, MedianMode::Fast);
    }

    #[test]
    fn test_median_blur_3x3_constant() {
        let mut data = vec![128u8; 9];
        let mut img = FusableImage::new(&mut data, 3, 3, 1);

        let m = MedianBlur::kernel3();
        m.execute(&mut img);

        assert!(img.data.iter().all(|&x| x == 128));
    }

    #[test]
    fn test_median_blur_3x3_salt_pepper() {
        let mut data = vec![128u8, 128u8, 128u8, 128u8, 0u8, 128u8, 128u8, 128u8, 128u8];
        let mut img = FusableImage::new(&mut data, 3, 3, 1);

        let m = MedianBlur::kernel3();
        m.execute(&mut img);

        // Salt-pepper noise should be removed (center should be ~128)
        assert!((img.data[4] as i16 - 128).abs() < 30);
    }

    #[test]
    fn test_median_blur_3x3_preserves_edges() {
        let mut data = vec![0u8, 0u8, 255u8, 0u8, 0u8, 255u8, 0u8, 0u8, 255u8];
        let mut img = FusableImage::new(&mut data, 3, 3, 1);

        let m = MedianBlur::kernel3();
        m.execute(&mut img);

        // Sliding histogram computes exact median:
        // For [0,0,0,0,0,0,255,255,255], median is 0 (5th element)
        // - Left/center pixels should be 0 (median favors zeros)
        // - Right side pixels should be 255 (median favors 255s on right edge)
        assert!(
            img.data[0] < 50,
            "Left side should be low, got {}",
            img.data[0]
        );
        assert!(
            img.data[4] < 50,
            "Center should be low (median of mostly zeros), got {}",
            img.data[4]
        );
        assert!(
            img.data[8] > 200,
            "Right side should be high (median of mostly 255s), got {}",
            img.data[8]
        );
    }

    #[test]
    fn test_median_blur_5x5_constant() {
        let mut data = vec![100u8; 25];
        let mut img = FusableImage::new(&mut data, 5, 5, 1);

        let m = MedianBlur::kernel5();
        m.execute(&mut img);

        assert!(img.data.iter().all(|&x| x == 100));
    }

    #[test]
    fn test_median_blur_5x5_salt_pepper() {
        let mut data = vec![128u8; 25];
        data[12] = 0;

        let mut img = FusableImage::new(&mut data, 5, 5, 1);

        let m = MedianBlur::kernel5();
        m.execute(&mut img);

        // The double 3x3 application should affect the center pixel
        assert_ne!(img.data[12], 0);
    }

    #[test]
    fn test_median_blur_rgb() {
        // Create a 3x3 RGB image with salt-pepper noise in center
        let mut data = vec![
            128u8, 128u8, 128u8, 128u8, 128u8, 128u8, 128u8, 128u8, 128u8, 128u8, 128u8, 128u8,
            128u8, 0u8, 255u8, 128u8, 128u8, 128u8, 128u8, 128u8, 128u8, 128u8, 128u8, 128u8,
            128u8, 128u8, 128u8,
        ];
        let mut img = FusableImage::new(&mut data, 3, 3, 3);

        let m = MedianBlur::kernel3();
        m.execute(&mut img);

        // Center pixel is at index: (1 * 3 + 1) * 3 = 12
        // RGB channels are at indices 12, 13, 14
        // Salt-pepper should be removed - all channels should be close to 128
        assert!((img.data[12] as i16 - 128).abs() < 30);
        assert!((img.data[13] as i16 - 128).abs() < 30);
        assert!((img.data[14] as i16 - 128).abs() < 30);
    }

    #[test]
    fn test_median_blur_access_pattern() {
        let m = MedianBlur::kernel3();
        assert_eq!(m.access(), AccessPattern::InPlace);
        assert_eq!(m.shape_effect(), ShapeEffect::Preserve);
    }

    #[test]
    fn test_median_blur_larger_image() {
        let mut data = vec![128u8; 100];
        data[10] = 0;
        data[50] = 255;
        data[90] = 0;

        let mut img = FusableImage::new(&mut data, 10, 10, 1);

        let m = MedianBlur::kernel3();
        m.execute(&mut img);

        // Salt-pepper noise should be removed
        assert_ne!(img.data[10], 0);
        assert_ne!(img.data[50], 255);
        assert_ne!(img.data[90], 0);
    }
}
